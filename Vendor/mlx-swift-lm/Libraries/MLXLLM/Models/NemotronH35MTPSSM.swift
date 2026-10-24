// Short captured windows with the canonical M=1 SSM arithmetic at every step.
// All prefix states are written in FP32; no parallel/prefill recurrence is used.
import Foundation
import MLX
import MLXLMCommon

private enum NemotronMTPSSMKernel {
    static let kernel = MLXFast.metalKernel(
        name: "nemotron_mtp_ssm_window",
        inputNames: ["X", "A_log", "B", "C", "D", "dt", "state_in"],
        outputNames: ["out", "state_out"],
        source: """
            auto n = thread_position_in_grid.z;
            auto h_idx = n % H;
            auto g_idx = n / G;
            constexpr int n_per_t = (Ds + 31) / 32;
            auto ds_idx = thread_position_in_threadgroup.x;
            auto d_idx = thread_position_in_grid.y;
            if constexpr (Dh % 8 != 0) { if (d_idx >= Dh) return; }
            float prior[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * ds_idx + i;
                if constexpr (Ds % 32 != 0) { if (s_idx >= Ds) continue; }
                prior[i] = state_in[n * Dh * Ds + d_idx * Ds + s_idx];
            }
            for (int t = 0; t < L; ++t) {
                auto x_ = static_cast<float>(X[(t * H + n) * Dh + d_idx]);
                auto C_ = C + (t * (H / G) + g_idx) * Ds;
                auto B_ = B + (t * (H / G) + g_idx) * Ds;
                auto dt_ = static_cast<float>(dt[t * H + n]);
                auto A = -fast::exp(static_cast<float>(A_log[h_idx]));
                auto dA = fast::exp(A * dt_);
                float acc = 0.0;
                for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * ds_idx + i;
                    if constexpr (Ds % 32 != 0) { if (s_idx >= Ds) continue; }
                    auto dB_by_x = x_ * dt_ * static_cast<float>(B_[s_idx]);
                    auto state = dA * prior[i] + dB_by_x;
                    state_out[((t * H + n) * Dh + d_idx) * Ds + s_idx] = static_cast<U>(state);
                    prior[i] = static_cast<U>(state);
                    acc += state * C_[s_idx];
                }
                acc = simd_sum(acc);
                if (thread_index_in_simdgroup == 0) {
                    out[(t * H + n) * Dh + d_idx] = static_cast<T>(acc + x_ * D[h_idx]);
                }
            }
            """)
}

func nemotronMTPSSMWindow(
    hiddenStates: MLXArray, ALog: MLXArray, B: MLXArray, C: MLXArray,
    D: MLXArray, dt: MLXArray, dtBias: MLXArray, state: MLXArray,
    timeStepLimit: (Float, Float)
) -> (output: MLXArray, capturedStates: MLXArray) {
    let (batch, length, heads, headDim) = hiddenStates.shape4
    let groups = B.dim(-2), stateDim = B.dim(-1)
    precondition(batch == 1 && (1...8).contains(length))
    precondition(groups > 0 && heads % groups == 0 && stateDim > 0)
    precondition(B.shape == [1, length, groups, stateDim] && C.shape == B.shape)
    precondition(state.dtype == .float32 && state.shape == [1, heads, headDim, stateDim])
    precondition(dt.shape == [1, length, heads])
    let kernel = NemotronMTPSSMKernel.kernel
    let outputs = kernel(
        [hiddenStates, ALog, B, C, D, nemotronPrefillComputeDt(dt, dtBias, timeStepLimit), state],
        template: [("T", hiddenStates.dtype), ("U", DType.float32),
                   ("Dh", headDim), ("Ds", stateDim), ("H", heads),
                   ("G", heads / groups), ("L", length)],
        grid: (32, headDim, heads), threadGroup: (32, 8, 1),
        outputShapes: [[1, length, heads, headDim], [length, heads, headDim, stateDim]],
        outputDTypes: [hiddenStates.dtype, .float32])
    return (outputs[0], outputs[1])
}

/// Compact verify (final-state-only window + exact prefix replay). On by
/// default; DARKBLOOM_NEMOTRON35_MTP_COMPACT_SSM=0 restores the captured stack.
let nemotronMTPCompactSSM =
    ProcessInfo.processInfo.environment["DARKBLOOM_NEMOTRON35_MTP_COMPACT_SSM"] != "0"

/// A bounded exact prefix recipe. It owns the same tape and initial array
/// charged by strictReplayRetainedRoots; it never contains another recipe.
struct NemotronMTPSSMPrefix: CBv2RecurrentSSMContinuation {
    let convolution: MLXArray
    let aLog: MLXArray
    let D: MLXArray
    let dt: MLXArray
    let initialState: MLXArray
    let heads: Int
    let headDim: Int
    let groups: Int
    let stateDim: Int
    let steps: Int

    func matches(heads: Int, headDim: Int, groups: Int, stateDim: Int, dtype: DType) -> Bool {
        self.heads == heads && self.headDim == headDim && self.groups == groups
            && self.stateDim == stateDim && convolution.dtype == dtype
    }

    func materializedState() -> MLXArray {
        nemotronMTPSSMCompact(
            convolution: convolution, ALog: aLog, D: D, dt: dt, state: initialState,
            heads: heads, headDim: headDim, groups: groups, stateDim: stateDim,
            steps: steps, writeOutput: false).finalState
    }
}

// Compact verify window. Per position the recurrence text is identical to
// `nemotron_mtp_ssm_window` (and therefore to the stock M=1 `ssm_kernel`):
// same dA, dB_by_x, state and acc expressions in the same order. X, B and C
// are read in place from the contiguous convolution output (no copies), and
// only the state after the last step is written. A strict accepted prefix is
// recovered at commit by the same kernel run for `L = keep` steps from the
// same input state, which reproduces the per-position state bit for bit.
private enum NemotronMTPCompactSSMKernel {
    static let kernel = MLXFast.metalKernel(
        name: "nemotron_mtp_ssm_compact",
        inputNames: ["conv", "A_log", "D", "dt", "state_in",
                     "prefix_conv", "prefix_A_log", "prefix_dt", "prefix_length"],
        outputNames: ["out", "state_out", "replayed_state"],
        source: """
            auto n = thread_position_in_grid.z;
            auto h_idx = n % H;
            auto g_idx = n / G;
            constexpr int n_per_t = (Ds + 31) / 32;
            auto ds_idx = thread_position_in_threadgroup.x;
            auto d_idx = thread_position_in_grid.y;
            if constexpr (Dh % 8 != 0) { if (d_idx >= Dh) return; }
            float prior[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * ds_idx + i;
                if constexpr (Ds % 32 != 0) { if (s_idx >= Ds) continue; }
                prior[i] = state_in[n * Dh * Ds + d_idx * Ds + s_idx];
            }
            if constexpr (WO) {
                // Reproduce the committed prefix in registers, then continue
                // into this verify. Save its exact boundary for a later
                // strict commit so repeated rejection cannot chain recipes.
                for (int t = 0; t < prefix_length; ++t) {
                    auto x_ = static_cast<float>(prefix_conv[t * CD + n * Dh + d_idx]);
                    auto B_ = prefix_conv + t * CD + BO + g_idx * Ds;
                    auto dt_ = static_cast<float>(prefix_dt[t * H + n]);
                    auto A = -fast::exp(static_cast<float>(prefix_A_log[h_idx]));
                    auto dA = fast::exp(A * dt_);
                    for (int i = 0; i < n_per_t; ++i) {
                        auto s_idx = n_per_t * ds_idx + i;
                        if constexpr (Ds % 32 != 0) { if (s_idx >= Ds) continue; }
                        auto dB_by_x = x_ * dt_ * static_cast<float>(B_[s_idx]);
                        auto state = dA * prior[i] + dB_by_x;
                        prior[i] = static_cast<U>(state);
                    }
                }
                if (prefix_length > 0) {
                    for (int i = 0; i < n_per_t; ++i) {
                        auto s_idx = n_per_t * ds_idx + i;
                        if constexpr (Ds % 32 != 0) { if (s_idx >= Ds) continue; }
                        replayed_state[(n * Dh + d_idx) * Ds + s_idx] = prior[i];
                    }
                }
            }
            for (int t = 0; t < L; ++t) {
                auto x_ = static_cast<float>(conv[t * CD + n * Dh + d_idx]);
                auto C_ = conv + t * CD + CO + g_idx * Ds;
                auto B_ = conv + t * CD + BO + g_idx * Ds;
                auto dt_ = static_cast<float>(dt[t * H + n]);
                auto A = -fast::exp(static_cast<float>(A_log[h_idx]));
                auto dA = fast::exp(A * dt_);
                float acc = 0.0;
                for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * ds_idx + i;
                    if constexpr (Ds % 32 != 0) { if (s_idx >= Ds) continue; }
                    auto dB_by_x = x_ * dt_ * static_cast<float>(B_[s_idx]);
                    auto state = dA * prior[i] + dB_by_x;
                    if (t == L - 1) {
                        state_out[(n * Dh + d_idx) * Ds + s_idx] = static_cast<U>(state);
                    }
                    prior[i] = static_cast<U>(state);
                    acc += state * C_[s_idx];
                }
                if constexpr (WO) {
                    acc = simd_sum(acc);
                    if (thread_index_in_simdgroup == 0) {
                        out[(t * H + n) * Dh + d_idx] = static_cast<T>(acc + x_ * D[h_idx]);
                    }
                }
            }
            """)
}

/// `convolution` is the window's activated conv output `[1, W, convDim]`
/// (x | B | C), `dt` the transformed timestep `[1, W, heads]` (FP32). Runs
/// the first `steps <= W` positions from `state`. With `writeOutput` the
/// result's `output` holds y for those positions `[1, steps, heads, headDim]`.
func nemotronMTPSSMCompact(
    convolution: MLXArray, ALog: MLXArray, D: MLXArray, dt: MLXArray,
    state: MLXArray, heads: Int, headDim: Int, groups: Int, stateDim: Int,
    steps: Int, writeOutput: Bool, prefix: NemotronMTPSSMPrefix? = nil
) -> (output: MLXArray, finalState: MLXArray, replayedState: MLXArray?) {
    let width = convolution.dim(1), convDim = convolution.dim(2)
    precondition(convolution.ndim == 3 && convolution.dim(0) == 1 && (1...width).contains(steps))
    precondition(groups > 0 && heads % groups == 0 && convDim == heads * headDim + 2 * groups * stateDim)
    precondition(dt.dtype == .float32 && dt.shape == [1, width, heads])
    precondition(state.dtype == .float32 && state.shape == [1, heads, headDim, stateDim])
    let replay = prefix.flatMap {
        writeOutput && $0.matches(heads: heads, headDim: headDim, groups: groups,
                                  stateDim: stateDim, dtype: convolution.dtype) ? $0 : nil
    }
    let prefixLength = replay?.steps ?? 0
    if let replay {
        precondition((1...7).contains(prefixLength) && prefixLength <= replay.convolution.dim(1))
        precondition(replay.dt.dtype == .float32
            && replay.dt.shape == [1, replay.convolution.dim(1), heads])
        precondition(replay.initialState.dtype == .float32
            && replay.initialState.shape == [1, heads, headDim, stateDim])
    }
    let outputs = NemotronMTPCompactSSMKernel.kernel(
        [convolution, ALog, D, dt, replay?.initialState ?? state,
         replay?.convolution ?? convolution, replay?.aLog ?? ALog,
         replay?.dt ?? dt, prefixLength],
        template: [("T", convolution.dtype), ("U", DType.float32),
                   ("Dh", headDim), ("Ds", stateDim), ("H", heads),
                   ("G", heads / groups), ("L", steps), ("CD", convDim),
                   ("BO", heads * headDim), ("CO", heads * headDim + groups * stateDim),
                   ("WO", writeOutput)],
        grid: (32, headDim, heads), threadGroup: (32, 8, 1),
        outputShapes: [writeOutput ? [1, steps, heads, headDim] : [1],
                       [1, heads, headDim, stateDim],
                       replay == nil ? [1] : [1, heads, headDim, stateDim]],
        outputDTypes: [convolution.dtype, .float32, .float32])
    return (outputs[0], outputs[1], replay == nil ? nil : outputs[2])
}
