// Short captured windows with the canonical M=1 SSM arithmetic at every step.
// All prefix states are written in FP32; no parallel/prefill recurrence is used.
import Foundation
import MLX

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
        inputNames: ["conv", "A_log", "D", "dt", "state_in"],
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
    steps: Int, writeOutput: Bool
) -> (output: MLXArray, finalState: MLXArray) {
    let width = convolution.dim(1), convDim = convolution.dim(2)
    precondition(convolution.ndim == 3 && convolution.dim(0) == 1 && (1...width).contains(steps))
    precondition(groups > 0 && heads % groups == 0 && convDim == heads * headDim + 2 * groups * stateDim)
    precondition(dt.dtype == .float32 && dt.shape == [1, width, heads])
    precondition(state.dtype == .float32 && state.shape == [1, heads, headDim, stateDim])
    let outputs = NemotronMTPCompactSSMKernel.kernel(
        [convolution, ALog, D, dt, state],
        template: [("T", convolution.dtype), ("U", DType.float32),
                   ("Dh", headDim), ("Ds", stateDim), ("H", heads),
                   ("G", heads / groups), ("L", steps), ("CD", convDim),
                   ("BO", heads * headDim), ("CO", heads * headDim + groups * stateDim),
                   ("WO", writeOutput)],
        grid: (32, headDim, heads), threadGroup: (32, 8, 1),
        outputShapes: [writeOutput ? [1, steps, heads, headDim] : [1], [1, heads, headDim, stateDim]],
        outputDTypes: [convolution.dtype, .float32])
    return (outputs[0], outputs[1])
}

let nemotronFuseConvDtIntoCompactSSM =
    ProcessInfo.processInfo.environment["NEMOTRON_MAMBA_FUSE_CONV_DT"] != "0"

private enum NemotronMTPCompactFusedKernel {
    static let header = """
        #define NEMO_ACTIVATE(channel, position, result)                       \\
          do {                                                                 \\
            float nemo_acc = 0;                                                \\
            for (int nemo_i = 0; nemo_i < 4; ++nemo_i) {                       \\
              int64_t nemo_row = (position) + int64_t(nemo_i);                 \\
              T nemo_value = nemo_row < 3                                      \\
                  ? oldState[nemo_row * oldState_strides[1] +                  \\
                        (channel) * oldState_strides[2]]                       \\
                  : convIn[(nemo_row - 3) * convIn_strides[1] +                \\
                        (channel) * convIn_strides[2]];                        \\
              nemo_acc += static_cast<float>(nemo_value) *                     \\
                  static_cast<float>(weight[(channel) * weight_strides[0] +    \\
                      int64_t(nemo_i) * weight_strides[1]]);                   \\
            }                                                                  \\
            T nemo_convolved = static_cast<T>(nemo_acc);                       \\
            T nemo_x = nemo_convolved + bias[(channel) * bias_strides[0]];     \\
            auto nemo_y = 1 / (1 + metal::exp(metal::abs(nemo_x)));            \\
            T nemo_sigmoid = nemo_x < 0 ? nemo_y : 1 - nemo_y;                 \\
            (result) = nemo_x * nemo_sigmoid;                                  \\
          } while (0)
        """
    static let kernel = MLXFast.metalKernel(
        name: "nemotron_mtp_ssm_conv_dt",
        inputNames: [
            "oldState", "convIn", "weight", "bias", "dtRaw", "dtBias", "limits",
            "A_log", "D", "state_in",
        ],
        outputNames: ["out", "state_out", "convolution", "padded", "dt_out"],
        source: """
            auto n = thread_position_in_grid.z;
            auto h_idx = n % H;
            auto g_idx = n / G;
            constexpr int n_per_t = (Ds + 31) / 32;
            auto ds_idx = thread_position_in_threadgroup.x;
            auto local_d = thread_position_in_threadgroup.y;
            auto d_idx = thread_position_in_grid.y;
            auto tg_y = threadgroup_position_in_grid.y;
            const uint lid = ds_idx + local_d * 32u;
            threadgroup T cache_x[8];
            threadgroup T cache_b[Ds];
            threadgroup T cache_c[Ds];

            if (ds_idx == 0) {
                int64_t x_channel = int64_t(n) * Dh + int64_t(d_idx);
                for (int h = 0; h < 3; ++h) {
                    padded[int64_t(h) * CD + x_channel] =
                        oldState[int64_t(h) * oldState_strides[1] +
                            x_channel * oldState_strides[2]];
                }
            }
            if (tg_y == 0 && (n % G) == 0 && lid < uint(Ds)) {
                int64_t bch = int64_t(BO) + int64_t(g_idx) * Ds + int64_t(lid);
                int64_t cch = int64_t(CO) + int64_t(g_idx) * Ds + int64_t(lid);
                for (int h = 0; h < 3; ++h) {
                    padded[int64_t(h) * CD + bch] =
                        oldState[int64_t(h) * oldState_strides[1] +
                            bch * oldState_strides[2]];
                    padded[int64_t(h) * CD + cch] =
                        oldState[int64_t(h) * oldState_strides[1] +
                            cch * oldState_strides[2]];
                }
            }

            float prior[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * ds_idx + i;
                if constexpr (Ds % 32 != 0) { if (s_idx >= Ds) continue; }
                prior[i] = state_in[n * Dh * Ds + d_idx * Ds + s_idx];
            }
            for (int t = 0; t < L; ++t) {
                if (ds_idx == 0) {
                    int64_t x_channel = int64_t(n) * Dh + int64_t(d_idx);
                    padded[(int64_t(t) + 3) * CD + x_channel] =
                        convIn[int64_t(t) * convIn_strides[1] +
                            x_channel * convIn_strides[2]];
                    T activated;
                    NEMO_ACTIVATE(x_channel, int64_t(t), activated);
                    cache_x[local_d] = activated;
                    convolution[int64_t(t) * CD + x_channel] = activated;
                }
                if (lid < uint(Ds)) {
                    int64_t bch = int64_t(BO) + int64_t(g_idx) * Ds + int64_t(lid);
                    int64_t cch = int64_t(CO) + int64_t(g_idx) * Ds + int64_t(lid);
                    T b_act;
                    T c_act;
                    NEMO_ACTIVATE(bch, int64_t(t), b_act);
                    NEMO_ACTIVATE(cch, int64_t(t), c_act);
                    cache_b[lid] = b_act;
                    cache_c[lid] = c_act;
                    if (tg_y == 0 && (n % G) == 0) {
                        padded[(int64_t(t) + 3) * CD + bch] =
                            convIn[int64_t(t) * convIn_strides[1] +
                                bch * convIn_strides[2]];
                        padded[(int64_t(t) + 3) * CD + cch] =
                            convIn[int64_t(t) * convIn_strides[1] +
                                cch * convIn_strides[2]];
                        convolution[int64_t(t) * CD + bch] = b_act;
                        convolution[int64_t(t) * CD + cch] = c_act;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float raw = static_cast<float>(dtRaw[int64_t(t) * dtRaw_strides[1] +
                    int64_t(n) * dtRaw_strides[2]]);
                float z = raw + static_cast<float>(dtBias[int64_t(n) * dtBias_strides[0]]);
                float maxv = metal::max(z, 0.0f);
                float minv = metal::min(z, 0.0f);
                constexpr float inf = metal::numeric_limits<float>::infinity();
                float sp = (minv == -inf || maxv == inf)
                    ? maxv
                    : maxv + metal::log1p(metal::exp(minv - maxv));
                float dt_ = metal::isnan(sp) ? sp : metal::min(metal::max(sp, limits[0]), limits[1]);
                if (tg_y == 0 && lid == 0) {
                    dt_out[int64_t(t) * H + int64_t(n)] = dt_;
                }
                auto x_ = static_cast<float>(cache_x[local_d]);
                auto A = -fast::exp(static_cast<float>(A_log[int64_t(h_idx) * A_log_strides[0]]));
                auto dA = fast::exp(A * dt_);
                float acc = 0.0;
                for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * ds_idx + i;
                    if constexpr (Ds % 32 != 0) { if (s_idx >= Ds) continue; }
                    auto dB_by_x = x_ * dt_ * static_cast<float>(cache_b[s_idx]);
                    auto state = dA * prior[i] + dB_by_x;
                    if (t == L - 1) {
                        state_out[(n * Dh + d_idx) * Ds + s_idx] = static_cast<U>(state);
                    }
                    prior[i] = static_cast<U>(state);
                    acc += state * static_cast<float>(cache_c[s_idx]);
                }
                acc = simd_sum(acc);
                if (thread_index_in_simdgroup == 0) {
                    out[(t * H + n) * Dh + d_idx] = static_cast<T>(
                        acc + x_ * static_cast<float>(D[int64_t(h_idx) * D_strides[0]]));
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            """,
        header: header,
        ensureRowContiguous: false)
}

func nemotronMTPSSMCompactFused(
    oldState: MLXArray, convInput: MLXArray, weight: MLXArray, bias: MLXArray,
    dtRaw: MLXArray, dtBias: MLXArray, dtMin: Float, dtMax: Float,
    ALog: MLXArray, D: MLXArray, state: MLXArray,
    heads: Int, headDim: Int, groups: Int, stateDim: Int
) -> (output: MLXArray, finalState: MLXArray, convolution: MLXArray, padded: MLXArray, dt: MLXArray)? {
    let length = convInput.dim(1)
    let convDim = convInput.dim(2)
    guard nemotronFuseConvDtIntoCompactSSM,
        oldState.shape == [1, 3, convDim], convInput.shape == [1, length, convDim],
        (2...8).contains(length), convDim == heads * headDim + 2 * groups * stateDim,
        groups > 0, heads % groups == 0, headDim % 8 == 0, stateDim > 0,
        weight.shape == [convDim, 4, 1], bias.shape == [convDim],
        dtRaw.shape == [1, length, heads], dtBias.shape == [heads],
        state.shape == [1, heads, headDim, stateDim], state.dtype == .float32,
        oldState.dtype == .bfloat16, convInput.dtype == .bfloat16,
        weight.dtype == .bfloat16, bias.dtype == .bfloat16,
        dtRaw.dtype == .bfloat16, ALog.shape == [heads], D.shape == [heads]
    else { return nil }
    let outputs = NemotronMTPCompactFusedKernel.kernel(
        [oldState, convInput, weight, bias, dtRaw, dtBias,
         MLXArray([dtMin, dtMax]), ALog, D, state],
        template: [
            ("T", DType.bfloat16), ("U", DType.float32),
            ("Dh", headDim), ("Ds", stateDim), ("H", heads),
            ("G", heads / groups), ("L", length), ("CD", convDim),
            ("BO", heads * headDim), ("CO", heads * headDim + groups * stateDim),
        ],
        grid: (32, headDim, heads), threadGroup: (32, 8, 1),
        outputShapes: [
            [1, length, heads, headDim],
            [1, heads, headDim, stateDim],
            [1, length, convDim],
            [1, length + 3, convDim],
            [1, length, heads],
        ],
        outputDTypes: [.bfloat16, .float32, .bfloat16, .bfloat16, .float32])
    return (outputs[0], outputs[1], outputs[2], outputs[3], outputs[4])
}
