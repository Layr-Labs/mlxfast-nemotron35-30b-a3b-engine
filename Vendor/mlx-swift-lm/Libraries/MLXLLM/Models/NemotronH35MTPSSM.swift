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


// Native compact-verification producer: preserve packed conv arithmetic and
// materialize only its existing FP32 dt tensor in the same dispatch. The
// compact scan and strict-prefix replay consume their unchanged inputs.
private enum NemotronPackedConvDtKernel {
    static let kernel = MLXFast.metalKernel(
        name: "nemotron_packed_conv_silu_dt",
        inputNames: ["oldState", "input", "weight", "bias", "rawDt", "dtBias", "dt_min", "dt_max"],
        outputNames: ["out", "padded", "timestep_out"],
        source: """
            int64_t channel = int64_t(thread_position_in_grid.x);
            int64_t position = int64_t(thread_position_in_grid.y);
            padded[(position + 3) * C + channel] =
                input[position * input_strides[1] + channel * input_strides[2]];
            if (position == 0) {
                for (int i = 0; i < 3; ++i) {
                    padded[int64_t(i) * C + channel] =
                        oldState[int64_t(i) * oldState_strides[1] + channel * oldState_strides[2]];
                }
            }
            float acc = 0;
            for (int i = 0; i < 4; ++i) {
                int64_t row = position + int64_t(i);
                T value = row < 3
                    ? oldState[row * oldState_strides[1] + channel * oldState_strides[2]]
                    : input[(row - 3) * input_strides[1] + channel * input_strides[2]];
                acc += static_cast<float>(value)
                    * weight[channel * weight_strides[0] + int64_t(i) * weight_strides[1]];
            }
            T convolved = static_cast<T>(acc);
            T x = convolved + bias[channel * bias_strides[0]];
            auto y = 1 / (1 + metal::exp(metal::abs(x)));
            T sigmoid = x < 0 ? y : 1 - y;
            out[position * C + channel] = x * sigmoid;
            if (channel < H) {
                float z = static_cast<float>(rawDt[position * rawDt_strides[1]
                    + channel * rawDt_strides[2]])
                    + static_cast<float>(dtBias[channel * dtBias_strides[0]]);
                float timestep = nemotronPackedDtLogAddExp(z, 0.0f);
                // Match Maximum(timestep, min), then Minimum(result, max):
                // left NaN propagates, ties select the right bound.
                float low = metal::isnan(timestep) ? timestep
                    : (timestep > dt_min ? timestep : dt_min);
                timestep_out[position * H + channel] = metal::isnan(low) ? low
                    : (low < dt_max ? low : dt_max);
            }
            """,
        header: """
            // Float LogAddExp from vendored binary_ops.h. The unqualified
            // log1p resolves to the same vendored utils.h Float overload.
            inline float nemotronPackedDtLogAddExp(float x, float y) {
                if (metal::isnan(x) || metal::isnan(y)) {
                    return metal::numeric_limits<float>::quiet_NaN();
                }
                constexpr float inf = metal::numeric_limits<float>::infinity();
                float maxval = metal::max(x, y);
                float minval = metal::min(x, y);
                return (minval == -inf || maxval == inf)
                    ? maxval : (maxval + log1p(metal::exp(minval - maxval)));
            }
            """,
        ensureRowContiguous: false)
}

func nemotronPackedConvAndDt(
    oldState: MLXArray, input: MLXArray, weight: MLXArray, bias: MLXArray,
    rawDt: MLXArray, dtBias: MLXArray, timeStepLimit: (Float, Float)
) -> (convolution: MLXArray, padded: MLXArray, dt: MLXArray)? {
    guard Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
        input.ndim == 3, input.dim(0) == 1, (2...8).contains(input.dim(1)),
        input.dim(2) == 6144, oldState.shape == [1, 3, 6144],
        input.dtype == .bfloat16, oldState.dtype == .bfloat16,
        weight.shape == [6144, 4, 1], weight.dtype == .bfloat16,
        bias.shape == [6144], bias.dtype == .bfloat16,
        rawDt.shape == [1, input.dim(1), 64], rawDt.dtype == .bfloat16,
        dtBias.shape == [64], [DType.bfloat16, .float16, .float32].contains(dtBias.dtype)
    else { return nil }
    let length = input.dim(1)
    let arrays = NemotronPackedConvDtKernel.kernel(
        [oldState, input, weight, bias, rawDt, dtBias,
         MLXArray(timeStepLimit.0), MLXArray(timeStepLimit.1)],
        template: [("T", DType.bfloat16), ("C", 6144), ("H", 64)],
        grid: (6144, length, 1), threadGroup: (256, 1, 1),
        outputShapes: [[1, length, 6144], [1, length + 3, 6144], [1, length, 64]],
        outputDTypes: [.bfloat16, .bfloat16, .float32])
    return (arrays[0], arrays[1], arrays[2])
}
