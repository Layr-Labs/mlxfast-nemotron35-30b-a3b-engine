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
    private static let source = """
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
            """
    static let kernel = MLXFast.metalKernel(
        name: "nemotron_mtp_ssm_compact",
        inputNames: ["conv", "A_log", "D", "dt", "state_in"],
        outputNames: ["out", "state_out"], source: source)

    private static func preparedSource() -> String? {
        let before = "auto A = -fast::exp(static_cast<float>(A_log[h_idx]));\n    auto dA = fast::exp(A * dt_);"
        guard source.components(separatedBy: before).count == 2 else { return nil }
        return source.replacingOccurrences(of: before, with: "auto dA = decay[t * H + n];")
    }
    static let preparedKernel: MLXFast.MLXFastKernel? = preparedSource().map {
        MLXFast.metalKernel(
            name: "nemotron_mtp_ssm_compact_prepared_decay",
            inputNames: ["conv", "A_log", "D", "dt", "state_in", "decay"],
            outputNames: ["out", "state_out"], source: $0)
    }
}

/// Every table belongs to this invocation. It contains original FP32 dt and
/// its original SSM decay, never a model/request/pointer-keyed result cache.
enum NemotronMTPDTDecay {
    static let source = """
            const uint i = thread_position_in_grid.x;
            if (i >= W * H) return;
            const uint t = i / H;
            const uint h = i % H;
            const float x = static_cast<float>(raw[t * raw_strides[1] + h * raw_strides[2]])
                + static_cast<float>(bias[h * bias_strides[0]]);
            // MLXNN.softplus is LogAddExp(x, 0). Match its NaN/Inf cases,
            // metal::exp and MLX utils' unqualified log1p, not a substitute.
            float value;
            if (metal::isnan(x)) {
                value = metal::numeric_limits<float>::quiet_NaN();
            } else {
                constexpr float inf = metal::numeric_limits<float>::infinity();
                const float maxval = metal::max(x, 0.0f);
                const float minval = metal::min(x, 0.0f);
                value = (minval == -inf || maxval == inf) ? maxval
                    : (maxval + log1p(metal::exp(minval - maxval)));
            }
            // ops.cpp clip = maximum first, then minimum. Float operators
            // propagate x's NaN and use the original asymmetric comparisons.
            value = metal::isnan(value) ? value : (value > lower[0] ? value : lower[0]);
            value = metal::isnan(value) ? value : (value < upper[0] ? value : upper[0]);
            table[i] = value;
            auto A = -fast::exp(static_cast<float>(A_log[h * A_log_strides[0]]));
            auto dA = fast::exp(A * value);
            table[W * H + i] = dA;
            """

    static let enabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_DT_DECAY_PREPARED"] != "0"
    private static let kernel = MLXFast.metalKernel(
        name: "nemotron_mtp_dt_decay_current_call",
        inputNames: ["raw", "bias", "A_log", "lower", "upper"],
        outputNames: ["table"],
        source: source, ensureRowContiguous: false)

    static func make(_ raw: MLXArray, bias: MLXArray, ALog: MLXArray,
                     limits: (Float, Float)) -> MLXArray? {
        guard enabled, Device.defaultDevice().deviceType == .gpu,
              StreamOrDevice.default == .gpu,
              raw.ndim == 3, raw.dim(0) == 1, (2...4).contains(raw.dim(1)), raw.dim(2) == 64,
              [.bfloat16, .float32].contains(raw.dtype),
              bias.shape == [64], [.bfloat16, .float32].contains(bias.dtype),
              ALog.shape == [64], ALog.dtype == .float32 else { return nil }
        return kernel([raw, bias, ALog, MLXArray(limits.0), MLXArray(limits.1)],
            template: [("W", raw.dim(1)), ("H", 64)],
            grid: (raw.dim(1) * 64, 1, 1), threadGroup: (32, 1, 1),
            outputShapes: [[2, raw.dim(1), 64]], outputDTypes: [.float32])[0]
    }

    static func unpack(_ info: MLXArray, width: Int, heads: Int)
        -> (dt: MLXArray, decay: MLXArray?) {
        if info.dtype == .float32, info.shape == [2, width, heads] {
            return (info[0..<1, 0..., 0...], info[1..<2, 0..., 0...])
        }
        return (info, nil)
    }
}

/// Add independent current-call timestep work to the native convolution launch.
/// The original convolution and timestep arithmetic strings stay unchanged.
private enum NemotronMTPPackedConvDTDecay {
    static let enabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_CONV_DT_FUSED"] != "0"

    private static func fusedSource() -> String? {
        let prefix = "const uint i = thread_position_in_grid.x;\nif (i >= W * H) return;\nconst uint t = i / H;\nconst uint h = i % H;\n"
        let source = NemotronMTPDTDecay.source
        let bias = "bias[h * bias_strides[0]]"
        guard source.hasPrefix(prefix), source.components(separatedBy: bias).count == 2
        else { return nil }
        let body = String(source.dropFirst(prefix.count))
            .replacingOccurrences(of: bias, with: "dtBias[h * dtBias_strides[0]]")
        return NemotronPackedConvKernel.source + "\nif (channel < H) {\n"
            + "const uint t = uint(position);\nconst uint h = uint(channel);\nconst uint i = t * H + h;\n"
            + body + "\n}\n"
    }

    private static let kernel: MLXFast.MLXFastKernel? = fusedSource().map {
        MLXFast.metalKernel(
            name: "nemotron_mtp_packed_conv_dt_decay",
            inputNames: ["oldState", "input", "weight", "bias", "raw", "dtBias", "A_log", "lower", "upper"],
            outputNames: ["out", "padded", "table"], source: $0,
            ensureRowContiguous: false)
    }

    static func make(
        _ oldState: MLXArray, _ input: MLXArray, weight: MLXArray, bias: MLXArray,
        raw: MLXArray, dtBias: MLXArray, ALog: MLXArray, limits: (Float, Float)
    ) -> (convolution: MLXArray, padded: MLXArray, timeInfo: MLXArray)? {
        guard enabled, NemotronMTPDTDecay.enabled,
              Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              input.ndim == 3, input.dim(0) == 1, (2...4).contains(input.dim(1)),
              input.dim(2) == 6144, input.dtype == .bfloat16,
              oldState.shape == [1, 3, 6144], oldState.dtype == .bfloat16,
              weight.shape == [6144, 4, 1], weight.dtype == .bfloat16,
              bias.shape == [6144], bias.dtype == .bfloat16,
              raw.shape == [1, input.dim(1), 64], [.bfloat16, .float32].contains(raw.dtype),
              dtBias.shape == [64], [.bfloat16, .float32].contains(dtBias.dtype),
              ALog.shape == [64], ALog.dtype == .float32, let kernel else { return nil }
        let width = input.dim(1)
        let arrays = kernel(
            [oldState, input, weight, bias, raw, dtBias, ALog, MLXArray(limits.0), MLXArray(limits.1)],
            template: [("T", DType.bfloat16), ("C", 6144), ("H", 64), ("W", width)],
            grid: (6144, width, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1, width, 6144], [1, width + 3, 6144], [2, width, 64]],
            outputDTypes: [.bfloat16, .bfloat16, .float32])
        return (arrays[0], arrays[1], arrays[2])
    }
}

extension NemotronHMamba2Mixer {
    func mtpPackedConvAndTime(
        _ oldState: MLXArray, _ input: MLXArray, rawDT: MLXArray, prepareDecay: Bool
    ) -> (convolution: MLXArray, padded: MLXArray, timeInfo: MLXArray) {
        let native = prepareDecay && numHeads == 64 && headDim == 64 && numGroups == 8
            && ssmStateSize == 128 && convDim == 6144
        if native, convKernelSize == 4, conv1d.groups == convDim,
           conv1d.stride == 1, conv1d.dilation == 1, conv1d.padding == 0,
           let bias = conv1d.bias,
           let result = NemotronMTPPackedConvDTDecay.make(
               oldState, input, weight: conv1d.weight, bias: bias,
               raw: rawDT, dtBias: dtBias, ALog: aLog, limits: timeStepLimit) {
            return result
        }
        let packed = packedConv(oldState, input)
        let timeInfo = (native
            ? NemotronMTPDTDecay.make(rawDT, bias: dtBias, ALog: aLog, limits: timeStepLimit) : nil)
            ?? nemotronPrefillComputeDt(rawDT, dtBias, timeStepLimit)
        return (packed.convolution, packed.padded, timeInfo)
    }
}

/// `convolution` is the window's activated conv output `[1, W, convDim]`
/// (x | B | C), `dt` the transformed timestep `[1, W, heads]` (FP32). Runs
/// the first `steps <= W` positions from `state`. With `writeOutput` the
/// result's `output` holds y for those positions `[1, steps, heads, headDim]`.
func nemotronMTPSSMCompact(
    convolution: MLXArray, ALog: MLXArray, D: MLXArray, dt: MLXArray,
    state: MLXArray, heads: Int, headDim: Int, groups: Int, stateDim: Int,
    steps: Int, writeOutput: Bool, preparedDecay: MLXArray? = nil
) -> (output: MLXArray, finalState: MLXArray) {
    let width = convolution.dim(1), convDim = convolution.dim(2)
    precondition(convolution.ndim == 3 && convolution.dim(0) == 1 && (1...width).contains(steps))
    precondition(groups > 0 && heads % groups == 0 && convDim == heads * headDim + 2 * groups * stateDim)
    precondition(dt.dtype == .float32 && dt.shape == [1, width, heads])
    precondition(state.dtype == .float32 && state.shape == [1, heads, headDim, stateDim])
    let kernel: MLXFast.MLXFastKernel
    let arguments: [MLXArray]
    if NemotronMTPDTDecay.enabled, heads == 64, headDim == 64, groups == 8, stateDim == 128,
       convolution.dtype == .bfloat16,
       let preparedDecay, preparedDecay.shape == dt.shape, preparedDecay.dtype == .float32,
       let preparedKernel = NemotronMTPCompactSSMKernel.preparedKernel {
        kernel = preparedKernel
        arguments = [convolution, ALog, D, dt, state, preparedDecay]
    } else {
        kernel = NemotronMTPCompactSSMKernel.kernel
        arguments = [convolution, ALog, D, dt, state]
    }
    let outputs = kernel(
        arguments,
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
