// Short captured windows with the canonical M=1 SSM arithmetic at every step.
// All prefix states are written in FP32; no parallel/prefill recurrence is used.
import Foundation
import MLX
import MLXLMCommon
import MLXNN

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
        source: source)

    static let source = """
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

/// One current forward launch: original compact SSM, then the norm group's
/// actual last producer performs the original gated RMS/prepared epilogue.
/// Strict-prefix replay retains the original compact/grouped kernels.
enum NemotronMTPSSMGatedJoin {
    private static let enabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_SSM_GATED_JOIN"] != "0"
        && ProcessInfo.processInfo.environment["NEMOTRON_MTP_GATED_PREPARED_OUT"] != "0"
    private struct Entry {
        let counter: MLXArray
        let norm: ObjectIdentifier
        let aLog: ObjectIdentifier
        let d: ObjectIdentifier
        let out: ObjectIdentifier
    }
    private static let lock = NSLock()
    nonisolated(unsafe) private static var entries: [ObjectIdentifier: Entry] = [:]
    private static let identity = MLXArray(1).asType(.bfloat16)

    private static func source() -> String {
        let release = """

          // All 256 producer threads publish their original BF16 outputs
          // and FP32 state before this CTA can take its group ticket.
          threadgroup_barrier(mem_flags::mem_device);
          atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
          threadgroup uint join_ticket[1];
          threadgroup float local_inv_mean[1];
          threadgroup float local_sums[32];
          threadgroup T rounded[512];
          const uint join_linear = thread_position_in_threadgroup.x
              + 32u * thread_position_in_threadgroup.y;
          const uint join_group = uint(n) / 8u;
          if (join_linear == 0) {
            join_ticket[0] = atomic_fetch_add_explicit(
                (device atomic_uint*)(counter + join_group), 1u, memory_order_relaxed);
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (join_ticket[0] != 63u) return;
          atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
          const device T* x = out;
          for (int join_row = 0; join_row < L; join_row++) {
        """
        let finish = """

            // Do not replace rounded/local reduction scratch until the
            // first SIMD finished packing the previous row.
            threadgroup_barrier(mem_flags::mem_threadgroup);
          }
          threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
          atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
          if (join_linear == 0) {
            atomic_store_explicit((device atomic_uint*)(counter + join_group),
                                  0u, memory_order_relaxed);
          }
        """
        return NemotronMTPCompactSSMKernel.source + release
            + NemotronHRMSNormGated.ssmPreparedJoinSource() + finish
    }

    private static let kernel = MLXFast.metalKernel(
        name: "nemotron_mtp_ssm_gated_prepared_join",
        inputNames: ["conv", "A_log", "D", "dt", "state_in", "gate", "weight",
                     "identity", "eps", "counter"],
        outputNames: ["out", "state_out", "joined_norm", "prepared"],
        source: source(), ensureRowContiguous: false, mutableInputs: ["counter"])

    private static func run(_ mixer: NemotronHMamba2Mixer, convolution: MLXArray,
                            dt: MLXArray, state: MLXArray, gate: MLXArray,
                            counter: MLXArray, steps: Int) -> [MLXArray]? {
        guard Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              type(of: mixer) == NemotronHMamba2Mixer.self,
              mixer.numHeads == 64, mixer.headDim == 64, mixer.numGroups == 8,
              mixer.ssmStateSize == 128, mixer.convDim == 6144,
              type(of: mixer.norm) == NemotronHRMSNormGated.self,
              mixer.norm.groupSize == 512, mixer.norm.eps > 0, mixer.norm.eps.isFinite,
              mixer.norm.weight.shape == [4096], mixer.norm.weight.dtype == .bfloat16,
              let q = mixer.outProj as? QuantizedLinear, type(of: q) == QuantizedLinear.self,
              q.mode == .affine, q.bits == 4, q.groupSize == 64, q.bias == nil,
              q.weight.shape == [2688, 512], q.weight.dtype == .uint32,
              q.scales.shape == [2688, 64], q.scales.dtype == .bfloat16,
              q.biases?.shape == [2688, 64], q.biases?.dtype == .bfloat16,
              convolution.shape == [1, 3, 6144], convolution.dtype == .bfloat16,
              Array(convolution.strides.suffix(2)) == [6144, 1],
              dt.shape == [1, 3, 64], dt.dtype == .float32,
              Array(dt.strides.suffix(2)) == [64, 1],
              mixer.aLog.shape == [64], mixer.aLog.strides == [1],
              mixer.D.shape == [64], mixer.D.strides == [1],
              state.shape == [1, 64, 64, 128], state.dtype == .float32,
              Array(state.strides.suffix(3)) == [8192, 128, 1],
              (1...3).contains(steps), gate.shape == [1, steps, 4096], gate.dtype == .bfloat16,
              counter.shape == [8], counter.dtype == .uint32
        else { return nil }
        return kernel(
            [convolution, mixer.aLog, mixer.D, dt, state, gate,
             mixer.norm.weight, identity, mixer.norm.eps, counter],
            template: [("T", DType.bfloat16), ("U", DType.float32), ("Dh", 64),
                       ("Ds", 128), ("H", 64), ("G", 8), ("L", steps), ("CD", 6144),
                       ("BO", 4096), ("CO", 5120), ("WO", true)],
            grid: (32, 64, 64), threadGroup: (32, 8, 1),
            outputShapes: [[1, steps, 64, 64], [1, 64, 64, 128],
                           [1, steps, 4096], [steps, 8, 17 * 32]],
            outputDTypes: [.bfloat16, .float32, .bfloat16, .float32])
    }

    static func apply(_ mixer: NemotronHMamba2Mixer, convolution: MLXArray,
                      dt: MLXArray, state: MLXArray, gate: MLXArray)
        -> (output: MLXArray, finalState: MLXArray, normalized: MLXArray, prepared: MLXArray)? {
        guard enabled, NemotronMTPExecution.batchedNorm,
              let entry = lock.withLock({ entries[ObjectIdentifier(mixer)] }),
              entry.norm == ObjectIdentifier(mixer.norm), entry.aLog == ObjectIdentifier(mixer.aLog),
              entry.d == ObjectIdentifier(mixer.D), entry.out == ObjectIdentifier(mixer.outProj),
              let values = run(mixer, convolution: convolution, dt: dt, state: state,
                               gate: gate, counter: entry.counter, steps: 3)
        else { return nil }
        guard NemotronMoELeader.selfTest else { return (values[0], values[1], values[2], values[3]) }
        let expected = nemotronMTPSSMCompact(
            convolution: convolution, ALog: mixer.aLog, D: mixer.D, dt: dt, state: state,
            heads: 64, headDim: 64, groups: 8, stateDim: 128, steps: 3, writeOutput: true)
        guard let pair = mixer.norm.preparedRows(expected.output.reshaped(1, 3, 4096), gate: gate)
        else { return nil }
        return (nemotronBitwiseOrNaN(values[0], expected.output),
                nemotronBitwiseOrNaN(values[1], expected.finalState),
                nemotronBitwiseOrNaN(values[2], pair.0), nemotronBitwiseOrNaN(values[3], pair.1))
    }

    static func qualify(_ model: Module) {
        lock.withLock { entries = [:] }
        guard enabled, nemotronMTPCompactSSM, NemotronMTPExecution.batchedM1,
              NemotronMTPExecution.batchedNorm, NemotronMultiRowQMV.enabled,
              NemotronMultiRowQMV.narrowEnabled, Device.defaultDevice().deviceType == .gpu,
              StreamOrDevice.default == .gpu else { return }
        let mixers = model.modules().compactMap { $0 as? NemotronHMamba2Mixer }
        guard mixers.count == 23 else { return }
        var built: [ObjectIdentifier: Entry] = [:]
        do {
            let matched = try withError { () -> Bool in
                for (layer, mixer) in mixers.enumerated() {
                    let counter = MLXArray.zeros([8], dtype: .uint32)
                    eval(counter)
                    for test in 0..<3 {
                        let keys = MLXRandom.split(
                            key: MLXRandom.key(UInt64(0x55a000 + layer * 8 + test)), into: 4)
                        let convolution = test == 2 ? MLXArray.zeros([1, 3, 6144], dtype: .bfloat16)
                            : MLXRandom.normal([1, 3, 6144], dtype: .bfloat16, key: keys[0])
                        let state = test == 2 ? MLXArray.zeros([1, 64, 64, 128], dtype: .float32)
                            : MLXRandom.normal([1, 64, 64, 128], dtype: .float32,
                                               scale: 0.1, key: keys[1])
                        let rawDt = MLXRandom.normal([1, 3, 64], dtype: .bfloat16, key: keys[2])
                        let dt = nemotronPrefillComputeDt(rawDt, mixer.dtBias, mixer.timeStepLimit)
                        // Match production's strided z view of the full in_proj.
                        let projected = MLXRandom.normal([1, 3, 10304], dtype: .bfloat16,
                                                          scale: 2, key: keys[3])
                        for keep in 1...3 {
                            let gate = projected[0..., 0..<keep, 0..<4096]
                            guard let got = run(mixer, convolution: convolution, dt: dt, state: state,
                                                gate: gate, counter: counter, steps: keep)
                            else { return false }
                            let expected = nemotronMTPSSMCompact(
                                convolution: convolution, ALog: mixer.aLog, D: mixer.D,
                                dt: dt, state: state, heads: 64, headDim: 64, groups: 8,
                                stateDim: 128, steps: keep, writeOutput: true)
                            let normed: MLXArray, packed: MLXArray
                            let y = expected.output.reshaped(1, keep, 4096)
                            if let pair = mixer.norm.preparedRows(y, gate: gate) {
                                normed = pair.0; packed = pair.1
                            } else {
                                normed = mixer.norm(y, gate: gate)
                                packed = NemotronHeadPreparedActivation.prepareContiguousRows(
                                    normed.reshaped(keep, 4096), pv: 16)
                            }
                            let same = logicalAnd(
                                (got[0].view(dtype: .uint16) .== expected.output.view(dtype: .uint16)).all(),
                                logicalAnd((got[1].view(dtype: .uint32) .== expected.finalState.view(dtype: .uint32)).all(),
                                logicalAnd((got[2].view(dtype: .uint16) .== normed.view(dtype: .uint16)).all(),
                                    (got[3].view(dtype: .uint32) .== packed.view(dtype: .uint32)).all())))
                            eval(got[0], got[1], got[2], got[3], same)
                            guard same.item(Bool.self), (counter .== MLXArray(UInt32(0))).all().item(Bool.self)
                            else { return false }
                        }
                    }
                    built[ObjectIdentifier(mixer)] = Entry(
                        counter: counter, norm: ObjectIdentifier(mixer.norm),
                        aLog: ObjectIdentifier(mixer.aLog), d: ObjectIdentifier(mixer.D),
                        out: ObjectIdentifier(mixer.outProj))
                }
                return true
            }
            guard matched else { return report(false, layers: 0) }
        } catch {
            return report(false, layers: 0)
        }
        lock.withLock { entries = built }
        report(true, layers: built.count)
    }

    private static func report(_ on: Bool, layers: Int) {
        FileHandle.standardError.write(
            Data("nemotron SSM gated join: \(on ? "on" : "off"), \(layers) qualified layers\n".utf8))
    }
}

// MARK: - Grouped strict-prefix replay

/// NEMOTRON_MTP_SSM_REPLAY_GROUP: when a round accepts a strict prefix, the
/// compact replay of every Mamba layer (`nemotronMTPSSMCompact(steps: keep,
/// writeOutput: false)`, one launch per layer) runs as ceil(layers / 6)
/// launches. Each launch covers up to six layers; a layer's operands are
/// selected by a per-threadgroup pointer table (`z / H`), and the body is the
/// compact kernel's source with only `n = z` -> `n = z % H` and the
/// `if constexpr (WO)` block (dead for replay) removed: every load, product,
/// sum, rounding point and store of a layer is the per-layer kernel's.
/// conv, dt, A_log, state_in and state_out per layer are five buffers, so
/// six layers (30) fit Metal's 31-entry buffer table.
enum NemotronMTPReplayGroupKernel {
    static let maxLayers = 6

    /// Balanced consecutive chunks of at most `maxLayers` (23 -> 6, 6, 6, 5).
    static func chunkSizes(_ count: Int) -> [Int] {
        guard count > 0 else { return [] }
        let chunks = (count + maxLayers - 1) / maxLayers
        return (0 ..< chunks).map { count / chunks + ($0 < count % chunks ? 1 : 0) }
    }

    private static let woBlock = """
            if constexpr (WO) {
                acc = simd_sum(acc);
                if (thread_index_in_simdgroup == 0) {
                    out[(t * H + n) * Dh + d_idx] = static_cast<T>(acc + x_ * D[h_idx]);
                }
            }

        """

    static func source(layers: Int) -> String {
        var s = NemotronMTPCompactSSMKernel.source
        func swap(_ old: String, _ new: String) {
            precondition(s.components(separatedBy: old).count == 2,
                         "ssm replay group source: \(old)")
            s = s.replacingOccurrences(of: old, with: new)
        }
        swap("auto n = thread_position_in_grid.z;", "auto n = thread_position_in_grid.z % H;")
        swap(woBlock, "")
        func list(_ name: String) -> String {
            (0 ..< layers).map { "\(name)\($0)" }.joined(separator: ", ")
        }
        let prelude = """
            const uint layer_ = thread_position_in_grid.z / H;
            const device T* conv_table_[\(layers)] = {\(list("conv"))};
            const device float* dt_table_[\(layers)] = {\(list("dt"))};
            const device AT* a_log_table_[\(layers)] = {\(list("A_log"))};
            const device U* state_in_table_[\(layers)] = {\(list("state_in"))};
            device U* state_out_table_[\(layers)] = {\(list("state_out"))};
            const device T* conv = conv_table_[layer_];
            const device float* dt = dt_table_[layer_];
            const device AT* A_log = a_log_table_[layer_];
            const device U* state_in = state_in_table_[layer_];
            device U* state_out = state_out_table_[layer_];

            """
        return prelude + s
    }

    nonisolated(unsafe) private static var kernels: [Int: MLXFast.MLXFastKernel] = [:]
    private static let lock = NSLock()
    private static func kernel(layers: Int) -> MLXFast.MLXFastKernel {
        lock.lock(); defer { lock.unlock() }
        if let kernel = kernels[layers] { return kernel }
        func names(_ name: String) -> [String] { (0 ..< layers).map { "\(name)\($0)" } }
        let kernel = MLXFast.metalKernel(
            name: "nemotron_mtp_ssm_replay_group\(layers)",
            inputNames: names("conv") + names("dt") + names("A_log") + names("state_in"),
            outputNames: names("state_out"),
            source: source(layers: layers))
        kernels[layers] = kernel
        return kernel
    }

    /// The final state after `keep` steps for every member, in member order.
    static func replay(_ members: [NemotronMTPReplayGroup.Member], keep: Int) -> [MLXArray] {
        guard let first = members.first else { return [] }
        let (heads, headDim, groups, stateDim) =
            (first.heads, first.headDim, first.groups, first.stateDim)
        let convDim = first.convolution.dim(2)
        var states: [MLXArray] = []
        states.reserveCapacity(members.count)
        var start = 0
        for size in chunkSizes(members.count) {
            let chunk = members[start ..< (start + size)]
            start += size
            let inputs = chunk.map(\.convolution) + chunk.map(\.dt) + chunk.map(\.aLog)
                + chunk.map(\.state)
            states += kernel(layers: size)(
                inputs,
                template: [("T", first.convolution.dtype), ("AT", first.aLog.dtype),
                           ("U", DType.float32),
                           ("Dh", headDim), ("Ds", stateDim), ("H", heads),
                           ("G", heads / groups), ("L", keep), ("CD", convDim),
                           ("BO", heads * headDim), ("CO", heads * headDim + groups * stateDim)],
                grid: (32, headDim, heads * size), threadGroup: (32, 8, 1),
                outputShapes: Array(repeating: [1, heads, headDim, stateDim], count: size),
                outputDTypes: Array(repeating: DType.float32, count: size))
        }
        return states
    }

    /// Compile every (chunk size, keep) instantiation `layerCount` Mamba
    /// layers can use (keep 1...7, the strict prefixes of windows 2...8) at
    /// load, so no JIT lands in a timed window. `convolution` is [1, 8, convDim].
    static func warm(layerCount: Int, convolution: MLXArray, aLog: MLXArray, D: MLXArray,
                     heads: Int, headDim: Int, groups: Int, stateDim: Int) -> [MLXArray] {
        guard CBv2CriticalPathCuts.ssmReplayGroup else { return [] }
        let width = convolution.dim(1)
        let member = NemotronMTPReplayGroup.Member(
            convolution: convolution,
            dt: MLXArray.zeros([1, width, heads], dtype: .float32), aLog: aLog, D: D,
            state: MLXArray.zeros([1, heads, headDim, stateDim], dtype: .float32),
            heads: heads, headDim: headDim, groups: groups, stateDim: stateDim)
        guard member.isValid else { return [] }
        var roots: [MLXArray] = []
        for size in Set(chunkSizes(layerCount)).sorted() {
            for keep in 1 ..< width {
                roots += replay(Array(repeating: member, count: size), keep: keep)
            }
        }
        return roots
    }
}

/// The compact replay operands of one verify forward's Mamba layers (one
/// recurrent evaluation), replayed together at commit. Each layer's replay
/// closure holds the group and its slot; the first one the commit calls
/// builds the grouped launches for every member, the rest read their slot.
final class NemotronMTPReplayGroup {
    struct Member {
        let convolution: MLXArray, dt: MLXArray, aLog: MLXArray, D: MLXArray, state: MLXArray
        let heads: Int, headDim: Int, groups: Int, stateDim: Int

        /// The operands `nemotronMTPSSMCompact` accepts.
        var isValid: Bool {
            let width = convolution.dim(1)
            return convolution.ndim == 3 && convolution.dim(0) == 1 && (2 ... 8).contains(width)
                && groups > 0 && heads % groups == 0
                && convolution.dim(2) == heads * headDim + 2 * groups * stateDim
                && dt.dtype == .float32 && dt.shape == [1, width, heads]
                && state.dtype == .float32 && state.shape == [1, heads, headDim, stateDim]
                && aLog.shape == [heads]
        }

        /// Same kernel instantiation as `other`.
        func matches(_ other: Member) -> Bool {
            convolution.shape == other.convolution.shape
                && convolution.dtype == other.convolution.dtype
                && aLog.dtype == other.aLog.dtype
                && heads == other.heads && headDim == other.headDim
                && groups == other.groups && stateDim == other.stateDim
        }
    }

    private weak var evaluation: CBv2RecurrentStateEvaluation?
    private var members: [Member] = []
    private var replayed: (keep: Int, states: [MLXArray])?
    nonisolated(unsafe) private static weak var open: NemotronMTPReplayGroup?

    private init(_ evaluation: CBv2RecurrentStateEvaluation) {
        self.evaluation = evaluation
    }

    /// Add `member` to `evaluation`'s open group (or open one). Returns nil
    /// when grouping is off or the operands are not the compact kernel's.
    static func join(_ evaluation: CBv2RecurrentStateEvaluation, _ member: Member)
        -> (group: NemotronMTPReplayGroup, slot: Int)? {
        guard CBv2CriticalPathCuts.ssmReplayGroup, member.isValid else { return nil }
        let group: NemotronMTPReplayGroup
        if let current = open, current.evaluation === evaluation, current.replayed == nil,
           current.members.first.map({ $0.matches(member) }) ?? true {
            group = current
        } else {
            group = NemotronMTPReplayGroup(evaluation)
            open = group
        }
        group.members.append(member)
        return (group, group.members.count - 1)
    }

    /// The state after `keep` steps of member `slot`: bitwise the per-layer
    /// `nemotronMTPSSMCompact(steps: keep, writeOutput: false).finalState`.
    func finalState(slot: Int, keep: Int) -> MLXArray {
        if replayed?.keep != keep {
            replayed = (keep, NemotronMTPReplayGroupKernel.replay(members, keep: keep))
        }
        let state = replayed!.states[slot]
        guard NemotronMoELeader.selfTest else { return state }
        let m = members[slot]
        return nemotronBitwiseOrNaN(state, nemotronMTPSSMCompact(
            convolution: m.convolution, ALog: m.aLog, D: m.D, dt: m.dt, state: m.state,
            heads: m.heads, headDim: m.headDim, groups: m.groups, stateDim: m.stateDim,
            steps: keep, writeOutput: false).finalState)
    }
}
