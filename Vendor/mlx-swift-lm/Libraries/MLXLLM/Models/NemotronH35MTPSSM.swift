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
