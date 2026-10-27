// Batched target verification with the ordinary one-token Mamba recurrence.
// Projections are built once over the window. Every confirmed-prefix state is
// staged for the engine's existing transactional commit/rollback mechanism.
import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// NEMOTRON_MTP_SSM_PENDING_FOLD: a strict-prefix commit replays every Mamba
/// layer's SSM state over the accepted steps, and the grouped replay reads and
/// writes each layer's FP32 state. The replayed state is an array the commit
/// only builds; this table maps it to its operands (an evaluated anchor state
/// and the accepted steps' conv/dt tape). When the next compact verify of that
/// layer receives it as its initial state, the fused SSM launch runs those
/// steps from the anchor before its window (`NemotronHCompactSSMGated
/// .pendingKernel`, the replay kernel's per-step statements) and the replay
/// array is never evaluated. Every other reader of the committed state still
/// evaluates the replay. Strict prefixes in a row extend the pending steps up
/// to `maxSteps`. Admitted at load only after the folded launch matched replay
/// + launch bitwise for every step count (`qualify`).
final class NemotronMTPPendingSSM {
    struct Pending {
        let anchor: MLXArray, conv: MLXArray, dt: MLXArray
        let steps: Int
    }
    static let maxSteps = 4
    static let requested =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_SSM_PENDING_FOLD"] != "0"
    nonisolated(unsafe) static var qualified = false
    static var enabled: Bool { requested && qualified }

    private final class Entry {
        weak var state: MLXArray?
        let pending: Pending
        init(_ state: MLXArray, _ pending: Pending) {
            self.state = state
            self.pending = pending
        }
    }
    private static let lock = NSLock()
    nonisolated(unsafe) private static var entries: [ObjectIdentifier: Entry] = [:]

    /// `state` is the replay of `pending.steps` steps of `pending` from its anchor.
    static func register(_ state: MLXArray, _ pending: Pending) {
        guard enabled, (1 ... maxSteps).contains(pending.steps) else { return }
        lock.withLock {
            entries = entries.filter { $0.value.state != nil }
            entries[ObjectIdentifier(state)] = Entry(state, pending)
        }
    }

    /// The pending replay `state` stands for, if it is one (removed: a second
    /// reader evaluates the replay).
    static func take(_ state: MLXArray) -> Pending? {
        guard enabled else { return nil }
        return lock.withLock {
            guard let entry = entries.removeValue(forKey: ObjectIdentifier(state)),
                  entry.state === state else { return nil }
            return entry.pending
        }
    }

    /// Load time: compile the folded launch for every step count and admit it
    /// only if its normed rows, prepared activation and final state equal the
    /// replay followed by the plain fused launch, bitwise, on seeded operands.
    static func qualify(_ mixer: NemotronHMamba2Mixer) {
        guard requested else { return }
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron mtp ssm pending fold: \(message)\n".utf8))
        }
        let (heads, dim, groups, stateDim) =
            (mixer.numHeads, mixer.headDim, mixer.numGroups, mixer.ssmStateSize)
        var seed: UInt64 = 0x9E37_79B9_7F4A_7C15
        func values(_ count: Int, _ low: Float, _ high: Float) -> [Float] {
            (0 ..< count).map { _ -> Float in
                seed = seed &* 6364136223846793005 &+ 1442695040888963407
                return low + (high - low) * Float(seed >> 40) / Float(1 << 24)
            }
        }
        func bits(_ a: MLXArray, _ b: MLXArray) -> Bool {
            let dtype: DType = a.dtype == .float32 ? .uint32 : .uint16
            return (a.view(dtype: dtype) .== b.view(dtype: dtype)).all().item(Bool.self)
        }
        let convDim = mixer.convDim
        let projectedWidth = mixer.intermediateSize + convDim + heads
        for steps in 1 ... maxSteps {
            let pconv = MLXArray(values(steps * convDim, -2, 2), [1, steps, convDim]).asType(.bfloat16)
            let pdt = MLXArray(values(steps * heads, 0.01, 0.5), [1, steps, heads])
            let conv = MLXArray(values(3 * convDim, -2, 2), [1, 3, convDim]).asType(.bfloat16)
            let dt = MLXArray(values(3 * heads, 0.01, 0.5), [1, 3, heads])
            let anchor = MLXArray(values(heads * dim * stateDim, -0.5, 0.5), [1, heads, dim, stateDim])
            let projected = MLXArray(values(3 * projectedWidth, -3, 3), [1, 3, projectedWidth])
                .asType(.bfloat16)
            let replayed = nemotronMTPSSMCompact(
                convolution: pconv, ALog: mixer.aLog, D: mixer.D, dt: pdt, state: anchor,
                heads: heads, headDim: dim, groups: groups, stateDim: stateDim,
                steps: steps, writeOutput: false).finalState
            guard let reference = mixer.norm.compactSSMPreparedRows(
                    convolution: conv, ALog: mixer.aLog, D: mixer.D, dt: dt, state: replayed,
                    projected: projected, heads: heads, headDim: dim, groups: groups,
                    stateDim: stateDim),
                  let folded = mixer.norm.compactSSMPreparedRows(
                    convolution: conv, ALog: mixer.aLog, D: mixer.D, dt: dt, state: anchor,
                    projected: projected, heads: heads, headDim: dim, groups: groups,
                    stateDim: stateDim, pending: (pconv, pdt, steps))
            else { return report("off (fused launch unavailable)") }
            eval(reference.0, reference.1, reference.2, folded.0, folded.1, folded.2)
            guard bits(reference.0, folded.0), bits(reference.1, folded.1), bits(reference.2, folded.2)
            else { return report("off (folded launch differs at \(steps) pending steps)") }
        }
        qualified = true
        report("on (1...\(maxSteps) pending steps bitwise equal to replay + launch)")
    }
}

/// Cache the complete numerical Mamba verify window, with explicit current
/// conv/SSM inputs and explicit replay-tape outputs. No evaluation object,
/// transaction, prefix-replay closure or request identity enters this graph.
final class NemotronH35MTPCompactGraphCache {
    private struct Entry {
        let identity: [ObjectIdentifier]
        let function: @Sendable ([MLXArray]) -> [MLXArray]
    }
    private static let enabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_COMPILED_MAMBA_WINDOW"] != "0"
    private let lock = NSLock()
    /// [plain, consuming a prepared in_proj activation] x [unjoined, joined]
    /// x pending steps (0 ... NemotronMTPPendingSSM.maxSteps): separate graphs.
    private var entries: [Int: Entry] = [:]
    private var joinTimed = false
    private var joinChecked = false
    private static let joinCheck = ProcessInfo.processInfo.environment["NEMOTRON_MTP_MAMBA_JOIN_CHECK"] == "1"
    /// Timing records are not written in this build (no file output).
    private static let joinTiming: String? = nil

    func supportsJoined(_ hidden: MLXArray, owner: NemotronHMamba2Mixer) -> Bool {
        Self.enabled && NemotronMTPExecution.batchedM1 && NemotronMTPExecution.batchedNorm
            && Device.defaultDevice().deviceType == .gpu && StreamOrDevice.default == .gpu
            && hidden.shape == [1, 3, 2688] && hidden.dtype == .bfloat16
            && Self.nativeIdentity(owner) != nil
    }

    func callAsFunction(
        _ hidden: MLXArray, convState: MLXArray, initialSSM: MLXArray,
        prepared: MLXArray? = nil, pending: NemotronMTPPendingSSM.Pending? = nil,
        owner: NemotronHMamba2Mixer,
        residual: MLXArray? = nil, nextNorm: RMSNorm? = nil
    ) -> [MLXArray] {
        let fold = pending.map { (conv: $0.conv, dt: $0.dt, steps: $0.steps) }
        let joins = residual != nil && nextNorm != nil
        func finish(_ values: [MLXArray], _ lhs: MLXArray?) -> [MLXArray] {
            guard joins, let lhs, let nextNorm else { return values }
            guard let joined = NemotronHResidualRMSNorm.applyPreparedRows(
                lhs, values[0], norm: nextNorm) else {
                preconditionFailure("Qualified Mamba compiled join became ineligible")
            }
            return [joined.0] + Array(values[1...4]) + [joined.1, joined.2]
        }
        guard Self.enabled, NemotronMTPExecution.batchedM1, NemotronMTPExecution.batchedNorm,
              Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              hidden.ndim == 3, hidden.dim(0) == 1, (2...4).contains(hidden.dim(1)),
              hidden.dim(2) == 2688, hidden.dtype == .bfloat16,
              convState.shape == [1, 3, 6144], convState.dtype == .bfloat16,
              initialSSM.shape == [1, 64, 64, 128], initialSSM.dtype == .float32,
              var identity = Self.nativeIdentity(owner)
        else {
            return finish(owner.mtpCompactArrays(
                hidden, convState: convState, initialSSM: initialSSM, prepared: prepared,
                pending: fold), residual)
        }
        if joins, let nextNorm {
            identity += [ObjectIdentifier(nextNorm), ObjectIdentifier(nextNorm.weight)]
        }
        let usesPrepared = prepared != nil
        let pendingSteps = pending?.steps ?? 0
        let slot = (usesPrepared ? 1 : 0) + (joins ? 2 : 0) + 4 * pendingSteps
        var retired: Entry?
        let current = lock.withLock { () -> Entry in
            if let entry = entries[slot], entry.identity == identity { return entry }
            let state = owner.innerState() + (joins ? [nextNorm!.weight] : [])
            let function: @Sendable ([MLXArray]) -> [MLXArray] = compile(inputs: [state]) {
                [weak owner] arrays in
                guard let owner else { preconditionFailure("Released Nemotron MTP Mamba-window owner") }
                let base = pendingSteps > 0 ? 5 : 3
                let values = owner.mtpCompactArrays(
                    arrays[0], convState: arrays[1], initialSSM: arrays[2],
                    prepared: usesPrepared ? arrays[base] : nil,
                    pending: pendingSteps > 0 ? (arrays[3], arrays[4], pendingSteps) : nil)
                guard joins, let nextNorm else { return values }
                guard let joined = NemotronHResidualRMSNorm.applyPreparedRows(
                    arrays[base + (usesPrepared ? 1 : 0)], values[0], norm: nextNorm) else {
                    preconditionFailure("Qualified Mamba compiled join became ineligible")
                }
                return [joined.0] + Array(values[1...4]) + [joined.1, joined.2]
            }
            let replacement = Entry(identity: identity, function: function)
            retired = entries[slot]
            entries[slot] = replacement
            return replacement
        }
        let result = withExtendedLifetime(retired) {
            current.function([hidden, convState, initialSSM]
                + (pending.map { [$0.conv, $0.dt] } ?? []) + (prepared.map { [$0] } ?? [])
                + (joins ? [residual!] : []))
        }
        guard result.count == (joins ? 7 : 5) else {
            // Retain any enclosing MLX error while avoiding empty indexing.
            return finish(owner.mtpCompactArrays(
                hidden, convState: convState, initialSSM: initialSSM, prepared: prepared,
                pending: fold), residual)
        }
        guard NemotronMoELeader.selfTest else { return result }
        // Reference: in_proj prepares its activation from `hidden` itself.
        let expected = finish(owner.mtpCompactArrays(
            hidden, convState: convState, initialSSM: initialSSM, pending: fold), residual)
        return zip(result, expected).enumerated().map {
            joins && $0.offset == 6 ? $0.element.0
                : nemotronBitwiseOrNaN($0.element.0, $0.element.1)
        }
    }

    func checkAndBenchmarkJoin(_ hidden: MLXArray, convState: MLXArray, initialSSM: MLXArray,
                              prepared: MLXArray?, residual: MLXArray, nextNorm: RMSNorm,
                              owner: NemotronHMamba2Mixer, layerIndex: Int, result: [MLXArray]) {
        let check = Self.joinCheck
        let directory = Self.joinTiming
        guard check || directory != nil else { return }
        func make(_ joined: Bool) -> ([MLXArray], (MLXArray, MLXArray, MLXArray)) {
            if joined {
                precondition(supportsJoined(hidden, owner: owner)
                    && NemotronHResidualRMSNorm.plumbingAvailable
                    && residual.shape == hidden.shape && residual.dtype == .bfloat16
                    && type(of: nextNorm) == RMSNorm.self
                    && nextNorm.weight.shape == [2688] && nextNorm.weight.dtype == .bfloat16)
            }
            let y = self(hidden, convState: convState, initialSSM: initialSSM,
                prepared: prepared, owner: owner,
                residual: joined ? residual : nil, nextNorm: joined ? nextNorm : nil)
            if joined { return (y, (y[0], y[5], y[6])) }
            let j = NemotronHResidualRMSNorm.applyPreparedRows(residual, y[0], norm: nextNorm)!
            return (y, j)
        }
        func roots(_ r: ([MLXArray], (MLXArray, MLXArray, MLXArray))) -> [MLXArray] {
            [r.1.0] + Array(r.0[1...4]) + [r.1.1, r.1.2]
        }
        if check {
            let expected = roots(make(false))
            for index in 0..<6 {
                let dtype: DType = result[index].dtype == .bfloat16 ? .uint16 : .uint32
                precondition((result[index].view(dtype: dtype) .== expected[index].view(dtype: dtype))
                    .all().item(Bool.self), "Mamba compiled join output/tape bit mismatch")
            }
            let got = result[6].reshaped([3, 11, 9, 32]).view(dtype: .uint32)
            let want = expected[6].reshaped([3, 11, 9, 32]).view(dtype: .uint32)
            precondition(MLX.logicalAnd(
                (got[0..., 0..<10, 0..., 0...] .== want[0..., 0..<10, 0..., 0...]).all(),
                (got[0..., 10, 0..., 0..<16] .== want[0..., 10, 0..., 0..<16]).all()).item(Bool.self),
                "Mamba compiled join live prepared bits mismatch")
            lock.withLock {
                if !joinChecked {
                    fputs("nemotron compiled Mamba join: residual, norm, live prepared and all replay tape bits exact\n", stderr)
                    joinChecked = true
                }
            }
        }
        guard let directory, lock.withLock({ () -> Bool in
            if joinTimed { return false }; joinTimed = true; return true
        }) else { return }
        eval([hidden, convState, initialSSM, residual] + (prepared.map { [$0] } ?? []))
        for i in 0..<16 { eval(roots(make(i % 2 == 0))) }
        var blocks: [[String: Any]] = []
        for block in 0..<13 {
            var measurements: [String: Any] = ["block": block]
            for joined in (block % 2 == 0 ? [false, true] : [true, false]) {
                var times: [Double] = []
                for _ in 0..<64 {
                    let start = DispatchTime.now().uptimeNanoseconds
                    let y = make(joined)
                    let elapsed = DispatchTime.now().uptimeNanoseconds - start
                    times.append(Double(elapsed) / 1000.0)
                    eval(roots(y))
                }
                measurements[joined ? "candidate_us" : "frontier_us"] = times
            }
            blocks.append(measurements)
        }
        let record: [String: Any] = ["layer": layerIndex, "blocks": blocks,
            "scope": "host graph construction only; GPU eval excluded",
            "prepared": prepared != nil, "rows": hidden.shape]
        let data = try! JSONSerialization.data(withJSONObject: record, options: [.sortedKeys])
        try! data.write(to: URL(fileURLWithPath: directory)
            .appendingPathComponent("layer-\(layerIndex).json"), options: .atomic)
    }

    private static func nativeIdentity(_ owner: NemotronHMamba2Mixer) -> [ObjectIdentifier]? {
        func linear(_ value: Linear) -> Bool {
            type(of: value) == Linear.self || type(of: value) == QuantizedLinear.self
        }
        guard type(of: owner) == NemotronHMamba2Mixer.self,
              owner.numHeads == 64, owner.headDim == 64, owner.ssmStateSize == 128,
              owner.numGroups == 8, owner.convKernelSize == 4, owner.convDim == 6144,
              type(of: owner.conv1d) == Conv1d.self,
              type(of: owner.norm) == NemotronHRMSNormGated.self,
              linear(owner.inProj), linear(owner.outProj) else { return nil }
        let modules: [Module] = [owner, owner.inProj, owner.conv1d, owner.norm, owner.outProj]
        return modules.map(ObjectIdentifier.init)
    }
}

extension NemotronHMamba2Mixer {
    func cbv2ForwardCapturedJoined(
        _ hidden: MLXArray, modelLayerIndex: Int,
        recurrentState: [CBv2RecurrentStateEvaluation], prepared: MLXArray?,
        residual: MLXArray, nextNorm: RMSNorm
    ) -> (MLXArray, MLXArray, MLXArray)? {
        guard nemotronMTPCompactSSM, recurrentState.count == 1,
              mtpCompactGraphCache.supportsJoined(hidden, owner: self),
              NemotronHResidualRMSNorm.plumbingAvailable,
              residual.shape == hidden.shape, residual.dtype == .bfloat16,
              type(of: nextNorm) == RMSNorm.self,
              nextNorm.weight.shape == [2688], nextNorm.weight.dtype == .bfloat16 else { return nil }
        let evaluation = recurrentState[0]
        let initial = evaluation.inputState(modelLayerIndex: modelLayerIndex)
        let conv = initial?.conv ?? MLXArray.zeros([1, 3, 6144], dtype: hidden.dtype)
        let ssm = initial?.ssm ?? MLXArray.zeros([1, 64, 64, 128], dtype: .float32)
        let y = mtpCompactForward(hidden, modelLayerIndex: modelLayerIndex,
            evaluation: evaluation, convState: conv, initialSSM: ssm,
            prepared: prepared, residual: residual, nextNorm: nextNorm)
        return (y[0], y[5], y[6])
    }

    func cbv2ForwardCaptured(
        _ hiddenStates: MLXArray, modelLayerIndex: Int,
        recurrentState: [CBv2RecurrentStateEvaluation], prepared: MLXArray? = nil
    ) -> MLXArray {
        let batch = hiddenStates.dim(0), length = hiddenStates.dim(1)
        precondition(batch == 1 && recurrentState.count == 1 && (1...8).contains(length))
        let evaluation = recurrentState[0]
        let initial = evaluation.inputState(modelLayerIndex: modelLayerIndex)
        let convState = initial?.conv ?? MLXArray.zeros(
            [batch, max(0, convKernelSize - 1), convDim], dtype: hiddenStates.dtype)
        let ssmState = initial?.ssm ?? MLXArray.zeros(
            [batch, numHeads, headDim, ssmStateSize], dtype: .float32)
        if nemotronMTPCompactSSM && length > 1 {
            return mtpCompactForward(hiddenStates, modelLayerIndex: modelLayerIndex,
                evaluation: evaluation, convState: convState, initialSSM: ssmState,
                prepared: prepared)[0]
        }
        let arrays = mtpCapturedArrays(hiddenStates, convState: convState, initialSSM: ssmState)
        do {
            try evaluation.stageCaptured(modelLayerIndex: modelLayerIndex,
                conv: arrays[1], ssm: arrays[2], positions: length)
        } catch {
            preconditionFailure("Nemotron MTP captured-state stage failed: \(error)")
        }
        return arrays[0]
    }

    /// Compact verify: the window kernel writes only the final SSM state and
    /// the window's y; a strict accepted prefix replays the same kernel for
    /// `keep` steps from the same input state at commit (bitwise equal to the
    /// captured per-position state). Conv states are views of `padded`.
    func mtpCompactForward(
        _ hiddenStates: MLXArray, modelLayerIndex: Int,
        evaluation: CBv2RecurrentStateEvaluation, convState: MLXArray, initialSSM: MLXArray,
        prepared: MLXArray? = nil, residual: MLXArray? = nil, nextNorm: RMSNorm? = nil
    ) -> [MLXArray] {
        let length = hiddenStates.dim(1)
        // NEMOTRON_MTP_SSM_PENDING_FOLD: a committed strict-prefix replay
        // arrives as its anchor and tape; the window's fused launch runs it.
        let pending = length == 3 ? NemotronMTPPendingSSM.take(initialSSM) : nil
        let start = pending?.anchor ?? initialSSM
        let arrays = mtpCompactGraphCache(
            hiddenStates, convState: convState, initialSSM: start, prepared: prepared,
            pending: pending, owner: self, residual: residual, nextNorm: nextNorm)
        if let residual, let nextNorm, pending == nil {
            mtpCompactGraphCache.checkAndBenchmarkJoin(hiddenStates, convState: convState,
                initialSSM: initialSSM, prepared: prepared, residual: residual, nextNorm: nextNorm,
                owner: self, layerIndex: modelLayerIndex, result: arrays)
        }
        let (padded, convolution, dt, finalSSM) = (arrays[1], arrays[2], arrays[3], arrays[4])
        let (heads, dim, groups, state) = (numHeads, headDim, numGroups, ssmStateSize)
        let (logA, skip) = (self.aLog, self.D)
        let history = convKernelSize - 1
        let tape = [padded, convolution, dt]
        let tapeBytes = tape.reduce(0) { $0 + $1.nbytes }
        // NEMOTRON_MTP_SSM_REPLAY_GROUP: replay with this forward's other
        // Mamba layers in grouped launches (same per-layer statements).
        let grouped = pending == nil ? NemotronMTPReplayGroup.join(evaluation, .init(
            convolution: convolution, dt: dt, aLog: logA, D: skip, state: initialSSM,
            heads: heads, headDim: dim, groups: groups, stateDim: state)) : nil
        let pendingTape = pending.map { [$0.conv, $0.dt] } ?? []
        let pendingBytes = pendingTape.reduce(0) { $0 + $1.nbytes }
        do {
            try evaluation.stagePrefixReplay(
                modelLayerIndex: modelLayerIndex, positions: length,
                finalConv: padded[0..., length ..< (length + history), 0...],
                finalSSM: finalSSM,
                materializedByteCount: tapeBytes + finalSSM.nbytes,
                evaluationRoots: tape,
                strictReplayRetainedByteCount: tapeBytes + start.nbytes + pendingBytes,
                strictReplayRetainedRoots: tape + [start] + pendingTape,
                replay: { keep in
                    let ssm: MLXArray
                    if let pending {
                        // The window started `pending.steps` steps past the
                        // anchor: replay those and the accepted ones from it.
                        let steps = pending.steps + keep
                        let tapeConv = concatenated([pending.conv, convolution[0..., ..<keep, 0...]], axis: 1)
                        let tapeDt = concatenated([pending.dt, dt[0..., ..<keep, 0...]], axis: 1)
                        ssm = nemotronMTPSSMCompact(
                            convolution: tapeConv, ALog: logA, D: skip, dt: tapeDt,
                            state: pending.anchor, heads: heads, headDim: dim, groups: groups,
                            stateDim: state, steps: steps, writeOutput: false).finalState
                        NemotronMTPPendingSSM.register(ssm, .init(
                            anchor: pending.anchor, conv: tapeConv, dt: tapeDt, steps: steps))
                    } else {
                        ssm = grouped.map { $0.group.finalState(slot: $0.slot, keep: keep) }
                            ?? nemotronMTPSSMCompact(
                            convolution: convolution, ALog: logA, D: skip, dt: dt,
                            state: initialSSM, heads: heads, headDim: dim, groups: groups,
                            stateDim: state, steps: keep, writeOutput: false).finalState
                        NemotronMTPPendingSSM.register(ssm, .init(
                            anchor: initialSSM, conv: convolution[0..., ..<keep, 0...],
                            dt: dt[0..., ..<keep, 0...], steps: keep))
                    }
                    return CBv2RecurrentLayerState(
                        conv: padded[0..., keep ..< (keep + history), 0...], ssm: ssm)
                })
        } catch {
            preconditionFailure("Nemotron MTP compact recurrent stage failed: \(error)")
        }
        return arrays
    }

    /// Original numerical graph, with request state supplied as dynamic arrays.
    /// Return its replay tape explicitly; staging/commit stay outside compile.
    /// `inputPrepared`: in_proj's K=2688 activation, prepared by the preceding
    /// join from these exact BF16 rows (otherwise prepared here).
    func mtpCompactArrays(
        _ hiddenStates: MLXArray, convState: MLXArray, initialSSM: MLXArray,
        prepared inputPrepared: MLXArray? = nil,
        pending: (conv: MLXArray, dt: MLXArray, steps: Int)? = nil
    ) -> [MLXArray] {
        let batch = hiddenStates.dim(0), length = hiddenStates.dim(1)
        // The window's conv, padded history and timestep come from in_proj's
        // own epilogue when the shapes allow it (same values, one launch fewer).
        var fusedWindow: (MLXArray, MLXArray, MLXArray)? = nil
        let projected: MLXArray
        if batch == 1, hiddenStates.ndim == 3, convKernelSize == 4, conv1d.groups == convDim,
           conv1d.stride == 1, conv1d.dilation == 1, conv1d.padding == 0,
           let convBias = conv1d.bias, let q = inProj as? QuantizedLinear,
           type(of: q) == QuantizedLinear.self,
           let fused = NemotronMambaInProjConvDt.apply(
               hiddenStates.reshaped(length, hiddenStates.dim(2)), q, prepared: inputPrepared,
               oldState: convState, convWeight: conv1d.weight, convBias: convBias,
               dtBias: dtBias, timeStepLimit: timeStepLimit,
               convStart: intermediateSize, convDim: convDim, heads: numHeads) {
            projected = fused.projected.reshaped(batch, length, -1)
            fusedWindow = (fused.convolution, fused.padded, fused.dt)
        } else {
            projected = nemotronMTPLinearRows(hiddenStates, inProj, prepared: inputPrepared)
        }
        let parts = split(projected, indices: [intermediateSize, intermediateSize + convDim], axis: -1)
        let gate = parts[0], input = parts[1]
        let rawDt = parts[2].reshaped(batch, length, numHeads)
        func separateWindow() -> (MLXArray, MLXArray, MLXArray) {
            packedConvDt(convState, input, rawDt) ?? {
                let (convolution, padded) = packedConv(convState, input)
                return (convolution, padded, nemotronPrefillComputeDt(rawDt, dtBias, timeStepLimit))
            }()
        }
        var (convolution, padded, dt) = fusedWindow ?? separateWindow()
        if fusedWindow != nil, NemotronMambaInProjConvDt.check {
            let reference = separateWindow()
            convolution = nemotronBitwiseOrNaN(convolution, reference.0)
            padded = nemotronBitwiseOrNaN(padded, reference.1)
            dt = nemotronBitwiseOrNaN(dt, reference.2)
        }
        let (heads, dim, groups, state) = (numHeads, headDim, numGroups, ssmStateSize)
        let (logA, skip) = (self.aLog, self.D)
        // NEMOTRON_SSM_GATED_FUSED: the compact SSM window, the gated RMS norm
        // and out_proj's prepared activation in one launch (see
        // NemotronHCompactSSMGated); the same normed rows, prepared activation
        // and final state as the two launches below.
        if length == 3, NemotronMTPExecution.batchedNorm,
           let q = outProj as? QuantizedLinear, q.mode == .affine,
           q.bits == 4, q.groupSize == 64, q.bias == nil,
           q.weight.shape == [2688, 512], q.weight.dtype == .uint32,
           q.scales.shape == [2688, 64], q.scales.dtype == .bfloat16,
           q.biases?.shape == [2688, 64], q.biases?.dtype == .bfloat16,
           var fused = norm.compactSSMPreparedRows(
               convolution: convolution, ALog: logA, D: skip, dt: dt, state: initialSSM,
               projected: projected, heads: heads, headDim: dim, groups: groups, stateDim: state,
               pending: pending) {
            if NemotronMoELeader.selfTest {
                // Reference: the two launches (compact SSM, then the gated
                // norm with the prepared output) on the same operands.
                let reference = nemotronMTPSSMCompact(
                    convolution: convolution, ALog: logA, D: skip, dt: dt,
                    state: replayedPending(pending, initialSSM),
                    heads: heads, headDim: dim, groups: groups, stateDim: state,
                    steps: length, writeOutput: true)
                if let pair = norm.preparedRows(
                    reference.output.reshaped(batch, length, heads * dim), gate: gate) {
                    fused = (nemotronBitwiseOrNaN(fused.0, pair.0),
                             nemotronBitwiseOrNaN(fused.1, pair.1),
                             nemotronBitwiseOrNaN(fused.2, reference.finalState))
                }
            }
            return [nemotronMTPLinearRows(fused.0, outProj, prepared: fused.1),
                    padded, convolution, dt, fused.2]
        }
        let result = nemotronMTPSSMCompact(
            convolution: convolution, ALog: logA, D: skip, dt: dt,
            state: replayedPending(pending, initialSSM),
            heads: heads, headDim: dim, groups: groups, stateDim: state,
            steps: length, writeOutput: true)
        let y = result.output.reshaped(batch, length, heads * dim)
        let normalized: MLXArray
        let prepared: MLXArray?
        if NemotronMTPExecution.batchedNorm,
           let q = outProj as? QuantizedLinear, q.mode == .affine,
           q.bits == 4, q.groupSize == 64, q.bias == nil,
           q.weight.shape == [2688, 512], q.weight.dtype == .uint32,
           q.scales.shape == [2688, 64], q.scales.dtype == .bfloat16,
           q.biases?.shape == [2688, 64], q.biases?.dtype == .bfloat16,
           let pair = norm.preparedRows(y, gate: gate) {
            normalized = pair.0
            prepared = pair.1
        } else {
            normalized = NemotronMTPExecution.batchedNorm
                ? norm(y, gate: gate)
                : concatenated((0..<length).map { position in
                norm(y[0..., position..<(position + 1), 0...],
                     gate: gate[0..., position..<(position + 1), 0...])
                }, axis: 1)
            prepared = nil
        }
        return [nemotronMTPLinearRows(normalized, outProj, prepared: prepared),
                padded, convolution, dt, result.finalState]
    }

    /// The window's start state: `anchor` itself, or `anchor` advanced over
    /// the pending tape by the compact replay (outside the fused launch).
    func replayedPending(
        _ pending: (conv: MLXArray, dt: MLXArray, steps: Int)?, _ anchor: MLXArray
    ) -> MLXArray {
        guard let pending else { return anchor }
        return nemotronMTPSSMCompact(
            convolution: pending.conv, ALog: aLog, D: D, dt: pending.dt, state: anchor,
            heads: numHeads, headDim: headDim, groups: numGroups, stateDim: ssmStateSize,
            steps: pending.steps, writeOutput: false).finalState
    }

    func mtpCapturedArrays(_ hiddenStates: MLXArray, convState: MLXArray,
                           initialSSM: MLXArray) -> [MLXArray] {
        let batch = hiddenStates.dim(0), length = hiddenStates.dim(1)
        var ssmState = initialSSM
        let projected = nemotronMTPLinearRows(hiddenStates, inProj)
        let parts = split(projected, indices: [intermediateSize, intermediateSize + convDim], axis: -1)
        let gate = parts[0], input = parts[1], dt = parts[2]
        // One causal depthwise convolution covers the complete window. Each
        // position still owns its exact history slice and ordinary M=1 SSM
        // update, so rejecting a suffix cannot advance persistent recurrence.
        let (convolution, padded) = packedConv(convState, input)
        // Per-position conv histories as one strided view of `padded`
        // (window p = rows p+1 ..< p+convKernelSize). Pure data movement:
        // identical values to stacking the per-position slices.
        let convStack = asStrided(
            padded, [length, convKernelSize - 1, convDim],
            strides: [convDim, convDim, 1], offset: convDim)
        var outputs: [MLXArray] = []
        let capturedSSM: MLXArray
        if NemotronMTPExecution.windowSSM && length > 1 {
            let values = split(convolution,
                indices: [intermediateSize, intermediateSize + numGroups * ssmStateSize], axis: -1)
            let result = nemotronMTPSSMWindow(
                hiddenStates: values[0].reshaped(batch, length, numHeads, headDim),
                ALog: aLog, B: values[1].reshaped(batch, length, numGroups, ssmStateSize),
                C: values[2].reshaped(batch, length, numGroups, ssmStateSize), D: D,
                dt: dt.reshaped(batch, length, numHeads), dtBias: dtBias,
                state: ssmState, timeStepLimit: timeStepLimit)
            capturedSSM = result.capturedStates
            if NemotronMTPExecution.batchedNorm {
                // [1, L, H, Dh] is contiguous: one reshape equals the
                // concatenation of the per-position flattened slices.
                outputs = [result.output.reshaped(batch, length, -1)]
            } else {
                outputs = (0..<length).map {
                    result.output[0..., $0..<($0 + 1), 0..., 0...].flattened(start: 2)
                }
            }
        } else {
            var ssmStates: [MLXArray] = []
            for position in 0..<length {
                let values = split(convolution[0..., position..<(position + 1), 0...],
                    indices: [intermediateSize, intermediateSize + numGroups * ssmStateSize], axis: -1)
                let (y, next) = ssmUpdate(
                    hiddenStates: values[0].reshaped(batch, 1, numHeads, headDim),
                    ALog: aLog, B: values[1].reshaped(batch, 1, numGroups, ssmStateSize),
                    C: values[2].reshaped(batch, 1, numGroups, ssmStateSize), D: D,
                    dt: dt[0..., position..<(position + 1), 0...].reshaped(batch, 1, numHeads),
                    dtBias: dtBias, state: ssmState, timeStepLimit: timeStepLimit, mask: nil)
                ssmState = next.asType(.float32)
                ssmStates.append(ssmState)
                outputs.append(y.asType(hiddenStates.dtype).flattened(start: 2))
            }
            capturedSSM = concatenated(ssmStates, axis: 0)
        }
        let normalized: MLXArray
        if NemotronMTPExecution.batchedNorm {
            normalized = norm(
                outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 1), gate: gate)
        } else {
            normalized = concatenated(outputs.enumerated().map { position, y in
                norm(y, gate: gate[0..., position..<(position + 1), 0...])
            }, axis: 1)
        }
        return [nemotronMTPLinearRows(normalized, outProj), convStack, capturedSSM]
    }
}


extension NemotronH35Model: CBv2RecurrentCaptureMTPForwardable {
    public func cbv2ForwardWithHiddenCaptured(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        precondition(positionIds == nil)
        if let fused = cbv2HiddenCapturedFused(
            tokens, caches: caches, recurrentState: recurrentState) {
            var output = lmHead.map { nemotronMTPHeadRows(fused.hidden, $0, prepared: fused.prepared) }
                ?? nemotronMTPMapRows(fused.hidden) { logits($0) }
            if NemotronMTPHeadTwice.enabled, let lmHead {
                // The same head launch again, ordered after the first; consumers wait for it.
                let again = nemotronMTPHeadRows(
                    depends(input: fused.hidden, dependencies: [output]), lmHead,
                    prepared: fused.prepared.map { depends(input: $0, dependencies: [output]) })
                output = depends(input: output, dependencies: [again])
            }
            return (output, fused.hidden)
        }
        let hidden = cbv2Hidden(tokens, caches: caches, recurrentState: recurrentState, captureMTP: true)
        let output = lmHead.map { nemotronMTPHeadRows(hidden, $0) }
            ?? nemotronMTPMapRows(hidden) { logits($0) }
        return (output, hidden)
    }
}

extension NemotronH35Model {
    /// Compile, at load time, every SSM verify-kernel instantiation an MTP
    /// round can use (window widths 2...8; compact replay prefixes 1...7), so
    /// a JIT compile never lands inside a timed decode window. Dummy inputs
    /// use the real parameter dtypes, hence the exact runtime instantiations.
    func warmMTPSSMKernels(activationDType: DType) {
        guard let mixer = modules().lazy.compactMap({ $0 as? NemotronHMamba2Mixer }).first
        else { return }
        let history = mixer.convKernelSize - 1
        let (heads, dim, groups, stateDim) =
            (mixer.numHeads, mixer.headDim, mixer.numGroups, mixer.ssmStateSize)
        let normInput = MLXArray.zeros([1, 1, heads * dim], dtype: activationDType)
        var roots: [MLXArray] = [mixer.norm(normInput, gate: normInput)]
        // NEMOTRON_SSM_GATED_FUSED: the zeroed arrival counters and the
        // fused launch's pipeline, here at load (zero operands leave the
        // counters at zero).
        NemotronHCompactSSMGated.prepareCounter()
        if let fused = mixer.norm.compactSSMPreparedRows(
            convolution: MLXArray.zeros([1, 3, mixer.convDim], dtype: .bfloat16),
            ALog: mixer.aLog, D: mixer.D,
            dt: MLXArray.zeros([1, 3, heads], dtype: .float32),
            state: MLXArray.zeros([1, heads, dim, stateDim], dtype: .float32),
            projected: MLXArray.zeros(
                [1, 3, mixer.intermediateSize + mixer.convDim + heads], dtype: .bfloat16),
            heads: heads, headDim: dim, groups: groups, stateDim: stateDim) {
            eval(fused.0, fused.1, fused.2)
        }
        NemotronHCompactSSMGated.qualifyShuffle(mixer)
        NemotronHCompactSSMGated.qualifyRowsTail(mixer)
        // NEMOTRON_MTP_SSM_PENDING_FOLD: compiles and checks every folded launch.
        NemotronMTPPendingSSM.qualify(mixer)
        for width in 2...8 {
            let padded = MLXArray.zeros([1, history + width, mixer.convDim], dtype: activationDType)
            let convolution = silu(mixer.conv1d(padded))
            let rawDt = MLXArray.zeros([1, width, heads], dtype: activationDType)
            let state = MLXArray.zeros([1, heads, dim, stateDim], dtype: .float32)
            if nemotronMTPCompactSSM {
                let dt = nemotronPrefillComputeDt(rawDt, mixer.dtBias, mixer.timeStepLimit)
                let window = nemotronMTPSSMCompact(
                    convolution: convolution, ALog: mixer.aLog, D: mixer.D, dt: dt,
                    state: state, heads: heads, headDim: dim, groups: groups,
                    stateDim: stateDim, steps: width, writeOutput: true)
                let replay = nemotronMTPSSMCompact(
                    convolution: convolution, ALog: mixer.aLog, D: mixer.D, dt: dt,
                    state: state, heads: heads, headDim: dim, groups: groups,
                    stateDim: stateDim, steps: width - 1, writeOutput: false)
                roots += [window.output, window.finalState, replay.finalState]
            } else if NemotronMTPExecution.windowSSM {
                let values = split(convolution,
                    indices: [mixer.intermediateSize, mixer.intermediateSize + groups * stateDim],
                    axis: -1)
                let window = nemotronMTPSSMWindow(
                    hiddenStates: values[0].reshaped(1, width, heads, dim),
                    ALog: mixer.aLog, B: values[1].reshaped(1, width, groups, stateDim),
                    C: values[2].reshaped(1, width, groups, stateDim), D: mixer.D,
                    dt: rawDt, dtBias: mixer.dtBias, state: state,
                    timeStepLimit: mixer.timeStepLimit)
                roots += [window.output, window.capturedStates]
            }
        }
        if nemotronMTPCompactSSM {
            let mambaLayers = modules().lazy.filter { $0 is NemotronHMamba2Mixer }.count
            roots += NemotronMTPReplayGroupKernel.warm(
                layerCount: mambaLayers,
                convolution: silu(mixer.conv1d(
                    MLXArray.zeros([1, history + 8, mixer.convDim], dtype: activationDType))),
                aLog: mixer.aLog, D: mixer.D,
                heads: heads, headDim: dim, groups: groups, stateDim: stateDim)
        }
        let packed = mixer.packedConv(
            MLXArray.zeros([1, history, mixer.convDim], dtype: activationDType),
            MLXArray.zeros([1, 1, mixer.convDim], dtype: activationDType))
        roots += [packed.convolution, packed.padded]
        if !roots.isEmpty { eval(roots) }
    }
}

/// Runs the captured verify's head launch a second time after the first.
/// NEMOTRON_MTP_HEAD_TWICE=0 turns it off.
enum NemotronMTPHeadTwice {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_MTP_HEAD_TWICE"] != "0"
}
