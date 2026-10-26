// Batched target verification with the ordinary one-token Mamba recurrence.
// Projections are built once over the window. Every confirmed-prefix state is
// staged for the engine's existing transactional commit/rollback mechanism.
import Foundation
import MLX
import MLXLMCommon
import MLXNN

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
    /// [plain, consuming a prepared in_proj activation] x [no fold, fold,
    /// fold storing the pre-window state]: separate graphs. A fold graph
    /// takes `initialSSM` as the chain root and the link operands as inputs.
    private var entries: [Entry?] = Array(repeating: nil, count: 6)

    func callAsFunction(
        _ hidden: MLXArray, convState: MLXArray, initialSSM: MLXArray,
        prepared: MLXArray? = nil, fold: NemotronMTPFoldInputs? = nil,
        owner: NemotronHMamba2Mixer
    ) -> [MLXArray] {
        guard Self.enabled, NemotronMTPExecution.batchedM1, NemotronMTPExecution.batchedNorm,
              Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              hidden.ndim == 3, hidden.dim(0) == 1, (2...4).contains(hidden.dim(1)),
              hidden.dim(2) == 2688, hidden.dtype == .bfloat16,
              convState.shape == [1, 3, 6144], convState.dtype == .bfloat16,
              initialSSM.shape == [1, 64, 64, 128], initialSSM.dtype == .float32,
              let identity = Self.nativeIdentity(owner)
        else {
            return owner.mtpCompactArrays(
                hidden, convState: convState, initialSSM: initialSSM, prepared: prepared,
                fold: fold)
        }
        let usesPrepared = prepared != nil
        let writePre = fold?.writePre ?? false
        let folds = fold != nil
        let slot = (usesPrepared ? 1 : 0) + (folds ? (writePre ? 4 : 2) : 0)
        var retired: Entry?
        let current = lock.withLock { () -> Entry in
            if let entry = entries[slot], entry.identity == identity { return entry }
            let state = owner.innerState()
            let function: @Sendable ([MLXArray]) -> [MLXArray] = compile(inputs: [state]) {
                [weak owner] arrays in
                guard let owner else { preconditionFailure("Released Nemotron MTP Mamba-window owner") }
                let links = usesPrepared ? 4 : 3
                return owner.mtpCompactArrays(
                    arrays[0], convState: arrays[1], initialSSM: arrays[2],
                    prepared: usesPrepared ? arrays[3] : nil,
                    fold: folds ? NemotronMTPFoldInputs(arrays[links...], writePre: writePre) : nil)
            }
            let replacement = Entry(identity: identity, function: function)
            retired = entries[slot]
            entries[slot] = replacement
            return replacement
        }
        let result = withExtendedLifetime(retired) {
            current.function([hidden, convState, initialSSM] + (prepared.map { [$0] } ?? [])
                + (fold?.arrays ?? []))
        }
        guard result.count == (writePre ? 6 : 5) else {
            // Retain any enclosing MLX error while avoiding empty indexing.
            return owner.mtpCompactArrays(
                hidden, convState: convState, initialSSM: initialSSM, prepared: prepared,
                fold: fold)
        }
        guard NemotronMoELeader.selfTest else { return result }
        // Reference: in_proj prepares its activation from `hidden` itself.
        let expected = owner.mtpCompactArrays(
            hidden, convState: convState, initialSSM: initialSSM, fold: fold)
        return zip(result, expected).map { nemotronBitwiseOrNaN($0.0, $0.1) }
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
    func cbv2ForwardCaptured(
        _ hiddenStates: MLXArray, modelLayerIndex: Int,
        recurrentState: [CBv2RecurrentStateEvaluation], prepared: MLXArray? = nil
    ) -> MLXArray {
        let batch = hiddenStates.dim(0), length = hiddenStates.dim(1)
        precondition(batch == 1 && recurrentState.count == 1 && (1...8).contains(length))
        let evaluation = recurrentState[0]
        // NEMOTRON_MTP_RFOLD: a strict-prefix input this window can replay in
        // its own launch is read as its chain; its replay is never built.
        if nemotronMTPCompactSSM, NemotronMTPFold.enabled,
           let chain = evaluation.inputFold(modelLayerIndex: modelLayerIndex)
               as? NemotronMTPFoldChain,
           mtpFoldWindow(hiddenStates, root: chain.root) {
            return mtpCompactForward(hiddenStates, modelLayerIndex: modelLayerIndex,
                evaluation: evaluation, convState: chain.conv, initialSSM: chain.root,
                prepared: prepared, chain: chain)
        }
        let initial = evaluation.inputState(modelLayerIndex: modelLayerIndex)
        let convState = initial?.conv ?? MLXArray.zeros(
            [batch, max(0, convKernelSize - 1), convDim], dtype: hiddenStates.dtype)
        let ssmState = initial?.ssm ?? MLXArray.zeros(
            [batch, numHeads, headDim, ssmStateSize], dtype: .float32)
        if nemotronMTPCompactSSM && length > 1 {
            return mtpCompactForward(hiddenStates, modelLayerIndex: modelLayerIndex,
                evaluation: evaluation, convState: convState, initialSSM: ssmState,
                prepared: prepared)
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
    /// With `chain` (NEMOTRON_MTP_RFOLD) `initialSSM` is the chain's root and
    /// the window launch first replays the chain's links.
    func mtpCompactForward(
        _ hiddenStates: MLXArray, modelLayerIndex: Int,
        evaluation: CBv2RecurrentStateEvaluation, convState: MLXArray, initialSSM: MLXArray,
        prepared: MLXArray? = nil, chain: NemotronMTPFoldChain? = nil
    ) -> MLXArray {
        let length = hiddenStates.dim(1)
        let linkInputs = chain.map(NemotronMTPFoldInputs.init)
        let arrays = mtpCompactGraphCache(
            hiddenStates, convState: convState, initialSSM: initialSSM, prepared: prepared,
            fold: linkInputs, owner: self)
        let (padded, convolution, dt, finalSSM) = (arrays[1], arrays[2], arrays[3], arrays[4])
        // A full chain's end state, stored by the launch: the next chain's root.
        let preWindow = linkInputs?.writePre == true ? arrays[5] : nil
        let (heads, dim, groups, state) = (numHeads, headDim, numGroups, ssmStateSize)
        let (logA, skip) = (self.aLog, self.D)
        let history = convKernelSize - 1
        let tape = [padded, convolution, dt] + (preWindow.map { [$0] } ?? [])
        let tapeBytes = tape.reduce(0) { $0 + $1.nbytes }
        // NEMOTRON_MTP_SSM_REPLAY_GROUP: replay with this forward's other
        // Mamba layers in grouped launches (same per-layer statements). A
        // folded window's input state exists only as its chain.
        let grouped = chain == nil ? NemotronMTPReplayGroup.join(evaluation, .init(
            convolution: convolution, dt: dt, aLog: logA, D: skip, state: initialSSM,
            heads: heads, headDim: dim, groups: groups, stateDim: state)) : nil
        let replayStart: () -> MLXArray = {
            guard let chain else { return initialSSM }
            return preWindow ?? chain.materialize(
                ALog: logA, D: skip, heads: heads, headDim: dim, groups: groups, stateDim: state)
        }
        // NEMOTRON_MTP_RFOLD: the strict prefix also as a chain the next
        // window can fold: this window's tape appended to its own chain, or
        // started at its input state (or at the stored end of a full chain).
        let fold: ((Int) -> AnyObject?)? = NemotronMTPFold.enabled && length == 3 ? { keep in
            guard (1 ..< length).contains(keep) else { return nil }
            let link = NemotronMTPFoldChain.Link(convolution: convolution, dt: dt, keep: keep)
            let conv = padded[0..., keep ..< (keep + history), 0...]
            if let chain, preWindow == nil {
                return NemotronMTPFoldChain(root: chain.root, links: chain.links + [link], conv: conv)
            }
            if preWindow != nil { NemotronMTPFold.rerooted += 1 }
            return NemotronMTPFoldChain(root: preWindow ?? initialSSM, links: [link], conv: conv)
        } : nil
        let retained = chain?.arrays ?? [initialSSM]
        do {
            try evaluation.stagePrefixReplay(
                modelLayerIndex: modelLayerIndex, positions: length,
                finalConv: padded[0..., length ..< (length + history), 0...],
                finalSSM: finalSSM,
                materializedByteCount: tapeBytes + finalSSM.nbytes,
                evaluationRoots: tape,
                strictReplayRetainedByteCount: retained.reduce(tapeBytes) { $0 + $1.nbytes },
                strictReplayRetainedRoots: tape + retained,
                fold: fold,
                replay: { keep in
                    CBv2RecurrentLayerState(
                        conv: padded[0..., keep ..< (keep + history), 0...],
                        ssm: grouped.map { $0.group.finalState(slot: $0.slot, keep: keep) }
                            ?? nemotronMTPSSMCompact(
                            convolution: convolution, ALog: logA, D: skip, dt: dt,
                            state: replayStart(), heads: heads, headDim: dim, groups: groups,
                            stateDim: state, steps: keep, writeOutput: false).finalState)
                })
        } catch {
            preconditionFailure("Nemotron MTP compact recurrent stage failed: \(error)")
        }
        if chain != nil { NemotronMTPFold.folded += 1 }
        return arrays[0]
    }

    /// The window runs `mtpCompactArrays`' fused SSM launch, the one launch
    /// with a fold variant: its guards, on the operands a three-row window
    /// of this mixer produces (bf16 conv rows and in_proj output, FP32 dt).
    func mtpFoldWindow(_ hidden: MLXArray, root: MLXArray) -> Bool {
        hidden.ndim == 3 && hidden.dim(0) == 1 && hidden.dim(1) == 3
            && hidden.dtype == .bfloat16 && mtpGatedOutProj
            && norm.compactSSMGatedAccepts(
                convolution: ([1, 3, convDim], .bfloat16), dt: ([1, 3, numHeads], .float32),
                state: (root.shape, root.dtype),
                projected: ([1, 3, intermediateSize + convDim + numHeads], .bfloat16),
                heads: numHeads, headDim: headDim, groups: numGroups, stateDim: ssmStateSize)
    }

    /// out_proj is the 4-bit affine bank whose rows the fused window prepares.
    var mtpGatedOutProj: Bool {
        guard let q = outProj as? QuantizedLinear else { return false }
        return q.mode == .affine && q.bits == 4 && q.groupSize == 64 && q.bias == nil
            && q.weight.shape == [2688, 512] && q.weight.dtype == .uint32
            && q.scales.shape == [2688, 64] && q.scales.dtype == .bfloat16
            && q.biases?.shape == [2688, 64] && q.biases?.dtype == .bfloat16
    }

    /// Original numerical graph, with request state supplied as dynamic arrays.
    /// Return its replay tape explicitly; staging/commit stay outside compile.
    /// `inputPrepared`: in_proj's K=2688 activation, prepared by the preceding
    /// join from these exact BF16 rows (otherwise prepared here).
    /// `fold`: `initialSSM` is a chain's root, run through the fused launch's
    /// fold variant (callers check `mtpFoldWindow` first); the result then
    /// also holds the chain's end state when `fold.writePre`.
    func mtpCompactArrays(
        _ hiddenStates: MLXArray, convState: MLXArray, initialSSM: MLXArray,
        prepared inputPrepared: MLXArray? = nil, fold: NemotronMTPFoldInputs? = nil
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
        if let fold {
            // Checked at load against the replay-then-window path
            // (`NemotronMTPFold.qualify`); keeps are device data here.
            guard length == 3, NemotronMTPExecution.batchedNorm, mtpGatedOutProj,
                  let folded = norm.compactSSMPreparedRowsFolded(
                    convolution: convolution, ALog: logA, D: skip, dt: dt, root: initialSSM,
                    projected: projected, heads: heads, headDim: dim, groups: groups,
                    stateDim: state, fold: fold)
            else { preconditionFailure("Nemotron rfold: a folded window outside the fused launch") }
            return [nemotronMTPLinearRows(folded.0, outProj, prepared: folded.1),
                    padded, convolution, dt, folded.2] + (folded.3.map { [$0] } ?? [])
        }
        // NEMOTRON_SSM_GATED_FUSED: the compact SSM window, the gated RMS norm
        // and out_proj's prepared activation in one launch (see
        // NemotronHCompactSSMGated); the same normed rows, prepared activation
        // and final state as the two launches below.
        if length == 3, NemotronMTPExecution.batchedNorm, mtpGatedOutProj,
           var fused = norm.compactSSMPreparedRows(
               convolution: convolution, ALog: logA, D: skip, dt: dt, state: initialSSM,
               projected: projected, heads: heads, headDim: dim, groups: groups, stateDim: state) {
            if NemotronMoELeader.selfTest {
                // Reference: the two launches (compact SSM, then the gated
                // norm with the prepared output) on the same operands.
                let reference = nemotronMTPSSMCompact(
                    convolution: convolution, ALog: logA, D: skip, dt: dt, state: initialSSM,
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
            convolution: convolution, ALog: logA, D: skip, dt: dt, state: initialSSM,
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
            let output = lmHead.map { nemotronMTPHeadRows(fused.hidden, $0, prepared: fused.prepared) }
                ?? nemotronMTPMapRows(fused.hidden) { logits($0) }
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
        NemotronMTPFold.qualify(
            mixer, mambaLayers: modules().lazy.filter { $0 is NemotronHMamba2Mixer }.count)
    }
}
