// Batched target verification with the ordinary one-token Mamba recurrence.
// Projections are built once over the window. Every confirmed-prefix state is
// staged for the engine's existing transactional commit/rollback mechanism.
import Foundation
import MLX
import MLXLMCommon
import MLXNN

extension NemotronHMamba2Mixer {
    func cbv2ForwardCaptured(
        _ hiddenStates: MLXArray, modelLayerIndex: Int,
        recurrentState: [CBv2RecurrentStateEvaluation]
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
                evaluation: evaluation, convState: convState, initialSSM: ssmState)
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
        evaluation: CBv2RecurrentStateEvaluation, convState: MLXArray, initialSSM: MLXArray
    ) -> MLXArray {
        let length = hiddenStates.dim(1)
        let core = NemotronMTPExecution.compiledMamba
            ? mtpCompiledCompactCache([hiddenStates, convState, initialSSM], owner: self)
            : mtpCompactCore(hiddenStates, convState: convState, initialSSM: initialSSM)
        let (output, padded, convolution, dt, finalState) = (core[0], core[1], core[2], core[3], core[4])
        let (heads, dim, groups, state) = (numHeads, headDim, numGroups, ssmStateSize)
        let (logA, skip) = (self.aLog, self.D)
        let history = convKernelSize - 1
        let tape = [padded, convolution, dt]
        let tapeBytes = tape.reduce(0) { $0 + $1.nbytes }
        do {
            try evaluation.stagePrefixReplay(
                modelLayerIndex: modelLayerIndex, positions: length,
                finalConv: padded[0..., length ..< (length + history), 0...],
                finalSSM: finalState,
                materializedByteCount: tapeBytes + finalState.nbytes,
                evaluationRoots: tape,
                strictReplayRetainedByteCount: tapeBytes + initialSSM.nbytes,
                strictReplayRetainedRoots: tape + [initialSSM],
                replay: { keep in
                    CBv2RecurrentLayerState(
                        conv: padded[0..., keep ..< (keep + history), 0...],
                        ssm: nemotronMTPSSMCompact(
                            convolution: convolution, ALog: logA, D: skip, dt: dt,
                            state: initialSSM, heads: heads, headDim: dim, groups: groups,
                            stateDim: state, steps: keep, writeOutput: false).finalState)
                })
        } catch {
            preconditionFailure("Nemotron MTP compact recurrent stage failed: \(error)")
        }
        return output
    }

    /// The stateless body of the compact verify: [mixer output, padded,
    /// convolution, dt, final SSM state]. Staging stays with the caller.
    func mtpCompactCore(_ hiddenStates: MLXArray, convState: MLXArray,
                        initialSSM: MLXArray) -> [MLXArray] {
        let batch = hiddenStates.dim(0), length = hiddenStates.dim(1)
        let projected = nemotronMTPLinearRows(hiddenStates, inProj)
        let parts = split(projected, indices: [intermediateSize, intermediateSize + convDim], axis: -1)
        let gate = parts[0], input = parts[1]
        let padded = concatenated([convState, input], axis: 1)
        let convolution = silu(conv1d(padded))
        let dt = computeDt(parts[2].reshaped(batch, length, numHeads), dtBias, timeStepLimit)
        let result = nemotronMTPSSMCompact(
            convolution: convolution, ALog: aLog, D: D, dt: dt, state: initialSSM,
            heads: numHeads, headDim: headDim, groups: numGroups, stateDim: ssmStateSize,
            steps: length, writeOutput: true)
        let y = result.output.reshaped(batch, length, numHeads * headDim)
        let normalized = NemotronMTPExecution.batchedNorm
            ? norm(y, gate: gate)
            : concatenated((0..<length).map { position in
                norm(y[0..., position..<(position + 1), 0...],
                     gate: gate[0..., position..<(position + 1), 0...])
            }, axis: 1)
        return [nemotronMTPLinearRows(normalized, outProj), padded, convolution, dt, result.finalState]
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
        let padded = concatenated([convState, input], axis: 1)
        let convolution = silu(conv1d(padded))
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


/// Stateless graph cache for `mtpCompactCore`, the same scheme as the MoE
/// verify cache. Conv/SSM inputs arrive as arguments; staging, offsets and
/// transactions are never captured.
final class NemotronH35MTPMambaGraphCache {
    private struct Entry {
        let identity: [ObjectIdentifier]
        let function: @Sendable ([MLXArray]) -> [MLXArray]
    }
    private let lock = NSLock()
    private var entry: Entry?

    func callAsFunction(_ arguments: [MLXArray], owner: NemotronHMamba2Mixer) -> [MLXArray] {
        let nativeIdentity = Self.nativeIdentity(owner)
        let parameters = nativeIdentity == nil ? owner.innerState() : nil
        let identity = nativeIdentity ?? (owner.modules().map(ObjectIdentifier.init)
            + (parameters ?? []).map(ObjectIdentifier.init))
        var retired: Entry?
        let current = lock.withLock { () -> Entry in
            if let entry, entry.identity == identity { return entry }
            let state = parameters ?? owner.innerState()
            let function: @Sendable ([MLXArray]) -> [MLXArray] = compile(inputs: [state]) {
                [weak owner] arguments in
                guard let owner else { preconditionFailure("Released Nemotron Mamba graph owner") }
                return owner.mtpCompactCore(arguments[0], convState: arguments[1], initialSSM: arguments[2])
            }
            let replacement = Entry(identity: identity, function: function)
            retired = entry
            entry = replacement
            return replacement
        }
        return withExtendedLifetime(retired) { current.function(arguments) }
    }

    private static func nativeIdentity(_ owner: NemotronHMamba2Mixer) -> [ObjectIdentifier]? {
        func linear(_ value: Linear) -> Bool {
            type(of: value) == Linear.self || type(of: value) == QuantizedLinear.self
        }
        guard type(of: owner) == NemotronHMamba2Mixer.self,
              type(of: owner.conv1d) == Conv1d.self,
              type(of: owner.norm) == NemotronHRMSNormGated.self,
              linear(owner.inProj), linear(owner.outProj) else { return nil }
        let modules: [Module] = [owner, owner.conv1d, owner.inProj, owner.outProj, owner.norm]
        return modules.map(ObjectIdentifier.init)
    }
}

extension NemotronH35Model: CBv2RecurrentCaptureMTPForwardable {
    public func cbv2ForwardWithHiddenCaptured(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        precondition(positionIds == nil)
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
        var roots: [MLXArray] = []
        for width in 2...8 {
            let padded = MLXArray.zeros([1, history + width, mixer.convDim], dtype: activationDType)
            let convolution = silu(mixer.conv1d(padded))
            let rawDt = MLXArray.zeros([1, width, heads], dtype: activationDType)
            let state = MLXArray.zeros([1, heads, dim, stateDim], dtype: .float32)
            if nemotronMTPCompactSSM {
                let dt = computeDt(rawDt, mixer.dtBias, mixer.timeStepLimit)
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
        if !roots.isEmpty { eval(roots) }
    }
}
