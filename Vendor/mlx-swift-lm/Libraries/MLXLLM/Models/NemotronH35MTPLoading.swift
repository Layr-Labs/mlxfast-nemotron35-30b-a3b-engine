import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum NemotronH35MTPError: Error, LocalizedError {
    case invalidArtifact(String)
    public var errorDescription: String? {
        switch self { case .invalidArtifact(let detail): "Invalid Nemotron Lightning MTP artifact: \(detail)" }
    }
}

extension NemotronH35MTPAssistant {
    public static func load(from directory: URL, target: NemotronH35Model) throws -> NemotronH35MTPAssistant {
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        guard configData.count <= 4 * 1024 * 1024 else { throw NemotronH35MTPError.invalidArtifact("config too large") }
        let args = try JSONDecoder().decode(NemotronH35Configuration.self, from: configData).target
        guard args.numNextnPredictLayers == 1, args.mtpLayersBlockType == ["attention", "moe"] else {
            throw NemotronH35MTPError.invalidArtifact("expected one attention/MoE prediction layer")
        }
        // A same-directory declaration is insufficient: all decoded architecture
        // fields must match the exact loaded target before sharing embeddings/head.
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let decoded = try JSONDecoder().decode(NemotronH35Configuration.self, from: configData)
        guard try encoder.encode(decoded) == encoder.encode(target.lightningConfiguration) else {
            throw NemotronH35MTPError.invalidArtifact("configuration differs from target")
        }
        struct Index: Decodable { let weight_map: [String: String] }
        let indexData = try Data(contentsOf: directory.appendingPathComponent("model.safetensors.index.json"))
        guard indexData.count <= 16 * 1024 * 1024 else { throw NemotronH35MTPError.invalidArtifact("index too large") }
        let selected = try JSONDecoder().decode(Index.self, from: indexData).weight_map.filter { $0.key.hasPrefix("mtp.") }
        guard !selected.isEmpty, selected.count <= 512 else { throw NemotronH35MTPError.invalidArtifact("missing or oversized MTP inventory") }
        var weights: [String: MLXArray] = [:]
        for shard in Set(selected.values).sorted() {
            guard shard == URL(fileURLWithPath: shard).lastPathComponent,
                !shard.contains(".."), shard.hasSuffix(".safetensors") else {
                throw NemotronH35MTPError.invalidArtifact("invalid shard path")
            }
            let (arrays, _) = try loadArraysAndMetadata(url: directory.appendingPathComponent(shard))
            for (key, expected) in selected where expected == shard {
                guard let value = arrays[key] else { throw NemotronH35MTPError.invalidArtifact("indexed tensor missing: \(key)") }
                weights[String(key.dropFirst(4))] = value
            }
        }
        // Official BF16 checkpoints store individual experts. Converted MLX
        // checkpoints already carry the stacked SwitchLinear parameters.
        for (source, dest) in [("up_proj", "fc1"), ("down_proj", "fc2")] {
            let prefix = "layers.1.mixer"
            if weights["\(prefix).experts.0.\(source).weight"] != nil {
                var experts: [MLXArray] = []
                for e in 0..<args.nRoutedExperts {
                    guard let value = weights.removeValue(forKey: "\(prefix).experts.\(e).\(source).weight") else {
                        throw NemotronH35MTPError.invalidArtifact("incomplete expert bank")
                    }
                    experts.append(value)
                }
                weights["\(prefix).switch_mlp.\(dest).weight"] = stacked(experts)
            }
        }
        let assistant = NemotronH35MTPAssistant(target: target)
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: configData)
        let scaled = Set(weights.keys.filter { $0.hasSuffix(".scales") }.map { String($0.dropLast(7)) })
        for path in scaled {
            guard base.perLayerQuantization?.quantization(layer: "mtp." + path) != nil else {
                throw NemotronH35MTPError.invalidArtifact("quantization missing for \(path)")
            }
        }
        quantize(model: assistant.module) { path, _ in
            scaled.contains(path) ? base.perLayerQuantization?.quantization(layer: "mtp." + path)?.asTuple : nil
        }
        try assistant.module.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(assistant.module)
        // The pinned metadata and every original tensor have already bound.
        // This helper receives only the assistant; target/shared weights are
        // inaccessible. It runs before any private layout or graph is built.
        try nemotronRequantizeMTPAttention(assistant.module)
        (assistant.module.layers[0] as? NemotronH35MTPAttentionBlock)?
            .prepareJoinedDraftQKV()
        (assistant.module.layers[1] as? NemotronH35MTPExpertBlock)?
            .prepareOwnedSparseDraftDown()
        assistant.prepareDraftHeadScreen()
        target.warmMTPSSMKernels(
            activationDType: target.mtpEmbedding(MLXArray([Int32(0)]).reshaped([1, 1])).dtype)
        // Load-time JIT of the exact verify kernels (kernel agent hook).
        NemotronRoutedSharedLaunch.verify(target)
        NemotronRoutedSharedFourRows.verify(target)
        NemotronMoELeader.warm(target)
        NemotronMTPRouterRows.warm(target)
        NemotronMultiRowQMV.warmNarrow()
        NemotronMultiRowQMV.verifyAttentionOutputAdding(
            (assistant.module.layers[0] as! NemotronH35MTPAttentionBlock).mixer.wo)
        NemotronMTPAttentionDirect.warm(target)
        // The general pass replaces one synthetic post-verify warm-up with
        // its real captured rows, then warms only the remaining widths.
        // Keep last: it ends with Memory.clearCache().
        try NemotronDraftAttentionGeometry.warm()
        assistant.warmUpExactShapes()
        return assistant
    }
}

/// Contract 3.3/4.4 permits on-load re-quantization of mtp.* head weights.
/// This fixed five-projection allowlist excludes embeddings, LM head, all
/// target layers, and even the assistant's existing 4-bit sparse expert bank.
/// New proposals are lossy; target verification still decides emitted tokens.
private func nemotronRequantizeMTPAttention(_ module: NemotronH35MTPModule) throws {
    let requested = ProcessInfo.processInfo.environment["NEMOTRON_MTP_HEAD_ATTENTION_BITS"] ?? "3"
    guard requested == "3", Device.defaultDevice().deviceType == .gpu,
          StreamOrDevice.default == .gpu,
          let block = module.layers.first as? NemotronH35MTPAttentionBlock
    else {
        print("[nemotron-mtp-attention-bits] original checkpoint layout")
        return
    }
    let paths = ["layers.0.eh_proj", "layers.0.mixer.q_proj", "layers.0.mixer.k_proj",
                 "layers.0.mixer.v_proj", "layers.0.mixer.o_proj"]
    let dimensions = [(2688, 5376), (4096, 2688), (256, 2688), (256, 2688), (2688, 4096)]
    func current() -> [Linear] {
        [block.projection, block.mixer.wq, block.mixer.wk, block.mixer.wv, block.mixer.wo]
    }
    let originals = current()
    var replacements: [QuantizedLinear]?
    do {
        replacements = try withError { () -> [QuantizedLinear]? in
            var prepared: [QuantizedLinear] = []
            for index in paths.indices {
                let (n, k) = dimensions[index]
                guard type(of: originals[index]) == QuantizedLinear.self,
                      let old = originals[index] as? QuantizedLinear,
                      old.mode == .affine, old.groupSize == 64, old.bits == 4, old.bias == nil,
                      old.weight.dtype == .uint32, old.weight.shape == [n, k / 8],
                      old.scales.dtype == .bfloat16, old.scales.shape == [n, k / 64],
                      let offsets = old.biases, offsets.dtype == .bfloat16,
                      offsets.shape == old.scales.shape else { return nil }
                // Reconstruct one bounded projection at a time using its
                // actual 4-bit metadata. Never treat packed UInt32 as dense.
                let dense = dequantized(old.weight, scales: old.scales, biases: offsets,
                    groupSize: old.groupSize, bits: old.bits, mode: old.mode)
                let next = QuantizedLinear(weight: dense, bias: nil,
                    groupSize: 64, bits: 3, mode: .affine)
                eval(next)
                guard next.weight.dtype == .uint32, next.weight.shape == [n, k * 3 / 32],
                      next.scales.dtype == .bfloat16, next.scales.shape == [n, k / 64],
                      let nextOffsets = next.biases, nextOffsets.dtype == .bfloat16,
                      nextOffsets.shape == next.scales.shape else { return nil }
                let rebuilt = dequantized(next.weight, scales: next.scales, biases: nextOffsets,
                    groupSize: next.groupSize, bits: next.bits, mode: next.mode)
                let originalFloat = dense.asType(.float32)
                let delta = rebuilt.asType(.float32) - originalFloat
                let energy = (originalFloat * originalFloat).sum().item(Float.self)
                let error = (delta * delta).sum().item(Float.self)
                // A bounded reconstruction sanity guard, not a token gate or
                // a claim that the 3-bit proposals equal the 4-bit proposals.
                guard energy.isFinite, energy > 0, error.isFinite, error <= energy * 0.0625,
                      isFinite(rebuilt).all().item(Bool.self) else { return nil }
                for seed in 0..<3 {
                    let x = MLXRandom.normal([1, 1, k], dtype: .bfloat16,
                        key: MLXRandom.key(UInt64(0x3b710 + 7 * index + seed)))
                    let output = next(x)
                    guard output.shape == [1, 1, n], output.dtype == .bfloat16,
                          isFinite(output).all().item(Bool.self) else { return nil }
                }
                prepared.append(next)
            }
            return prepared
        }
    } catch {
        print("[nemotron-mtp-attention-bits] original layout (3-bit preparation error)")
        return
    }
    guard let replacements, replacements.count == paths.count else {
        print("[nemotron-mtp-attention-bits] original layout (3-bit sanity guard declined)")
        return
    }
    let updates: [(String, Module)] = paths.indices.map { (paths[$0], replacements[$0]) }
    do {
        try module.update(modules: ModuleChildren.unflattened(updates), verify: [.noUnusedKeys])
        guard zip(current(), replacements).allSatisfy({ ObjectIdentifier($0.0) == ObjectIdentifier($0.1) })
        else { throw NemotronH35MTPError.invalidArtifact("MTP attention replacement identity") }
    } catch {
        // A failed partial update may not leave a mixed head installed.
        let restore: [(String, Module)] = paths.indices.map { (paths[$0], originals[$0]) }
        do {
            try module.update(modules: ModuleChildren.unflattened(restore), verify: [.noUnusedKeys])
            guard zip(current(), originals).allSatisfy({ ObjectIdentifier($0.0) == ObjectIdentifier($0.1) })
            else { throw NemotronH35MTPError.invalidArtifact("MTP attention restoration identity") }
        } catch {
            throw NemotronH35MTPError.invalidArtifact("MTP attention restoration failed")
        }
        print("[nemotron-mtp-attention-bits] original layout (replacement restored)")
        return
    }
    print("[nemotron-mtp-attention-bits] 3 (five lossy assistant projections; target verification required)")
}

// Load-time warm-up of every decode/verify shape, outside any timed window.
// The first MTP rounds of a fresh process otherwise pay the metalKernel JIT
// (multi-row qmv per R/FAST, windowed SSM per L, top-two), the compile()
// trace of every MoE layer's verify graph per row count, and pipeline-state
// creation. Everything here runs on THROWAWAY state: a private contiguous KV
// backend + layer caches, a private recurrent request state and a private
// assistant request state, all released before returning. Only process-wide
// caches keyed by graph/shape are populated (kernel libraries, pipelines,
// compiled graphs), and those produce the same arithmetic whoever fills them.
// The allocator's free-buffer cache is then dropped (Memory.clearCache()).
extension NemotronH35MTPAssistant {
    func warmUpExactShapes() {
        qualifyDeviceHeadGraph()
        qualifyDeviceThirdDraft()
        let environment = ProcessInfo.processInfo.environment
        guard environment["DARKBLOOM_NEMOTRON35_MTP_WARMUP"] != "0" else {
            warmPostVerifySpeculation()
            return
        }
        let requested = Int(environment["DARKBLOOM_NEMOTRON35_MTP_WARMUP_DEPTH"] ?? "") ?? 3
        let depth = max(1, min(requested, maximumDraftTokens ?? 7, 7))
        let promptLength = max(8, Int(environment["DARKBLOOM_NEMOTRON35_MTP_WARMUP_PROMPT"] ?? "") ?? 488)
        let prefillLength = Int(environment["DARKBLOOM_NEMOTRON35_MTP_WARMUP_PREFILL"] ?? "") ?? 1024
        // This records the one attempted replacement even if a later
        // operation throws, so the catch path cannot add another warm pass.
        var postVerifyWarmupWidth: Int?
        do {
            // A tile-aligned timed-prefill length (the seed length is not
            // aligned, so its kernels use the other alignment variants).
            if prefillLength > 0 { try warmUpPrefill(length: prefillLength) }
            try runWarmUp(depth: depth, promptLength: promptLength,
                          postVerifyWarmupWidth: &postVerifyWarmupWidth)
        } catch {
            FileHandle.standardError.write(Data("nemotron mtp warm-up skipped: \(error)\n".utf8))
        }
        warmPostVerifySpeculation(skippingWidth: postVerifyWarmupWidth)
        Memory.clearCache()
    }

    private func warmUpPrefill(length: Int) throws {
        let kinds = target.cbv2LayerKinds
        let backend = CBv2ContiguousKVBackend(
            config: CBv2ContiguousBackendConfig(bytesCapacity: 1 << 30))
        let rows = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: length, maxLength: length + 8)
        let layerCaches = kinds.enumerated().map { storageIndex, kind in
            CBv2LayerCache(layerIndex: kind.modelLayerIndex ?? storageIndex, kind: kind)
        }
        for (cache, row) in zip(layerCaches, rows) { if let row { cache.appendRow(row) } }
        let recurrent = try CBv2RecurrentRequestState(spec: target.cbv2RecurrentStateSpec)
        defer {
            for cache in layerCaches { cache.setRows([]) }
            backend.release(rows)
            try? recurrent.release()
        }
        let vocabulary = target.configuration.vocabSize
        let prompt = MLXArray((0..<length).map { Int32(($0 &* 104729 &+ 7) % vocabulary) })
            .reshaped(1, length)
        let evaluation = try recurrent.bind()
        let logits = target.cbv2Forward(prompt, caches: layerCaches, recurrentState: [evaluation])
        let roots = try evaluation.evaluate()
        eval([argMax(logits[0..., -1, 0...], axis: -1)] + roots
            + layerCaches.flatMap { $0.innerState() })
        try evaluation.commit()
    }

    private func runWarmUp(
        depth: Int, promptLength: Int, postVerifyWarmupWidth: inout Int?
    ) throws {
        let kinds = target.cbv2LayerKinds
        let maxLength = promptLength + 4 * (depth + 2) + 16
        let backend = CBv2ContiguousKVBackend(
            config: CBv2ContiguousBackendConfig(bytesCapacity: 1 << 30))
        let rows = try backend.makeSequenceState(
            layerKinds: kinds, promptLength: promptLength, maxLength: maxLength)
        let layerCaches = kinds.enumerated().map { storageIndex, kind in
            CBv2LayerCache(layerIndex: kind.modelLayerIndex ?? storageIndex, kind: kind)
        }
        for (cache, row) in zip(layerCaches, rows) { if let row { cache.appendRow(row) } }
        let caches: [KVCache] = layerCaches
        let recurrent = try CBv2RecurrentRequestState(spec: target.cbv2RecurrentStateSpec)
        let draftState = makeRequestState()
        defer {
            releaseRequestState(draftState)
            for cache in layerCaches { cache.setRows([]) }
            backend.release(rows)
            try? recurrent.release()
        }
        try configureRequestState(draftState, maximumSequenceLength: maxLength)
        let storageRows = rows.compactMap { $0 }
        let vocabulary = target.configuration.vocabSize
        func cacheState() -> [MLXArray] { layerCaches.flatMap { $0.innerState() } }

        // Seed-path prefill (one chunk, as the engine schedules the seed).
        let prompt = MLXArray((0..<promptLength).map { Int32(($0 &* 7919 &+ 101) % vocabulary) })
            .reshaped(1, promptLength)
        let prefillEvaluation = try recurrent.bind()
        let prefill = target.cbv2ForwardWithHiddenForPrefill(
            prompt, caches: caches, recurrentState: [prefillEvaluation], positionIds: nil,
            requirement: .lastPositionLogits)
        var seed = argMax(prefill.logits[0..., -1, 0...], axis: -1).asType(.int32).reshaped(1, 1)
        let prefillRoots = try prefillEvaluation.evaluate()
        eval([seed, prefill.lastHidden] + prefillRoots + cacheState())
        try prefillEvaluation.commit()
        observeCommittedTarget(
            CBv2MTPCommittedTargetObservation(tokens: prompt, hidden: prefill.lastHidden),
            requestState: draftState)
        var carry = prefill.lastHidden[0..., (promptLength - 1)..<promptLength, 0...]

        // Plain one-token target step (non-captured decode shape).
        let stepEvaluation = try recurrent.bind()
        let step = target.cbv2ForwardWithHidden(
            seed, caches: caches, recurrentState: [stepEvaluation], positionIds: nil)
        let stepRoots = try stepEvaluation.evaluate()
        eval([step.logits, step.lastHidden] + stepRoots + cacheState())
        try stepEvaluation.commit()
        observeCommittedTarget(
            CBv2MTPCommittedTargetObservation(tokens: seed, hidden: step.lastHidden),
            requestState: draftState)
        seed = argMax(step.logits[0..., -1, 0...], axis: -1).asType(.int32).reshaped(1, 1)
        carry = step.lastHidden
        let seedPolicy = target.cbv2MTPTopTwo(step.logits.reshaped(1, 1, vocabulary))
        _ = seedPolicy.values.asType(.float32).asArray(Float.self)

        // MTP rounds as (drafts, confirmed window positions): the deepest
        // width first (prompt-history priming), then partial, full and zero
        // acceptance (one-token accepted-history replay, every finalize
        // branch), then every shallower width.
        var rounds = [(depth, min(2, depth + 1)), (depth, depth + 1), (depth, 1)]
        for k in (1..<depth).reversed() { rounds += [(k, k + 1), (k, 1)] }
        for (k, confirmed) in rounds {
            var tokens = seed, hidden = carry
            var drafts: [MLXArray] = []
            var draftRows: [MLXArray] = []
            for _ in 0..<k {
                let result = draftStep(tokens: tokens, hidden: hidden, shortlist: nil,
                                       requestState: draftState)
                let next = result.tokens.reshaped([1])
                asyncEval([next, result.hidden] + evaluationTargets(for: draftState))
                draftRows.append(concatenated([next], axis: 0))
                drafts.append(next.reshaped([1, 1]))
                tokens = next.reshaped([1, 1])
                hidden = result.hidden
            }
            let draftIDs = stacked(draftRows, axis: 1)
            for row in storageRows { row.beginSpeculativeWrite() }
            for cache in layerCaches {
                cache.mtpSerializesRectangularAttention = true
                cache.mtpBatchesRectangularAttention = prefersBatchedRectangularAttention
            }
            let window = concatenated([seed] + drafts, axis: 1)
            let evaluation = try recurrent.bind()
            let verify = target.cbv2ForwardWithHiddenCaptured(
                window, caches: caches, recurrentState: [evaluation], positionIds: nil)
            for cache in layerCaches {
                cache.mtpSerializesRectangularAttention = false
                cache.mtpBatchesRectangularAttention = false
            }
            let width = k + 1
            let topTwo = target.cbv2MTPTopTwo(verify.logits.reshaped(1, width, vocabulary))
            let ids = topTwo.ids.reshaped(1, width, 2).asType(.int32)
            let values = topTwo.values.reshaped(1, width, 2).asType(.float32)
            let packet = concatenated(
                [draftIDs.reshaped([-1]), ids[0..., 0..., 0].reshaped([-1])], axis: 0)
            let verifyRoots = try evaluation.evaluate()
            asyncEval([packet, values, verify.lastHidden] + verifyRoots
                + cacheState() + evaluationTargets(for: draftState))
            // Replace ONE existing synthetic width with real verifier rows.
            // The third deep round is the inherited zero-acceptance case:
            // the assistant is already started and the selected offset is 0.
            let warmPostVerify = postVerifyWarmupWidth == nil && k == depth
                && confirmed == 1 && canWarmPostVerifySpeculation(width: width)
            if warmPostVerify {
                postVerifyWarmupWidth = width
                speculateAfterVerify(
                    targetTokens: packet[k ..< (k + width)].reshaped([1, width]),
                    targetHidden: verify.lastHidden, requestState: draftState)
                // The device-picked first draft (gathered row, same tail) for
                // the row this warm-up round confirms; finalize adopts it.
                if CBv2MTPFirstDraftPrestage.deviceAccepted {
                    speculateNextFirstDraft(
                        acceptedDrafts: MLXArray([Int32(confirmed - 1)]), requestState: draftState)
                }
            }
            let hostPacket = packet.asArray(Int32.self)
            _ = values.asArray(Float.self)
            let keep = confirmed
            try evaluation.commit(keepPositions: keep)
            for row in storageRows {
                if width > keep { row.rollback(width - keep) }
                row.commitSpeculativeWrite()
            }
            noteHostAcceptedDrafts(keep - 1, requestState: draftState)
            finalizeRound(
                requestState: draftState, confirmedInputTokens: keep,
                committedDraftTokens: window[0..., 1..<keep],
                committedTargetHidden: verify.lastHidden[0..., 0..<(keep - 1), 0...])
            seed = ids[0..., (keep - 1)..<keep, 0].reshaped(1, 1)
            carry = verify.lastHidden[0..., (keep - 1)..<keep, 0...]
            if warmPostVerify {
                prestageFirstDraft(
                    token: Int(hostPacket[k + keep - 1]), hidden: carry,
                    requestState: draftState)
            }
            eval([seed, carry] + evaluationTargets(for: draftState))
        }
        // Official-worker private load smoke for the new three-entry prestage
        // protocol. Consume the exact chained hidden objects, then discard its
        // ordinary transaction. No request state or warm tokens escape.
        if depth == 3, NemotronMTPFourRows.enabled {
            prestageDrafts(token: Int(seed.item(Int32.self)), hidden: carry,
                requestState: draftState, depth: 3)
            var tokens = seed, hidden = carry
            for _ in 0..<3 {
                let next = draftStep(tokens: tokens, hidden: hidden, shortlist: nil,
                    requestState: draftState)
                tokens = next.tokens.reshaped(1, 1)
                hidden = next.hidden
            }
            eval([tokens, hidden] + evaluationTargets(for: draftState))
        }
        // Depth one has no shallower round to consume the prestaged row.
        // Release that private transaction through the ordinary discard path.
        discardRound(requestState: draftState)
        Stream.gpu.synchronize()
    }
}
