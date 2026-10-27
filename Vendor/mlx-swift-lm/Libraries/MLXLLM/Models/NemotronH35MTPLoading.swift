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
        assistant.prepareDraftHeadScreen()
        target.warmMTPSSMKernels(
            activationDType: target.mtpEmbedding(MLXArray([Int32(0)]).reshaped([1, 1])).dtype)
        CBv2MTPDeviceAcceptWalk.warm()
        NemotronMTPDraftRowPick.qualify()
        // Load-time JIT of the exact verify kernels (kernel agent hook).
        NemotronSharedUpPacking.verify(target)
        NemotronZSkip.prepare(target)
        NemotronSharedDownZSkip.prepare(target)
        NemotronSBDict.prepare(target)
        NemotronMoELeader.qualifyFC1Unroll(target)
        NemotronSharedUpPacking.prepareUnpacked(target)
        NemotronFC2SharedColaunch.prepare(target)
        NemotronSBDict.prepareDraft(assistant.module)
        NemotronBankBinding.warm(target, extra: NemotronZSkip.copyArrays + NemotronSBDict.indexArrays)
        NemotronBankBinding.warm(assistant.module, extra: NemotronSBDict.draftBankIndexArrays)
        target.prepareVerifyEntry()
        NemotronHResidualRMSNorm.checkProductsJoin(target)
        NemotronZSkip.prepareDraft(assistant.module)
        NemotronMoELeader.warm(target)
        NemotronMTPRouterRows.warm(target)
        NemotronMTPRouterRowsSelect.qualify(target)
        NemotronMTPRouterRowsSelect.qualifyDraft(assistant.module)
        NemotronU2.qualify(target: target, head: assistant.module)
        NemotronRouterSharedUpColaunch.qualify(target)
        cbv2TopTwoRowsPacketWarm(logitsDType: .bfloat16)
        NemotronMultiRowQMV.warmNarrow()
        NemotronMultiRowQMV.verifyAttentionOutputAdding(
            (assistant.module.layers[0] as! NemotronH35MTPAttentionBlock).mixer.wo)
        NemotronMTPFrontRowsQKV.qualify(assistant.module.layers[0] as! NemotronH35MTPAttentionBlock)
        NemotronMTPAttentionDirect.warm(target)
        NemotronMTPVerifySDPA.qualify(target)
        NemotronPrefillConvDt.qualify(target)
        NemotronFreshStateUpdate.qualify()
        NemotronPrefillCountingSort.qualify(target)
        NemotronPrefillCombineAdd.qualify()
        NemotronPrefillSharedRelu2.qualify(target)
        NemotronPrefillDirectFC1.qualify(target)
        NemotronPrefillDirectFC1.qualifyDict(target)
        NemotronSeedHeadDict.qualify(target)
        NemotronPrefillSharedUpGather.qualify(target)
        NemotronPrefillFC2Copy.qualify(target)
        NemotronPrefillSharedDownCopy.qualify(target)
        NemotronPrefillDenseDict.qualify(target)
        NemotronPrefillMoEJoin.qualify(target)
        NemotronPrefillDtxDecay4.qualify()
        NemotronPrefillScanExp.qualify()
        NemotronPrefillBoundaryJoin.qualify(target)
        NemotronPrefillFreshWindowOut.qualify()
        NemotronPrefillWindowOutSG.qualify()
        NemotronPrefillFreshStateFold.qualify()
        NemotronPrefillGatedRMSWide.qualify()
        NemotronPrefillScanDecaySkip.qualify()
        NemotronPrefillScanDecayGemm.qualify()
        NemotronPrefillStateGemm.qualify()
        NemotronPrefillSeedI32.qualify(target)
        // After every seed qualify (the GPU is warm): retiled seed fc1/fc2.
        NemotronSeedRetile.tune(target)
        // The general pass replaces one synthetic post-verify warm-up with
        // its real captured rows, then warms only the remaining widths.
        // Keep last: it ends with Memory.clearCache().
        try NemotronDraftAttentionGeometry.warm()
        assistant.warmUpExactShapes()
        assistant.prepareDecodeResidency()
        return assistant
    }
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
        let environment = ProcessInfo.processInfo.environment
        guard environment["DARKBLOOM_NEMOTRON35_MTP_WARMUP"] != "0" else {
            warmPostVerifySpeculation()
            return
        }
        let requested = Int(environment["DARKBLOOM_NEMOTRON35_MTP_WARMUP_DEPTH"] ?? "") ?? 2
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
        if depth == 2, NemotronMTPPendingSSM.warmsSteps, NemotronMTPPendingSSM.enabled {
            rounds += [(depth, depth + 1)]
                + Array(repeating: (depth, 1), count: NemotronMTPPendingSSM.limit + 1)
                + [(depth, depth + 1)]
        }
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
        // Depth one has no shallower round to consume the prestaged row.
        // Release that private transaction through the ordinary discard path.
        discardRound(requestState: draftState)
        Stream.gpu.synchronize()
    }
}

import Metal

@_silgen_name("nemotron_decode_backing")
private func nemotronDecodeBacking(_ opaque: UnsafeMutableRawPointer) -> UnsafeMutableRawPointer?

@available(macOS 15.0, *)
final class NemotronDecodeResidency: @unchecked Sendable {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_MTP_DECODE_RESIDENCY"] != "0"
    static let seedLayers = 0
    let arrays: [MLXArray]
    let buffers: [any MTLBuffer]
    let seedSet: any MTLResidencySet
    let decodeSets: [any MTLResidencySet]
    let owners: [ObjectIdentifier]
    let cut: Int
    let queue = DispatchQueue(label: "nemotron.decode.memory")
    /// NEMOTRON_MTP_RESIDENCY_END=1 ends every set's request when the lease finishes; by default the
    /// requests are left standing between requests.
    static let endsRequests = ProcessInfo.processInfo.environment["NEMOTRON_MTP_RESIDENCY_END"] == "1"

    /// The target's own parameter buffers, one set per backbone group (embeddings, layers in order, then the
    /// rest), requested on their own serial queue when a request's lease begins and ended with the lease.
    static let targetEnabled = ProcessInfo.processInfo.environment["NEMOTRON_MTP_TARGET_RESIDENCY"] != "0"
    let targetSets: [any MTLResidencySet]
    let targetQueue = DispatchQueue(label: "nemotron.target.memory")
    /// Load-time decode-only backing outside the head and copy sets (the draft head's routed-FC2 copy and the
    /// scale-bias pair dictionaries), requested on the target queue after the target's own groups.
    static let extrasEnabled = ProcessInfo.processInfo.environment["NEMOTRON_MTP_EXTRA_RESIDENCY"] != "0"
    let extraSets: [any MTLResidencySet]
    /// Backing first read after the request's first verify (the head's routed experts with
    /// `NemotronMTPFirstRoundDraft`), requested on the target queue after the extras.
    let lateSets: [any MTLResidencySet]
    /// With target sets, the per-layer copy sets are requested on the target queue right after the extras, each
    /// marking its layer ready when its request returns, instead of at the first verify graph.
    static let copiesAfterExtras = ProcessInfo.processInfo.environment["NEMOTRON_MTP_COPIES_AFTER_HEAD"] != "0"
    private(set) var copiesEnqueued = false
    /// The buffers of the residency sets are locked in memory (`mlock`) once at load, so a lease's request maps
    /// pages that stay wired between requests. NEMOTRON_MTP_MLOCK=targets locks only the target's groups;
    /// NEMOTRON_MTP_MLOCK=0 locks nothing.
    static let lockMode = ProcessInfo.processInfo.environment["NEMOTRON_MTP_MLOCK"] ?? "all"
    /// `mlock` of every distinct buffer of `sets`: (buffers locked, bytes locked).
    static func lockPages(_ sets: [any MTLResidencySet]) -> (Int, Int) {
        var seen = Set<ObjectIdentifier>()
        var count = 0, bytes = 0
        for set in sets {
            for allocation in set.allAllocations {
                guard let buffer = allocation as? any MTLBuffer,
                      seen.insert(ObjectIdentifier(buffer)).inserted,
                      buffer.storageMode == .shared, buffer.length > 0,
                      mlock(buffer.contents(), buffer.length) == 0
                else { continue }
                count += 1
                bytes += buffer.length
            }
        }
        return (count, bytes)
    }

    /// The target's parameters grouped in backbone order: embeddings, layers 0..., then the rest.
    /// With `NemotronPrefillFC2Copy` verified, a MoE layer's routed fc2 bank is replaced in its group by
    /// that layer's decode copy (codes, pair index, pair table): the request prefill reads the copy, the
    /// decode reads the copy, and nothing in the request reads the bank.
    /// NEMOTRON_RESIDENCY_SPLIT_EXPERTS=0 keeps one set per backbone layer.
    static let splitExpertGroups = ProcessInfo.processInfo.environment["NEMOTRON_RESIDENCY_SPLIT_EXPERTS"] != "0"

    static func targetGroups(_ model: Module) -> [[MLXArray]] {
        var copies: [Int: [MLXArray]] = [:]
        if NemotronPrefillFC2Copy.verified {
            for (name, module) in model.namedModules() {
                let parts = name.split(separator: ".")
                guard let i = parts.firstIndex(of: "layers"), i + 1 < parts.count, let n = Int(parts[i + 1]),
                      let moe = module as? NemotronHMoE,
                      let copy = NemotronZSkip.copy(for: moe.switchMLP.fc2), copy.dict else { continue }
                copies[n] = [copy.codes, copy.scales, copy.biases]
            }
        }
        // With `NemotronPrefillDirectFC1.dictVerified`, the request prefill's fc1 reads the bank's pair
        // dictionary (as the decode does) instead of its BF16 scales and biases.
        var fc1Dicts: [Int: [MLXArray]] = [:]
        if NemotronPrefillDirectFC1.dictVerified {
            for (name, module) in model.namedModules() {
                let parts = name.split(separator: ".")
                guard let i = parts.firstIndex(of: "layers"), i + 1 < parts.count, let n = Int(parts[i + 1]),
                      let moe = module as? NemotronHMoE,
                      let d = NemotronSBDict.dict(moe.switchMLP.fc1) else { continue }
                fc1Dicts[n] = [d.index, d.table]
            }
        }
        // With `splitExpertGroups` a MoE layer's group is three consecutive sets in the order the request
        // prefill reads them: the layer's own small arrays (router, shared expert, norms), its routed fc1
        // (bank words and their scale/bias form), then its routed fc2 (the decode copy when verified).
        let split = Self.splitExpertGroups
        func key(_ layer: Int, _ part: Int) -> Int { split ? 3 * layer + part : layer }
        // With `NemotronPrefillSharedDownCopy.verified`, the request prefill's shared down reads the layer's
        // shared-down decode copy (as the verify does) instead of the dense weights.
        var sharedCopies: [Int: [MLXArray]] = [:]
        if NemotronPrefillSharedDownCopy.verified {
            for (name, module) in model.namedModules() {
                let parts = name.split(separator: ".")
                guard let i = parts.firstIndex(of: "layers"), i + 1 < parts.count, let n = Int(parts[i + 1]),
                      let moe = module as? NemotronHMoE, let down = moe.sharedExperts?.downProj,
                      let c = NemotronSharedDownZSkip.copyArrays(for: down) else { continue }
                sharedCopies[n] = [c.codes, c.scales, c.biases]
            }
        }
        // With a `NemotronPrefillDenseDict` kind admitted, the request prefill reads that kind's pair dictionaries
        // (as the verify does) instead of its BF16 scales and biases.
        var denseDicts: [Int: [MLXArray]] = [:]
        var denseMetadata = Set<ObjectIdentifier>()
        for (name, module) in model.namedModules() {
            let parts = name.split(separator: ".")
            guard let i = parts.firstIndex(of: "layers"), i + 1 < parts.count, let n = Int(parts[i + 1]) else { continue }
            var dense: [(String, Linear)] = []
            if let mamba = module as? NemotronHMamba2Mixer {
                dense = [("in_proj", mamba.inProj), ("out_proj", mamba.outProj)]
            } else if let moe = module as? NemotronHMoE, let up = moe.sharedExperts?.upProj {
                dense = [("up_proj", up)]
            }
            for (kind, linear) in dense where NemotronPrefillDenseDict.kinds.contains(kind) {
                guard let q = linear as? QuantizedLinear, let d = NemotronSBDict.dict(q), let b = q.biases else { continue }
                denseDicts[n, default: []] += [d.index, d.table]
                denseMetadata.formUnion([ObjectIdentifier(q.scales), ObjectIdentifier(b)])
            }
        }
        var groups: [Int: [MLXArray]] = [:]
        for (path, a) in model.parameters().flattened() {
            let parts = path.split(separator: ".")
            var g = Int.max
            var layer: Int? = nil
            if parts.contains("embeddings") { g = split ? -3 : -1 }
            if let i = parts.firstIndex(of: "layers"), i + 1 < parts.count, let n = Int(parts[i + 1]) {
                layer = n
                g = key(n, 0)
            }
            if denseMetadata.contains(ObjectIdentifier(a)) { continue }
            if let n = layer, sharedCopies[n] != nil, let s = parts.firstIndex(of: "shared_experts"), s + 1 < parts.count,
               parts[s + 1] == "down_proj" {
                continue
            }
            if let n = layer, let s = parts.firstIndex(of: "switch_mlp"), s + 2 < parts.count {
                if parts[s + 1] == "fc2" {
                    if copies[n] != nil { continue }
                    g = key(n, 2)
                } else if parts[s + 1] == "fc1" {
                    if fc1Dicts[n] != nil, parts[s + 2] == "scales" || parts[s + 2] == "biases" { continue }
                    g = key(n, 1)
                }
            }
            groups[g, default: []].append(a)
        }
        for (n, arrays) in copies { groups[key(n, 2), default: []].append(contentsOf: arrays) }
        for (n, arrays) in fc1Dicts { groups[key(n, 1), default: []].append(contentsOf: arrays) }
        // The shared-down decode copies and the dense pair dictionaries belong to the layer's own small-array set.
        for (n, arrays) in sharedCopies { groups[key(n, 0), default: []].append(contentsOf: arrays) }
        for (n, arrays) in denseDicts { groups[key(n, 0), default: []].append(contentsOf: arrays) }
        FileHandle.standardError.write(Data(
            "nemotron target residency: \(denseMetadata.count) dense scale/bias arrays replaced by their pair dictionaries\n".utf8))
        FileHandle.standardError.write(Data(
            "nemotron target residency: \(sharedCopies.count) MoE shared down projections replaced by their decode copies\n".utf8))
        FileHandle.standardError.write(Data(
            "nemotron target residency: \(fc1Dicts.count) MoE fc1 scales/biases replaced by their pair dictionaries\n".utf8))
        FileHandle.standardError.write(Data(
            "nemotron target residency: \(copies.count) MoE fc2 banks replaced by their decode copies\n".utf8))
        return groups.keys.sorted().map { groups[$0]! }
    }

    /// One committed set per group; none at all when a set cannot be made or a buffer cannot be found.
    private static func makeTargetSets(_ device: any MTLDevice, _ target: [[MLXArray]]) -> [any MTLResidencySet] {
        var sets: [any MTLResidencySet] = []
        var seen = Set<ObjectIdentifier>()
        for group in target {
            let descriptor = MTLResidencySetDescriptor()
            descriptor.initialCapacity = group.count
            guard let set = try? device.makeResidencySet(descriptor: descriptor) else { return [] }
            for a in group {
                guard let ctx = a.ctx.ctx, let ptr = nemotronDecodeBacking(ctx),
                      let buffer = Unmanaged<AnyObject>.fromOpaque(ptr).takeUnretainedValue() as? any MTLBuffer
                else { return [] }
                if seen.insert(ObjectIdentifier(buffer)).inserted { set.addAllocation(buffer) }
            }
            set.commit()
            sets.append(set)
        }
        return sets
    }

    init?(head: [MLXArray], layers: [[MLXArray]], owners: [ObjectIdentifier], target: [[MLXArray]],
          extras: [MLXArray], late: [MLXArray] = []) {
        guard Self.enabled, let device = MTLCreateSystemDefaultDevice() else { return nil }
        let targetSets = Self.targetEnabled ? Self.makeTargetSets(device, target) : []
        let lateSets = Self.extrasEnabled && !late.isEmpty ? Self.makeTargetSets(device, [late]) : []
        // Without a late set the banks go back with the extras (requested before the decode).
        let extraArrays = lateSets.isEmpty ? extras + late : extras
        let extraSets = Self.extrasEnabled && !extraArrays.isEmpty ? Self.makeTargetSets(device, [extraArrays]) : []
        self.lateSets = lateSets
        FileHandle.standardError.write(Data(
            "nemotron target residency: \(targetSets.count) sets, decode extras: \(extraSets.count) set, late: \(lateSets.count) set\n".utf8))
        self.targetSets = targetSets
        self.extraSets = extraSets
        let cut = min(Self.seedLayers, layers.count)
        guard owners.count == layers.count else { return nil }
        let groups = [head + layers.prefix(cut).flatMap { $0 }] + Array(layers.dropFirst(cut))
        var buffers: [any MTLBuffer] = []
        var seen = Set<ObjectIdentifier>()
        var sets: [any MTLResidencySet] = []
        for (i, group) in groups.enumerated() {
            let descriptor = MTLResidencySetDescriptor()
            descriptor.initialCapacity = group.count
            guard let set = try? device.makeResidencySet(descriptor: descriptor) else { return nil }
            var bytes = 0
            for a in group {
                guard let ctx = a.ctx.ctx, let ptr = nemotronDecodeBacking(ctx),
                      let buffer = Unmanaged<AnyObject>.fromOpaque(ptr).takeUnretainedValue() as? any MTLBuffer
                else { return nil }
                if seen.insert(ObjectIdentifier(buffer)).inserted {
                    buffers.append(buffer); bytes += buffer.length; set.addAllocation(buffer)
                }
            }
            set.commit()
            sets.append(set)
        }
        self.arrays = head + layers.flatMap { $0 }
        self.buffers = buffers
        self.seedSet = sets[0]
        self.decodeSets = Array(sets.dropFirst())
        self.owners = owners
        self.cut = cut
        if Self.lockMode != "0" {
            let locked = Self.lockPages(Self.lockMode == "targets" ? targetSets : targetSets + extraSets + lateSets + sets)
            FileHandle.standardError.write(Data(
                "nemotron residency mlock: \(locked.0) buffers, \(locked.1 >> 20) MiB\n".utf8))
        }
    }

    final class Lease: CBv2MTPMemoryLease {
        let owner: NemotronDecodeResidency
        private var decoded = false
        private var finished = false
        private var rounds = 0
        /// The request's CPU keep-busy (`NemotronCPUWarm`), from the lease's begin (the seed) to round 4.
        private let warm = NemotronCPUWarm.enabled ? NemotronCPUWarm() : nil
        init(_ owner: NemotronDecodeResidency) { self.owner = owner }
        func beginDecode() {
            guard !finished else { return }
            rounds += 1
            if rounds > NemotronCPUWarm.rounds { warm?.stop() }
            if !decoded {
                decoded = true
                for (i, set) in (owner.copiesEnqueued ? [] : owner.decodeSets).enumerated() {
                    owner.request(set, phase: "layer", ready: [owner.owners[owner.cut + i]])
                }
            }
            _ = NemotronZSkip.snapshotCopyReadiness()
        }
        func finish() {
            guard !finished else { return }
            finished = true
            warm?.stopAndJoin()
            NemotronMTPFirstRoundDraft.ready()
            let owner = owner
            guard NemotronDecodeResidency.endsRequests else {
                // The requests stay standing; the next lease requests the same sets again.
                owner.targetQueue.sync {}
                owner.queue.sync { NemotronZSkip.endCopyReadiness() }
                return
            }
            owner.targetQueue.sync {
                for set in owner.targetSets + owner.extraSets + owner.lateSets { set.endResidency() }
            }
            owner.queue.sync {
                owner.seedSet.endResidency()
                if decoded || owner.copiesEnqueued { for set in owner.decodeSets { set.endResidency() } }
                NemotronZSkip.endCopyReadiness()
            }
        }
        deinit { finish() }
    }
    private func request(_ set: any MTLResidencySet, phase: String, ready: [ObjectIdentifier]) {
        queue.async {
            set.requestResidency()
            NemotronZSkip.markCopyReady(ready)
        }
    }
    /// One serial order for the lease's first requests: the target's groups (the seed's own weights, in
    /// backbone order), then the MTP head's set, then the decode-only extras. With no target sets the head
    /// set keeps its own queue.
    /// The lease this process built at load, for `earlyBegin` from an engine's initializer.
    nonisolated(unsafe) static weak var current: NemotronDecodeResidency?
    /// NEMOTRON_RESIDENCY_EARLY=0 skips it: the target groups' residency requests are enqueued (in the same
    /// backbone order, on the same serial queue) when a request's engine is built, before the engine is
    /// assembled; the lease's `begin` that follows within `earlyWindow` then enqueues only the head set, the
    /// extras and the copies after them, so the serial order is the base's.
    static let earlyEnabled = ProcessInfo.processInfo.environment["NEMOTRON_RESIDENCY_EARLY"] != "0"
    private static let earlyWindow: TimeInterval = 5
    private let earlyLock = NSLock()
    private var earlyIssuedAt: Date?

    func earlyBegin() {
        guard Self.earlyEnabled, !targetSets.isEmpty else { return }
        earlyLock.withLock { earlyIssuedAt = Date() }
        for set in targetSets { targetQueue.async { set.requestResidency() } }
    }

    func begin() -> Lease {
        let early = earlyLock.withLock { () -> Bool in
            defer { earlyIssuedAt = nil }
            guard let at = earlyIssuedAt else { return false }
            return Date().timeIntervalSince(at) < Self.earlyWindow
        }
        NemotronZSkip.beginCopyReadiness(owners)
        copiesEnqueued = false
        let copiesInTarget = NemotronPrefillFC2Copy.verified && !targetSets.isEmpty
        guard !targetSets.isEmpty else {
            for set in extraSets + lateSets { targetQueue.async { set.requestResidency() } }
            request(seedSet, phase: "seed", ready: Array(owners.prefix(cut)))
            return Lease(self)
        }
        if !early { for set in targetSets { targetQueue.async { set.requestResidency() } } }
        if copiesInTarget {
            // The request prefill reads every layer's copy (they replace the fc2 banks in the target
            // groups), so the first verify already finds them bound: no round falls back to a bank.
            NemotronZSkip.markCopyReady(owners)
        }
        let head = seedSet, ready = Array(owners.prefix(cut))
        targetQueue.async {
            head.requestResidency()
            NemotronZSkip.markCopyReady(ready)
        }
        for set in extraSets { targetQueue.async { set.requestResidency() } }
        if !lateSets.isEmpty {
            NemotronMTPFirstRoundDraft.begin()
            let late = lateSets
            targetQueue.async {
                for set in late { set.requestResidency() }
                NemotronMTPFirstRoundDraft.ready()
            }
        }
        if Self.copiesAfterExtras {
            copiesEnqueued = true
            for (i, set) in decodeSets.enumerated() {
                let ready = [owners[cut + i]]
                targetQueue.async {
                    set.requestResidency()
                    NemotronZSkip.markCopyReady(ready)
                }
            }
        }
        return Lease(self)
    }
}

// Second measurement of ff544a18.

// Second measurement entry of this tree.

// Second measurement of 208838c4.
// e-man07: third measurement of c28 (b550b079, 2dbff4f2).

// Measurement entry 2 of this tree.

/// A request's CPU keep-busy (terrapinelf's `8e4dac9d`, carried unchanged). The host builds the first decode
/// rounds right after the seed, a phase in which it mostly waits on the GPU; after an idle period the CPU then
/// builds those rounds at a lowered clock, and the GPU waits for their command buffers. From the MTP lease's
/// begin (the request's seed) until the fourth decode round begins, or the lease finishes if sooner, two
/// user-interactive threads stay busy so that the cores that build the first rounds run at speed. Nothing runs
/// between requests, and no value changes. Off by default; NEMOTRON_MTP_CPU_WARM=1 enables it.
final class NemotronCPUWarm: @unchecked Sendable {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_MTP_CPU_WARM"] == "1"
    /// Decode rounds covered after the seed.
    static let rounds = 3
    private static let threads = 2
    private let lock = NSLock()
    private var stopped = false
    private let exited = DispatchGroup()

    init() {
        for i in 0 ..< Self.threads {
            exited.enter()
            let thread = Thread { [self] in
                while !self.isStopped {}
                self.exited.leave()
            }
            thread.qualityOfService = .userInteractive
            thread.name = "nemotron.cpu-warm.\(i)"
            thread.start()
        }
    }

    private var isStopped: Bool { lock.withLock { stopped } }

    /// Idempotent and non-blocking: the threads leave their loops at their next check.
    func stop() {
        lock.withLock { stopped = true }
    }

    /// `stop`, then wait until both threads have left their loops (the lease's finish).
    func stopAndJoin() {
        stop()
        exited.wait()
    }
}

// Measurement entry of the carried 3f932e3c tree.
// e-man07: measurement 2 of c44.
// Measurement 3
