import Foundation
import Cmlx
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
        // Load-time JIT of the exact verify kernels (kernel agent hook).
        NemotronSharedUpPacking.verify(target)
        NemotronZSkip.prepare(target)
        NemotronSBDict.prepare(target)
        NemotronBankBinding.warm(target, extra: NemotronZSkip.copyArrays + NemotronSBDict.indexArrays)
        NemotronBankBinding.warm(assistant.module)
        NemotronHResidualRMSNorm.checkProductsJoin(target)
        NemotronZSkip.prepareDraft(assistant.module)
        NemotronMoELeader.warm(target)
        NemotronMTPRouterRows.warm(target)
        NemotronMTPRouterRowsSelect.qualify(target)
        cbv2TopTwoRowsPacketWarm(logitsDType: .bfloat16)
        NemotronMultiRowQMV.warmNarrow()
        NemotronMultiRowQMV.verifyAttentionOutputAdding(
            (assistant.module.layers[0] as! NemotronH35MTPAttentionBlock).mixer.wo)
        NemotronMTPAttentionDirect.warm(target)
        NemotronMTPVerifySDPA.qualify(target)
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

// arm64 libc++ layout, measured with offsetof against the pinned array.h and
// allocator.h using Xcode 26.4.1: array.array_desc_=0, ArrayDesc.data=272,
// ArrayDesc.offset=288, Data.buffer=0, Buffer.ptr_=0. shared_ptr stores its
// element pointer first. No ownership is taken through this ABI view.
private func nemotronDecodeBacking(_ a: MLXArray) -> (any MTLBuffer)? {
    func aligned(_ p: UnsafeRawPointer?) -> Bool {
        guard let p else { return false }
        return UInt(bitPattern: p) >= 4096 && UInt(bitPattern: p) % 8 == 0
    }
    guard MemoryLayout<UnsafeRawPointer>.size == 8,
          let ctx = a.ctx.ctx, aligned(UnsafeRawPointer(ctx)),
          let info = try? a.evaluatedBufferInfo(), info.allocatedBytes >= a.nbytes,
          info.dataOffset >= 0, a.nbytes > 0 else { return nil }
    var identity: UInt = 0
    var canCache = false
    guard _mlx_array_constant_cache_identity(&identity, &canCache, a.ctx) == 0,
          identity != 0 else { return nil }
    let desc = ctx.load(as: UnsafeRawPointer?.self)
    guard aligned(desc), let desc, UInt(bitPattern: desc) == identity else { return nil }
    let offset = desc.load(fromByteOffset: 288, as: Int64.self)
    guard offset >= 0, UInt64(offset) == UInt64(info.dataOffset) else { return nil }
    let data = desc.load(fromByteOffset: 272, as: UnsafeRawPointer?.self)
    guard aligned(data), let data else { return nil }
    let ptr = data.load(as: UnsafeRawPointer?.self)
    guard aligned(ptr), let ptr,
          let dataPtr = mlx_array_data_uint8(a.ctx) else { return nil }
    let address = UInt(bitPattern: dataPtr)
    // Check agreement through the walk before messaging the candidate object.
    guard address >= UInt(offset), (address - UInt(offset)) % 8 == 0,
          let buffer = Unmanaged<AnyObject>.fromOpaque(ptr).takeUnretainedValue() as? any MTLBuffer
    else { return nil }
    let contents = UInt(bitPattern: buffer.contents())
    guard buffer.length >= a.nbytes, buffer.length == info.allocatedBytes,
          address >= contents, address - contents < UInt(buffer.length),
          address - contents == UInt(offset),
          UInt(a.nbytes) <= UInt(buffer.length) - (address - contents) else { return nil }
    return buffer
}

@available(macOS 15.0, *)
final class NemotronDecodeResidency: @unchecked Sendable {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_MTP_DECODE_RESIDENCY"] != "0"
    private final class ValidationLog: @unchecked Sendable {
        let lock = NSLock()
        var failed = false
        func reject() {
            lock.lock(); defer { lock.unlock() }
            if !failed {
                failed = true
                FileHandle.standardError.write(Data("[decode-residency] backing validation failed; no sets, original route disabled\n".utf8))
            }
        }
    }
    private static let validationLog = ValidationLog()
    static let seedLayers = 0
    let arrays: [MLXArray]
    let buffers: [any MTLBuffer]
    let seedSet: any MTLResidencySet
    let decodeSets: [any MTLResidencySet]
    let owners: [ObjectIdentifier]
    let cut: Int
    let queue = DispatchQueue(label: "nemotron.decode.memory")

    /// The target's own parameter buffers, one set per backbone group (embeddings, layers in order, then the
    /// rest), requested on their own serial queue when a request's lease begins and ended with the lease.
    static let targetEnabled = ProcessInfo.processInfo.environment["NEMOTRON_MTP_TARGET_RESIDENCY"] != "0"
    let targetSets: [any MTLResidencySet]
    let targetQueue = DispatchQueue(label: "nemotron.target.memory")

    /// The target's parameters grouped in backbone order: embeddings, layers 0..., then the rest.
    static func targetGroups(_ model: Module) -> [[MLXArray]] {
        var groups: [Int: [MLXArray]] = [:]
        for (key, a) in model.parameters().flattened() {
            let parts = key.split(separator: ".")
            var g = Int.max
            if parts.contains("embeddings") { g = -1 }
            if let i = parts.firstIndex(of: "layers"), i + 1 < parts.count, let n = Int(parts[i + 1]) { g = n }
            groups[g, default: []].append(a)
        }
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
                guard let buffer = nemotronDecodeBacking(a) else { return [] }
                if seen.insert(ObjectIdentifier(buffer)).inserted { set.addAllocation(buffer) }
            }
            set.commit()
            sets.append(set)
        }
        return sets
    }

    init?(head: [MLXArray], layers: [[MLXArray]], owners: [ObjectIdentifier], target: [[MLXArray]]) {
        guard Self.enabled, let device = MTLCreateSystemDefaultDevice() else { return nil }
        let targetSets = Self.targetEnabled ? Self.makeTargetSets(device, target) : []
        FileHandle.standardError.write(Data("nemotron target residency: \(targetSets.count) sets\n".utf8))
        self.targetSets = targetSets
        let cut = min(Self.seedLayers, layers.count)
        guard owners.count == layers.count else { return nil }
        let groups = [head + layers.prefix(cut).flatMap { $0 }] + Array(layers.dropFirst(cut))
        // Validate the entire inventory before creating any residency set. A
        // failure returns nil, so the inherited ready-owner gate cannot choose
        // the original-bank route for this request.
        var validated: [[any MTLBuffer]] = []
        var lookupCount = 0
        for group in groups {
            var backing: [any MTLBuffer] = []
            for a in group {
                guard let buffer = nemotronDecodeBacking(a) else {
                    Self.validationLog.reject()
                    return nil
                }
                backing.append(buffer)
                lookupCount += 1
            }
            validated.append(backing)
        }
        var buffers: [any MTLBuffer] = []
        var seen = Set<ObjectIdentifier>()
        var sets: [any MTLResidencySet] = []
        for (i, group) in validated.enumerated() {
            let descriptor = MTLResidencySetDescriptor()
            descriptor.initialCapacity = group.count
            guard let set = try? device.makeResidencySet(descriptor: descriptor) else { return nil }
            var bytes = 0
            for buffer in group {
                if seen.insert(ObjectIdentifier(buffer)).inserted {
                    buffers.append(buffer); bytes += buffer.length; set.addAllocation(buffer)
                }
            }
            set.commit()
            sets.append(set)
        }
        FileHandle.standardError.write(Data("[decode-residency] backing validation passed: \(lookupCount) lookups, \(buffers.count) unique buffers\n".utf8))
        self.arrays = head + layers.flatMap { $0 }
        self.buffers = buffers
        self.seedSet = sets[0]
        self.decodeSets = Array(sets.dropFirst())
        self.owners = owners
        self.cut = cut
    }

    final class Lease: CBv2MTPMemoryLease {
        let owner: NemotronDecodeResidency
        private var decoded = false
        private var finished = false
        init(_ owner: NemotronDecodeResidency) { self.owner = owner }
        func beginDecode() {
            guard !finished else { return }
            if !decoded {
                decoded = true
                for (i, set) in owner.decodeSets.enumerated() {
                    owner.request(set, phase: "layer", ready: [owner.owners[owner.cut + i]])
                }
            }
            _ = NemotronZSkip.snapshotCopyReadiness()
        }
        func finish() {
            guard !finished else { return }
            finished = true
            let owner = owner
            owner.targetQueue.sync {
                for set in owner.targetSets { set.endResidency() }
            }
            owner.queue.sync {
                owner.seedSet.endResidency()
                if decoded { for set in owner.decodeSets { set.endResidency() } }
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
    func begin() -> Lease {
        for set in targetSets {
            targetQueue.async { set.requestResidency() }
        }
        NemotronZSkip.beginCopyReadiness(owners)
        request(seedSet, phase: "seed", ready: Array(owners.prefix(cut)))
        return Lease(self)
    }
}
