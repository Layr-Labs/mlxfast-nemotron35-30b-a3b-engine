// Native Nemotron Lightning embedded MTP head: post-norm target hidden +
// next-token embedding -> fusion -> attention -> MoE -> final norm.
// Tensor contract cross-checked against NVIDIA's BF16 checkpoint and
// jundot/omlx patches/mlx_lm_mtp/nemotron_h_model.py (2026-09-08).
import Foundation
import MLX
import MLXLMCommon
import MLXNN

enum NemotronMTPDraftTail {
    private static func enabled(_ key: String) -> Bool {
        let environment = ProcessInfo.processInfo.environment
        return environment[key] != "0" && environment["NEMOTRON_MTP_DRAFT_FUSED"] != "0"
    }
    static let fusedQKV = enabled("NEMOTRON_MTP_DRAFT_QKV")
    static let exactTail = enabled("NEMOTRON_MTP_DRAFT_TAIL")
}

enum NemotronMTPFrontNorms {
    static let enabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_DRAFT_FRONT"] != "0"
        && ProcessInfo.processInfo.environment["NEMOTRON_MTP_DRAFT_FUSED"] != "0"
    private static let kernel = MLXFast.metalKernel(
        name: "nemotron_mtp_front_norms_concat_bf16",
        inputNames: ["e", "h", "ew", "hw", "axis_size", "e_eps", "h_eps"],
        outputNames: ["out"],
        source: """
            constexpr uint n_reads = 4;
            const uint gid = threadgroup_position_in_grid.x;
            const uint lid = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;
            const uint row = gid / 2;
            const bool second = (gid % 2) != 0;
            const device T* x = second ? h : e;
            const device T* weight = second ? hw : ew;
            const float eps = second ? h_eps : e_eps;
            threadgroup float local_inv_mean[1];
            threadgroup float local_sums[32];
            float acc = 0;
            float thread_x[n_reads];
            const size_t offset = size_t(row) * axis_size + lid * n_reads;
            for (uint i = 0; i < n_reads; i++) {
                thread_x[i] = x[offset + i];
                acc += thread_x[i] * thread_x[i];
            }
            acc = simd_sum(acc);
            if (simd_group_id == 0) {
                local_sums[simd_lane_id] = 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_lane_id == 0) {
                local_sums[simd_group_id] = acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group_id == 0) {
                acc = simd_sum(local_sums[simd_lane_id]);
                if (simd_lane_id == 0) {
                    local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const size_t target = size_t(row) * (2 * size_t(axis_size))
                + (second ? size_t(axis_size) : 0) + lid * n_reads;
            for (uint i = 0; i < n_reads; i++) {
                out[target + i] = weight[lid * n_reads + i]
                    * static_cast<T>(thread_x[i] * local_inv_mean[0]);
            }
            """, ensureRowContiguous: true)

    nonisolated(unsafe) private static var axis: MLXArray?
    nonisolated(unsafe) private static var epsilons: [UInt32: MLXArray] = [:]
    private static let lock = NSLock()
    private static func scalars(_ e: Float, _ h: Float) -> (MLXArray, MLXArray, MLXArray) {
        lock.lock(); defer { lock.unlock() }
        let a = axis ?? MLXArray(UInt32(2688))
        axis = a
        func value(_ eps: Float) -> MLXArray {
            if let cached = epsilons[eps.bitPattern] { return cached }
            let created = MLXArray(eps)
            epsilons[eps.bitPattern] = created
            return created
        }
        return (a, value(e), value(h))
    }

    static func apply(embedding: MLXArray, hidden: MLXArray, enorm: RMSNorm, hnorm: RMSNorm)
        -> MLXArray?
    {
        guard enabled, Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              embedding.ndim == 3, embedding.dim(0) == 1, (1...512).contains(embedding.dim(1)),
              embedding.dim(2) == 2688, hidden.shape == embedding.shape,
              embedding.dtype == .bfloat16, hidden.dtype == .bfloat16,
              type(of: enorm) == RMSNorm.self, type(of: hnorm) == RMSNorm.self,
              enorm.weight.shape == [2688], hnorm.weight.shape == [2688],
              enorm.weight.dtype == .bfloat16, hnorm.weight.dtype == .bfloat16
        else { return nil }
        let rows = embedding.dim(1)
        let (a, e, h) = scalars(enorm.eps, hnorm.eps)
        let joined = kernel([embedding, hidden, enorm.weight, hnorm.weight, a, e, h],
            template: [("T", DType.bfloat16)],
            grid: (672 * 2 * rows, 1, 1), threadGroup: (672, 1, 1),
            outputShapes: [[1, rows, 5376]], outputDTypes: [.bfloat16])[0]
        guard NemotronMoELeader.selfTest else { return joined }
        return nemotronBitwiseOrNaN(
            joined, concatenated([enorm(embedding), hnorm(hidden)], axis: -1))
    }
}

final class NemotronH35MTPAttentionBlock: Module {
    let hiddenSize: Int
    @ModuleInfo(key: "eh_proj") var projection: Linear
    @ModuleInfo(key: "enorm") var embeddingNorm: RMSNorm
    @ModuleInfo(key: "hnorm") var hiddenNorm: RMSNorm
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "mixer") var mixer: NemotronHAttention
    private let draftFrontCache = NemotronH35MTPDraftFrontCache()

    init(_ args: NemotronHConfiguration) {
        hiddenSize = args.hiddenSize
        _projection.wrappedValue = Linear(2 * args.hiddenSize, args.hiddenSize, bias: false)
        _embeddingNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
        _hiddenNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
        _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
        _mixer.wrappedValue = NemotronHAttention(args)
    }

    private func fuse(hidden: MLXArray, embedding: MLXArray) -> MLXArray {
        projection(joinedNorms(hidden: hidden, embedding: embedding))
    }

    private func joinedNorms(hidden: MLXArray, embedding: MLXArray) -> MLXArray {
        NemotronMTPFrontNorms.apply(
            embedding: embedding, hidden: hidden, enorm: embeddingNorm, hnorm: hiddenNorm)
            ?? concatenated([embeddingNorm(embedding), hiddenNorm(hidden)], axis: -1)
    }

    func projectedKVDType(hiddenDType: DType) -> DType {
        let hidden = MLXArray.zeros([1, 1, hiddenSize], dtype: hiddenDType)
        let embedding = MLXArray.zeros(like: hidden)
        return mixer.wk(norm(fuse(hidden: hidden, embedding: embedding))).dtype
    }

    func callAsFunction(hidden: MLXArray, embedding: MLXArray, cache: KVCache) -> MLXArray {
        let fused = fuse(hidden: hidden, embedding: embedding)
        let mask = createAttentionMask(h: fused, cache: cache)
        return fused + mixer(norm(fused), attentionMask: mask, ssmMask: nil, cache: cache)
    }

    func pagedForward(
        hidden: MLXArray, embedding: MLXArray,
        cache: any CBv2AttendingLayerCache
    ) -> MLXArray {
        let parts = pagedForwardParts(hidden: hidden, embedding: embedding, cache: cache)
        return parts.residual + parts.attention
    }

    func pagedForwardParts(
        hidden: MLXArray, embedding: MLXArray,
        cache: any CBv2AttendingLayerCache
    ) -> (residual: MLXArray, attention: MLXArray) {
        let front = draftFrontCache(hidden: hidden, embedding: embedding, owner: self)
        return (front[0], mixer.cbv2Forward(
            front[1], cache: cache, targetQKV: NemotronMTPDraftTail.fusedQKV))
    }

    /// The part of `pagedForwardParts` that reads no cache: the fused
    /// residual, its norm and that norm's Q/K/V, as [residual, norm, q, k, v].
    func pagedDraftInputs(hidden: MLXArray, embedding: MLXArray) -> [MLXArray] {
        let front = draftFrontCache(hidden: hidden, embedding: embedding, owner: self)
        let (q, k, v) = mixer.cbv2Projections(front[1], targetQKV: NemotronMTPDraftTail.fusedQKV)
        return [front[0], front[1], q, k, v]
    }

    /// `pagedForwardRows`' cache-free front: the fused residual rows and their
    /// Q [1, heads, L, headDim] and K/V [1, kvHeads, L, headDim].
    func frontRows(hidden: MLXArray, embedding: MLXArray)
        -> (fused: MLXArray, queries: MLXArray, keys: MLXArray, values: MLXArray)
    {
        let joined = joinedNorms(hidden: hidden, embedding: embedding)
        let fused = nemotronMTPLinearRows(joined, projection)
        let x = norm(fused)
        let batch = x.dim(0), length = x.dim(1)
        let queries = nemotronMTPLinearRows(x, mixer.wq)
            .reshaped(batch, length, mixer.numHeads, mixer.headDim).transposed(0, 2, 1, 3)
        let keys = nemotronMTPLinearRows(x, mixer.wk)
            .reshaped(batch, length, mixer.numKeyValueHeads, mixer.headDim).transposed(0, 2, 1, 3)
        let values = nemotronMTPLinearRows(x, mixer.wv)
            .reshaped(batch, length, mixer.numKeyValueHeads, mixer.headDim).transposed(0, 2, 1, 3)
        return (fused, queries, keys, values)
    }

    /// One row's attention-block output (residual + projected attention) from
    /// its query [1, heads, 1, headDim] over explicit K/V with an additive mask.
    func maskedRowOutput(
        fused: MLXArray, queries: MLXArray, keys: MLXArray, values: MLXArray, mask: MLXArray
    ) -> MLXArray {
        let output = MLXFast.scaledDotProductAttention(
            queries: queries.asType(keys.dtype), keys: keys, values: values,
            scale: mixer.scale, mask: .array(mask))
        let merged = output.transposed(0, 2, 1, 3).reshaped(1, 1, -1)
        return fused + nemotronMTPLinearRows(merged, mixer.wo)
    }

    /// The draft row's own K/V, [1, kvHeads, 1, headDim], from `pagedDraftInputs`.
    func draftKeyValues(inputs: [MLXArray]) -> (keys: MLXArray, values: MLXArray) {
        (inputs[3].reshaped(1, 1, mixer.numKeyValueHeads, mixer.headDim).transposed(0, 2, 1, 3),
         inputs[4].reshaped(1, 1, mixer.numKeyValueHeads, mixer.headDim).transposed(0, 2, 1, 3))
    }

    /// The attention of `pagedForwardParts(inputs:)` over explicit `keys` and
    /// `values` with an additive `mask` (no cache access); output projected.
    func maskedDraftAttention(
        inputs: [MLXArray], keys: MLXArray, values: MLXArray, mask: MLXArray
    ) -> MLXArray {
        let queries = inputs[2].reshaped(1, 1, mixer.numHeads, mixer.headDim).transposed(0, 2, 1, 3)
            .asType(keys.dtype)
        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: mixer.scale, mask: .array(mask))
        return mixer.wo(output.transposed(0, 2, 1, 3).reshaped(1, 1, -1))
    }

    func devicePagedSecondAttention(
        inputs: [MLXArray], cache: PagedLayerCache, accepted: MLXArray,
        trustedOffset: Int, postVerifyRows: Int
    ) -> MLXArray? {
        let own = draftKeyValues(inputs: inputs)
        let queries = inputs[2].reshaped(1, mixer.numHeads, mixer.headDim)
        guard let output = cache.nemotronDeviceSecondAttention(
            queries: queries, ownKeys: own.keys.squeezed(axis: 2),
            ownValues: own.values.squeezed(axis: 2), acceptedDrafts: accepted,
            trustedOffset: trustedOffset, postVerifyRows: postVerifyRows, scale: mixer.scale)
        else { return nil }
        return mixer.wo(output.reshaped(1, 1, -1))
    }

    /// `pagedForwardParts` from `pagedDraftInputs` of the same inputs.
    func pagedForwardParts(
        inputs: [MLXArray], cache: any CBv2AttendingLayerCache
    ) -> (residual: MLXArray, attention: MLXArray) {
        (inputs[0], mixer.cbv2Forward(
            inputs[1], cache: cache, targetQKV: NemotronMTPDraftTail.fusedQKV,
            projections: (inputs[2], inputs[3], inputs[4])))
    }

    fileprivate func uncompiledDraftFront(hidden: MLXArray, embedding: MLXArray) -> [MLXArray] {
        let fused = fuse(hidden: hidden, embedding: embedding)
        return [fused, norm(fused)]
    }

    /// Trusted target rows only need the head's K/V projections. Their
    /// discarded query, attention output, MoE and final norm cannot affect KV.
    func appendTrustedKV(hidden: MLXArray, embedding: MLXArray, cache: KVCache) -> [MLXArray] {
        let x = norm(fuse(hidden: hidden, embedding: embedding))
        let batch = x.dim(0), length = x.dim(1)
        let keys = mixer.wk(x).reshaped(batch, length, mixer.numKeyValueHeads, mixer.headDim).transposed(0, 2, 1, 3)
        let values = mixer.wv(x).reshaped(batch, length, mixer.numKeyValueHeads, mixer.headDim).transposed(0, 2, 1, 3)
        let (k, v) = cache.update(keys: keys, values: values)
        return [k, v]
    }


    func appendTrustedKV(
        hidden: MLXArray, embedding: MLXArray, cache: PagedLayerCache
    ) -> [MLXArray] {
        let x = norm(fuse(hidden: hidden, embedding: embedding))
        let batch = x.dim(0), length = x.dim(1)
        let keys = mixer.wk(x).reshaped(batch, length, mixer.numKeyValueHeads, mixer.headDim)
            .transposed(0, 2, 1, 3)
        let values = mixer.wv(x).reshaped(batch, length, mixer.numKeyValueHeads, mixer.headDim)
            .transposed(0, 2, 1, 3)
        return cache.appendTrusted(keys: keys, values: values)
    }

    /// Causal forward over several trusted (token, target hidden) rows at
    /// once, returning the projected K/V too. Projections use the multi-row
    /// exact qmv (per-row stock M=1 arithmetic); norms are row-local.
    func pagedForwardRows(
        hidden: MLXArray, embedding: MLXArray, cache: any CBv2AttendingLayerCache
    ) -> (output: MLXArray, keys: MLXArray, values: MLXArray) {
        let joined = joinedNorms(hidden: hidden, embedding: embedding)
        let fused = nemotronMTPLinearRows(joined, projection)
        let x = norm(fused)
        let batch = x.dim(0), length = x.dim(1)
        let queries = nemotronMTPLinearRows(x, mixer.wq)
            .reshaped(batch, length, mixer.numHeads, mixer.headDim).transposed(0, 2, 1, 3)
        let keys = nemotronMTPLinearRows(x, mixer.wk)
            .reshaped(batch, length, mixer.numKeyValueHeads, mixer.headDim).transposed(0, 2, 1, 3)
        let values = nemotronMTPLinearRows(x, mixer.wv)
            .reshaped(batch, length, mixer.numKeyValueHeads, mixer.headDim).transposed(0, 2, 1, 3)
        let attended = cache.updateAndAttend(
            queries: queries, keys: keys, values: values, scale: mixer.scale, sinks: nil)
        let merged = attended.transposed(0, 2, 1, 3).reshaped(batch, length, -1)
        if let output = NemotronMultiRowQMV.applyAttentionOutputAdding(
            merged, mixer.wo, residual: fused) {
            return (output, keys, values)
        }
        return (fused + nemotronMTPLinearRows(merged, mixer.wo), keys, values)
    }
}

/// Only the one-token draft's stateless norm/concat/projection/norm prefix.
/// The original attention projections and all cache operations stay outside.
private final class NemotronH35MTPDraftFrontCache {
    private struct Entry {
        let identity: [ObjectIdentifier]
        let function: @Sendable ([MLXArray]) -> [MLXArray]
    }
    private static let enabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_COMPILED_DRAFT_FRONT"] != "0"
    private let lock = NSLock()
    private var entry: Entry?

    func callAsFunction(
        hidden: MLXArray, embedding: MLXArray, owner: NemotronH35MTPAttentionBlock
    ) -> [MLXArray] {
        guard Self.enabled,
              Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              hidden.shape == [1, 1, 2688], hidden.dtype == .bfloat16,
              embedding.shape == hidden.shape, embedding.dtype == hidden.dtype,
              type(of: owner.embeddingNorm) == RMSNorm.self,
              type(of: owner.hiddenNorm) == RMSNorm.self,
              type(of: owner.norm) == RMSNorm.self,
              (type(of: owner.projection) == Linear.self
                || type(of: owner.projection) == QuantizedLinear.self)
        else { return owner.uncompiledDraftFront(hidden: hidden, embedding: embedding) }
        let modules: [Module] = [owner, owner.embeddingNorm, owner.hiddenNorm,
                                 owner.projection, owner.norm]
        let identity = modules.map(ObjectIdentifier.init)
        var retired: Entry?
        let current = lock.withLock { () -> Entry in
            if let entry, entry.identity == identity { return entry }
            let state = modules.dropFirst().flatMap { $0.innerState() }
            let function: @Sendable ([MLXArray]) -> [MLXArray] = compile(inputs: [state]) {
                [weak owner] arrays in
                guard let owner else { preconditionFailure("Released Nemotron MTP draft-front owner") }
                return owner.uncompiledDraftFront(hidden: arrays[0], embedding: arrays[1])
            }
            let replacement = Entry(identity: identity, function: function)
            retired = entry
            entry = replacement
            return replacement
        }
        let result = withExtendedLifetime(retired) { current.function([hidden, embedding]) }
        return result.count == 2 ? result
            : owner.uncompiledDraftFront(hidden: hidden, embedding: embedding)
    }
}

final class NemotronH35MTPExpertBlock: Module {
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "mixer") var mixer: NemotronHMoE
    @ModuleInfo(key: "final_layernorm") var finalNorm: RMSNorm
    private let oneTokenTailCache = NemotronH35MTPOneTokenTailCache()
    fileprivate var sparseDraftDown: NemotronMTPDraftSparseDown?

    /// The loader calls this only for its own mtp.layers.1 after weight update,
    /// before any expert-tail graph is traced. Target MoEs are never passed.
    func prepareOwnedSparseDraftDown() {
        guard mixer.numExpertsPerTok == 6 else {
            sparseDraftDown = nil
            return
        }
        sparseDraftDown = NemotronMTPDraftSparseDown.prepare(mixer.switchMLP, finalNorm: finalNorm)
        print("[nemotron-mtp-owned-sparse-fc2] \(sparseDraftDown == nil ? "off" : "on (worker bitwise admission passed)")")
        print("[nemotron-mtp-owned-fc2-fold-rms] \(sparseDraftDown?.foldJoinQualified == true ? "on (worker bitwise admission passed)" : "off")")
    }

    init(_ args: NemotronHConfiguration) {
        _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
        _mixer.wrappedValue = NemotronHMoE(args)
        _finalNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if NemotronMTPExecution.compiledMoE,
           x.ndim == 3, x.dim(0) == 1, x.dim(1) == 1 {
            return oneTokenTailCache([x], mode: exactTailAvailable(x) ? 1 : 0, owner: self)
        }
        return exactTailAvailable(x) ? exactOneTokenTail(x, normalized: nil)
            : uncompiledOneTokenTail(x)
    }

    func callAsFunction(residual: MLXArray, attention: MLXArray) -> MLXArray {
        if exactTailAvailable(residual), attention.shape == residual.shape,
           let joined = NemotronHResidualRMSNorm.apply(residual, attention, norm: norm) {
            if NemotronMTPExecution.compiledMoE {
                return oneTokenTailCache([joined.0, joined.1], mode: 2, owner: self)
            }
            return exactOneTokenTail(joined.0, normalized: joined.1)
        }
        return self(residual + attention)
    }

    /// Same operations as the joined entry above, without a nested compiled
    /// cache. The whole-second graph supplies all parameter/copy handles.
    fileprivate func uncompiledJoinedTail(residual: MLXArray, attention: MLXArray) -> MLXArray {
        if exactTailAvailable(residual), attention.shape == residual.shape,
           let joined = NemotronHResidualRMSNorm.apply(residual, attention, norm: norm) {
            return exactOneTokenTail(joined.0, normalized: joined.1)
        }
        let x = residual + attention
        return exactTailAvailable(x) ? exactOneTokenTail(x, normalized: nil)
            : uncompiledOneTokenTail(x)
    }

    fileprivate func uncompiledOneTokenTail(_ x: MLXArray) -> MLXArray {
        finalNorm(x + mixer(norm(x)))
    }

    private func exactTailAvailable(_ x: MLXArray) -> Bool {
        NemotronMTPDraftTail.exactTail && NemotronMoELeader.enabled
            && x.shape == [1, 1, 2688] && x.dtype == .bfloat16
            && type(of: norm) == RMSNorm.self && type(of: finalNorm) == RMSNorm.self
            && type(of: mixer) == NemotronHMoE.self && mixer.sharedExperts != nil
            && mixer.numExpertsPerTok == 6
            && Device.defaultDevice().deviceType == .gpu && StreamOrDevice.default == .gpu
    }

    fileprivate func exactOneTokenTail(_ x: MLXArray, normalized: MLXArray?) -> MLXArray {
        let h = normalized ?? norm(x)
        let (indices, scores) = mixer.gate(h)
        if let plan = sparseDraftDown, plan.foldJoinQualified,
           let sharedExperts = mixer.sharedExperts,
           let products = plan.productsForJoinedTail(h, indices, scores, mixer.switchMLP),
           let joined = NemotronHResidualRMSNorm.applyDraftExpertFold(
               x, products: products, shared: sharedExperts(h), norm: finalNorm) {
            return joined.1
        }
        let sparse = sparseDraftDown?.run(h, indices, scores, mixer.switchMLP)
        guard let sharedExperts = mixer.sharedExperts,
              let routed = sparse ?? NemotronMoELeader.switchRowsAndWeightedReduce(
                  h, indices, scores, mixer.switchMLP)
        else { return finalNorm(x + mixer(h)) }
        let shared = sharedExperts(h)
        if let joined = NemotronHResidualRMSNorm.applySum(x, routed, shared: shared, norm: finalNorm) {
            return joined.1
        }
        return finalNorm(x + (routed + shared))
    }

    /// Row-batched twin of `callAsFunction` through the verify MoE row path.
    func forwardRows(_ x: MLXArray) -> MLXArray {
        finalNorm(x + mixer.mtpForwardRows(norm(x)))
    }
}

/// Cache only the assistant's stateless one-token expert tail. Attention,
/// request state, KV writes, offsets and transactions remain outside the graph.
private final class NemotronH35MTPOneTokenTailCache {
    private struct Entry {
        let identity: [ObjectIdentifier]
        let usesFoldJoin: Bool
        let function: @Sendable ([MLXArray]) -> [MLXArray]
    }
    private let lock = NSLock()
    private var entries: [Entry?] = [nil, nil, nil]

    func callAsFunction(_ inputs: [MLXArray], mode: Int, owner: NemotronH35MTPExpertBlock)
        -> MLXArray
    {
        // Native parameter wrappers keep their array objects when updated;
        // capture those handles as compiler inputs, as the MoE row cache does.
        // Child replacement invalidates the entry. Unknown child types use
        // reflective module and parameter identities instead.
        let nativeIdentity = Self.nativeIdentity(owner)
        let parameters = nativeIdentity == nil ? owner.innerState() : nil
        let identity = (nativeIdentity ?? (owner.modules().map(ObjectIdentifier.init)
            + (parameters ?? []).map(ObjectIdentifier.init)))
            + (owner.sparseDraftDown.map { [ObjectIdentifier($0)] } ?? [])
        let usesFoldJoin = owner.sparseDraftDown?.foldJoinQualified == true
        var retired: Entry?
        let current = lock.withLock { () -> Entry in
            if let entry = entries[mode], entry.identity == identity,
               entry.usesFoldJoin == usesFoldJoin { return entry }
            let state = (parameters ?? owner.innerState())
                + (owner.sparseDraftDown?.compilationState ?? [])
            let function: @Sendable ([MLXArray]) -> [MLXArray] = compile(inputs: [state]) {
                [weak owner] arrays in
                guard let owner else { preconditionFailure("Released Nemotron MTP expert-tail owner") }
                switch mode {
                case 0: return [owner.uncompiledOneTokenTail(arrays[0])]
                case 1: return [owner.exactOneTokenTail(arrays[0], normalized: nil)]
                default: return [owner.exactOneTokenTail(arrays[0], normalized: arrays[1])]
                }
            }
            let replacement = Entry(identity: identity, usesFoldJoin: usesFoldJoin, function: function)
            retired = entries[mode]
            entries[mode] = replacement
            return replacement
        }
        // The compiler has its own locks; retire old entries after releasing
        // this cache lock, including when a module tree has been replaced.
        let result = withExtendedLifetime(retired) { current.function(inputs) }
        guard result.count == 1 else {
            switch mode {
            case 0: return owner.uncompiledOneTokenTail(inputs[0])
            case 1: return owner.exactOneTokenTail(inputs[0], normalized: nil)
            default: return owner.exactOneTokenTail(inputs[0], normalized: inputs[1])
            }
        }
        return result[0]
    }

    fileprivate static func nativeIdentity(_ owner: NemotronH35MTPExpertBlock) -> [ObjectIdentifier]? {
        func linear(_ value: Linear) -> Bool {
            type(of: value) == Linear.self || type(of: value) == QuantizedLinear.self
        }
        func switched(_ value: SwitchLinear) -> Bool {
            type(of: value) == SwitchLinear.self || type(of: value) == QuantizedSwitchLinear.self
        }
        let mixer = owner.mixer
        guard type(of: owner.norm) == RMSNorm.self,
              type(of: owner.finalNorm) == RMSNorm.self,
              type(of: mixer) == NemotronHMoE.self,
              type(of: mixer.gate) == NemotronHMoEGate.self,
              type(of: mixer.switchMLP) == NemotronHSwitchMLP.self,
              switched(mixer.switchMLP.fc1), switched(mixer.switchMLP.fc2) else { return nil }
        var modules: [Module] = [owner, owner.norm, mixer, owner.finalNorm,
                                 mixer.gate, mixer.switchMLP,
                                 mixer.switchMLP.fc1, mixer.switchMLP.fc2]
        if let shared = mixer.sharedExperts {
            guard type(of: shared) == NemotronHMLP.self,
                  linear(shared.upProj), linear(shared.downProj) else { return nil }
            modules += [shared, shared.upProj, shared.downProj]
        }
        return modules.map(ObjectIdentifier.init)
    }
}

/// Bounded model-owned execution graph, with no request/pool/cache captured.
/// The common plan supplies its ten request inputs; the residual is eleventh.
/// Attention fence publication and all KV/transaction updates stay outside.
private final class NemotronH35MTPWholeSecondCache {
    static let enabled = ProcessInfo.processInfo.environment[
        "NEMOTRON_MTP_DEVICE_SECOND_WHOLE_GRAPH"] != "0"
        && ProcessInfo.processInfo.environment["NEMOTRON_MTP_DEVICE_SECOND_GRAPH"] != "0"
    private struct Entry {
        let key: [Int]
        let identity: [ObjectIdentifier]
        let usesFoldJoin: Bool
        let function: @Sendable ([MLXArray]) -> [MLXArray]
    }
    private let lock = NSLock()
    private var entry: Entry?

    func run(_ plan: NemotronDeviceSecondAttentionPlan, residual: MLXArray,
             attention: NemotronH35MTPAttentionBlock, expert: NemotronH35MTPExpertBlock)
        -> [MLXArray]
    {
        let expertIdentity = NemotronH35MTPOneTokenTailCache.nativeIdentity(expert)
        let parameters = expertIdentity == nil ? expert.innerState() : nil
        let identity = [ObjectIdentifier(attention), ObjectIdentifier(attention.mixer),
                        ObjectIdentifier(attention.mixer.wo)]
            + (expertIdentity ?? (expert.modules().map(ObjectIdentifier.init)
                + (parameters ?? []).map(ObjectIdentifier.init)))
            + (expert.sparseDraftDown.map { [ObjectIdentifier($0)] } ?? [])
        let usesFoldJoin = expert.sparseDraftDown?.foldJoinQualified == true
        // Bind the pure builder, NEVER `plan`, which also owns current arrays.
        let buildAttention = plan.build
        let build: ([MLXArray]) -> [MLXArray] = {
            [weak attention, weak expert] arrays in
            guard let attention, let expert else {
                preconditionFailure("Released Nemotron whole-second graph owner")
            }
            let result = buildAttention(Array(arrays.prefix(10)))
            let projected = attention.mixer.wo(result[0].reshaped(1, 1, -1))
            let hidden = expert.uncompiledJoinedTail(residual: arrays[10], attention: projected)
            return [hidden, result[1]]
        }
        var retired: Entry?
        let current = lock.withLock { () -> Entry in
            if let entry, entry.key == plan.key, entry.identity == identity,
               entry.usesFoldJoin == usesFoldJoin { return entry }
            let state = attention.mixer.wo.innerState()
                + (parameters ?? expert.innerState())
                + (expert.sparseDraftDown?.compilationState ?? [])
            let function: @Sendable ([MLXArray]) -> [MLXArray] = compile(inputs: [state], build)
            let replacement = Entry(key: plan.key, identity: identity,
                usesFoldJoin: usesFoldJoin, function: function)
            retired = entry
            entry = replacement
            return replacement
        }
        let inputs = plan.inputs + [residual]
        let result = withExtendedLifetime(retired) { current.function(inputs) }
        return result.count == 2 ? result : build(inputs)
    }
}

/// Entire cache-free front, native read-only attention and expert tail in one
/// graph. The two current model inputs and seven current cache metadata arrays
/// are explicit; own K/V are outputs for the unchanged host-adoption boundary.
/// A separate bounded third-draft cache adds two explicit previous K/V inputs.
private final class NemotronH35MTPSecondFrontCache {
    static let enabled = NemotronH35MTPWholeSecondCache.enabled
        && ProcessInfo.processInfo.environment["NEMOTRON_MTP_SECOND_FRONT_WHOLE_GRAPH"] != "0"
    private struct Entry {
        let key: [Int]
        let identity: [ObjectIdentifier]
        let usesFoldJoin: Bool
        let function: @Sendable ([MLXArray]) -> [MLXArray]
    }
    private let lock = NSLock()
    private var entry: Entry?

    func run(_ plan: NemotronDeviceSecondFrontPlan, hidden: MLXArray, embedding: MLXArray,
             attention: NemotronH35MTPAttentionBlock, expert: NemotronH35MTPExpertBlock)
        -> [MLXArray]
    {
        let expertIdentity = NemotronH35MTPOneTokenTailCache.nativeIdentity(expert)
        let parameters = expertIdentity == nil ? expert.innerState() : nil
        let identity = attention.modules().map(ObjectIdentifier.init)
            + (expertIdentity ?? (expert.modules().map(ObjectIdentifier.init)
                + (parameters ?? []).map(ObjectIdentifier.init)))
            + (expert.sparseDraftDown.map { [ObjectIdentifier($0)] } ?? [])
        let usesFoldJoin = expert.sparseDraftDown?.foldJoinQualified == true
        let buildAttention = plan.build
        let build: ([MLXArray]) -> [MLXArray] = {
            [weak attention, weak expert] arrays in
            guard let attention, let expert else {
                preconditionFailure("Released Nemotron second-front graph owner")
            }
            // Uncompiled original operations; no nested front-cache boundary.
            let front = attention.uncompiledDraftFront(hidden: arrays[0], embedding: arrays[1])
            let (q, k, v) = attention.mixer.cbv2Projections(
                front[1], targetQKV: NemotronMTPDraftTail.fusedQKV)
            let query = q.reshaped(1, attention.mixer.numHeads, attention.mixer.headDim)
            let keys = k.reshaped(1, attention.mixer.numKeyValueHeads, attention.mixer.headDim)
            let values = v.reshaped(1, attention.mixer.numKeyValueHeads, attention.mixer.headDim)
            let attentionInputs = [query, keys, values] + Array(arrays.dropFirst(2))
            let result = buildAttention(attentionInputs)
            let projected = attention.mixer.wo(result[0].reshaped(1, 1, -1))
            let out = expert.uncompiledJoinedTail(residual: front[0], attention: projected)
            return [out, result[1], keys.expandedDimensions(axis: 2), values.expandedDimensions(axis: 2)]
        }
        var retired: Entry?
        let current = lock.withLock { () -> Entry in
            if let entry, entry.key == plan.key, entry.identity == identity,
               entry.usesFoldJoin == usesFoldJoin { return entry }
            let state = attention.innerState() + (parameters ?? expert.innerState())
                + (expert.sparseDraftDown?.compilationState ?? [])
            let function: @Sendable ([MLXArray]) -> [MLXArray] = compile(inputs: [state], build)
            let replacement = Entry(key: plan.key, identity: identity,
                usesFoldJoin: usesFoldJoin, function: function)
            retired = entry
            entry = replacement
            return replacement
        }
        let inputs = [hidden, embedding] + plan.inputs
        let result = withExtendedLifetime(retired) { current.function(inputs) }
        return result.count == 4 ? result : build(inputs)
    }
}

final class NemotronH35MTPModule: Module {
    // Heterogeneous array preserves the official mtp.layers.{0,1} namespace.
    @ModuleInfo(key: "layers") var layers: [Module]
    private let wholeSecondCache = NemotronH35MTPWholeSecondCache()
    private let secondFrontCache = NemotronH35MTPSecondFrontCache()
    private let thirdFrontCache = NemotronH35MTPSecondFrontCache()
    init(_ args: NemotronHConfiguration) {
        _layers.wrappedValue = [NemotronH35MTPAttentionBlock(args), NemotronH35MTPExpertBlock(args)]
    }
    func projectedKVDType(hiddenDType: DType) -> DType {
        (layers[0] as! NemotronH35MTPAttentionBlock).projectedKVDType(hiddenDType: hiddenDType)
    }
    func callAsFunction(hidden: MLXArray, embedding: MLXArray, cache: KVCache) -> MLXArray {
        let x = (layers[0] as! NemotronH35MTPAttentionBlock)(hidden: hidden, embedding: embedding, cache: cache)
        return (layers[1] as! NemotronH35MTPExpertBlock)(x)
    }
    func pagedForward(
        hidden: MLXArray, embedding: MLXArray,
        cache: any CBv2AttendingLayerCache
    ) -> MLXArray {
        let parts = (layers[0] as! NemotronH35MTPAttentionBlock).pagedForwardParts(
            hidden: hidden, embedding: embedding, cache: cache)
        return (layers[1] as! NemotronH35MTPExpertBlock)(
            residual: parts.residual, attention: parts.attention)
    }
    func pagedDraftInputs(hidden: MLXArray, embedding: MLXArray) -> [MLXArray] {
        (layers[0] as! NemotronH35MTPAttentionBlock).pagedDraftInputs(
            hidden: hidden, embedding: embedding)
    }
    func draftKeyValues(inputs: [MLXArray]) -> (keys: MLXArray, values: MLXArray) {
        (layers[0] as! NemotronH35MTPAttentionBlock).draftKeyValues(inputs: inputs)
    }
    func frontRows(hidden: MLXArray, embedding: MLXArray)
        -> (fused: MLXArray, queries: MLXArray, keys: MLXArray, values: MLXArray)
    {
        (layers[0] as! NemotronH35MTPAttentionBlock).frontRows(hidden: hidden, embedding: embedding)
    }
    func maskedRowOutput(
        fused: MLXArray, queries: MLXArray, keys: MLXArray, values: MLXArray, mask: MLXArray
    ) -> MLXArray {
        (layers[0] as! NemotronH35MTPAttentionBlock).maskedRowOutput(
            fused: fused, queries: queries, keys: keys, values: values, mask: mask)
    }
    /// The one-token draft step from `pagedDraftInputs` with its attention over
    /// explicit, masked K/V instead of the paged head cache.
    func maskedSecondDraft(
        inputs: [MLXArray], keys: MLXArray, values: MLXArray, mask: MLXArray
    ) -> MLXArray {
        let attention = (layers[0] as! NemotronH35MTPAttentionBlock).maskedDraftAttention(
            inputs: inputs, keys: keys, values: values, mask: mask)
        return (layers[1] as! NemotronH35MTPExpertBlock)(residual: inputs[0], attention: attention)
    }
    /// Read the post-verify selected row through the same native, explicit
    /// attention/WO/expert graph. Its own K/V is already one of the slab rows.
    func devicePagedPickedFirstDraft(
        fused: MLXArray, queries: MLXArray, keys: MLXArray, values: MLXArray,
        cache: PagedLayerCache, accepted: MLXArray, trustedOffset: Int, postVerifyRows: Int
    ) -> MLXArray? {
        guard NemotronH35MTPPickedFirstAttention.nativePaged else { return nil }
        let block = layers[0] as! NemotronH35MTPAttentionBlock
        return devicePagedDraft(residual: fused,
            queries: queries.reshaped(1, block.mixer.numHeads, block.mixer.headDim),
            ownKeys: keys.squeezed(axis: 2), ownValues: values.squeezed(axis: 2),
            cache: cache, accepted: accepted, trustedOffset: trustedOffset,
            postVerifyRows: postVerifyRows, ownRowDelta: 1)
    }

    func devicePagedSecondDraft(
        inputs: [MLXArray], cache: PagedLayerCache, accepted: MLXArray,
        trustedOffset: Int, postVerifyRows: Int
    ) -> MLXArray? {
        guard inputs.count == 5 else { return nil }
        let block = layers[0] as! NemotronH35MTPAttentionBlock
        let own = block.draftKeyValues(inputs: inputs)
        return devicePagedDraft(residual: inputs[0],
            queries: inputs[2].reshaped(1, block.mixer.numHeads, block.mixer.headDim),
            ownKeys: own.keys.squeezed(axis: 2), ownValues: own.values.squeezed(axis: 2),
            cache: cache, accepted: accepted, trustedOffset: trustedOffset,
            postVerifyRows: postVerifyRows, ownRowDelta: 2)
    }

    /// No current projection is constructed outside the admitted full graph.
    /// A rejected layout/parameter signature retains the old five-input route.
    func devicePagedSecondFromHidden(
        hidden: MLXArray, embedding: MLXArray, cache: PagedLayerCache,
        accepted: MLXArray, trustedOffset: Int, postVerifyRows: Int,
        precedingKeys: MLXArray? = nil, precedingValues: MLXArray? = nil
    ) -> (out: MLXArray, keys: MLXArray, values: MLXArray)? {
        guard NemotronH35MTPSecondFrontCache.enabled,
              hidden.shape == [1, 1, 2688], hidden.dtype == .bfloat16,
              embedding.shape == hidden.shape, embedding.dtype == .bfloat16
        else { return nil }
        let attention = layers[0] as! NemotronH35MTPAttentionBlock
        let expert = layers[1] as! NemotronH35MTPExpertBlock
        func projection(_ layer: Linear, n: Int, k: Int) -> Bool {
            if type(of: layer) == QuantizedLinear.self, let q = layer as? QuantizedLinear {
                guard q.mode == .affine, q.bits == 4, q.groupSize == 64, q.bias == nil,
                      q.weight.dtype == .uint32, q.weight.shape == [n, k / 8],
                      q.scales.dtype == .bfloat16, q.scales.shape == [n, k / 64],
                      let offsets = q.biases, offsets.dtype == .bfloat16,
                      offsets.shape == q.scales.shape else { return false }
                return true
            }
            return type(of: layer) == Linear.self && layer.weight.dtype == .bfloat16
                && layer.weight.shape == [n, k] && layer.bias == nil
        }
        guard attention.hiddenSize == 2688, attention.mixer.numHeads == 32,
              attention.mixer.numKeyValueHeads == 2, attention.mixer.headDim == 128,
              projection(attention.projection, n: 2688, k: 5376),
              projection(attention.mixer.wq, n: 4096, k: 2688),
              projection(attention.mixer.wk, n: 256, k: 2688),
              projection(attention.mixer.wv, n: 256, k: 2688),
              projection(attention.mixer.wo, n: 2688, k: 4096),
              [attention.embeddingNorm, attention.hiddenNorm, attention.norm].allSatisfy({
                  type(of: $0) == RMSNorm.self && $0.weight.dtype == .bfloat16
                      && $0.weight.shape == [2688]
              }) else { return nil }
        var outputs: [MLXArray]?
        let result = cache.nemotronDeviceSecondFromFront(
            acceptedDrafts: accepted, trustedOffset: trustedOffset,
            postVerifyRows: postVerifyRows, scale: attention.mixer.scale,
            precedingKeys: precedingKeys, precedingValues: precedingValues) { plan in
                let graph = precedingKeys == nil ? self.secondFrontCache : self.thirdFrontCache
                let made = graph.run(plan, hidden: hidden, embedding: embedding,
                    attention: attention, expert: expert)
                guard made.count == 4 else { return [] }
                outputs = made
                return [made[0], made[1]]
            }
        guard result != nil, let outputs else { return nil }
        return (outputs[0], outputs[2], outputs[3])
    }

    /// All current data is passed to the common plan; only immutable model
    /// state joins the trace. First/second semantics live in explicit config.
    private func devicePagedDraft(
        residual: MLXArray, queries: MLXArray, ownKeys: MLXArray, ownValues: MLXArray,
        cache: PagedLayerCache, accepted: MLXArray, trustedOffset: Int,
        postVerifyRows: Int, ownRowDelta: Int
    ) -> MLXArray? {
        let attention = layers[0] as! NemotronH35MTPAttentionBlock
        let expert = layers[1] as! NemotronH35MTPExpertBlock
        let execution: ((NemotronDeviceSecondAttentionPlan) -> [MLXArray])?
        if NemotronH35MTPWholeSecondCache.enabled,
           residual.shape == [1, 1, 2688], residual.dtype == .bfloat16,
           (type(of: attention.mixer.wo) == Linear.self
                || type(of: attention.mixer.wo) == QuantizedLinear.self) {
            execution = { plan in
                self.wholeSecondCache.run(plan, residual: residual, attention: attention, expert: expert)
            }
        } else {
            execution = nil
        }
        guard let result = cache.nemotronDeviceSecondAttention(
            queries: queries, ownKeys: ownKeys, ownValues: ownValues,
            acceptedDrafts: accepted, trustedOffset: trustedOffset,
            postVerifyRows: postVerifyRows, scale: attention.mixer.scale,
            ownRowDelta: ownRowDelta, execution: execution) else { return nil }
        if execution != nil { return result }
        return expert(residual: residual, attention: attention.mixer.wo(result.reshaped(1, 1, -1)))
    }
    func pagedForward(inputs: [MLXArray], cache: any CBv2AttendingLayerCache) -> MLXArray {
        let parts = (layers[0] as! NemotronH35MTPAttentionBlock).pagedForwardParts(
            inputs: inputs, cache: cache)
        return (layers[1] as! NemotronH35MTPExpertBlock)(
            residual: parts.residual, attention: parts.attention)
    }
    func appendTrustedKV(hidden: MLXArray, embedding: MLXArray, cache: KVCache) -> [MLXArray] {
        (layers[0] as! NemotronH35MTPAttentionBlock).appendTrustedKV(hidden: hidden, embedding: embedding, cache: cache)
    }
    func appendTrustedKV(
        hidden: MLXArray, embedding: MLXArray, cache: PagedLayerCache
    ) -> [MLXArray] {
        (layers[0] as! NemotronH35MTPAttentionBlock).appendTrustedKV(
            hidden: hidden, embedding: embedding, cache: cache)
    }
    /// Keep the promoted full-width attention/KV arithmetic and its overlap.
    func pagedAttentionRows(
        hidden: MLXArray, embedding: MLXArray, cache: any CBv2AttendingLayerCache
    ) -> (output: MLXArray, keys: MLXArray, values: MLXArray) {
        (layers[0] as! NemotronH35MTPAttentionBlock).pagedForwardRows(
            hidden: hidden, embedding: embedding, cache: cache)
    }

    func finishPagedAttentionRow(_ attention: MLXArray) -> MLXArray {
        precondition(attention.ndim == 3 && attention.dim(0) == 1 && attention.dim(1) == 1)
        return (layers[1] as! NemotronH35MTPExpertBlock)(attention)
    }

    func pagedForwardRows(
        hidden: MLXArray, embedding: MLXArray, cache: any CBv2AttendingLayerCache
    ) -> (output: MLXArray, keys: MLXArray, values: MLXArray) {
        let a = (layers[0] as! NemotronH35MTPAttentionBlock).pagedForwardRows(
            hidden: hidden, embedding: embedding, cache: cache)
        return ((layers[1] as! NemotronH35MTPExpertBlock).forwardRows(a.output), a.keys, a.values)
    }
}

extension NemotronH35Model: CBv2RecurrentMTPForwardable {
    public func cbv2ForwardWithHidden(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        precondition(positionIds == nil, "Nemotron Lightning does not apply RoPE")
        let hidden = cbv2Hidden(tokens, caches: caches, recurrentState: recurrentState)
        // Unlike Qwen35/Gemma, this head is trained on POST-norm_f hidden.
        return (logits(hidden), hidden)
    }
}

extension NemotronH35Model: CBv2MTPPolicyTopTwoProviding {
    public func cbv2MTPTopTwo(_ logits: MLXArray) -> (ids: MLXArray, values: MLXArray) {
        cbv2TopTwoRows(logits)
    }
}

extension NemotronH35Model: CBv2RecurrentPrefillHiddenForwardable {
    public func cbv2ForwardWithHiddenForPrefill(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        precondition(positionIds == nil)
        let hidden = cbv2Hidden(
            tokens, caches: caches, recurrentState: recurrentState, recyclePrefillChunks: true,
            earlyEvalInterval: nemotronPrefillEvalInterval)
        let last = hidden[0..., (hidden.dim(1) - 1)..., 0...]
        switch requirement {
        case .evaluationOnly: return (last[0..., 0..., 0..<1], hidden)
        case .lastPositionLogits: return (logits(last), hidden)
        }
    }
}
