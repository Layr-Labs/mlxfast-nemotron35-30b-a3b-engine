import Foundation
// Native Nemotron Lightning embedded MTP head: post-norm target hidden +
// next-token embedding -> fusion -> attention -> MoE -> final norm.
// Tensor contract cross-checked against NVIDIA's BF16 checkpoint and
// jundot/omlx patches/mlx_lm_mtp/nemotron_h_model.py (2026-09-08).
import MLX
import MLXLMCommon
import MLXNN

enum NemotronMTPDualRMSJoin {
    private static let kernel = MLXFast.metalKernel(
        name: "nemotron_mtp_dual_rms_join",
        inputNames: ["embedding", "hidden", "e_weight", "h_weight", "axis_size", "e_eps", "h_eps"],
        outputNames: ["joined"],
        source: """
            // Copyright © 2024 Apple Inc. Native rms_single_row arithmetic.
            constexpr int SIMD_SIZE = 32;
            constexpr int N_READS = 4;
            uint gid = threadgroup_position_in_grid.x;
            uint lid = thread_position_in_threadgroup.x;
            uint simd_lane_id = thread_index_in_simdgroup;
            uint simd_group_id = simdgroup_index_in_threadgroup;
            uint row = gid / 2;
            uint side = gid % 2;
            const device T* x = side == 0 ? embedding : hidden;
            const device T* w = side == 0 ? e_weight : h_weight;
            int64_t row_stride = side == 0 ? embedding_strides[1] : hidden_strides[1];
            int64_t feature_stride = side == 0 ? embedding_strides[2] : hidden_strides[2];
            uint w_stride = uint(side == 0 ? e_weight_strides[0] : h_weight_strides[0]);
            float eps = side == 0 ? e_eps : h_eps;
            threadgroup float local_inv_mean[1];
            threadgroup float local_sums[SIMD_SIZE];
            float acc = 0;
            float thread_x[N_READS];
            x += int64_t(row) * row_stride + int64_t(lid * N_READS) * feature_stride;
            w += w_stride * lid * N_READS;
            if (lid * N_READS + N_READS <= axis_size) {
                for (int i = 0; i < N_READS; i++) {
                    thread_x[i] = x[int64_t(i) * feature_stride];
                    acc += thread_x[i] * thread_x[i];
                }
            } else {
                for (int i = 0; i < N_READS; i++) {
                    thread_x[i] = (lid * N_READS + i < axis_size)
                        ? (float)x[int64_t(i) * feature_stride] : 0;
                    acc += thread_x[i] * thread_x[i];
                }
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
            device T* out = joined + size_t(row) * 5376 + side * 2688 + lid * N_READS;
            if (lid * N_READS + N_READS <= axis_size) {
                for (int i = 0; i < N_READS; i++) {
                    out[i] = w[w_stride * i] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
                }
            } else {
                for (int i = 0; i < N_READS; i++) {
                    if ((lid * N_READS + i) < axis_size) {
                        out[i] = w[w_stride * i] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
                    }
                }
            }
            """, ensureRowContiguous: false)

    static func apply(
        embedding: MLXArray, hidden: MLXArray, embeddingNorm: RMSNorm, hiddenNorm: RMSNorm
    ) -> MLXArray? {
        guard Device.defaultDevice() == .gpu, StreamOrDevice.default == .gpu,
            type(of: embeddingNorm) == RMSNorm.self, type(of: hiddenNorm) == RMSNorm.self,
            embedding.ndim == 3, embedding.dim(0) == 1, (2...512).contains(embedding.dim(1)),
            embedding.dim(2) == 2688, hidden.shape == embedding.shape,
            embedding.dtype == .bfloat16, hidden.dtype == .bfloat16,
            embeddingNorm.weight.shape == [2688], hiddenNorm.weight.shape == [2688],
            embeddingNorm.weight.dtype == .bfloat16, hiddenNorm.weight.dtype == .bfloat16
        else { return nil }
        let rows = embedding.dim(1)
        return kernel(
            [embedding, hidden, embeddingNorm.weight, hiddenNorm.weight,
             MLXArray(UInt32(2688)), MLXArray(embeddingNorm.eps), MLXArray(hiddenNorm.eps)],
            template: [("T", DType.bfloat16)], grid: (2 * rows * 672, 1, 1),
            threadGroup: (672, 1, 1), outputShapes: [[1, rows, 5376]], outputDTypes: [.bfloat16])[0]
    }
}

final class NemotronH35MTPAttentionBlock: Module {
    let hiddenSize: Int
    @ModuleInfo(key: "eh_proj") var projection: Linear
    @ModuleInfo(key: "enorm") var embeddingNorm: RMSNorm
    @ModuleInfo(key: "hnorm") var hiddenNorm: RMSNorm
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "mixer") var mixer: NemotronHAttention

    init(_ args: NemotronHConfiguration) {
        hiddenSize = args.hiddenSize
        _projection.wrappedValue = Linear(2 * args.hiddenSize, args.hiddenSize, bias: false)
        _embeddingNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
        _hiddenNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
        _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
        _mixer.wrappedValue = NemotronHAttention(args)
    }

    private func fuse(hidden: MLXArray, embedding: MLXArray) -> MLXArray {
        let joined = concatenated([embeddingNorm(embedding), hiddenNorm(hidden)], axis: -1)
        return NemotronMultiRowQMV.regularHeadR1(joined, projection) ?? projection(joined)
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
        let fused = fuse(hidden: hidden, embedding: embedding)
        return fused + mixer.cbv2Forward(norm(fused), cache: cache)
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
        let fused: MLXArray
        if let joined = NemotronMTPDualRMSJoin.apply(
            embedding: embedding, hidden: hidden, embeddingNorm: embeddingNorm, hiddenNorm: hiddenNorm)
        {
            fused = projection(joined)
        } else {
            fused = fuse(hidden: hidden, embedding: embedding)
        }
        let x = norm(fused)
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
        let joined = concatenated([embeddingNorm(embedding), hiddenNorm(hidden)], axis: -1)
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
        return (fused + nemotronMTPLinearRows(merged, mixer.wo), keys, values)
    }
}

final class NemotronH35MTPExpertBlock: Module {
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "mixer") var mixer: NemotronHMoE
    @ModuleInfo(key: "final_layernorm") var finalNorm: RMSNorm
    private let selectedTailCache = NemotronH35MTPSelectedTailCache()

    init(_ args: NemotronHConfiguration) {
        _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
        _mixer.wrappedValue = NemotronHMoE(args)
        _finalNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { finalNorm(x + mixer(norm(x))) }

    func selectedTail(_ x: MLXArray) -> MLXArray {
        guard NemotronMTPExecution.compiledMoE,
            Device.defaultDevice() == .gpu, StreamOrDevice.default == .gpu,
            x.ndim == 3, x.dim(0) == 1, x.dim(1) == 1, x.dim(2) == 2688,
            x.dtype == .bfloat16
        else { return self(x) }
        return selectedTailCache(x, owner: self)
    }

    /// Row-batched twin of `callAsFunction` through the verify MoE row path.
    func forwardRows(_ x: MLXArray) -> MLXArray {
        finalNorm(x + mixer.mtpForwardRows(norm(x)))
    }
}

/// Cache only the selected stateless expert tail; attention and prefix logits stay outside.
private final class NemotronH35MTPSelectedTailCache {
    private struct Entry {
        let identity: [ObjectIdentifier]
        let function: @Sendable (MLXArray) -> MLXArray
    }
    private let lock = NSLock()
    private var entry: Entry?

    func callAsFunction(_ x: MLXArray, owner: NemotronH35MTPExpertBlock) -> MLXArray {
        guard let identity = Self.nativeIdentity(owner) else { return owner(x) }
        var retired: Entry?
        let current = lock.withLock { () -> Entry in
            if let entry, entry.identity == identity { return entry }
            let state = owner.innerState()
            let function: @Sendable (MLXArray) -> MLXArray = compile(inputs: [state]) { [weak owner] x in
                guard let owner else { preconditionFailure("Released Nemotron selected expert-tail owner") }
                return owner(x)
            }
            let replacement = Entry(identity: identity, function: function)
            retired = entry
            entry = replacement
            return replacement
        }
        return withExtendedLifetime(retired) { current.function(x) }
    }

    private static func nativeIdentity(_ owner: NemotronH35MTPExpertBlock) -> [ObjectIdentifier]? {
        let mixer = owner.mixer
        guard type(of: owner.norm) == RMSNorm.self,
            type(of: owner.finalNorm) == RMSNorm.self,
            type(of: mixer) == NemotronHMoE.self,
            type(of: mixer.gate) == NemotronHMoEGate.self,
            type(of: mixer.switchMLP) == NemotronHSwitchMLP.self,
            type(of: mixer.switchMLP.fc1) == QuantizedSwitchLinear.self,
            type(of: mixer.switchMLP.fc2) == QuantizedSwitchLinear.self,
            let shared = mixer.sharedExperts, type(of: shared) == NemotronHMLP.self,
            type(of: shared.upProj) == QuantizedLinear.self,
            type(of: shared.downProj) == QuantizedLinear.self
        else { return nil }
        let modules: [Module] = [owner, owner.norm, mixer, owner.finalNorm,
            mixer.gate, mixer.switchMLP, mixer.switchMLP.fc1, mixer.switchMLP.fc2,
            shared, shared.upProj, shared.downProj]
        return modules.map(ObjectIdentifier.init)
    }
}

final class NemotronH35MTPModule: Module {
    // Heterogeneous array preserves the official mtp.layers.{0,1} namespace.
    @ModuleInfo(key: "layers") var layers: [Module]
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
        let x = (layers[0] as! NemotronH35MTPAttentionBlock).pagedForward(
            hidden: hidden, embedding: embedding, cache: cache)
        return (layers[1] as! NemotronH35MTPExpertBlock)(x)
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
    /// Retain every attention/KV row; finalize selects one ordinary tail.
    func pagedAttentionRows(
        hidden: MLXArray, embedding: MLXArray, cache: any CBv2AttendingLayerCache
    ) -> (output: MLXArray, keys: MLXArray, values: MLXArray) {
        (layers[0] as! NemotronH35MTPAttentionBlock).pagedForwardRows(
            hidden: hidden, embedding: embedding, cache: cache)
    }

    func finishPagedAttentionRow(_ attention: MLXArray) -> MLXArray {
        precondition(attention.ndim == 3 && attention.dim(0) == 1 && attention.dim(1) == 1)
        return (layers[1] as! NemotronH35MTPExpertBlock).selectedTail(attention)
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
        return (NemotronMultiRowQMV.regularHeadR1(hidden, lmHead) ?? logits(hidden), hidden)
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
        let hidden = cbv2Hidden(tokens, caches: caches, recurrentState: recurrentState)
        let last = hidden[0..., (hidden.dim(1) - 1)..., 0...]
        switch requirement {
        case .evaluationOnly: return (last[0..., 0..., 0..<1], hidden)
        case .lastPositionLogits:
            return (NemotronMultiRowQMV.regularHeadR1(last, lmHead) ?? logits(last), hidden)
        }
    }
}
