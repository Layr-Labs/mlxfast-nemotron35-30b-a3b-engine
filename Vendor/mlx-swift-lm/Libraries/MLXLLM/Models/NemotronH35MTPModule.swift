// Native Nemotron Lightning embedded MTP head: post-norm target hidden +
// next-token embedding -> fusion -> attention -> MoE -> final norm.
// Tensor contract cross-checked against NVIDIA's BF16 checkpoint and
// jundot/omlx patches/mlx_lm_mtp/nemotron_h_model.py (2026-09-08).
import MLX
import MLXLMCommon
import MLXNN


/// Row-local native BF16 RMS reductions with the concatenation as the output store.
private enum NemotronMTPPairedNorm {
    static let kernel = MLXFast.metalKernel(
        name: "nemotron_mtp_paired_input_rms", inputNames: ["embedding", "hidden", "ew", "hw", "eps"],
        outputNames: ["joined"], source: """
            const uint gid = threadgroup_position_in_grid.x;
            const uint row = gid / 2;
            const uint bank = gid % 2;
            const uint lid = thread_position_in_threadgroup.x;
            const uint lane = thread_index_in_simdgroup;
            const uint simd = simdgroup_index_in_threadgroup;
            const device T* input = bank == 0 ? embedding : hidden;
            const device T* weight = bank == 0 ? ew : hw;
            threadgroup float local_sums[32];
            threadgroup float inverse[1];
            float values[4];
            float acc = 0;
            const uint feature = lid * 4;
            for (uint i = 0; i < 4; i++) {
                values[i] = input[row * 2688 + feature + i];
                acc += values[i] * values[i];
            }
            acc = simd_sum(acc);
            if (simd == 0) { local_sums[lane] = 0; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) { local_sums[simd] = acc; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd == 0) {
                acc = simd_sum(local_sums[lane]);
                if (lane == 0) { inverse[0] = metal::precise::rsqrt(acc / 2688 + eps); }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = 0; i < 4; i++) {
                joined[row * 5376 + bank * 2688 + feature + i] = weight[feature + i]
                    * static_cast<T>(values[i] * inverse[0]);
            }
            """, ensureRowContiguous: true)

    static func apply(_ embedding: MLXArray, _ hidden: MLXArray, en: RMSNorm, hn: RMSNorm) -> MLXArray? {
        guard Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              embedding.ndim == 3, embedding.dim(0) == 1, embedding.dim(1) > 0,
              embedding.dim(2) == 2688, hidden.shape == embedding.shape,
              embedding.dtype == .bfloat16, hidden.dtype == .bfloat16,
              type(of: en) == RMSNorm.self, type(of: hn) == RMSNorm.self,
              en.eps == hn.eps, en.weight.shape == [2688], hn.weight.shape == [2688],
              en.weight.dtype == .bfloat16, hn.weight.dtype == .bfloat16 else { return nil }
        return kernel([embedding, hidden, en.weight, hn.weight, en.eps],
            template: [("T", DType.bfloat16)],
            grid: (672 * embedding.dim(1) * 2, 1, 1), threadGroup: (672, 1, 1),
            outputShapes: [[1, embedding.dim(1), 5376]], outputDTypes: [.bfloat16])[0]
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

    private func joinedInputs(hidden: MLXArray, embedding: MLXArray) -> MLXArray {
        NemotronMTPPairedNorm.apply(embedding, hidden, en: embeddingNorm, hn: hiddenNorm)
            ?? concatenated([embeddingNorm(embedding), hiddenNorm(hidden)], axis: -1)
    }

    private func fuse(hidden: MLXArray, embedding: MLXArray) -> MLXArray {
        projection(joinedInputs(hidden: hidden, embedding: embedding))
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
        cache: any CBv2AttendingLayerCache,
        finishing: ((MLXArray, MLXArray) -> MLXArray)? = nil
    ) -> MLXArray {
        let fused = fuse(hidden: hidden, embedding: embedding)
        let attended = mixer.cbv2Forward(norm(fused), cache: cache)
        return finishing?(fused, attended) ?? (fused + attended)
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
        hidden: MLXArray, embedding: MLXArray, cache: any CBv2AttendingLayerCache,
        finishing: ((MLXArray, MLXArray) -> MLXArray)? = nil
    ) -> (output: MLXArray, keys: MLXArray, values: MLXArray) {
        let joined = joinedInputs(hidden: hidden, embedding: embedding)
        let fused = nemotronMTPLinearRows(joined, projection)
        let x = norm(fused)
        let batch = x.dim(0), length = x.dim(1)
        let (q, k, v) = NemotronTargetQKV.apply(
            x, query: mixer.wq, key: mixer.wk, value: mixer.wv, captureMTP: true
        ) ?? (
            nemotronMTPLinearRows(x, mixer.wq),
            nemotronMTPLinearRows(x, mixer.wk),
            nemotronMTPLinearRows(x, mixer.wv)
        )
        let queries = q
            .reshaped(batch, length, mixer.numHeads, mixer.headDim).transposed(0, 2, 1, 3)
        let keys = k
            .reshaped(batch, length, mixer.numKeyValueHeads, mixer.headDim).transposed(0, 2, 1, 3)
        let values = v
            .reshaped(batch, length, mixer.numKeyValueHeads, mixer.headDim).transposed(0, 2, 1, 3)
        let attended = cache.updateAndAttend(
            queries: queries, keys: keys, values: values, scale: mixer.scale, sinks: nil)
        let merged = attended.transposed(0, 2, 1, 3).reshaped(batch, length, -1)
        let projected = nemotronMTPLinearRows(merged, mixer.wo)
        return (finishing?(fused, projected) ?? (fused + projected), keys, values)
    }
}

final class NemotronH35MTPExpertBlock: Module {
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "mixer") var mixer: NemotronHMoE
    @ModuleInfo(key: "final_layernorm") var finalNorm: RMSNorm

    init(_ args: NemotronHConfiguration) {
        _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
        _mixer.wrappedValue = NemotronHMoE(args)
        _finalNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
    }

    func forwardResidual(_ lhs: MLXArray, _ rhs: MLXArray, rows: Bool) -> MLXArray {
        let residual: MLXArray
        let normalized: MLXArray
        if let pair = NemotronHResidualRMSNorm.apply(lhs, rhs, norm: norm) {
            residual = pair.0; normalized = pair.1
        } else {
            residual = lhs + rhs; normalized = norm(residual)
        }
        let mixed = rows ? mixer.mtpForwardRows(normalized) : mixer(normalized)
        return NemotronHResidualRMSNorm.apply(residual, mixed, norm: finalNorm)?.1
            ?? finalNorm(residual + mixed)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { finalNorm(x + mixer(norm(x))) }

    /// Row-batched twin of `callAsFunction` through the verify MoE row path.
    func forwardRows(_ x: MLXArray) -> MLXArray {
        finalNorm(x + mixer.mtpForwardRows(norm(x)))
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
        let expert = layers[1] as! NemotronH35MTPExpertBlock
        return (layers[0] as! NemotronH35MTPAttentionBlock).pagedForward(
            hidden: hidden, embedding: embedding, cache: cache,
            finishing: { expert.forwardResidual($0, $1, rows: false) })
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
    func pagedForwardRows(
        hidden: MLXArray, embedding: MLXArray, cache: any CBv2AttendingLayerCache
    ) -> (output: MLXArray, keys: MLXArray, values: MLXArray) {
        let expert = layers[1] as! NemotronH35MTPExpertBlock
        return (layers[0] as! NemotronH35MTPAttentionBlock).pagedForwardRows(
            hidden: hidden, embedding: embedding, cache: cache,
            finishing: { expert.forwardResidual($0, $1, rows: true) })
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
        let hidden = cbv2Hidden(tokens, caches: caches, recurrentState: recurrentState)
        let last = hidden[0..., (hidden.dim(1) - 1)..., 0...]
        switch requirement {
        case .evaluationOnly: return (last[0..., 0..., 0..<1], hidden)
        case .lastPositionLogits: return (logits(last), hidden)
        }
    }
}
