// Native Nemotron Lightning embedded MTP head: post-norm target hidden +
// next-token embedding -> fusion -> attention -> MoE -> final norm.
// Tensor contract cross-checked against NVIDIA's BF16 checkpoint and
// jundot/omlx patches/mlx_lm_mtp/nemotron_h_model.py (2026-09-08).
import MLX
import MLXLMCommon
import MLXNN

/// The two row-local RMS norms write directly into their halves of the fusion
/// input. Each half keeps rms_single_row's 4 values/lane and 672-thread tree.
/// The strided reads also handle a target-hidden slice without a staging copy.
private enum NemotronMTPJoinedNorm {
    static let kernel = MLXFast.metalKernel(
        name: "nemotron_mtp_joined_rms_bf16",
        inputNames: ["embedding", "hidden", "ew", "hw", "eeps", "heps"],
        outputNames: ["joined"],
        source: """
            constexpr uint axis_size = 2688;
            constexpr uint n_reads = 4;
            const uint row = threadgroup_position_in_grid.x;
            const uint bank = threadgroup_position_in_grid.y;
            const uint lid = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;
            const device T* x = bank == 0 ? embedding : hidden;
            const device T* w = bank == 0 ? ew : hw;
            const size_t row_stride = bank == 0 ? embedding_strides[1] : hidden_strides[1];
            const size_t feature_stride = bank == 0 ? embedding_strides[2] : hidden_strides[2];
            const size_t weight_stride = bank == 0 ? ew_strides[0] : hw_strides[0];
            const float eps = bank == 0 ? eeps : heps;
            threadgroup float local_inv_mean[1];
            threadgroup float local_sums[32];
            float acc = 0;
            float thread_x[n_reads];
            for (uint i = 0; i < n_reads; i++) {
                thread_x[i] = x[size_t(row) * row_stride + (lid * n_reads + i) * feature_stride];
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
            for (uint i = 0; i < n_reads; i++) {
                joined[(size_t(row) * 2 + bank) * axis_size + lid * n_reads + i] =
                    w[(lid * n_reads + i) * weight_stride]
                    * static_cast<T>(thread_x[i] * local_inv_mean[0]);
            }
            """, ensureRowContiguous: false)

    static func apply(_ embedding: MLXArray, _ hidden: MLXArray,
                      embeddingNorm: RMSNorm, hiddenNorm: RMSNorm) -> MLXArray? {
        guard Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              hidden.ndim == 3, hidden.dim(0) == 1, hidden.dim(1) > 0, hidden.dim(2) == 2688,
              embedding.shape == hidden.shape, hidden.dtype == .bfloat16, embedding.dtype == .bfloat16,
              type(of: embeddingNorm) == RMSNorm.self, type(of: hiddenNorm) == RMSNorm.self,
              embeddingNorm.weight.shape == [2688], hiddenNorm.weight.shape == [2688],
              embeddingNorm.weight.dtype == .bfloat16, hiddenNorm.weight.dtype == .bfloat16
        else { return nil }
        return kernel([embedding, hidden, embeddingNorm.weight, hiddenNorm.weight,
                       NemotronScalarInputs.float(embeddingNorm.eps),
                       NemotronScalarInputs.float(hiddenNorm.eps)],
            template: [("T", DType.bfloat16)],
            grid: (672 * hidden.dim(1), 2, 1), threadGroup: (672, 1, 1),
            outputShapes: [[1, hidden.dim(1), 5376]], outputDTypes: [.bfloat16])[0]
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

    private func joinedNorm(hidden: MLXArray, embedding: MLXArray) -> MLXArray {
        NemotronMTPJoinedNorm.apply(embedding, hidden,
            embeddingNorm: embeddingNorm, hiddenNorm: hiddenNorm)
            ?? concatenated([embeddingNorm(embedding), hiddenNorm(hidden)], axis: -1)
    }

    private func fuse(hidden: MLXArray, embedding: MLXArray) -> MLXArray {
        projection(joinedNorm(hidden: hidden, embedding: embedding))
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
        nextNorm: RMSNorm? = nil
    ) -> (output: MLXArray, keys: MLXArray, values: MLXArray, preparedNorm: MLXArray?) {
        let joined = joinedNorm(hidden: hidden, embedding: embedding)
        let fused = nemotronMTPLinearRows(joined, projection)
        let x = norm(fused)
        let batch = x.dim(0), length = x.dim(1)
        // Preserve native M=1 QMV: published assistant component timing found
        // the banked single-row QKV kernel slower than separate dispatches.
        let projections = length > 1 ? NemotronTargetQKV.apply(
            x, query: mixer.wq, key: mixer.wk, value: mixer.wv, captureMTP: true) : nil
        let queries = (projections?.0 ?? nemotronMTPLinearRows(x, mixer.wq))
            .reshaped(batch, length, mixer.numHeads, mixer.headDim).transposed(0, 2, 1, 3)
        let keys = (projections?.1 ?? nemotronMTPLinearRows(x, mixer.wk))
            .reshaped(batch, length, mixer.numKeyValueHeads, mixer.headDim).transposed(0, 2, 1, 3)
        let values = (projections?.2 ?? nemotronMTPLinearRows(x, mixer.wv))
            .reshaped(batch, length, mixer.numKeyValueHeads, mixer.headDim).transposed(0, 2, 1, 3)
        let attended = cache.updateAndAttend(
            queries: queries, keys: keys, values: values, scale: mixer.scale, sinks: nil)
        let merged = attended.transposed(0, 2, 1, 3).reshaped(batch, length, -1)
        let output = nemotronMTPLinearRows(merged, mixer.wo)
        if let nextNorm, let pair = NemotronHResidualRMSNorm.apply(fused, output, norm: nextNorm) {
            return (pair.0, keys, values, pair.1)
        }
        return (fused + output, keys, values, nil)
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

    func callAsFunction(_ x: MLXArray) -> MLXArray { finalNorm(x + mixer(norm(x))) }

    /// Row-batched twin of `callAsFunction` through the verify MoE row path.
    func forwardRows(_ x: MLXArray, preparedNorm: MLXArray? = nil) -> MLXArray {
        let output = mixer.mtpForwardRows(preparedNorm ?? norm(x))
        if let pair = NemotronHResidualRMSNorm.apply(x, output, norm: finalNorm) { return pair.1 }
        return finalNorm(x + output)
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
        // A one-token draft can use the same exact M=1 kernels as the
        // post-verify rows, including compiled MoE and fused epilogues.
        if hidden.ndim == 3, hidden.dim(0) == 1, hidden.dim(1) == 1,
           hidden.dtype == .bfloat16, embedding.dtype == .bfloat16,
           Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu {
            return pagedForwardRows(hidden: hidden, embedding: embedding, cache: cache).output
        }
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
    func pagedForwardRows(
        hidden: MLXArray, embedding: MLXArray, cache: any CBv2AttendingLayerCache
    ) -> (output: MLXArray, keys: MLXArray, values: MLXArray) {
        let expert = layers[1] as! NemotronH35MTPExpertBlock
        let a = (layers[0] as! NemotronH35MTPAttentionBlock).pagedForwardRows(
            hidden: hidden, embedding: embedding, cache: cache, nextNorm: expert.norm)
        return (expert.forwardRows(a.output, preparedNorm: a.preparedNorm), a.keys, a.values)
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
