// Native Nemotron Lightning embedded MTP head: post-norm target hidden +
// next-token embedding -> fusion -> attention -> MoE -> final norm.
// Tensor contract cross-checked against NVIDIA's BF16 checkpoint and
// jundot/omlx patches/mlx_lm_mtp/nemotron_h_model.py (2026-09-08).
import MLX
import MLXLMCommon
import MLXNN

// One dispatch over the original three affine buffers. Each SIMD group keeps
// the native M=1 qmv lane split, K order, tail, reduction and BF16 store.
enum NemotronMTPQKV {
    private static let source = """
      constexpr int results_per_simdgroup = 4;
      constexpr int values_per_thread = 8;
      constexpr int block_size = values_per_thread * 32;
      constexpr int group_size = 64;
      typedef float U;
      const uint tile = threadgroup_position_in_grid.y;
      const uint simd_gid = simdgroup_index_in_threadgroup;
      const uint simd_lid = thread_index_in_simdgroup;
      const uint local_tile = tile < 512 ? tile : (tile < 544 ? tile - 512 : tile - 544);
      const device uint32_t* w = tile < 512 ? qw : (tile < 544 ? kw : vw);
      const device T* scales = tile < 512 ? qs : (tile < 544 ? ks : vs);
      const device T* biases = tile < 512 ? qb : (tile < 544 ? kb : vb);
      device T* y = tile < 512 ? qy : (tile < 544 ? ky : vy);
      const int in_vec_size = K;
      const int in_vec_size_w = in_vec_size / 2;
      const int in_vec_size_g = in_vec_size / group_size;
      const int out_row = local_tile * 8 + simd_gid * results_per_simdgroup;
      const device uint8_t* ws = (const device uint8_t*)w;
      ws += out_row * in_vec_size_w + simd_lid * 4;
      scales += out_row * in_vec_size_g + simd_lid / 8;
      biases += out_row * in_vec_size_g + simd_lid / 8;
      const device T* xp = x + simd_lid * values_per_thread;
      y += out_row;
      thread U x_thread[values_per_thread];
      thread U result[results_per_simdgroup] = {0};
      int k = 0;
      for (; k < in_vec_size - block_size; k += block_size) {
        U sum = nmr_load_vector<T, U, values_per_thread>(xp, x_thread);
        for (int row = 0; row < results_per_simdgroup; row++) {
          auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
          const device T* sl = scales + row * in_vec_size_g;
          const device T* bl = biases + row * in_vec_size_g;
          U s = sl[0];
          U b = bl[0];
          result[row] += nmr_qdot<U, values_per_thread>(wl, x_thread, s, b, sum);
        }
        ws += block_size / 2;
        scales += block_size / group_size;
        biases += block_size / group_size;
        xp += block_size;
      }
      const int remaining = clamp(
          static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
          0, values_per_thread);
      if (remaining > 0) {
        U sum = nmr_load_vector_safe<T, U, values_per_thread>(xp, x_thread, remaining);
        for (int row = 0; row < results_per_simdgroup; row++) {
          auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
          const device T* sl = scales + row * in_vec_size_g;
          const device T* bl = biases + row * in_vec_size_g;
          U s = sl[0];
          U b = bl[0];
          result[row] += nmr_qdot_safe<U, values_per_thread>(
              wl, x_thread, s, b, sum, remaining);
        }
      }
      for (int row = 0; row < results_per_simdgroup; row++) {
        result[row] = simd_sum(result[row]);
        if (simd_lid == 0) {
          y[row] = static_cast<T>(result[row]);
        }
      }
    """
    private static let kernel = MLXFast.metalKernel(
        name: "nemotron_mtp_qkv_affine4_gs64",
        inputNames: ["qw", "qs", "qb", "kw", "ks", "kb", "vw", "vs", "vb", "x", "K"],
        outputNames: ["qy", "ky", "vy"], source: source,
        header: nemotronMultiRowQMVHeader)

    private static func buffers(_ linear: Linear, rows: Int) -> [MLXArray]? {
        guard let q = linear as? QuantizedLinear,
            q.mode == .affine, q.bits == 4, q.groupSize == 64, q.bias == nil,
            q.weight.dtype == .uint32, q.weight.shape == [rows, 336],
            q.scales.dtype == .bfloat16, q.scales.shape == [rows, 42],
            let offsets = q.biases, offsets.dtype == .bfloat16,
            offsets.shape == [rows, 42]
        else { return nil }
        return [q.weight, q.scales, offsets]
    }

    static func apply(
        _ x: MLXArray, query: Linear, key: Linear, value: Linear
    ) -> (MLXArray, MLXArray, MLXArray)? {
        guard Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
            x.shape == [1, 1, 2688], x.dtype == .bfloat16,
            let q = buffers(query, rows: 4096), let k = buffers(key, rows: 256),
            let v = buffers(value, rows: 256)
        else { return nil }
        let kArray = NemotronMultiRowQMV.scalarPair(2688, 4608).0
        let output = kernel(q + k + v + [x, kArray], template: [("T", x.dtype)],
            grid: (32, 1152, 1), threadGroup: (32, 2, 1),
            outputShapes: [[1, 1, 4096], [1, 1, 256], [1, 1, 256]],
            outputDTypes: [x.dtype, x.dtype, x.dtype])
        return (output[0], output[1], output[2])
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
        if let joined = NemotronMTPInputNorm.apply(
            embedding: embedding, hidden: hidden,
            embeddingNorm: embeddingNorm, hiddenNorm: hiddenNorm) {
            return projection(joined)
        }
        return projection(concatenated([embeddingNorm(embedding), hiddenNorm(hidden)], axis: -1))
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
        let x = norm(fused)
        if mixer.numHeads == 32, mixer.numKeyValueHeads == 2, mixer.headDim == 128,
            let (q, k, v) = NemotronMTPQKV.apply(
                x, query: mixer.wq, key: mixer.wk, value: mixer.wv) {
            let queries = q.reshaped(1, 1, 32, 128).transposed(0, 2, 1, 3)
            let keys = k.reshaped(1, 1, 2, 128).transposed(0, 2, 1, 3)
            let values = v.reshaped(1, 1, 2, 128).transposed(0, 2, 1, 3)
            let output = cache.updateAndAttend(
                queries: queries, keys: keys, values: values, scale: mixer.scale, sinks: nil)
            let merged = output.transposed(0, 2, 1, 3).reshaped(1, 1, -1)
            return fused + mixer.wo(merged)
        }
        return fused + mixer.cbv2Forward(x, cache: cache)
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

    func appendTrustedKVRows(
        hidden: MLXArray, embedding: MLXArray, cache: PagedLayerCache
    ) -> [MLXArray] {
        let kv = prepareTrustedKVRows(hidden: hidden, embedding: embedding)
        return cache.appendTrusted(keys: kv.keys, values: kv.values)
    }

    func prepareTrustedKVRows(
        hidden: MLXArray, embedding: MLXArray
    ) -> (keys: MLXArray, values: MLXArray) {
        let e = nemotronMTPNormRows(embedding) { embeddingNorm($0) }
        let h = nemotronMTPNormRows(hidden) { hiddenNorm($0) }
        let fused = nemotronMTPLinearRows(concatenated([e, h], axis: -1), projection)
        let x = nemotronMTPNormRows(fused) { norm($0) }
        let batch = x.dim(0), length = x.dim(1)
        let keys = nemotronMTPLinearRows(x, mixer.wk)
            .reshaped(batch, length, mixer.numKeyValueHeads, mixer.headDim)
            .transposed(0, 2, 1, 3)
        let values = nemotronMTPLinearRows(x, mixer.wv)
            .reshaped(batch, length, mixer.numKeyValueHeads, mixer.headDim)
            .transposed(0, 2, 1, 3)
        return (keys, values)
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

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normalized = norm(x)
        // The draft head keeps R1. Reuse the exact verify graph only for the
        // native child modules; custom module overrides use the ordinary call.
        let shared = mixer.sharedExperts
        let nativeShared = shared.map {
            type(of: $0) == NemotronHMLP.self
                && type(of: $0.upProj) == QuantizedLinear.self
                && type(of: $0.downProj) == QuantizedLinear.self
        } ?? true
        let useRows = NemotronMTPExecution.batchedM1
            && x.shape == [1, 1, 2688] && x.dtype == .bfloat16
            && normalized.shape == x.shape && normalized.dtype == .bfloat16
            && type(of: mixer) == NemotronHMoE.self
            && type(of: mixer.gate) == NemotronHMoEGate.self
            && type(of: mixer.switchMLP) == NemotronHSwitchMLP.self
            && type(of: mixer.switchMLP.fc1) == QuantizedSwitchLinear.self
            && type(of: mixer.switchMLP.fc2) == QuantizedSwitchLinear.self
            && nativeShared
        let experts = useRows ? mixer.mtpForwardRows(normalized) : mixer(normalized)
        return finalNorm(x + experts)
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
    func appendTrustedKVRows(
        hidden: MLXArray, embedding: MLXArray, cache: PagedLayerCache
    ) -> [MLXArray] {
        (layers[0] as! NemotronH35MTPAttentionBlock).appendTrustedKVRows(
            hidden: hidden, embedding: embedding, cache: cache)
    }
    func prepareTrustedKVRows(
        hidden: MLXArray, embedding: MLXArray
    ) -> (keys: MLXArray, values: MLXArray) {
        (layers[0] as! NemotronH35MTPAttentionBlock).prepareTrustedKVRows(
            hidden: hidden, embedding: embedding)
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

// Two independent native RMS reductions write the E/H projection input directly.
// Keep the BF16 normalization round before multiplying each learned weight.
enum NemotronMTPInputNorm {
    private static let kernel = MLXFast.metalKernel(
        name: "nemotron_mtp_input_norm_join_bf16",
        inputNames: ["embedding", "hidden", "embedding_weight", "hidden_weight", "eps"],
        outputNames: ["out"],
        source: """
            constexpr uint axis_size = 2688;
            constexpr uint n_reads = 4;
            const uint gid = threadgroup_position_in_grid.x;
            const uint lid = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;
            const device T* x = gid == 0 ? embedding : hidden;
            const device T* w = gid == 0 ? embedding_weight : hidden_weight;
            x += lid * n_reads;
            w += lid * n_reads;
            threadgroup float local_inv_mean[1];
            threadgroup float local_sums[32];
            float acc = 0;
            float thread_x[n_reads];
            for (uint i = 0; i < n_reads; i++) {
                thread_x[i] = x[i];
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
                    local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps[gid]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            device T* y = out + gid * axis_size + lid * n_reads;
            for (uint i = 0; i < n_reads; i++) {
                y[i] = w[i] * static_cast<T>(thread_x[i] * local_inv_mean[0]);
            }
            """, ensureRowContiguous: true)

    static func apply(
        embedding: MLXArray, hidden: MLXArray,
        embeddingNorm: RMSNorm, hiddenNorm: RMSNorm
    ) -> MLXArray? {
        guard Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
            embedding.shape == [1, 1, 2688], hidden.shape == embedding.shape,
            embedding.dtype == .bfloat16, hidden.dtype == .bfloat16,
            embeddingNorm.weight.shape == [2688], hiddenNorm.weight.shape == [2688],
            embeddingNorm.weight.dtype == .bfloat16, hiddenNorm.weight.dtype == .bfloat16
        else { return nil }
        let eps = MLXArray([embeddingNorm.eps, hiddenNorm.eps])
        return kernel([embedding, hidden, embeddingNorm.weight, hiddenNorm.weight, eps],
            template: [("T", embedding.dtype)], grid: (1344, 1, 1), threadGroup: (672, 1, 1),
            outputShapes: [[1, 1, 5376]], outputDTypes: [embedding.dtype])[0]
    }
}
