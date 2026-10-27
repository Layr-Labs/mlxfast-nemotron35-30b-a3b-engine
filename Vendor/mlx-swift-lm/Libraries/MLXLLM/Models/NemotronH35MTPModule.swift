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
    static let oneLaunchRouter = enabled("NEMOTRON_MTP_DRAFT_ROUTER_FUSED")
    static let sharedEarly = enabled("NEMOTRON_MTP_DRAFT_SD_EARLY")
}

/// Draft encode order. In each draft step the head's routed fc2 (zero-skip slot kernel) and its shared-expert
/// down (R = 1 multi-row QMV) do not depend on each other. The default graph (record, and t131's shared-early
/// `depends` form) encodes {shared up, router} | {shared down, FC1} | {fc2}. Here the fc2 binds the shared up
/// rows and the shared down binds the FC1 rows (one extra input each, never read), and the join binds `shared`
/// before the products, so MLX encodes {router, shared up} | {FC1} | {shared down, fc2}: the router leads its
/// interval and the FC1 runs alone. Every kernel body is the default graph's source text, so every output is
/// bitwise the same; `qualify` checks it on the head's real weights at load and keeps the default graph on any
/// difference. NEMOTRON_DRAFT_COLAUNCH=0 keeps it too.
enum NemotronDraftColaunch {
    static let requested = ProcessInfo.processInfo.environment["NEMOTRON_DRAFT_COLAUNCH"] != "0"
    nonisolated(unsafe) private(set) static var active = false

    /// Load time, after `NemotronZSkip.prepareDraft` and before any draft graph is traced: the draft tail in
    /// this order against the default graph, bit for bit, on seeded rows through the head's real weights
    /// (router, FC1, fc2 copy, shared up/down, join), plus the fc2, shared-down and join twins alone on seeded
    /// relu2 rows over a density sweep. Any miss leaves the default graph.
    static func qualify(_ head: Module) {
        active = false
        guard requested, !NemotronMoELeader.selfTest,
              Device.defaultDevice().deviceType == .gpu,
              let block = head.modules().compactMap({ $0 as? NemotronH35MTPExpertBlock }).first,
              NemotronMTPDraftTail.exactTail, NemotronZSkip.draftDepAvailable(block.mixer.switchMLP.fc2),
              let shared = block.mixer.sharedExperts
        else {
            if requested {
                FileHandle.standardError.write(Data("nemotron draft colaunch off: not applicable\n".utf8))
            }
            return
        }
        var same = MLXArray(true)
        var cases = 0
        // Whole tail through the real weights.
        for seed in 0 ..< 6 {
            let key = MLXRandom.key(UInt64(0xdc01 + seed))
            let x = (MLXRandom.normal([1, 1, 2688], key: key) * Float(seed < 3 ? 1 : 4)).asType(.bfloat16)
            let a = block.exactOneTokenTail(x, normalized: nil, ordered: false)
            let b = block.exactOneTokenTail(x, normalized: nil, ordered: true)
            same = logicalAnd(same, (a.view(dtype: .uint16) .== b.view(dtype: .uint16)).all())
            cases += 1
        }
        // The fc2, shared-down and join twins alone, seeded relu2-like rows, densities 0 ... 100%.
        let fc2 = block.mixer.switchMLP.fc2
        for (i, density) in [Float(0), 0.1, 0.47, 0.83, 1].enumerated() {
            let keys = MLXRandom.split(key: MLXRandom.key(UInt64(0xdc11 + i)), into: 5)
            func relu2Rows(_ shape: [Int], _ k: MLXArray) -> MLXArray {
                let v = MLXRandom.normal(shape, key: k) * Float(0.25)
                let keep = MLXRandom.uniform(Float(0) ..< 1, shape, key: MLXRandom.split(key: k, into: 2)[1])
                    .< MLXArray(density)
                return which(keep, v * v, MLXArray(Float(i % 2 == 0 ? 0 : -0.0))).asType(.bfloat16)
            }
            let act = relu2Rows([6, 1856], keys[0])
            let sx = relu2Rows([1, 3712], keys[1])
            var routes: [UInt32] = [5, 33, 64, 90, 101, 127]
            for j in routes.indices { routes[j] = (routes[j] + UInt32(7 * i)) % 128 }
            let inds = MLXArray(routes)
            let scores = MLXRandom.uniform(Float(0.01) ..< 0.5, [6], key: keys[2])
            guard let p0 = NemotronZSkip.draftProducts(act, inds, scores, owner: fc2),
                  let p1 = NemotronZSkip.draftProducts(act, inds, scores, owner: fc2, dep: sx),
                  let q = shared.downProj as? QuantizedLinear,
                  let s0 = NemotronMultiRowQMV.apply(sx, q),
                  let s1 = shared.draftDown(sx, dep: act)
            else {
                FileHandle.standardError.write(Data("nemotron draft colaunch off: a twin is unavailable\n".utf8))
                return
            }
            let lhs = MLXRandom.normal([1, 1, 2688], key: keys[3]).asType(.bfloat16)
            let s0r = s0.reshaped(1, 1, 2688)
            guard let j0 = NemotronHResidualRMSNorm.applySum(lhs, p0, shared: s0r, norm: block.finalNorm),
                  let j1 = NemotronHResidualRMSNorm.applySum(lhs, p0, shared: s0r, norm: block.finalNorm,
                                                              sharedFirst: true)
            else { return }
            same = logicalAnd(same, logicalAnd(
                (p0.view(dtype: .uint32) .== p1.view(dtype: .uint32)).all(),
                (s0r.view(dtype: .uint16) .== s1.view(dtype: .uint16)).all()))
            same = logicalAnd(same, logicalAnd(
                (j0.0.view(dtype: .uint16) .== j1.0.view(dtype: .uint16)).all(),
                (j0.1.view(dtype: .uint16) .== j1.1.view(dtype: .uint16)).all()))
            cases += 1
        }
        var ok = false
        do {
            try withError { ok = same.item(Bool.self) }
        } catch {
            FileHandle.standardError.write(Data("nemotron draft colaunch off: \(error)\n".utf8))
            return
        }
        guard ok else {
            FileHandle.standardError.write(Data(
                "nemotron draft colaunch off: the draft tail differs from the default graph\n".utf8))
            return
        }
        active = true
        FileHandle.standardError.write(Data(
            "nemotron draft colaunch on: \(cases) cases bit-identical\n".utf8))
    }
}

enum NemotronMTPFrontNorms {
    static let enabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_DRAFT_FRONT"] != "0"
        && ProcessInfo.processInfo.environment["NEMOTRON_MTP_DRAFT_FUSED"] != "0"
    private static let source = """
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
            """
    private static let kernel = MLXFast.metalKernel(
        name: NemotronHResidualRMSNorm.oneBarrier
            ? "nemotron_mtp_front_norms_concat_bf16_1b" : "nemotron_mtp_front_norms_concat_bf16",
        inputNames: ["e", "h", "ew", "hw", "axis_size", "e_eps", "h_eps"],
        outputNames: ["out"],
        source: NemotronHResidualRMSNorm.oneBarrierSource(source), ensureRowContiguous: true)

    /// The same launch also emitting eh_proj's prepared K=5376 activation
    /// (the lane-coalesced FP32 buffer `nemotron_head_prepare_activation`
    /// builds, PV 8: per 8-value chunk the nmr_load_vector values and sum,
    /// from the BF16-rounded outputs). A chunk never straddles the two
    /// halves, so each threadgroup prepares its own 336 chunks.
    private static func preparedSource() -> String {
        let scratch = "threadgroup float local_sums[32];"
        // The literal's indentation is stripped to the closing delimiter's.
        let store = "    out[target + i] = weight[lid * n_reads + i]\n"
            + "        * static_cast<T>(thread_x[i] * local_inv_mean[0]);\n}"
        precondition(source.components(separatedBy: scratch).count == 2)
        precondition(source.components(separatedBy: store).count == 2)
        return source.replacingOccurrences(of: scratch, with: scratch + "\nthreadgroup T rounded[2688];")
            .replacingOccurrences(of: store, with: """
                T value = weight[lid * n_reads + i]
                    * static_cast<T>(thread_x[i] * local_inv_mean[0]);
                out[target + i] = value;
                rounded[lid * n_reads + i] = value;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lid < 336) {
                const int chunk = (second ? 336 : 0) + int(lid);
                float values[8];
                float total = 0;
                for (int i = 0; i < 8; i += 4) {
                    total += float(rounded[lid * 8 + i]) + float(rounded[lid * 8 + i + 1])
                        + float(rounded[lid * 8 + i + 2]) + float(rounded[lid * 8 + i + 3]);
                    values[i] = rounded[lid * 8 + i];
                    values[i + 1] = rounded[lid * 8 + i + 1] / 16.0f;
                    values[i + 2] = rounded[lid * 8 + i + 2] / 256.0f;
                    values[i + 3] = rounded[lid * 8 + i + 3] / 4096.0f;
                }
                if (NG) total += 1.0f;
                const int base = (int(row) * 21 + chunk / 32) * 9 * 32 + chunk % 32;
                for (int i = 0; i < 8; i++) prepared[base + i * 32] = values[i];
                prepared[base + 8 * 32] = total;
            }
""")
    }
    private static let preparedKernel = MLXFast.metalKernel(
        name: NemotronHResidualRMSNorm.oneBarrier
            ? "nemotron_mtp_front_norms_concat_prepared_bf16_1b" : "nemotron_mtp_front_norms_concat_prepared_bf16",
        inputNames: ["e", "h", "ew", "hw", "axis_size", "e_eps", "h_eps"],
        outputNames: ["out", "prepared"],
        source: NemotronHResidualRMSNorm.oneBarrierSource(preparedSource()), ensureRowContiguous: true)

    /// Post-verify rows (R = 3): (joined norms, eh_proj's prepared input).
    static func applyPrepared(embedding: MLXArray, hidden: MLXArray, enorm: RMSNorm, hnorm: RMSNorm)
        -> (MLXArray, MLXArray)?
    {
        guard NemotronMTPDraftJoins.enabled, embedding.ndim == 3, embedding.dim(1) == 3,
              NemotronMultiRowQMV.enabled, NemotronMultiRowQMV.narrowEnabled,
              let reference = eligible(embedding: embedding, hidden: hidden, enorm: enorm, hnorm: hnorm)
        else { return nil }
        let rows = embedding.dim(1)
        let (a, e, h) = scalars(enorm.eps, hnorm.eps)
        let outputs = preparedKernel([embedding, hidden, enorm.weight, hnorm.weight, a, e, h],
            template: [("T", DType.bfloat16), ("NG", NemotronMTPDraftJoins.negativeFront)],
            grid: (672 * 2 * rows, 1, 1), threadGroup: (672, 1, 1),
            outputShapes: [[1, rows, 5376], [rows, 21, 9 * 32]], outputDTypes: [.bfloat16, .float32])
        guard NemotronMoELeader.selfTest else { return (outputs[0], outputs[1]) }
        let expected = reference()
        let packed = NemotronHeadPreparedActivation.pack(expected.reshaped(rows, 5376))
        let ok = MLX.logicalAnd(
            (outputs[0].view(dtype: .uint16) .== expected.view(dtype: .uint16)).all(),
            (outputs[1].view(dtype: .uint32) .== packed.view(dtype: .uint32)).all())
        return (which(ok, outputs[0], MLXArray(Float.nan).asType(.bfloat16)),
                which(ok, outputs[1], MLXArray(Float.nan)))
    }

    private static func eligible(embedding: MLXArray, hidden: MLXArray, enorm: RMSNorm, hnorm: RMSNorm)
        -> (() -> MLXArray)?
    {
        guard enabled, Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              embedding.ndim == 3, embedding.dim(0) == 1, (1...512).contains(embedding.dim(1)),
              embedding.dim(2) == 2688, hidden.shape == embedding.shape,
              embedding.dtype == .bfloat16, hidden.dtype == .bfloat16,
              type(of: enorm) == RMSNorm.self, type(of: hnorm) == RMSNorm.self,
              enorm.weight.shape == [2688], hnorm.weight.shape == [2688],
              enorm.weight.dtype == .bfloat16, hnorm.weight.dtype == .bfloat16
        else { return nil }
        return { concatenated([enorm(embedding), hnorm(hidden)], axis: -1) }
    }

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
        nemotronR1Linear(joinedNorms(hidden: hidden, embedding: embedding), projection)
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
        // Draft-only joins at R = 3: the front norms also emit eh_proj's
        // prepared input, and the attention norm wq's (NEMOTRON_MTP_DRAFT_JOINS).
        let front = NemotronMTPFrontNorms.applyPrepared(
            embedding: embedding, hidden: hidden, enorm: embeddingNorm, hnorm: hiddenNorm)
        let joined = front?.0 ?? joinedNorms(hidden: hidden, embedding: embedding)
        let fused = nemotronMTPLinearRows(joined, projection, prepared: front?.1)
        var (x, xPrepared): (MLXArray, MLXArray?) = (norm(fused), nil)
        if NemotronMTPDraftJoins.enabled,
           let pair = NemotronHResidualRMSNorm.normPreparedRows(fused, norm: norm) {
            (x, xPrepared) = pair
            if NemotronMTPDraftJoins.negativeNorm {
                xPrepared = NemotronHeadPreparedActivation.pack(fused.reshaped(3, 2688))
            }
        }
        let batch = x.dim(0), length = x.dim(1)
        // Q, K and V in one launch once qualified (NEMOTRON_MTP_DRAFT_ROWS_QKV).
        let (q, k, v) = NemotronMTPFrontRowsQKV.apply(x, prepared: xPrepared, mixer: mixer)
            ?? NemotronMTPFrontRowsQKV.separate(x, prepared: xPrepared, mixer: mixer)
        let queries = q
            .reshaped(batch, length, mixer.numHeads, mixer.headDim).transposed(0, 2, 1, 3)
        let keys = k
            .reshaped(batch, length, mixer.numKeyValueHeads, mixer.headDim).transposed(0, 2, 1, 3)
        let values = v
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
        return fused + nemotronR1Linear(merged, mixer.wo)
    }

    func maskedRowParts(
        fused: MLXArray, queries: MLXArray, keys: MLXArray, values: MLXArray, mask: MLXArray
    ) -> (MLXArray, MLXArray) {
        let output = MLXFast.scaledDotProductAttention(
            queries: queries.asType(keys.dtype), keys: keys, values: values,
            scale: mixer.scale, mask: .array(mask))
        let merged = output.transposed(0, 2, 1, 3).reshaped(1, 1, -1)
        return (fused, nemotronR1Linear(merged, mixer.wo))
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
        return nemotronR1Linear(output.transposed(0, 2, 1, 3).reshaped(1, 1, -1), mixer.wo)
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

/// The head's routed banks (fc1 bank words with their pair dictionary, the fc2 copy) get their own residency
/// set, requested after the decode extras (`NemotronDecodeResidency.lateSets`). The request's first-round drafts
/// (the draft steps before its first verify) that are built while that request has not returned run the head's
/// block without its routed experts (the shared expert alone), so the first verify does not wait for the banks'
/// mapping; every later draft is the base's. Drafts only propose: the target verifies every token, so only
/// those first-round drafts' acceptance can change. Outside a lease the banks count as ready.
/// NEMOTRON_MTP_FIRST_ROUND_SHARED=0 keeps the routed experts (and the banks in the head and extras sets).
enum NemotronMTPFirstRoundDraft {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_MTP_FIRST_ROUND_SHARED"] != "0"
    private static let lock = NSLock()
    nonisolated(unsafe) private static var pending = false
    /// Set by `NemotronH35MTPAssistant.draftStep` around a first-round head step.
    nonisolated(unsafe) static var firstRound = false
    /// A first-round head step while the lease's late set has not returned: the block skips its routed experts.
    static var active: Bool { enabled && firstRound && lock.withLock { pending } }
    /// The lease's begin, before its late set is enqueued.
    static func begin() { lock.withLock { pending = true } }
    /// The late set's request returned (or the lease finished).
    static func ready() { lock.withLock { pending = false } }
}

final class NemotronH35MTPExpertBlock: Module {
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "mixer") var mixer: NemotronHMoE
    @ModuleInfo(key: "final_layernorm") var finalNorm: RMSNorm
    private let oneTokenTailCache = NemotronH35MTPOneTokenTailCache()

    init(_ args: NemotronHConfiguration) {
        _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
        _mixer.wrappedValue = NemotronHMoE(args)
        _finalNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if NemotronMTPFirstRoundDraft.active { return sharedOnlyTail(x) }
        if NemotronMTPExecution.compiledMoE,
           x.ndim == 3, x.dim(0) == 1, x.dim(1) == 1 {
            return oneTokenTailCache([x], mode: exactTailAvailable(x) ? 1 : 0, owner: self)
        }
        return exactTailAvailable(x) ? exactOneTokenTail(x, normalized: nil)
            : uncompiledOneTokenTail(x)
    }

    func callAsFunction(residual: MLXArray, attention: MLXArray) -> MLXArray {
        if NemotronMTPFirstRoundDraft.active { return sharedOnlyTail(residual + attention) }
        if exactTailAvailable(residual), attention.shape == residual.shape,
           let joined = NemotronHResidualRMSNorm.apply(residual, attention, norm: norm) {
            if NemotronMTPExecution.compiledMoE {
                return oneTokenTailCache([joined.0, joined.1], mode: 2, owner: self)
            }
            return exactOneTokenTail(joined.0, normalized: joined.1)
        }
        return self(residual + attention)
    }

    fileprivate func uncompiledOneTokenTail(_ x: MLXArray) -> MLXArray {
        finalNorm(x + mixer(norm(x)))
    }

    /// The block without its routed experts, for the request's first-round drafts
    /// (`NemotronMTPFirstRoundDraft`): the shared expert alone.
    private func sharedOnlyTail(_ x: MLXArray) -> MLXArray {
        guard let shared = mixer.sharedExperts else { return finalNorm(x) }
        return finalNorm(x + shared(norm(x)))
    }

    private func exactTailAvailable(_ x: MLXArray) -> Bool {
        NemotronMTPDraftTail.exactTail && NemotronMoELeader.enabled
            && x.shape == [1, 1, 2688] && x.dtype == .bfloat16
            && type(of: norm) == RMSNorm.self && type(of: finalNorm) == RMSNorm.self
            && type(of: mixer) == NemotronHMoE.self && mixer.sharedExperts != nil
            && mixer.numExpertsPerTok == 6
            && Device.defaultDevice().deviceType == .gpu && StreamOrDevice.default == .gpu
    }

    /// `ordered`: the `NemotronDraftColaunch` graph (nil: when qualified; false: the t71 graph).
    func exactOneTokenTail(_ x: MLXArray, normalized: MLXArray?, ordered: Bool? = nil) -> MLXArray {
        let h = normalized ?? norm(x)
        let gate = mixer.gate
        let fused = NemotronMTPDraftTail.oneLaunchRouter && gate.topK == 6 && gate.nGroup == 1
            && gate.topkGroup == 1 && gate.normTopkProb && gate.routedScalingFactor == 2.5
            ? NemotronMTPRouterRowsSelect.run(h, gate.weight, bias: gate.eSCB, draft: true) : nil
        let (indices, scores) = fused ?? gate(h)
        // `NemotronDraftColaunch`: the same kernels and arithmetic; only the eval tape's encode order changes.
        if ordered ?? NemotronDraftColaunch.active, let sharedExperts = mixer.sharedExperts,
           NemotronZSkip.draftDepAvailable(mixer.switchMLP.fc2),
           let sx = sharedExperts.draftUpActivation(h) {
            var fc1: MLXArray? = nil
            if let routed = NemotronMoELeader.switchRowsAndWeightedReduce(
                   h, indices, scores, mixer.switchMLP, products: true,
                   draftDep: sx, activation: { fc1 = $0 }),
               routed.shape == [6 * 2688], let fc1,
               let shared = sharedExperts.draftDown(sx, dep: fc1),
               let joined = NemotronHResidualRMSNorm.applySum(
                   x, routed, shared: shared, norm: finalNorm, sharedFirst: true) {
                return joined.1
            }
        }
        // The shared rows are a dependency of the routed slot products (NEMOTRON_MTP_DRAFT_SD_EARLY=0: not).
        let early = NemotronMTPDraftTail.sharedEarly ? mixer.sharedExperts.map { $0(h) } : nil
        guard let sharedExperts = mixer.sharedExperts,
              let routed = NemotronMoELeader.switchRowsAndWeightedReduce(
                h, indices, scores, mixer.switchMLP, products: true, after: early)
        else { return finalNorm(x + mixer(h)) }
        let shared = early ?? sharedExperts(h)
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
        let identity = nativeIdentity ?? (owner.modules().map(ObjectIdentifier.init)
            + (parameters ?? []).map(ObjectIdentifier.init))
        var retired: Entry?
        let current = lock.withLock { () -> Entry in
            if let entry = entries[mode], entry.identity == identity { return entry }
            let state = parameters ?? owner.innerState()
            let function: @Sendable ([MLXArray]) -> [MLXArray] = compile(inputs: [state]) {
                [weak owner] arrays in
                guard let owner else { preconditionFailure("Released Nemotron MTP expert-tail owner") }
                switch mode {
                case 0: return [owner.uncompiledOneTokenTail(arrays[0])]
                case 1: return [owner.exactOneTokenTail(arrays[0], normalized: nil)]
                default: return [owner.exactOneTokenTail(arrays[0], normalized: arrays[1])]
                }
            }
            let replacement = Entry(identity: identity, function: function)
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

    private static func nativeIdentity(_ owner: NemotronH35MTPExpertBlock) -> [ObjectIdentifier]? {
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

    static let deviceDraftJoin =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_DEVICE_DRAFT_JOIN"] != "0"

    func finishMaskedRow(
        fused: MLXArray, queries: MLXArray, keys: MLXArray, values: MLXArray, mask: MLXArray
    ) -> MLXArray {
        let block = layers[0] as! NemotronH35MTPAttentionBlock
        guard Self.deviceDraftJoin else {
            return finishPagedAttentionRow(block.maskedRowOutput(
                fused: fused, queries: queries, keys: keys, values: values, mask: mask))
        }
        let (residual, attention) = block.maskedRowParts(
            fused: fused, queries: queries, keys: keys, values: values, mask: mask)
        precondition(residual.ndim == 3 && residual.dim(0) == 1 && residual.dim(1) == 1)
        return (layers[1] as! NemotronH35MTPExpertBlock)(residual: residual, attention: attention)
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

    public func cbv2MTPTopTwoPacket(_ logits: MLXArray, window: MLXArray)
        -> (ids: MLXArray, values: MLXArray, packet: MLXArray)?
    {
        cbv2TopTwoRowsPacket(logits, window: window)
    }
}

extension NemotronH35Model: CBv2RecurrentPrefillHiddenForwardable {
    public func cbv2ForwardWithHiddenForPrefill(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        precondition(positionIds == nil)
        NemotronHPrefillX5.requestPrefill = true
        defer { NemotronHPrefillX5.requestPrefill = false }
        let hidden = cbv2Hidden(tokens, caches: caches, recurrentState: recurrentState)
        let last = hidden[0..., (hidden.dim(1) - 1)..., 0...]
        switch requirement {
        case .evaluationOnly: return (last[0..., 0..., 0..<1], hidden)
        case .lastPositionLogits: return (logits(last), hidden)
        }
    }
}

/// Draft-only prepared operands on the post-verify front (`frontRows`).
/// NEMOTRON_MTP_DRAFT_JOINS=0 keeps the separate prepare launches; =negb /
/// =negc are negative controls (each perturbs only its own piece's draft
/// values, so acceptance must change while tokens stay the target's).
enum NemotronMTPDraftJoins {
    private static let setting = ProcessInfo.processInfo.environment["NEMOTRON_MTP_DRAFT_JOINS"]
    static let enabled = setting != "0"
    static let negativeFront = setting == "negb"
    static let negativeNorm = setting == "negc"
}

/// The post-verify front's Q/K/V (`frontRows`, R = 3 or 4) in one launch: the
/// verify's fused Q/K/V (`NemotronTargetQKV.apply` on captured rows, fed the
/// attention norm's prepared rows) on the head's wq, wk and wv instead of the
/// prepared wq projection and the wk and wv projections. Each Q, K and V tile
/// keeps its projection's row body; used only after `qualify` found q, k and v
/// bitwise equal to the three launches on the head's weights.
/// NEMOTRON_MTP_DRAFT_ROWS_QKV=0 (or NEMOTRON_MTP_DRAFT_FUSED=0) keeps them.
enum NemotronMTPFrontRowsQKV {
    static let enabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_DRAFT_ROWS_QKV"] != "0"
        && ProcessInfo.processInfo.environment["NEMOTRON_MTP_DRAFT_FUSED"] != "0"
    nonisolated(unsafe) static var verified = false

    static func apply(_ x: MLXArray, prepared: MLXArray?, mixer: NemotronHAttention)
        -> (MLXArray, MLXArray, MLXArray)?
    {
        verified ? launch(x, prepared: prepared, mixer: mixer) : nil
    }

    private static func launch(_ x: MLXArray, prepared: MLXArray?, mixer: NemotronHAttention)
        -> (MLXArray, MLXArray, MLXArray)?
    {
        guard enabled, mixer.numHeads == 32, mixer.numKeyValueHeads == 2, mixer.headDim == 128,
              x.ndim == 3, x.dim(0) == 1, [3, 4].contains(x.dim(1))
        else { return nil }
        return NemotronTargetQKV.apply(
            x, query: mixer.wq, key: mixer.wk, value: mixer.wv, captureMTP: true, prepared: prepared)
    }

    /// `frontRows`' three projections without the fused launch.
    static func separate(_ x: MLXArray, prepared: MLXArray?, mixer: NemotronHAttention)
        -> (MLXArray, MLXArray, MLXArray)
    {
        (nemotronMTPLinearRows(x, mixer.wq, prepared: prepared),
         nemotronMTPLinearRows(x, mixer.wk), nemotronMTPLinearRows(x, mixer.wv))
    }

    /// Load time, before the post-verify warm-up: the fused launch against the
    /// three projections, bit for bit in q, k and v, with the head's weights and
    /// attention norm on seeded residual rows: R = 3 with the norm's prepared
    /// rows (as `frontRows` passes them) and without, and R = 4.
    static func qualify(_ block: NemotronH35MTPAttentionBlock) {
        verified = false
        guard enabled else { return }
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron mtp draft rows qkv fused: \(message)\n".utf8))
        }
        var ok = MLXArray(true)
        let cases = [(3, true), (3, true), (3, false), (4, false), (4, false)]
        for (seed, (rows, joined)) in cases.enumerated() {
            let key = MLXRandom.key(UInt64(0x9f40 + seed))
            let fused = (MLXRandom.normal([1, rows, 2688], key: key) * Float(1 + seed))
                .asType(.bfloat16)
            var (x, prepared): (MLXArray, MLXArray?) = (block.norm(fused), nil)
            if joined, NemotronMTPDraftJoins.enabled,
               let pair = NemotronHResidualRMSNorm.normPreparedRows(fused, norm: block.norm) {
                (x, prepared) = pair
            }
            guard let a = launch(x, prepared: prepared, mixer: block.mixer) else {
                return report("off (the head's projections do not fit the fused launch)")
            }
            let b = separate(x, prepared: prepared, mixer: block.mixer)
            for (u, w) in [(a.0, b.0), (a.1, b.1), (a.2, b.2)] {
                ok = MLX.logicalAnd(ok, (u.view(dtype: .uint16) .== w.view(dtype: .uint16)).all())
            }
            eval(ok)
        }
        verified = ok.item(Bool.self)
        report(verified
            ? "on (q, k, v bitwise equal to the three launches in \(cases.count) cases at R = 3, 4)"
            : "off (q, k or v differs from the three launches)")
    }
}
