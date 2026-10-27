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

    /// NEMOTRON_MTP_FRONT_SHUFFLE=0 keeps the threadgroup copy and barrier ahead of the prepared epilogue.
    /// Otherwise chunk c of a half (values 8c ..< 8c + 8) is lanes 2c and 2c + 1 of one SIMD group: each lane
    /// keeps its four rounded outputs, reads its partner's through simd_shuffle_xor, and the pair writes the
    /// chunk's nine words (the even lane its four and the total, the odd lane its four) with the same
    /// expressions and order. Used only after `qualifyShuffle` found it bitwise equal to the copy form.
    static let shuffle = ProcessInfo.processInfo.environment["NEMOTRON_MTP_FRONT_SHUFFLE"] != "0"
    nonisolated(unsafe) private static var shuffleVerified = false
    private static func preparedShuffleSource() -> String {
        let decl = "float thread_x[n_reads];"
        let store = "    out[target + i] = weight[lid * n_reads + i]\n"
            + "        * static_cast<T>(thread_x[i] * local_inv_mean[0]);\n}"
        precondition(source.components(separatedBy: decl).count == 2)
        precondition(source.components(separatedBy: store).count == 2)
        return source.replacingOccurrences(of: decl, with: decl + "\nT nmf_r[n_reads];")
            .replacingOccurrences(of: store, with: """
                T value = weight[lid * n_reads + i]
                    * static_cast<T>(thread_x[i] * local_inv_mean[0]);
                out[target + i] = value;
                nmf_r[i] = value;
            }
            {
                const bool odd = (lid & 1u) != 0;
                T other[4];
                for (int i = 0; i < 4; i++) {
                    other[i] = as_type<T>(simd_shuffle_xor(as_type<ushort>(nmf_r[i]), ushort(1)));
                }
                const int chunk = (second ? 336 : 0) + int(lid >> 1);
                const int base = (int(row) * 21 + chunk / 32) * 9 * 32 + chunk % 32;
                if (!odd) {
                    float total = 0;
                    total += float(nmf_r[0]) + float(nmf_r[1]) + float(nmf_r[2]) + float(nmf_r[3]);
                    total += float(other[0]) + float(other[1]) + float(other[2]) + float(other[3]);
                    if (NG) total += 1.0f;
                    prepared[base + 0 * 32] = float(nmf_r[0]);
                    prepared[base + 1 * 32] = float(nmf_r[1]) / 16.0f;
                    prepared[base + 2 * 32] = float(nmf_r[2]) / 256.0f;
                    prepared[base + 3 * 32] = float(nmf_r[3]) / 4096.0f;
                    prepared[base + 8 * 32] = total;
                } else {
                    prepared[base + 4 * 32] = float(nmf_r[0]);
                    prepared[base + 5 * 32] = float(nmf_r[1]) / 16.0f;
                    prepared[base + 6 * 32] = float(nmf_r[2]) / 256.0f;
                    prepared[base + 7 * 32] = float(nmf_r[3]) / 4096.0f;
                }
            }
""")
    }
    private static let shuffleKernel = MLXFast.metalKernel(
        name: NemotronHResidualRMSNorm.oneBarrier
            ? "nemotron_mtp_front_norms_concat_prepared_bf16_sh_1b" : "nemotron_mtp_front_norms_concat_prepared_bf16_sh",
        inputNames: ["e", "h", "ew", "hw", "axis_size", "e_eps", "h_eps"],
        outputNames: ["out", "prepared"],
        source: NemotronHResidualRMSNorm.oneBarrierSource(preparedShuffleSource()), ensureRowContiguous: true)

    /// Load time: the shuffle-form launch against the copy-form launch, bit for bit in the joined rows and in
    /// eh_proj's prepared activation, on seeded rows with the head's own norms at R = 1 ... 4. Admits the
    /// shuffle form only if every case matches.
    static func qualifyShuffle(_ block: NemotronH35MTPAttentionBlock) {
        shuffleVerified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron mtp front norms shuffle: \(message)\n".utf8))
        }
        guard shuffle else { return report("off (setting)") }
        var ok = MLXArray(true)
        var cases = 0
        do {
            try withError {
                for rows in 1 ... 4 {
                    for seed in 0 ..< 2 {
                        let keys = MLXRandom.split(key: MLXRandom.key(UInt64(0x7f50 + 8 * rows + seed)), into: 2)
                        let embedding = (MLXRandom.normal([1, rows, 2688], key: keys[0]) * Float(1 + 3 * seed))
                            .asType(.bfloat16)
                        let hidden = (MLXRandom.normal([1, rows, 2688], key: keys[1]) * Float(4 - 3 * seed))
                            .asType(.bfloat16)
                        guard let a = launch(embedding: embedding, hidden: hidden, enorm: block.embeddingNorm,
                                             hnorm: block.hiddenNorm, shuffled: false),
                              let b = launch(embedding: embedding, hidden: hidden, enorm: block.embeddingNorm,
                                             hnorm: block.hiddenNorm, shuffled: true)
                        else { ok = MLXArray(false); return }
                        ok = ok .&& (a.0.view(dtype: .uint16) .== b.0.view(dtype: .uint16)).all()
                            .&& (a.1.view(dtype: .uint32) .== b.1.view(dtype: .uint32)).all()
                        cases += 1
                    }
                }
                eval(ok)
            }
        } catch {
            return report("off (\(error))")
        }
        shuffleVerified = ok.item(Bool.self)
        report(shuffleVerified ? "on (\(cases) cases bitwise equal to the copy form at R = 1 ... 4)"
               : "off (differs from the copy form)")
    }

    /// The prepared front-norms launch in the copy or the shuffle form.
    private static func launch(embedding: MLXArray, hidden: MLXArray, enorm: RMSNorm, hnorm: RMSNorm,
                               shuffled: Bool) -> (MLXArray, MLXArray)? {
        guard eligible(embedding: embedding, hidden: hidden, enorm: enorm, hnorm: hnorm) != nil else { return nil }
        let rows = embedding.dim(1)
        let (a, e, h) = scalars(enorm.eps, hnorm.eps)
        let outputs = (shuffled ? shuffleKernel : preparedKernel)(
            [embedding, hidden, enorm.weight, hnorm.weight, a, e, h],
            template: [("T", DType.bfloat16), ("NG", NemotronMTPDraftJoins.negativeFront)],
            grid: (672 * 2 * rows, 1, 1), threadGroup: (672, 1, 1),
            outputShapes: [[1, rows, 5376], [rows, 21, 9 * 32]], outputDTypes: [.bfloat16, .float32])
        return (outputs[0], outputs[1])
    }

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
        let outputs = (shuffleVerified ? shuffleKernel : preparedKernel)(
            [embedding, hidden, enorm.weight, hnorm.weight, a, e, h],
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

    /// NEMOTRON_MTP_FRONT_EMBED_LANE16=0 keeps the embedding-rows launch (`NemotronHEmbeddingRows`) ahead of the front
    /// norms. Otherwise the embedding half's threadgroup dequantizes its token's row itself, with the embedding
    /// kernel's BF16 arithmetic; this twin uses one coefficient loader per 16-lane half-SIMD and uchar2 codes.
    /// It keeps the same id wrap, coefficient group, byte/nibble identity and `T value = scale * d + bias`,
    /// the BF16 value that kernel stores, and takes `thread_x[i]` from that value as the record takes it from `e`.
    /// The hidden half and everything after the reads are the record's text. Plain, prepared and shuffle forms;
    /// used only after `qualifyEmbed` found each bitwise equal to the embedding-rows launch plus the front norms.
    static let embed = ProcessInfo.processInfo.environment["NEMOTRON_MTP_FRONT_EMBED_LANE16"] != "0"
    nonisolated(unsafe) private static var embedVerified = false

    private static func embedSource(_ s: String) -> String {
        let pointer = "const device T* x = second ? h : e;"
        let read = """
            for (uint i = 0; i < n_reads; i++) {
                thread_x[i] = x[offset + i];
                acc += thread_x[i] * thread_x[i];
            }
            """
        precondition(s.components(separatedBy: pointer).count == 2, "front norms embed pointer")
        precondition(s.components(separatedBy: read).count == 2, "front norms embed read")
        return s.replacingOccurrences(of: pointer, with: "const device T* x = h;")
            .replacingOccurrences(of: read, with: """
                if (second) {
                \(read)
                } else {
                    int64_t id = int64_t(ids[row]);
                    if (id < 0) {
                        id += int64_t(V);
                    }
                    // Four dimensions per lane: lanes 0...15 share one affine
                    // coefficient pair, lanes 16...31 the next. Load only at
                    // each half-SIMD leader, then broadcast the exact BF16 bits.
                    // uchar2 preserves byte order without a packed-word endian assumption.
                    const device uchar2* erow2 = (const device uchar2*)
                        ((const device uint8_t*)w + id * int64_t(BYTES));
                    const uchar2 codes = erow2[lid];
                    ushort nmfe_s = 0, nmfe_b = 0;
                    if ((simd_lane_id & 15u) == 0u) {
                        const int64_t group = int64_t(lid / 16u);
                        nmfe_s = as_type<ushort>(scales[id * int64_t(G) + group]);
                        nmfe_b = as_type<ushort>(biases[id * int64_t(G) + group]);
                    }
                    const ushort leader = ushort(simd_lane_id & 16u);
                    const T scale = as_type<T>(simd_shuffle(nmfe_s, leader));
                    const T bias = as_type<T>(simd_shuffle(nmfe_b, leader));
                    for (uint i = 0; i < n_reads; i++) {
                        const uint val = i < 2u ? uint(codes.x) : uint(codes.y);
                        const uint8_t d = (val >> (4u * (i & 1u))) & 0x0f;
                        T value = scale * d + bias;
                        thread_x[i] = value;
                        acc += thread_x[i] * thread_x[i];
                    }
                }
                """)
    }
    private static let embedInputs = ["ids", "w", "scales", "biases", "h", "ew", "hw", "axis_size", "e_eps", "h_eps"]
    private static func embedName(_ base: String) -> String {
        NemotronHResidualRMSNorm.oneBarrier ? base + "_1b" : base
    }
    private static let embedKernel = MLXFast.metalKernel(
        name: embedName("nemotron_mtp_front_norms_embed_lane16_concat_bf16"),
        inputNames: embedInputs, outputNames: ["out"],
        source: NemotronHResidualRMSNorm.oneBarrierSource(embedSource(source)), ensureRowContiguous: true)
    private static let embedPreparedKernel = MLXFast.metalKernel(
        name: embedName("nemotron_mtp_front_norms_embed_lane16_concat_prepared_bf16"),
        inputNames: embedInputs, outputNames: ["out", "prepared"],
        source: NemotronHResidualRMSNorm.oneBarrierSource(embedSource(preparedSource())), ensureRowContiguous: true)
    private static let embedShuffleKernel = MLXFast.metalKernel(
        name: embedName("nemotron_mtp_front_norms_embed_lane16_concat_prepared_bf16_sh"),
        inputNames: embedInputs, outputNames: ["out", "prepared"],
        source: NemotronHResidualRMSNorm.oneBarrierSource(embedSource(preparedShuffleSource())),
        ensureRowContiguous: true)

    private enum Form { case plain, prepared, shuffle }

    /// A front-norms launch of `form` without the row-count guard (the load check sweeps the vocabulary in
    /// wide launches): on the embedding rows `e`, or reading the rows of `o` itself.
    private static func formLaunch(_ form: Form, e: MLXArray?, o: NemotronHEmbeddingRows.Operands?,
                                   hidden: MLXArray, enorm: RMSNorm, hnorm: RMSNorm) -> [MLXArray] {
        let rows = hidden.dim(1)
        let (a, ee, he) = scalars(enorm.eps, hnorm.eps)
        let tail = [hidden, enorm.weight, hnorm.weight, a, ee, he]
        var template: [(String, any KernelTemplateArg)] = [("T", DType.bfloat16)]
        if form != .plain { template.append(("NG", NemotronMTPDraftJoins.negativeFront)) }
        let grid = (672 * 2 * rows, 1, 1), threadGroup = (672, 1, 1)
        let shapes: [[Int]] = form == .plain ? [[1, rows, 5376]] : [[1, rows, 5376], [rows, 21, 9 * 32]]
        let dtypes: [DType] = form == .plain ? [.bfloat16] : [.bfloat16, .float32]
        if let o {
            let k = form == .plain ? embedKernel : form == .prepared ? embedPreparedKernel : embedShuffleKernel
            return k(o.arrays + tail, template: template + o.template, grid: grid, threadGroup: threadGroup,
                     outputShapes: shapes, outputDTypes: dtypes)
        }
        let k = form == .plain ? kernel : form == .prepared ? preparedKernel : shuffleKernel
        return k([e!] + tail, template: template, grid: grid, threadGroup: threadGroup,
                 outputShapes: shapes, outputDTypes: dtypes)
    }

    /// The conditions under which `apply` / `applyPrepared` launch the front norms on these rows' embedding
    /// (`NemotronHEmbeddingRows.rows(o)`), stated on the operands.
    static func serves(_ o: NemotronHEmbeddingRows.Operands, hidden: MLXArray, enorm: RMSNorm,
                                    hnorm: RMSNorm) -> Bool {
        embedVerified && enabled && Device.defaultDevice().deviceType == .gpu && StreamOrDevice.default == .gpu
            && o.dims == 2688 && (1...512).contains(o.count)
            && hidden.shape == [1, o.count, 2688] && hidden.dtype == .bfloat16
            && type(of: enorm) == RMSNorm.self && type(of: hnorm) == RMSNorm.self
            && enorm.weight.shape == [2688] && hnorm.weight.shape == [2688]
            && enorm.weight.dtype == .bfloat16 && hnorm.weight.dtype == .bfloat16
    }

    /// `applyPrepared` reading the token rows of `o` itself; nil where it does not serve them (the caller then
    /// builds the embedding rows and takes `applyPrepared`).
    static func applyPrepared(operands o: NemotronHEmbeddingRows.Operands, hidden: MLXArray, enorm: RMSNorm,
                              hnorm: RMSNorm) -> (MLXArray, MLXArray)? {
        guard NemotronMTPDraftJoins.enabled, o.count == 3,
              NemotronMultiRowQMV.enabled, NemotronMultiRowQMV.narrowEnabled,
              serves(o, hidden: hidden, enorm: enorm, hnorm: hnorm)
        else { return nil }
        let form: Form = shuffleVerified ? .shuffle : .prepared
        let outputs = formLaunch(form, e: nil, o: o, hidden: hidden, enorm: enorm, hnorm: hnorm)
        guard NemotronMoELeader.selfTest else { return (outputs[0], outputs[1]) }
        let expected = formLaunch(form, e: NemotronHEmbeddingRows.rows(o), o: nil, hidden: hidden,
                                  enorm: enorm, hnorm: hnorm)
        let ok = MLX.logicalAnd(
            (outputs[0].view(dtype: .uint16) .== expected[0].view(dtype: .uint16)).all(),
            (outputs[1].view(dtype: .uint32) .== expected[1].view(dtype: .uint32)).all())
        return (which(ok, outputs[0], MLXArray(Float.nan).asType(.bfloat16)),
                which(ok, outputs[1], MLXArray(Float.nan)))
    }

    /// `apply` reading the token rows of `o` itself; nil where it does not serve them.
    static func apply(operands o: NemotronHEmbeddingRows.Operands, hidden: MLXArray, enorm: RMSNorm,
                      hnorm: RMSNorm) -> MLXArray? {
        guard serves(o, hidden: hidden, enorm: enorm, hnorm: hnorm) else { return nil }
        let joined = formLaunch(.plain, e: nil, o: o, hidden: hidden, enorm: enorm, hnorm: hnorm)[0]
        guard NemotronMoELeader.selfTest else { return joined }
        return nemotronBitwiseOrNaN(joined, formLaunch(
            .plain, e: NemotronHEmbeddingRows.rows(o), o: nil, hidden: hidden, enorm: enorm, hnorm: hnorm)[0])
    }

    /// Load time, after `qualifyShuffle`: every form that can run (plain; the copy form; the shuffle form when it
    /// qualified) against the embedding-rows launch plus the same form on its rows, bit for bit in the joined rows
    /// and the prepared activation. Every vocabulary row once (2048 rows per launch, seeded hidden rows), then
    /// R = 1 ... 4 with int32 and uint32 ids, negative ids (-V, -V/2, -1) and the last row, at three hidden
    /// scales. Admits the reading forms only if every case matches.
    static func qualifyEmbed(_ block: NemotronH35MTPAttentionBlock, target: NemotronHModel) {
        embedVerified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron mtp front norms embed lane16: \(message)\n".utf8))
        }
        guard embed else { return report("off (setting)") }
        let (enorm, hnorm) = (block.embeddingNorm, block.hiddenNorm)
        guard let probe = target.mtpEmbeddingOperands(MLXArray([Int32(0)]).reshaped(1, 1)),
              eligible(embedding: NemotronHEmbeddingRows.rows(probe),
                       hidden: MLXArray.zeros([1, 1, 2688], dtype: .bfloat16), enorm: enorm, hnorm: hnorm) != nil
        else { return report("off (no embedding-rows launch ahead of the front norms)") }
        let vocabulary = probe.vocabulary
        let forms: [Form] = shuffleVerified ? [.plain, .prepared, .shuffle] : [.plain, .prepared]
        var cases = 0
        func same(_ ids: MLXArray, scale: Float, seed: UInt64) -> Bool {
            // Up to 512 rows as the call sites build them; wider sweeps on the same arrays.
            guard let o = ids.size <= 512 ? target.mtpEmbeddingOperands(ids.reshaped(1, ids.size))
                : .with([ids, probe.weight, probe.scales, probe.biases])
            else { return false }
            let rows = ids.size
            let hidden = (MLXRandom.normal([1, rows, 2688], key: MLXRandom.key(seed)) * scale).asType(.bfloat16)
            let e = NemotronHEmbeddingRows.rows(o)
            var ok = MLXArray(true)
            for form in forms {
                let a = formLaunch(form, e: nil, o: o, hidden: hidden, enorm: enorm, hnorm: hnorm)
                let b = formLaunch(form, e: e, o: nil, hidden: hidden, enorm: enorm, hnorm: hnorm)
                ok = ok .&& (a[0].view(dtype: .uint16) .== b[0].view(dtype: .uint16)).all()
                if form != .plain { ok = ok .&& (a[1].view(dtype: .uint32) .== b[1].view(dtype: .uint32)).all() }
                cases += 1
            }
            return ok.item(Bool.self)
        }
        var admitted = false
        do {
            admitted = try withError { () -> Bool in
                var start = 0
                while start < vocabulary {
                    let count = min(2048, vocabulary - start)
                    let ids = MLXArray(Int32(start) ..< Int32(start + count))
                    if !same(ids, scale: start % 4096 == 0 ? 1 : 8, seed: UInt64(0xfe00 + start / 2048)) {
                        return false
                    }
                    start += count
                }
                var state: UInt64 = 0x9E37_79B9_7F4A_7C15
                func draw() -> Int {
                    state = state &* 6364136223846793005 &+ 1442695040888963407
                    return Int(state >> 33) % vocabulary
                }
                let v = Int32(vocabulary)
                for rows in 1 ... 4 {
                    for (i, scale) in [Float(1), 4, 16].enumerated() {
                        let signed = (0 ..< rows).map { _ in Int32(draw()) }
                        let unsigned = (0 ..< rows).map { _ in UInt32(draw()) }
                        let edges = Array([-v, -(v / 2), -1, v - 1].prefix(rows))
                        let seed = UInt64(0xfe80 + 8 * rows + i)
                        if !same(MLXArray(signed), scale: scale, seed: seed)
                            || !same(MLXArray(unsigned), scale: scale, seed: seed + 64)
                            || !same(MLXArray(edges), scale: scale, seed: seed + 128) {
                            return false
                        }
                    }
                }
                return true
            }
        } catch {
            return report("off (\(error))")
        }
        embedVerified = admitted
        report(admitted
            ? "on (all \(vocabulary) rows and \(cases) launches bitwise equal to the embedding rows + front norms, forms \(forms.count))"
            : "off (differs from the embedding rows + front norms)")
    }

    /// After the primitive word gate and QKV qualification: the only new
    /// production call path, on real head weights, against its unchanged
    /// materialized-embedding entry. Neither entry writes any cache/state.
    static func qualifyEmbedFullRows(_ block: NemotronH35MTPAttentionBlock, target: NemotronHModel) {
        guard embedVerified,
              let probe = target.mtpEmbeddingOperands(MLXArray([Int32(0)]).reshaped(1, 1))
        else { return }
        let vocabulary = Int32(probe.vocabulary)
        var cases = 0
        do {
            let matches = try withError { () -> Bool in
                for seed in 0..<4 {
                    for (i, scale) in [Float(0), 1, 8].enumerated() {
                        let values: [Int32] = seed == 0 ? [0, 1, vocabulary - 1]
                            : seed == 1 ? [-vocabulary, -(vocabulary / 2), -1]
                            : [Int32((7919 * seed) % Int(vocabulary)),
                               Int32((104729 * seed) % Int(vocabulary)), vocabulary - Int32(seed + 1)]
                        let ids = MLXArray(values).reshaped(1, 3)
                        let hidden = (MLXRandom.normal([1, 3, 2688],
                            key: MLXRandom.key(UInt64(0x1a16_0000 + seed * 8 + i))) * scale).asType(.bfloat16)
                        guard let operands = target.mtpEmbeddingOperands(ids),
                              let actual = block.frontRows(hidden: hidden, operands: operands)
                        else { return false }
                        let expected = block.frontRows(hidden: hidden, embedding: target.mtpEmbedding(ids))
                        let aa = [actual.fused, actual.queries, actual.keys, actual.values]
                        let bb = [expected.fused, expected.queries, expected.keys, expected.values]
                        var same = MLXArray(true)
                        for (a, b) in zip(aa, bb) {
                            guard a.shape == b.shape, a.dtype == b.dtype else { return false }
                            same = same .&& (a.view(dtype: .uint16) .== b.view(dtype: .uint16)).all()
                        }
                        if !same.item(Bool.self) { return false }
                        cases += 1
                    }
                }
                return true
            }
            embedVerified = matches
        } catch { embedVerified = false }
        FileHandle.standardError.write(Data(
            "nemotron mtp front embed lane16 full rows: \(embedVerified ? "on" : "off") (\(cases) cases)\n".utf8))
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
        return frontRows(joined: joined, prepared: front?.1)
    }

    /// Only the post-verify R3 front: no embedding array is materialized when
    /// the fused reader qualifies. The original projection/QKV continuation is shared.
    func frontRows(hidden: MLXArray, operands: NemotronHEmbeddingRows.Operands)
        -> (fused: MLXArray, queries: MLXArray, keys: MLXArray, values: MLXArray)?
    {
        guard let front = NemotronMTPFrontNorms.applyPrepared(
            operands: operands, hidden: hidden, enorm: embeddingNorm, hnorm: hiddenNorm)
        else { return nil }
        return frontRows(joined: front.0, prepared: front.1)
    }

    private func frontRows(joined: MLXArray, prepared: MLXArray?)
        -> (fused: MLXArray, queries: MLXArray, keys: MLXArray, values: MLXArray)
    {
        let fused = nemotronMTPLinearRows(joined, projection, prepared: prepared)
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
        let output = maskedAttention(queries: queries, keys: keys, values: values, mask: mask)
        let merged = output.transposed(0, 2, 1, 3).reshaped(1, 1, -1)
        return fused + nemotronR1Linear(merged, mixer.wo)
    }

    func maskedRowParts(
        fused: MLXArray, queries: MLXArray, keys: MLXArray, values: MLXArray, mask: MLXArray
    ) -> (MLXArray, MLXArray) {
        let output = maskedAttention(queries: queries, keys: keys, values: values, mask: mask)
        let merged = output.transposed(0, 2, 1, 3).reshaped(1, 1, -1)
        return (fused, nemotronR1Linear(merged, mixer.wo))
    }

    /// A draft row's attention over the head cache `keys` / `values` (and, for the second draft, its `own`
    /// K/V row appended as the last key) with an additive mask: the one-launch twin
    /// (`NemotronMTPDraftSDPA`) once qualified, else the stock call (on the appended copies).
    private func maskedAttention(
        queries: MLXArray, keys: MLXArray, values: MLXArray, own: (keys: MLXArray, values: MLXArray)? = nil,
        mask: MLXArray
    ) -> MLXArray {
        let queries = queries.asType(keys.dtype)
        return NemotronMTPDraftSDPA.attend(
            queries: queries, keys: keys, values: values, own: own, scale: mixer.scale, mask: mask)
            ?? NemotronMTPDraftSDPA.stock(
                queries: queries, keys: keys, values: values, own: own, scale: mixer.scale, mask: mask)
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

    /// `maskedDraftAttention` over the history `keys` / `values` with the draft's `own` K/V row as the last
    /// key: the stock call takes `concatenated([keys, own])`, the qualified twin reads the row in place.
    func maskedDraftAttention(
        inputs: [MLXArray], keys: MLXArray, values: MLXArray, own: (keys: MLXArray, values: MLXArray),
        mask: MLXArray
    ) -> MLXArray {
        let queries = inputs[2].reshaped(1, 1, mixer.numHeads, mixer.headDim).transposed(0, 2, 1, 3)
        let output = maskedAttention(queries: queries, keys: keys, values: values, own: own, mask: mask)
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
    func frontRows(hidden: MLXArray, operands: NemotronHEmbeddingRows.Operands)
        -> (fused: MLXArray, queries: MLXArray, keys: MLXArray, values: MLXArray)?
    {
        (layers[0] as! NemotronH35MTPAttentionBlock).frontRows(hidden: hidden, operands: operands)
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
    /// `maskedSecondDraft` over the history K/V plus the draft's `own` row as the last key, without building
    /// the appended copies when the one-launch draft attention qualified (`NemotronMTPDraftSDPA`).
    func maskedSecondDraft(
        inputs: [MLXArray], historyKeys: MLXArray, historyValues: MLXArray,
        own: (keys: MLXArray, values: MLXArray), mask: MLXArray
    ) -> MLXArray {
        let attention = (layers[0] as! NemotronH35MTPAttentionBlock).maskedDraftAttention(
            inputs: inputs, keys: historyKeys, values: historyValues, own: own, mask: mask)
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
        case .lastPositionLogits: return (NemotronSeedHeadDict.apply(last, lmHead) ?? logits(last), hidden)
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

// MARK: - Draft attention in one launch per draft step

/// The draft steps' attention (`maskedRowOutput` / `maskedRowParts` for the first draft, `maskedDraftAttention`
/// for the second): one query row over the head cache with an additive float mask, which the stock call runs as
/// the one-pass `sdpa_vector` (1024 threads per query head; below 1024 keys every device takes that kernel).
/// This launch is that kernel's text with its function constants folded to the draft call's values (float
/// mask, not causal, no sinks) and its scalar arguments taken from the inputs' shapes and strides, with three
/// changes that move no arithmetic:
/// - Key loop: each SIMD group issues the mask, K and V loads of its next four keys (i, i + 32, i + 64,
///   i + 96) first, then forms their four scores (the same per-lane products and `simd_sum`) and applies the
///   same masked online-softmax updates in key order; the remainder runs the stock loop.
/// - Aggregation: every SIMD group stores its maximum, exp-sum and all four partial outputs (output i in its
///   own threadgroup slice) before one barrier; then the same factor, exp-sum and output sums are formed. The
///   stock text uses one barrier for the maxima and two per output value.
/// - The second draft's own K/V row (`OWN`): key L (the last) is read from the draft's own [1, 2, 1, 128]
///   projections instead of from a copy of the history with that row appended, so the two concatenations
///   (one copy each of K and V history) are not built.
/// Used only after `qualify` found every output bitwise equal to the stock call on seeded inputs.
/// NEMOTRON_MTP_DRAFT_SDPA=0 keeps the stock call (and the concatenations).
enum NemotronMTPDraftSDPA {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_MTP_DRAFT_SDPA"] != "0"
    nonisolated(unsafe) private(set) static var verified = false

    private static let source = """
      // One query row (head tid.x) over N keys: the history keys [0, L) and, with OWN, the draft's own row as
      // key L. Below this prologue the text is sdpa_vector's with the draft call's function constants.
      constexpr int BN = 32;
      constexpr int BD = 32;
      constexpr int qk_per_thread = D / BD;
      constexpr int v_per_thread = V / BD;
      typedef float U;
      const uint3 tid = threadgroup_position_in_grid;
      const uint simd_gid = simdgroup_index_in_threadgroup;
      const uint simd_lid = thread_index_in_simdgroup;
      const int L = keys_shape[2];
      const int N = OWN ? L + 1 : L;
      const int gqa_factor = int(threadgroups_per_grid.x) / keys_shape[1];
      const int mask_kv_seq_stride = fmask_shape[3] > 1 ? int(fmask_strides[3]) : 0;
      int inner_k_stride = BN * int(keys_strides[2]);
      int inner_v_stride = BN * int(values_strides[2]);

      thread U q[qk_per_thread];
      thread U k[qk_per_thread];
      thread U o[v_per_thread];

      threadgroup U outputs[v_per_thread * BN * BD];
      threadgroup U max_scores[BN];
      threadgroup U sum_exp_scores[BN];

      const int q_batch_head_idx = tid.x;
      const int kv_head_idx = q_batch_head_idx / gqa_factor;
      const int o_offset = q_batch_head_idx;
      const device T* qp = queries + q_batch_head_idx * queries_strides[1]
          + simd_lid * qk_per_thread * queries_strides[3];
      keys += kv_head_idx * keys_strides[1] + simd_gid * keys_strides[2] + simd_lid * qk_per_thread;
      values += kv_head_idx * values_strides[1] + simd_gid * values_strides[2] + simd_lid * v_per_thread;
      const device T* own_kp = OWN_KEYS;
      const device T* own_vp = OWN_VALUES;
      auto fm = fmask + simd_gid * mask_kv_seq_stride;

      out += o_offset * V + simd_gid * v_per_thread;

      // Read the query and 0 the output accumulator
      for (int i = 0; i < qk_per_thread; i++) {
        q[i] = static_cast<U>(scale) * qp[i * queries_strides[3]];
      }
      for (int i = 0; i < v_per_thread; i++) {
        o[i] = 0;
      }

      U max_score = Limits<U>::finite_min;
      U sum_exp_score = 0;

      // Four keys per step: their mask, K and V loads first, then the per-key
      // updates in key order with the stock statements.
      int i = simd_gid;
      for (; i + 3 * BN < N; i += 4 * BN) {
        T mk[4];
        U kk[4][qk_per_thread];
        U vv[4][v_per_thread];
        for (int u = 0; u < 4; u++) {
          mk[u] = fm[u * BN * mask_kv_seq_stride];
          const bool own = OWN && i + u * BN == L;
          const device T* kp = own ? own_kp : keys + u * inner_k_stride;
          const device T* vp = own ? own_vp : values + u * inner_v_stride;
          for (int j = 0; j < qk_per_thread; j++) {
            kk[u][j] = kp[j];
          }
          for (int j = 0; j < v_per_thread; j++) {
            vv[u][j] = vp[j];
          }
        }
        U key_scores[4];
        for (int u = 0; u < 4; u++) {
          U score = 0;
          for (int j = 0; j < qk_per_thread; j++) {
            score += q[j] * kk[u][j];
          }
          key_scores[u] = simd_sum(score);
        }
        for (int u = 0; u < 4; u++) {
          if (mk[u] >= Limits<T>::finite_min) {
            U score = key_scores[u];
            score += static_cast<U>(mk[u]);

            U new_max = max(max_score, score);
            U factor = fast::exp(max_score - new_max);
            U exp_score = fast::exp(score - new_max);

            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;

            for (int j = 0; j < v_per_thread; j++) {
              o[j] = o[j] * factor + exp_score * vv[u][j];
            }
          }
        }
        keys += 4 * inner_k_stride;
        values += 4 * inner_v_stride;
        fm += 4 * BN * mask_kv_seq_stride;
      }
      for (; i < N; i += BN) {
        bool use_key = (fm[0] >= Limits<T>::finite_min);
        if (use_key) {
          const device T* kp = (OWN && i == L) ? own_kp : keys;
          const device T* vp = (OWN && i == L) ? own_vp : values;
          // Read the key
          for (int j = 0; j < qk_per_thread; j++) {
            k[j] = kp[j];
          }

          // Compute the i-th score
          U score = 0;
          for (int j = 0; j < qk_per_thread; j++) {
            score += q[j] * k[j];
          }
          score = simd_sum(score);
          score += static_cast<U>(fm[0]);

          // Update the accumulators
          U new_max = max(max_score, score);
          U factor = fast::exp(max_score - new_max);
          U exp_score = fast::exp(score - new_max);

          max_score = new_max;
          sum_exp_score = sum_exp_score * factor + exp_score;

          // Update the output accumulator
          for (int j = 0; j < v_per_thread; j++) {
            o[j] = o[j] * factor + exp_score * vp[j];
          }
        }

        // Move the pointers to the next kv
        keys += inner_k_stride;
        values += inner_v_stride;
        fm += BN * mask_kv_seq_stride;
      }

      // Every SIMD group's maximum, exp-sum and four partial outputs (output i
      // in slice i, transposed as the stock store), then one barrier.
      if (simd_lid == 0) {
        max_scores[simd_gid] = max_score;
        sum_exp_scores[simd_gid] = sum_exp_score;
      }
      for (int i = 0; i < v_per_thread; i++) {
        outputs[i * BN * BD + simd_lid * BD + simd_gid] = o[i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      max_score = max_scores[simd_lid];
      U new_max = simd_max(max_score);
      U factor = fast::exp(max_score - new_max);
      sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

      // Now we need to aggregate all the outputs
      for (int i = 0; i < v_per_thread; i++) {
        o[i] = simd_sum(outputs[i * BN * BD + simd_gid * BD + simd_lid] * factor);
        o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
      }

      // And write the output
      if (simd_lid == 0) {
        for (int i = 0; i < v_per_thread; i++) {
          out[i] = static_cast<T>(o[i]);
        }
      }
      """

    private static func source(own: Bool) -> String {
        own ? source
            .replacingOccurrences(of: "OWN_KEYS", with:
                "own_keys + kv_head_idx * own_keys_strides[1] + simd_lid * qk_per_thread")
            .replacingOccurrences(of: "OWN_VALUES", with:
                "own_values + kv_head_idx * own_values_strides[1] + simd_lid * v_per_thread")
            : source
            .replacingOccurrences(of: "OWN_KEYS", with: "nullptr")
            .replacingOccurrences(of: "OWN_VALUES", with: "nullptr")
    }
    private static let kernel = MLXFast.metalKernel(
        name: "nemotron_mtp_draft_sdpa_u4_1b",
        inputNames: ["queries", "keys", "values", "fmask", "scale"], outputNames: ["out"],
        source: source(own: false), ensureRowContiguous: false)
    private static let ownKernel = MLXFast.metalKernel(
        name: "nemotron_mtp_draft_sdpa_own_u4_1b",
        inputNames: ["queries", "keys", "values", "fmask", "scale", "own_keys", "own_values"],
        outputNames: ["out"], source: source(own: true), ensureRowContiguous: false)

    nonisolated(unsafe) private static var scales: [UInt32: MLXArray] = [:]
    private static let lock = NSLock()
    private static func scaleArray(_ scale: Float) -> MLXArray {
        lock.lock(); defer { lock.unlock() }
        if let array = scales[scale.bitPattern] { return array }
        let array = MLXArray(scale)
        scales[scale.bitPattern] = array
        return array
    }

    /// The stock call this launch replaces: `own` (keys, values) appended to the history first.
    static func stock(
        queries: MLXArray, keys: MLXArray, values: MLXArray, own: (keys: MLXArray, values: MLXArray)?,
        scale: Float, mask: MLXArray
    ) -> MLXArray {
        MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: own.map { concatenated([keys, $0.keys], axis: 2) } ?? keys,
            values: own.map { concatenated([values, $0.values], axis: 2) } ?? values,
            scale: scale, mask: .array(mask))
    }

    /// queries [1, 32, 1, 128] (any strides), keys / values the history [1, 2, L, 128] (head dim contiguous),
    /// `own` the second draft's own row [1, 2, 1, 128] (key L), mask [1, 1, 1, N] with N = L (+ 1 with
    /// `own`), all BF16. Returns the attention [1, 32, 1, 128], bitwise the stock call's; nil until `qualify`
    /// passed or outside these shapes (every N here is below 1024, where the stock call is one-pass).
    static func attend(
        queries: MLXArray, keys: MLXArray, values: MLXArray, own: (keys: MLXArray, values: MLXArray)? = nil,
        scale: Float, mask: MLXArray
    ) -> MLXArray? {
        guard verified else { return nil }
        return launch(queries: queries, keys: keys, values: values, own: own, scale: scale, mask: mask)
    }

    private static func launch(
        queries: MLXArray, keys: MLXArray, values: MLXArray, own: (keys: MLXArray, values: MLXArray)?,
        scale: Float, mask: MLXArray
    ) -> MLXArray? {
        let length = keys.ndim == 4 ? keys.dim(2) : 0
        let n = length + (own == nil ? 0 : 1)
        guard enabled, Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              queries.shape == [1, 32, 1, 128], queries.dtype == .bfloat16,
              keys.ndim == 4, keys.dim(0) == 1, keys.dim(1) == 2, keys.dim(3) == 128, length > 0,
              keys.dtype == .bfloat16, values.shape == keys.shape, values.dtype == .bfloat16,
              keys.strides[3] == 1, values.strides[3] == 1,
              (1 ..< 1024).contains(n), mask.shape == [1, 1, 1, n], mask.dtype == .bfloat16
        else { return nil }
        if let own {
            guard own.keys.shape == [1, 2, 1, 128], own.values.shape == [1, 2, 1, 128],
                  own.keys.dtype == .bfloat16, own.values.dtype == .bfloat16,
                  own.keys.strides[3] == 1, own.values.strides[3] == 1
            else { return nil }
        }
        let inputs = [queries, keys, values, mask, scaleArray(scale)]
            + (own.map { [$0.keys, $0.values] } ?? [])
        let out = (own == nil ? kernel : ownKernel)(
            inputs, template: [("T", DType.bfloat16), ("D", 128), ("V", 128), ("OWN", own != nil)],
            grid: (32 * 1024, 1, 1), threadGroup: (1024, 1, 1),
            outputShapes: [[1, 32, 1, 128]], outputDTypes: [.bfloat16])[0]
        guard NemotronMoELeader.selfTest else { return out }
        return nemotronBitwiseOrNaN(
            out, stock(queries: queries, keys: keys, values: values, own: own, scale: scale, mask: mask))
    }

    /// Load time, before any draft graph is traced: the launch against the stock call at the head's attention
    /// scale on seeded inputs laid out as the draft passes them (the transposed Q row, history slices of a
    /// longer cache buffer and contiguous gathers, the own row as the transposed K/V projection), key counts
    /// across the 32-key and four-key-step boundaries up to 1023, query and key scales 1 to 16, masks with a
    /// -inf tail (and the own row kept) as the drafts build them plus scattered -inf keys. Engages only if
    /// every output is bitwise equal; an MLX error keeps the stock call.
    static func qualify(_ block: NemotronH35MTPAttentionBlock) {
        verified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron mtp draft sdpa: \(message)\n".utf8))
        }
        guard enabled else { return report("off (setting)") }
        let mixer = block.mixer
        guard mixer.numHeads == 32, mixer.numKeyValueHeads == 2, mixer.headDim == 128
        else { return report("off (the head attention is not 32 x 2 x 128)") }
        let scale = mixer.scale
        let lengths = [1, 2, 7, 8, 31, 32, 33, 63, 64, 65, 95, 96, 97, 98, 127, 128, 129, 130, 131, 255, 256,
                       257, 488, 489, 490, 491, 552, 600, 616, 617, 700, 1000, 1021, 1022]
        let inputScales: [(Float, Float)] = [(1, 1), (4, 4), (16, 8)]
        var checked = 0
        do {
            let matches = try withError { () -> Bool in
                for own in [false, true] {
                    for (i, length) in lengths.enumerated() {
                        let n = length + (own ? 1 : 0)
                        guard n < 1024 else { continue }
                        let seed = UInt64(0xd5d9_0000 + (own ? 4096 : 0) + i)
                        let (qScale, kvScale) = inputScales[i % inputScales.count]
                        let keys5 = MLXRandom.split(key: MLXRandom.key(seed), into: 6)
                        let q = MLXRandom.normal([1, 4096], dtype: .bfloat16, scale: qScale, key: keys5[0])
                        let queries = q.reshaped(1, 1, 32, 128).transposed(0, 2, 1, 3)
                        let capacity = length + 37
                        let kBuffer = MLXRandom.normal(
                            [1, 2, capacity, 128], dtype: .bfloat16, scale: kvScale, key: keys5[1])
                        let vBuffer = MLXRandom.normal(
                            [1, 2, capacity, 128], dtype: .bfloat16, scale: kvScale, key: keys5[2])
                        // Odd cases: slices of a longer buffer; even: contiguous copies.
                        var keys = kBuffer[.ellipsis, ..<length, 0...]
                        var values = vBuffer[.ellipsis, ..<length, 0...]
                        if i % 2 == 0 {
                            keys = MLX.contiguous(keys)
                            values = MLX.contiguous(values)
                        }
                        let ownRow: (keys: MLXArray, values: MLXArray)? = own
                            ? (MLXRandom.normal([1, 256], dtype: .bfloat16, scale: kvScale, key: keys5[3])
                                .reshaped(1, 1, 2, 128).transposed(0, 2, 1, 3),
                               MLXRandom.normal([1, 256], dtype: .bfloat16, scale: kvScale, key: keys5[4])
                                .reshaped(1, 1, 2, 128).transposed(0, 2, 1, 3))
                            : nil
                        // Kept: positions below length - tail (and the own row); every fifth case also drops
                        // scattered keys.
                        let tail = min(i % 4, length - 1)
                        let position = MLXArray(Int32(0) ..< Int32(n))
                        var keep = position .< Int32(length - tail)
                        if own { keep = keep .|| (position .== Int32(length)) }
                        if i % 5 == 4 {
                            let drop = MLXRandom.uniform(Float(0) ..< 1, [n], key: keys5[5]) .< MLXArray(Float(0.2))
                            keep = keep .&& (logicalNot(drop) .|| (position .== Int32(0)))
                        }
                        let mask = which(keep, MLXArray(Float(0)), MLXArray(-Float.infinity))
                            .asType(.bfloat16).reshaped(1, 1, 1, n)
                        guard let out = launch(
                            queries: queries, keys: keys, values: values, own: ownRow, scale: scale, mask: mask)
                        else { return false }
                        let expected = stock(
                            queries: queries, keys: keys, values: values, own: ownRow, scale: scale, mask: mask)
                        if !(out.view(dtype: .uint16) .== expected.view(dtype: .uint16)).all().item(Bool.self) {
                            return false
                        }
                        checked += 1
                    }
                }
                return true
            }
            verified = matches
        } catch {
            return report("off (\(error))")
        }
        report(verified
            ? "on (\(checked) cases bitwise equal to the stock call; key loop unroll4, one-barrier aggregation, own row in place)"
            : "off (differs from the stock call after \(checked) cases)")
    }
}
