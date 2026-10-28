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

    // MARK: The front norms reading the token rows themselves

    /// NEMOTRON_MTP_FRONT_EMBED=0 keeps the embedding-rows launch (`NemotronHEmbeddingRows`) ahead of the front
    /// norms. Otherwise the embedding half's threadgroup dequantizes its token's row itself, with the embedding
    /// kernel's statements (the same id wrap, group, byte and nibble expressions and `T value = scale * d + bias`,
    /// the BF16 value that kernel stores), and takes `thread_x[i]` from that value as the record takes it from `e`.
    /// The hidden half and everything after the reads are the record's text. Plain, prepared and shuffle forms;
    /// used only after `qualifyEmbed` found each bitwise equal to the embedding-rows launch plus the front norms.
    static let embed = ProcessInfo.processInfo.environment["NEMOTRON_MTP_FRONT_EMBED"] != "0"
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
                    const device uint8_t* erow = (const device uint8_t*)w + id * int64_t(BYTES);
                    for (uint i = 0; i < n_reads; i++) {
                        const uint column = lid * n_reads + i;
                        T scale = scales[id * int64_t(G) + int64_t(column / 64)];
                        T bias = biases[id * int64_t(G) + int64_t(column / 64)];
                        uint val = erow[column / 2];
                        uint8_t d;
                        d = (val >> (4 * (column % 2))) & 0x0f;
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
        name: embedName("nemotron_mtp_front_norms_embed_concat_bf16"),
        inputNames: embedInputs, outputNames: ["out"],
        source: NemotronHResidualRMSNorm.oneBarrierSource(embedSource(source)), ensureRowContiguous: true)
    private static let embedPreparedKernel = MLXFast.metalKernel(
        name: embedName("nemotron_mtp_front_norms_embed_concat_prepared_bf16"),
        inputNames: embedInputs, outputNames: ["out", "prepared"],
        source: NemotronHResidualRMSNorm.oneBarrierSource(embedSource(preparedSource())), ensureRowContiguous: true)
    private static let embedShuffleKernel = MLXFast.metalKernel(
        name: embedName("nemotron_mtp_front_norms_embed_concat_prepared_bf16_sh"),
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
            FileHandle.standardError.write(Data("nemotron mtp front norms embed: \(message)\n".utf8))
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

    // MARK: The plain form reading the token from its screen's chunk bests

    /// With the ticket-free draft screen (`NemotronHeadScreen.noTicketEnabled`), the plain reading form's twin
    /// for the second draft's front takes the first draft's chunk bests (`pv`, `pi`; NCH of them) instead of
    /// its pick: every SIMD group of the embedding half forms the pick itself with the final kernel's statements
    /// (`NemotronHeadScreen.pickStatements`: the same lane loop, comparator and xor butterfly, lane 0's id) and
    /// reads that row. Used only after `qualifyPick` found it bitwise equal to the plain reading form on the
    /// final kernel's pick.
    nonisolated(unsafe) private static var pickVerified = false
    private static func pickSource(_ s: String, hoisted: Bool = false) -> String {
        let read = "int64_t id = int64_t(ids[row]);"
        precondition(s.components(separatedBy: read).count == 2, "front norms pick read")
        let pick = hoisted
            ? NemotronHeadScreen.pickStatementsHoisted(
                values: "pv", ids: "pi", count: "NCH", maxCount: "NCH", result: "nfp_pick")
            : NemotronHeadScreen.pickStatements(values: "pv", ids: "pi", count: "NCH", result: "nfp_pick")
        return s.replacingOccurrences(of: read, with: pick + "int64_t id = int64_t(nfp_pick);")
    }
    private static let pickKernel = MLXFast.metalKernel(
        name: embedName("nemotron_mtp_front_norms_pick_concat_bf16"),
        inputNames: ["pv", "pi", "w", "scales", "biases", "h", "ew", "hw", "axis_size", "e_eps", "h_eps"],
        outputNames: ["out"],
        source: NemotronHResidualRMSNorm.oneBarrierSource(pickSource(embedSource(source))),
        ensureRowContiguous: true)
    /// The pick twin with the pick's loads hoisted (`NemotronHeadScreen.pickStatementsHoisted`); used once
    /// `qualifyPickHoist` found it bitwise equal to the loop form.
    nonisolated(unsafe) private static var pickHoistVerified = false
    private static let pickHoistKernel = MLXFast.metalKernel(
        name: embedName("nemotron_mtp_front_norms_pick_hoist_concat_bf16"),
        inputNames: ["pv", "pi", "w", "scales", "biases", "h", "ew", "hw", "axis_size", "e_eps", "h_eps"],
        outputNames: ["out"],
        source: NemotronHResidualRMSNorm.oneBarrierSource(pickSource(embedSource(source), hoisted: true)),
        ensureRowContiguous: true)

    /// Whether `apply(bests:rows:...)` serves one row whose token is the pick of `bests`.
    static func servesPick(_ bests: NemotronScreenPicks.Bests, rows o: NemotronHEmbeddingRows.Operands,
                           hidden: MLXArray, enorm: RMSNorm, hnorm: RMSNorm) -> Bool {
        pickVerified && o.count == 1 && serves(o, hidden: hidden, enorm: enorm, hnorm: hnorm)
            && bests.values.ndim == 1 && bests.values.dtype == .float32 && (1...4096).contains(bests.count)
            && bests.ids.shape == bests.values.shape && bests.ids.dtype == .int32
    }

    /// `apply(operands:)` for one row whose token is the pick of `bests`, formed in the launch; `o` gives the
    /// embedding rows (its ids are not read). nil where it does not serve them.
    static func apply(bests: NemotronScreenPicks.Bests, rows o: NemotronHEmbeddingRows.Operands, hidden: MLXArray,
                      enorm: RMSNorm, hnorm: RMSNorm) -> MLXArray? {
        guard servesPick(bests, rows: o, hidden: hidden, enorm: enorm, hnorm: hnorm) else { return nil }
        let joined = pickLaunch(bests, rows: o, hidden: hidden, enorm: enorm, hnorm: hnorm, hoisted: pickHoistVerified)
        guard NemotronMoELeader.selfTest else { return joined }
        let reference = NemotronHEmbeddingRows.Operands.with(
            [NemotronHeadScreen.pick(values: bests.values, ids: bests.ids), o.weight, o.scales, o.biases])
        return nemotronBitwiseOrNaN(joined, formLaunch(
            .plain, e: nil, o: reference, hidden: hidden, enorm: enorm, hnorm: hnorm)[0])
    }

    private static func pickLaunch(_ bests: NemotronScreenPicks.Bests, rows o: NemotronHEmbeddingRows.Operands,
                                   hidden: MLXArray, enorm: RMSNorm, hnorm: RMSNorm, hoisted: Bool = false)
        -> MLXArray
    {
        let (a, ee, he) = scalars(enorm.eps, hnorm.eps)
        return (hoisted ? pickHoistKernel : pickKernel)(
            [bests.values, bests.ids, o.weight, o.scales, o.biases, hidden, enorm.weight, hnorm.weight, a, ee, he],
            template: [("T", DType.bfloat16)] + o.template + [("NCH", bests.count)],
            grid: (672 * 2, 1, 1), threadGroup: (672, 1, 1),
            outputShapes: [[1, 1, 5376]], outputDTypes: [.bfloat16])[0]
    }

    /// Load time, after `qualifyEmbed` and the draft screen's plan: the pick twin against the plain reading form
    /// on the final kernel's pick of the same bests, bit for bit in the joined row, for each chunk count the
    /// ticket-free screen reduces. Bests: Gaussian values, exact ties, all-equal values, NaNs (one in the first
    /// slot), infinities; distinct ids drawn per chunk over the vocabulary. Admits the twin only if every case
    /// matches.
    static func qualifyPick(_ block: NemotronH35MTPAttentionBlock, target: NemotronHModel) {
        pickVerified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron mtp front norms pick: \(message)\n".utf8))
        }
        guard NemotronHeadScreen.noTicketEnabled else { return report("off (setting)") }
        let counts = NemotronScreenPicks.chunkCounts
        guard !counts.isEmpty else { return report("off (no ticket-free draft screen)") }
        guard embedVerified,
              let probe = target.mtpEmbeddingOperands(MLXArray([Int32(0)]).reshaped(1, 1))
        else { return report("off (the reading form is off)") }
        let (enorm, hnorm) = (block.embeddingNorm, block.hiddenNorm)
        var cases = 0
        var admitted = false
        do {
            admitted = try withError { () -> Bool in
                var ok = MLXArray(true)
                for n in counts {
                    for kind in 0 ..< 6 {
                        let bests = NemotronScreenPicks.seededBests(
                            count: n, vocabulary: probe.vocabulary, kind: kind, seed: UInt64(0x5e00 + 8 * n + kind))
                        let hidden = (MLXRandom.normal([1, 1, 2688], key: MLXRandom.key(UInt64(0x5e80 + kind)))
                            * Float(1 + kind)).asType(.bfloat16)
                        let rows = NemotronHEmbeddingRows.Operands.with(
                            [NemotronHeadScreen.pick(values: bests.values, ids: bests.ids),
                             probe.weight, probe.scales, probe.biases])
                        let a = pickLaunch(bests, rows: rows, hidden: hidden, enorm: enorm, hnorm: hnorm)
                        let b = formLaunch(.plain, e: nil, o: rows, hidden: hidden, enorm: enorm, hnorm: hnorm)[0]
                        ok = ok .&& (a.view(dtype: .uint16) .== b.view(dtype: .uint16)).all()
                        cases += 1
                    }
                }
                return ok.item(Bool.self)
            }
        } catch {
            return report("off (\(error))")
        }
        pickVerified = admitted
        report(admitted
            ? "on (\(cases) cases bitwise equal to the reading form on the final kernel's pick, chunk counts \(counts))"
            : "off (differs from the reading form on the final kernel's pick)")
    }

    /// Load time, after `qualifyPick` admitted the pick twin: the hoisted-load twin against the loop form, bit for
    /// bit in the joined row, for each chunk count the ticket-free screen reduces and two counts that leave the
    /// last slot of some lanes empty (113, 200), over the same six kinds of bests (NaN in slot 0 included).
    static func qualifyPickHoist(_ block: NemotronH35MTPAttentionBlock, target: NemotronHModel) {
        pickHoistVerified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron mtp front norms pick hoist: \(message)\n".utf8))
        }
        guard NemotronHeadScreen.pickHoistEnabled else { return report("off (setting)") }
        guard pickVerified,
              let probe = target.mtpEmbeddingOperands(MLXArray([Int32(0)]).reshaped(1, 1))
        else { return report("off (the pick twin is off)") }
        let counts = NemotronScreenPicks.chunkCounts + [113, 200]
        let (enorm, hnorm) = (block.embeddingNorm, block.hiddenNorm)
        var cases = 0
        var admitted = false
        do {
            admitted = try withError { () -> Bool in
                var ok = MLXArray(true)
                for n in counts {
                    for kind in 0 ..< 6 {
                        let bests = NemotronScreenPicks.seededBests(
                            count: n, vocabulary: probe.vocabulary, kind: kind, seed: UInt64(0x6e00 + 8 * n + kind))
                        let hidden = (MLXRandom.normal([1, 1, 2688], key: MLXRandom.key(UInt64(0x6e80 + kind)))
                            * Float(1 + kind)).asType(.bfloat16)
                        let a = pickLaunch(bests, rows: probe, hidden: hidden, enorm: enorm, hnorm: hnorm, hoisted: true)
                        let b = pickLaunch(bests, rows: probe, hidden: hidden, enorm: enorm, hnorm: hnorm)
                        ok = ok .&& (a.view(dtype: .uint16) .== b.view(dtype: .uint16)).all()
                        cases += 1
                    }
                }
                return ok.item(Bool.self)
            }
        } catch {
            return report("off (\(error))")
        }
        pickHoistVerified = admitted
        report(admitted
            ? "on (\(cases) cases bitwise equal to the loop form, counts \(counts))"
            : "off (differs from the loop form)")
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

/// The one-token draft front's attention norm (`uncompiledDraftFront`: the norm of the fused row, ahead of the
/// second draft's Q/K/V) through the tree's prepared norm launch at R = 1 (`normPreparedRows`' kernel, the
/// family the three-row front already uses: one barrier with NEMOTRON_MTP_JOIN_ONE_BARRIER) instead of the stock
/// RMSNorm launch; only its normalized row is read. Used only after `qualify` found it bitwise equal to the stock
/// launch. NEMOTRON_MTP_DRAFT2_NORM1B=0 keeps the stock launch.
enum NemotronMTPDraftFrontNorm {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_MTP_DRAFT2_NORM1B"] != "0"
    nonisolated(unsafe) private static var verified = false

    static func apply(_ x: MLXArray, norm: RMSNorm) -> MLXArray? {
        verified ? NemotronHResidualRMSNorm.normPreparedRow(x, norm: norm) : nil
    }

    /// Load time: the R = 1 prepared-norm launch against the stock RMSNorm launch on the head's own norm, bit for
    /// bit, on seeded rows at five scales (1/64 ... 4096), a constant row and a row with one large value.
    static func qualify(_ block: NemotronH35MTPAttentionBlock) {
        verified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron mtp draft front norm 1b: \(message)\n".utf8))
        }
        guard enabled else { return report("off (setting)") }
        var ok = MLXArray(true)
        var cases = 0
        do {
            try withError {
                var rows: [MLXArray] = []
                for seed in 0 ..< 3 {
                    for scale in [Float(1.0 / 64), 0.25, 1, 16, 4096] {
                        let key = MLXRandom.key(UInt64(0xd2f0 + 8 * seed) + UInt64(rows.count))
                        rows.append((MLXRandom.normal([1, 1, 2688], key: key) * scale).asType(.bfloat16))
                    }
                }
                rows.append(MLXArray.ones([1, 1, 2688], dtype: .bfloat16) * MLXArray(Float(3)).asType(.bfloat16))
                let spike = MLXRandom.normal([1, 1, 2688], key: MLXRandom.key(0xd2ff)) * Float(0.01)
                rows.append(which(MLXArray(Int32(0) ..< 2688).reshaped(1, 1, 2688) .== MLXArray(Int32(1337)),
                                  MLXArray(Float(300)), spike).asType(.bfloat16))
                for x in rows {
                    guard let got = NemotronHResidualRMSNorm.normPreparedRow(x, norm: block.norm) else {
                        ok = MLXArray(false); return
                    }
                    ok = ok .&& (got.view(dtype: .uint16) .== block.norm(x).view(dtype: .uint16)).all()
                    cases += 1
                }
                eval(ok)
            }
        } catch {
            return report("off (\(error))")
        }
        verified = ok.item(Bool.self)
        report(verified ? "on (\(cases) cases bitwise equal to the stock rms norm at R = 1)"
               : "off (differs from the stock rms norm)")
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

    /// `pagedDraftInputs` with the front norms reading the token's embedding row themselves
    /// (`NemotronMTPFrontNorms.apply(operands:)`); nil where that launch does not serve the operands.
    func pagedDraftInputs(hidden: MLXArray, operands: NemotronHEmbeddingRows.Operands) -> [MLXArray]? {
        guard let front = draftFrontCache(hidden: hidden, operands: operands, owner: self) else { return nil }
        let (q, k, v) = mixer.cbv2Projections(front[1], targetQKV: NemotronMTPDraftTail.fusedQKV)
        return [front[0], front[1], q, k, v]
    }

    /// `pagedDraftInputs(hidden:operands:)` whose token is the pick of `bests` (a ticket-free draft screen's chunk
    /// bests), formed inside the front norms; `operands` gives the embedding rows. nil where they do not serve it.
    func pagedDraftInputs(
        hidden: MLXArray, operands: NemotronHEmbeddingRows.Operands, bests: NemotronScreenPicks.Bests
    ) -> [MLXArray]? {
        guard let front = draftFrontCache(hidden: hidden, rows: operands, bests: bests, owner: self) else { return nil }
        let (q, k, v) = mixer.cbv2Projections(front[1], targetQKV: NemotronMTPDraftTail.fusedQKV)
        return [front[0], front[1], q, k, v]
    }

    /// `pagedForwardRows`' cache-free front: the fused residual rows and their
    /// Q [1, heads, L, headDim] and K/V [1, kvHeads, L, headDim]. With `cache`,
    /// the Q/K/V launch also appends the K/V rows to it when it can
    /// (`NemotronMTPFrontRowsKVDirect`); `written` then holds the roots
    /// `appendTrusted` would return, else it is nil and nothing was appended.
    func frontRows(hidden: MLXArray, embedding: MLXArray, cache: PagedLayerCache? = nil)
        -> (fused: MLXArray, queries: MLXArray, keys: MLXArray, values: MLXArray, written: [MLXArray]?)
    {
        // Draft-only joins at R = 3: the front norms also emit eh_proj's
        // prepared input, and the attention norm wq's (NEMOTRON_MTP_DRAFT_JOINS).
        let front = NemotronMTPFrontNorms.applyPrepared(
            embedding: embedding, hidden: hidden, enorm: embeddingNorm, hnorm: hiddenNorm)
        return frontRows(
            joined: front?.0 ?? joinedNorms(hidden: hidden, embedding: embedding), prepared: front?.1, cache: cache)
    }

    /// `frontRows` with the front norms reading the token rows themselves
    /// (`NemotronMTPFrontNorms.applyPrepared(operands:)`); nil where that launch does not serve the operands.
    func frontRows(hidden: MLXArray, operands: NemotronHEmbeddingRows.Operands, cache: PagedLayerCache? = nil)
        -> (fused: MLXArray, queries: MLXArray, keys: MLXArray, values: MLXArray, written: [MLXArray]?)?
    {
        guard let front = NemotronMTPFrontNorms.applyPrepared(
            operands: operands, hidden: hidden, enorm: embeddingNorm, hnorm: hiddenNorm) else { return nil }
        return frontRows(joined: front.0, prepared: front.1, cache: cache)
    }

    private func frontRows(joined: MLXArray, prepared: MLXArray?, cache: PagedLayerCache?)
        -> (fused: MLXArray, queries: MLXArray, keys: MLXArray, values: MLXArray, written: [MLXArray]?)
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
        let (q, k, v): (MLXArray, MLXArray, MLXArray)
        var written: [MLXArray]? = nil
        if let cache, let direct = NemotronMTPFrontRowsKVDirect.apply(
            x, prepared: xPrepared, mixer: mixer, cache: cache) {
            (q, k, v) = (direct.q, direct.k, direct.v)
            written = direct.written
        } else {
            // Q, K and V in one launch once qualified (NEMOTRON_MTP_DRAFT_ROWS_QKV).
            (q, k, v) = NemotronMTPFrontRowsQKV.apply(x, prepared: xPrepared, mixer: mixer)
                ?? NemotronMTPFrontRowsQKV.separate(x, prepared: xPrepared, mixer: mixer)
        }
        let queries = q
            .reshaped(batch, length, mixer.numHeads, mixer.headDim).transposed(0, 2, 1, 3)
        let keys = k
            .reshaped(batch, length, mixer.numKeyValueHeads, mixer.headDim).transposed(0, 2, 1, 3)
        let values = v
            .reshaped(batch, length, mixer.numKeyValueHeads, mixer.headDim).transposed(0, 2, 1, 3)
        return (fused, queries, keys, values, written)
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
    /// `maskedRowOutput` / `maskedRowParts` reading the head cache pages in place (`read`, from
    /// `NemotronMTPDraftSDPA.directRead`); nil when the paged launch does not serve them (the caller then takes
    /// the snapshot's copy).
    func maskedRowOutput(
        fused: MLXArray, queries: MLXArray, read: CBv2PagedDirectRead, row: PagedSequenceKV, mask: MLXArray
    ) -> MLXArray? {
        maskedRowParts(fused: fused, queries: queries, read: read, row: row, mask: mask).map { $0.0 + $0.1 }
    }

    func maskedRowParts(
        fused: MLXArray, queries: MLXArray, read: CBv2PagedDirectRead, row: PagedSequenceKV, mask: MLXArray
    ) -> (MLXArray, MLXArray)? {
        guard let output = NemotronMTPDraftSDPA.attendPaged(
            queries: queries.asType(read.dtype), read: read, row: row, scale: mixer.scale, mask: mask)
        else { return nil }
        let merged = output.transposed(0, 2, 1, 3).reshaped(1, 1, -1)
        return (fused, nemotronR1Linear(merged, mixer.wo))
    }

    /// The first draft's projected attention for the post-verify row the device accept walk selects
    /// (`NemotronMTPDraftSDPA.attendPagedWalk` over `queryRows` [1, heads, W, headDim]): `maskedRowParts`' attention
    /// of that row under its first mask, without the row pick. Nil when the walk launch does not serve them.
    func walkedRowAttention(
        queryRows: MLXArray, walk: CBv2MTPDeviceAcceptWalk.Operands, firstSpeculated: Int,
        read: CBv2PagedDirectRead, row: PagedSequenceKV
    ) -> MLXArray? {
        guard let output = NemotronMTPDraftSDPA.attendPagedWalk(
            queries: queryRows.asType(read.dtype), packet: walk.packet, count: walk.count, drafts: walk.drafts,
            targets: walk.targets, firstSpeculated: firstSpeculated, read: read, row: row, scale: mixer.scale)
        else { return nil }
        let merged = output.transposed(0, 2, 1, 3).reshaped(1, 1, -1)
        return nemotronR1Linear(merged, mixer.wo)
    }

    /// `maskedDraftAttention(inputs:read:row:own:mask:)` under the second mask of the row the device accept walk
    /// selects, formed in the launch (`NemotronMTPDraftSDPA.attendPagedWalk`); nil when it does not serve them.
    func maskedDraftAttention(
        inputs: [MLXArray], read: CBv2PagedDirectRead, row: PagedSequenceKV,
        own: (keys: MLXArray, values: MLXArray), walk: CBv2MTPDeviceAcceptWalk.Operands, firstSpeculated: Int
    ) -> MLXArray? {
        let queries = inputs[2].reshaped(1, 1, mixer.numHeads, mixer.headDim).transposed(0, 2, 1, 3)
            .asType(read.dtype)
        guard let output = NemotronMTPDraftSDPA.attendPagedWalk(
            queries: queries, packet: walk.packet, count: walk.count, drafts: walk.drafts, targets: walk.targets,
            firstSpeculated: firstSpeculated, read: read, row: row, own: own, scale: mixer.scale)
        else { return nil }
        return nemotronR1Linear(output.transposed(0, 2, 1, 3).reshaped(1, 1, -1), mixer.wo)
    }

    /// `maskedDraftAttention` over the head cache pages in place plus the draft's `own` row as the last key; nil
    /// when the paged launch does not serve them.
    func maskedDraftAttention(
        inputs: [MLXArray], read: CBv2PagedDirectRead, row: PagedSequenceKV,
        own: (keys: MLXArray, values: MLXArray), mask: MLXArray
    ) -> MLXArray? {
        let queries = inputs[2].reshaped(1, 1, mixer.numHeads, mixer.headDim).transposed(0, 2, 1, 3)
            .asType(read.dtype)
        guard let output = NemotronMTPDraftSDPA.attendPaged(
            queries: queries, read: read, row: row, own: own, scale: mixer.scale, mask: mask)
        else { return nil }
        return nemotronR1Linear(output.transposed(0, 2, 1, 3).reshaped(1, 1, -1), mixer.wo)
    }

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
        return [fused, draftNorm(fused)]
    }

    /// The draft front's attention-norm row: the tree's one-barrier prepared norm at R = 1 once qualified
    /// (`NemotronMTPDraftFrontNorm`), else the stock RMSNorm launch.
    private func draftNorm(_ fused: MLXArray) -> MLXArray {
        NemotronMTPDraftFrontNorm.apply(fused, norm: norm) ?? norm(fused)
    }

    /// `uncompiledDraftFront` with the front norms reading the token's row themselves; where they do not
    /// serve the operands, the embedding-rows launch and `uncompiledDraftFront`'s own front norms.
    fileprivate func uncompiledDraftFront(hidden: MLXArray, operands: NemotronHEmbeddingRows.Operands) -> [MLXArray] {
        guard let joined = NemotronMTPFrontNorms.apply(
            operands: operands, hidden: hidden, enorm: embeddingNorm, hnorm: hiddenNorm)
        else { return uncompiledDraftFront(hidden: hidden, embedding: NemotronHEmbeddingRows.rows(operands)) }
        let fused = nemotronR1Linear(joined, projection)
        return [fused, draftNorm(fused)]
    }

    fileprivate func draftFrontReadsRows(hidden: MLXArray, operands: NemotronHEmbeddingRows.Operands) -> Bool {
        NemotronMTPFrontNorms.serves(operands, hidden: hidden, enorm: embeddingNorm, hnorm: hiddenNorm)
    }

    /// `uncompiledDraftFront` whose token is the pick of `bests`, formed in the front norms; where they do not
    /// serve it, the final kernel's pick and the reading form.
    fileprivate func uncompiledDraftFront(
        hidden: MLXArray, rows: NemotronHEmbeddingRows.Operands, bests: NemotronScreenPicks.Bests
    ) -> [MLXArray] {
        guard let joined = NemotronMTPFrontNorms.apply(
            bests: bests, rows: rows, hidden: hidden, enorm: embeddingNorm, hnorm: hiddenNorm)
        else {
            return uncompiledDraftFront(hidden: hidden, operands: .with(
                [NemotronHeadScreen.pick(values: bests.values, ids: bests.ids), rows.weight, rows.scales, rows.biases]))
        }
        let fused = nemotronR1Linear(joined, projection)
        return [fused, draftNorm(fused)]
    }

    fileprivate func draftFrontReadsPick(
        hidden: MLXArray, rows: NemotronHEmbeddingRows.Operands, bests: NemotronScreenPicks.Bests
    ) -> Bool {
        NemotronMTPFrontNorms.servesPick(bests, rows: rows, hidden: hidden, enorm: embeddingNorm, hnorm: hiddenNorm)
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

    private var rowsEntry: Entry?

    /// The same front with the front norms reading the token's embedding row (`NemotronHEmbeddingRows`
    /// operands as inputs: ids, packed rows, scales, biases); nil where those norms do not serve the operands.
    func callAsFunction(
        hidden: MLXArray, operands: NemotronHEmbeddingRows.Operands, owner: NemotronH35MTPAttentionBlock
    ) -> [MLXArray]? {
        guard owner.draftFrontReadsRows(hidden: hidden, operands: operands) else { return nil }
        guard Self.enabled,
              Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              hidden.shape == [1, 1, 2688], hidden.dtype == .bfloat16, operands.count == 1,
              type(of: owner.embeddingNorm) == RMSNorm.self,
              type(of: owner.hiddenNorm) == RMSNorm.self,
              type(of: owner.norm) == RMSNorm.self,
              (type(of: owner.projection) == Linear.self
                || type(of: owner.projection) == QuantizedLinear.self)
        else { return owner.uncompiledDraftFront(hidden: hidden, operands: operands) }
        let modules: [Module] = [owner, owner.embeddingNorm, owner.hiddenNorm,
                                 owner.projection, owner.norm]
        let identity = modules.map(ObjectIdentifier.init)
        var retired: Entry?
        let current = lock.withLock { () -> Entry in
            if let rowsEntry, rowsEntry.identity == identity { return rowsEntry }
            let state = modules.dropFirst().flatMap { $0.innerState() }
            let function: @Sendable ([MLXArray]) -> [MLXArray] = compile(inputs: [state]) {
                [weak owner] arrays in
                guard let owner else { preconditionFailure("Released Nemotron MTP draft-front owner") }
                return owner.uncompiledDraftFront(
                    hidden: arrays[0], operands: .with(Array(arrays[1...])))
            }
            let replacement = Entry(identity: identity, function: function)
            retired = rowsEntry
            rowsEntry = replacement
            return replacement
        }
        let result = withExtendedLifetime(retired) { current.function([hidden] + operands.arrays) }
        return result.count == 2 ? result
            : owner.uncompiledDraftFront(hidden: hidden, operands: operands)
    }

    private var pickEntry: Entry?

    /// The same front whose token is the pick of `bests` (inputs: the hidden row, the chunk bests' values and
    /// ids, the packed rows, scales and biases); nil where the front norms do not serve it.
    func callAsFunction(
        hidden: MLXArray, rows: NemotronHEmbeddingRows.Operands, bests: NemotronScreenPicks.Bests,
        owner: NemotronH35MTPAttentionBlock
    ) -> [MLXArray]? {
        guard owner.draftFrontReadsPick(hidden: hidden, rows: rows, bests: bests) else { return nil }
        guard Self.enabled,
              Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              hidden.shape == [1, 1, 2688], hidden.dtype == .bfloat16,
              type(of: owner.embeddingNorm) == RMSNorm.self,
              type(of: owner.hiddenNorm) == RMSNorm.self,
              type(of: owner.norm) == RMSNorm.self,
              (type(of: owner.projection) == Linear.self
                || type(of: owner.projection) == QuantizedLinear.self)
        else { return owner.uncompiledDraftFront(hidden: hidden, rows: rows, bests: bests) }
        let modules: [Module] = [owner, owner.embeddingNorm, owner.hiddenNorm,
                                 owner.projection, owner.norm]
        let identity = modules.map(ObjectIdentifier.init)
        var retired: Entry?
        let current = lock.withLock { () -> Entry in
            if let pickEntry, pickEntry.identity == identity { return pickEntry }
            let state = modules.dropFirst().flatMap { $0.innerState() }
            let function: @Sendable ([MLXArray]) -> [MLXArray] = compile(inputs: [state]) {
                [weak owner] arrays in
                guard let owner else { preconditionFailure("Released Nemotron MTP draft-front owner") }
                let bests = NemotronScreenPicks.Bests(values: arrays[1], ids: arrays[2])
                return owner.uncompiledDraftFront(
                    hidden: arrays[0],
                    rows: .with([NemotronHeadScreen.pick(values: arrays[1], ids: arrays[2]),
                                 arrays[3], arrays[4], arrays[5]]),
                    bests: bests)
            }
            let replacement = Entry(identity: identity, function: function)
            retired = pickEntry
            pickEntry = replacement
            return replacement
        }
        let result = withExtendedLifetime(retired) {
            current.function([hidden, bests.values, bests.ids, rows.weight, rows.scales, rows.biases])
        }
        return result.count == 2 ? result
            : owner.uncompiledDraftFront(hidden: hidden, rows: rows, bests: bests)
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
        if NemotronMTPJoinRouter.draftEnabled, NemotronMTPExecution.compiledMoE,
           exactTailAvailable(residual), attention.shape == residual.shape, attention.dtype == residual.dtype {
            return oneTokenTailCache([residual, attention], mode: 3, owner: self)
        }
        if exactTailAvailable(residual), attention.shape == residual.shape,
           let joined = NemotronHResidualRMSNorm.apply(residual, attention, norm: norm) {
            if NemotronMTPExecution.compiledMoE {
                return oneTokenTailCache([joined.0, joined.1], mode: 2, owner: self)
            }
            return exactOneTokenTail(joined.0, normalized: joined.1)
        }
        return self(residual + attention)
    }

    /// Whether `callAsFunction(fusedRows:walk:attention:)` serves a [1, 1, 2688] BF16 attention row: the
    /// conditions `callAsFunction(residual:attention:)` joins under, for the walked fused row.
    func walkedTailServes(fusedRows: MLXArray, walk: CBv2MTPDeviceAcceptWalk.Operands) -> Bool {
        !NemotronMTPFirstRoundDraft.active
            && NemotronMTPDraftTail.exactTail && NemotronMoELeader.enabled
            && type(of: norm) == RMSNorm.self && type(of: finalNorm) == RMSNorm.self
            && type(of: mixer) == NemotronHMoE.self && mixer.sharedExperts != nil
            && mixer.numExpertsPerTok == 6
            && Device.defaultDevice().deviceType == .gpu && StreamOrDevice.default == .gpu
            && NemotronHResidualRMSNorm.walkServes(
                fusedRows, packet: walk.packet, count: walk.count, drafts: walk.drafts, targets: walk.targets,
                norm: norm)
    }

    /// `callAsFunction(residual:attention:)` with the residual the fused row the device accept walk selects
    /// from `fusedRows` [1, W, 2688] (`NemotronHResidualRMSNorm.applyWalk`), so the row is not gathered first;
    /// the same tail follows the join. Nil when it does not serve them.
    func callAsFunction(fusedRows: MLXArray, walk: CBv2MTPDeviceAcceptWalk.Operands, attention: MLXArray)
        -> MLXArray?
    {
        guard walkedTailServes(fusedRows: fusedRows, walk: walk), exactTailAvailable(attention),
              let joined = NemotronHResidualRMSNorm.applyWalk(
                  fusedRows, attention, packet: walk.packet, count: walk.count, drafts: walk.drafts,
                  targets: walk.targets, norm: norm)
        else { return nil }
        if NemotronMTPExecution.compiledMoE {
            return oneTokenTailCache([joined.0, joined.1], mode: 2, owner: self)
        }
        return exactOneTokenTail(joined.0, normalized: joined.1)
    }

    /// NEMOTRON_MTP_DRAFT_JOIN_ROUTER: the residual join with the attention row and the router in one launch
    /// (`NemotronMTPJoinRouter`, R = 1), then the same exact tail on its residual, normalized row and routing.
    fileprivate func joinedOneTokenTail(_ residual: MLXArray, _ attention: MLXArray) -> MLXArray? {
        let gate = mixer.gate
        guard gate.topK == 6, gate.nGroup == 1, gate.topkGroup == 1, gate.normTopkProb,
              gate.routedScalingFactor == 2.5,
              let joined = NemotronMTPJoinRouter.run(residual, attention, norm: norm, weight: gate.weight,
                                                     bias: gate.eSCB, draft: true)
        else { return nil }
        return exactOneTokenTail(joined.residual, normalized: joined.normalized,
                                 routing: (joined.indices, joined.weights))
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
    func exactOneTokenTail(_ x: MLXArray, normalized: MLXArray?, ordered: Bool? = nil,
                           routing: (MLXArray, MLXArray)? = nil) -> MLXArray {
        let h = normalized ?? norm(x)
        let gate = mixer.gate
        let fused = routing == nil && NemotronMTPDraftTail.oneLaunchRouter && gate.topK == 6 && gate.nGroup == 1
            && gate.topkGroup == 1 && gate.normTopkProb && gate.routedScalingFactor == 2.5
            ? NemotronMTPRouterRowsSelect.run(h, gate.weight, bias: gate.eSCB, draft: true) : nil
        let (indices, scores) = routing ?? fused ?? gate(h)
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
    private var entries: [Entry?] = [nil, nil, nil, nil]

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
                case 3:
                    guard let tail = owner.joinedOneTokenTail(arrays[0], arrays[1]) else {
                        preconditionFailure("Qualified draft join router became ineligible")
                    }
                    return [tail]
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
            case 3: return owner.joinedOneTokenTail(inputs[0], inputs[1])
                ?? owner.uncompiledOneTokenTail(inputs[0] + inputs[1])
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
    /// `pagedDraftInputs` on the token's embedding operands (the front norms read the row); nil where they
    /// do not serve them.
    func pagedDraftInputs(hidden: MLXArray, operands: NemotronHEmbeddingRows.Operands) -> [MLXArray]? {
        (layers[0] as! NemotronH35MTPAttentionBlock).pagedDraftInputs(hidden: hidden, operands: operands)
    }
    /// `pagedDraftInputs` whose token is the pick of a ticket-free screen's chunk bests; nil where the front
    /// norms do not serve it.
    func pagedDraftInputs(
        hidden: MLXArray, operands: NemotronHEmbeddingRows.Operands, bests: NemotronScreenPicks.Bests
    ) -> [MLXArray]? {
        (layers[0] as! NemotronH35MTPAttentionBlock).pagedDraftInputs(hidden: hidden, operands: operands, bests: bests)
    }
    func draftKeyValues(inputs: [MLXArray]) -> (keys: MLXArray, values: MLXArray) {
        (layers[0] as! NemotronH35MTPAttentionBlock).draftKeyValues(inputs: inputs)
    }
    func frontRows(hidden: MLXArray, embedding: MLXArray, cache: PagedLayerCache? = nil)
        -> (fused: MLXArray, queries: MLXArray, keys: MLXArray, values: MLXArray, written: [MLXArray]?)
    {
        (layers[0] as! NemotronH35MTPAttentionBlock).frontRows(hidden: hidden, embedding: embedding, cache: cache)
    }
    /// `frontRows` on the tokens' embedding operands (the front norms read the rows); nil where they do not
    /// serve them.
    func frontRows(hidden: MLXArray, operands: NemotronHEmbeddingRows.Operands, cache: PagedLayerCache? = nil)
        -> (fused: MLXArray, queries: MLXArray, keys: MLXArray, values: MLXArray, written: [MLXArray]?)?
    {
        (layers[0] as! NemotronH35MTPAttentionBlock).frontRows(hidden: hidden, operands: operands, cache: cache)
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
    /// `maskedSecondDraft` reading the head cache pages in place (`NemotronMTPDraftSDPA.directRead`); nil when
    /// the paged launch does not serve them (the caller then takes the snapshot's copy).
    func maskedSecondDraft(
        inputs: [MLXArray], read: CBv2PagedDirectRead, row: PagedSequenceKV,
        own: (keys: MLXArray, values: MLXArray), mask: MLXArray
    ) -> MLXArray? {
        guard let attention = (layers[0] as! NemotronH35MTPAttentionBlock).maskedDraftAttention(
            inputs: inputs, read: read, row: row, own: own, mask: mask) else { return nil }
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

    /// `finishMaskedRow` reading the head cache pages in place (`NemotronMTPDraftSDPA.directRead`); nil when the
    /// paged launch does not serve them (the caller then takes the snapshot's copy).
    func finishMaskedRow(
        fused: MLXArray, queries: MLXArray, read: CBv2PagedDirectRead, row: PagedSequenceKV, mask: MLXArray
    ) -> MLXArray? {
        let block = layers[0] as! NemotronH35MTPAttentionBlock
        guard Self.deviceDraftJoin else {
            return block.maskedRowOutput(fused: fused, queries: queries, read: read, row: row, mask: mask)
                .map { finishPagedAttentionRow($0) }
        }
        guard let (residual, attention) = block.maskedRowParts(
            fused: fused, queries: queries, read: read, row: row, mask: mask) else { return nil }
        precondition(residual.ndim == 3 && residual.dim(0) == 1 && residual.dim(1) == 1)
        return (layers[1] as! NemotronH35MTPExpertBlock)(residual: residual, attention: attention)
    }

    /// `finishMaskedRow(fused:queries:read:row:mask:)` for the post-verify row the device accept walk selects,
    /// without the row pick (`NemotronMTPDraftSDPA.walkVerified`): the attention reads that row's query from
    /// `queryRows` and masks from the walk, and the join reads that row of `fusedRows`. Nil, with nothing
    /// built, unless both launches serve the round.
    func finishWalkedRow(
        fusedRows: MLXArray, queryRows: MLXArray, walk: CBv2MTPDeviceAcceptWalk.Operands, firstSpeculated: Int,
        read: CBv2PagedDirectRead, row: PagedSequenceKV
    ) -> MLXArray? {
        let block = layers[0] as! NemotronH35MTPAttentionBlock
        let expert = layers[1] as! NemotronH35MTPExpertBlock
        guard Self.deviceDraftJoin, expert.walkedTailServes(fusedRows: fusedRows, walk: walk),
              NemotronMTPDraftSDPA.walkServes(
                  queryRows: queryRows, packet: walk.packet, count: walk.count, drafts: walk.drafts,
                  targets: walk.targets, read: read),
              firstSpeculated >= 0, firstSpeculated + queryRows.dim(2) == read.count,
              let attention = block.walkedRowAttention(
                  queryRows: queryRows, walk: walk, firstSpeculated: firstSpeculated, read: read, row: row)
        else { return nil }
        return expert(fusedRows: fusedRows, walk: walk, attention: attention)
    }

    /// `maskedSecondDraft(inputs:read:row:own:mask:)` under the second mask of the row the device accept walk
    /// selects, formed in the launch; nil when the walk launch does not serve them.
    func maskedSecondDraft(
        inputs: [MLXArray], read: CBv2PagedDirectRead, row: PagedSequenceKV,
        own: (keys: MLXArray, values: MLXArray), walk: CBv2MTPDeviceAcceptWalk.Operands, firstSpeculated: Int
    ) -> MLXArray? {
        guard let attention = (layers[0] as! NemotronH35MTPAttentionBlock).maskedDraftAttention(
            inputs: inputs, read: read, row: row, own: own, walk: walk, firstSpeculated: firstSpeculated)
        else { return nil }
        return (layers[1] as! NemotronH35MTPExpertBlock)(residual: inputs[0], attention: attention)
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

/// The post-verify front's Q/K/V launch (`NemotronMTPFrontRowsQKV`) storing its three K/V rows straight into the
/// head cache's pages, so the segment write `appendTrusted` dispatches for them (`cbv2_segment_write_strided`)
/// does not run. `PagedLayerCache.appendTrustedDirect` reserves the rows' destinations with the segment write's
/// own checks and page reservation, the launch (`NemotronTargetQKV.applyWritingPaged`) stores each K/V value it
/// writes to its outputs at the segment write's addresses as well, binds the group's write fence and returns its
/// successor (fence + 1, as the segment write does), and the cache then advances the frontier, offsets and fence
/// as `appendTrusted` does. q, k and v come from the same body as the fused launch; the pages receive the same
/// bits at the same addresses. Used only after `qualify` matched. NEMOTRON_MTP_FRONT_KV_DIRECT=0 keeps the
/// segment write.
enum NemotronMTPFrontRowsKVDirect {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_MTP_FRONT_KV_DIRECT"] != "0"
    nonisolated(unsafe) static var verified = false

    /// `frontRows`' Q/K/V with the K/V rows appended to `cache` by the same launch: (q, k, v, the roots
    /// `appendTrusted` would return), or nil when nothing was launched or appended.
    static func apply(_ x: MLXArray, prepared: MLXArray?, mixer: NemotronHAttention, cache: PagedLayerCache)
        -> (q: MLXArray, k: MLXArray, v: MLXArray, written: [MLXArray])?
    {
        guard enabled, verified, NemotronMTPFrontRowsQKV.enabled, NemotronMTPFrontRowsQKV.verified,
              mixer.numHeads == 32, mixer.numKeyValueHeads == 2, mixer.headDim == 128,
              x.ndim == 3, x.dim(0) == 1, x.dim(1) == 3
        else { return nil }
        var launched: (MLXArray, MLXArray, MLXArray)? = nil
        let written = cache.appendTrustedDirect(3) { write in
            guard write.count == 3, write.kvHeads == 2, write.headDim == 128, write.pageSize == 16,
                  write.dtype == .bfloat16,
                  let out = NemotronTargetQKV.applyWritingPaged(
                      x, query: mixer.wq, key: mixer.wk, value: mixer.wv, prepared: prepared,
                      storage: write.storage, valueOffset: write.valueOffset, pages: write.pages,
                      slots: write.slots, fence: write.writeFence)
            else { return nil }
            launched = (out.0, out.1, out.2)
            return (out.1, out.2, out.3)
        }
        guard let written, let launched else { return nil }
        return (launched.0, launched.1, launched.2, written)
    }

    /// Load time, after `NemotronMTPFrontRowsQKV.qualify` and before the post-verify warm-up: on the head's
    /// weights and attention norm, seeded residual rows at R = 3 with the norm's prepared rows (as `frontRows`
    /// passes them) and without, the launch against the fused Q/K/V launch, bit for bit in q, k and v; on seeded
    /// paged segments with scattered local pages and rows within one page or split across two, every stored
    /// element equal to the k / v row it belongs to and every other element of the segment unchanged; and the
    /// returned fence equal to the bound one plus one. Engages only if every case matches; an MLX error keeps the
    /// segment write.
    static func qualify(_ block: NemotronH35MTPAttentionBlock) {
        verified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron mtp front kv direct: \(message)\n".utf8))
        }
        guard enabled else { return report("off (setting)") }
        guard NemotronMTPFrontRowsQKV.verified else { return report("off (the fused front Q/K/V launch is off)") }
        let mixer = block.mixer
        // (seed, prepared rows, (page, slot) of each row) with 12-page segments; never local page 0.
        let cases: [(Int, Bool, [(Int32, Int32)])] = [
            (0, true, [(5, 14), (5, 15), (9, 0)]),
            (1, true, [(3, 0), (3, 1), (3, 2)]),
            (2, false, [(11, 13), (11, 14), (11, 15)]),
            (3, true, [(1, 15), (7, 0), (7, 1)]),
        ]
        let segmentPages = 12
        let valueOffset = segmentPages * 2 * 16 * 128
        var checked = 0
        do {
            let matched = try withError { () -> Bool in
                for (seed, joined, destination) in cases {
                    let key = MLXRandom.key(UInt64(0x9f80 + seed))
                    let fused = (MLXRandom.normal([1, 3, 2688], key: key) * Float(1 + seed)).asType(.bfloat16)
                    var (x, prepared): (MLXArray, MLXArray?) = (block.norm(fused), nil)
                    if joined, NemotronMTPDraftJoins.enabled,
                       let pair = NemotronHResidualRMSNorm.normPreparedRows(fused, norm: block.norm) {
                        (x, prepared) = pair
                    }
                    // Two arrays from the same key: the segment the launch writes and its pristine twin.
                    let storageKey = MLXRandom.key(UInt64(0x9f90 + seed))
                    func segment() -> MLXArray {
                        MLXRandom.normal([2, segmentPages, 2, 16, 128], dtype: .bfloat16, key: storageKey)
                    }
                    let storage = segment()
                    let pristine = segment()
                    let fence = MLXArray([Int32(40 + seed)])
                    eval(storage, pristine, fence)
                    guard let expected = NemotronTargetQKV.apply(
                              x, query: mixer.wq, key: mixer.wk, value: mixer.wv, captureMTP: true,
                              prepared: prepared),
                          let got = NemotronTargetQKV.applyWritingPaged(
                              x, query: mixer.wq, key: mixer.wk, value: mixer.wv, prepared: prepared,
                              storage: storage, valueOffset: valueOffset, pages: destination.map { $0.0 },
                              slots: destination.map { $0.1 }, fence: fence)
                    else { return false }
                    eval(got.0, got.1, got.2, got.3, expected.0, expected.1, expected.2)
                    var same = true
                    for (a, b) in [(got.0, expected.0), (got.1, expected.1), (got.2, expected.2)] {
                        same = same && a.shape == b.shape && a.dtype == b.dtype
                            && (a.view(dtype: .uint16) .== b.view(dtype: .uint16)).all().item(Bool.self)
                    }
                    same = same && got.3.shape == [1] && got.3.item(Int32.self) == Int32(41 + seed)
                    // The segment as the segment write leaves it: each row's K / V values at its slot.
                    var want = pristine.view(dtype: .uint16).asArray(UInt16.self)
                    let keys = expected.1.view(dtype: .uint16).asArray(UInt16.self)
                    let values = expected.2.view(dtype: .uint16).asArray(UInt16.self)
                    for (r, (page, slot)) in destination.enumerated() {
                        for head in 0 ..< 2 {
                            for d in 0 ..< 128 {
                                let at = ((Int(page) * 2 + head) * 16 + Int(slot)) * 128 + d
                                want[at] = keys[r * 256 + head * 128 + d]
                                want[valueOffset + at] = values[r * 256 + head * 128 + d]
                            }
                        }
                    }
                    let have = storage.view(dtype: .uint16).asArray(UInt16.self)
                    same = same && have == want
                    if !same { return false }
                    checked += 1
                }
                return true
            }
            verified = matched
        } catch {
            return report("off (\(error))")
        }
        report(verified
            ? "on (\(checked) cases: q, k, v bitwise equal to the fused launch, segment bytes equal to the segment write, fence + 1)"
            : "off (differs after \(checked) cases)")
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

// MARK: - Draft attention reading the head cache pages in place

/// The draft attention launch (`NemotronMTPDraftSDPA`) reading the head cache's keys and values from its paged
/// storage in place (`PagedSequenceKV.directRead`) instead of from `snapshot()`'s gathered copy, so the gather
/// launch (`cbv2_segment_read_output`) is not run. Key i of the history is slot (first_slot + i) % PS of local page
/// pages[(first_slot + i) / PS], read at the segment read's element addresses; everything else is the launch's
/// text (same key order per SIMD group, scores, mask, online-softmax updates, one-barrier aggregation, own row).
/// The group's write fence is bound to the launch (the read orders after every prior write, as the gather's
/// does), and one element of the output is folded back into it (`publishDirectRead`), so every later write orders
/// after the read. Used only after `qualifyPaged` found it bitwise equal to the contiguous launch on seeded paged
/// storage. NEMOTRON_MTP_DRAFT_SDPA_PAGED=0 keeps the gather and the contiguous launch.
extension NemotronMTPDraftSDPA {
    static let pagedEnabled = ProcessInfo.processInfo.environment["NEMOTRON_MTP_DRAFT_SDPA_PAGED"] != "0"
    nonisolated(unsafe) private(set) static var pagedVerified = false

    private static let pagedSource: String? = {
        var s = source
        let swaps: [(String, String)] = [
            ("const int L = keys_shape[2];\n",
             "const int L = prm[0];\nconst int first_slot = prm[1];\nconst size_t vbase = size_t(uint(prm[2]));\n"),
            ("const int gqa_factor = int(threadgroups_per_grid.x) / keys_shape[1];\n",
             "const int gqa_factor = int(threadgroups_per_grid.x) / KVH;\n"),
            ("int inner_k_stride = BN * int(keys_strides[2]);\nint inner_v_stride = BN * int(values_strides[2]);\n", ""),
            ("keys += kv_head_idx * keys_strides[1] + simd_gid * keys_strides[2] + simd_lid * qk_per_thread;\n"
                + "values += kv_head_idx * values_strides[1] + simd_gid * values_strides[2] + simd_lid * v_per_thread;\n",
             "// Key i: slot (first_slot + i) % PS of local page pages[(first_slot + i) / PS], the segment read's\n"
                + "// element address; the value plane starts vbase elements later.\n"
                + "auto nmp_at = [&](int key) -> size_t {\n"
                + "  const int off = first_slot + key;\n"
                + "  return ((size_t(pages[off / PS]) * KVH + kv_head_idx) * PS + size_t(off % PS)) * D;\n"
                + "};\n"
                + "const device T* kst = storage + simd_lid * qk_per_thread;\n"
                + "const device T* vst = storage + vbase + simd_lid * v_per_thread;\n"),
            ("    const device T* kp = own ? own_kp : keys + u * inner_k_stride;\n"
                + "    const device T* vp = own ? own_vp : values + u * inner_v_stride;\n",
             "    const size_t at = own ? 0 : nmp_at(i + u * BN);\n"
                + "    const device T* kp = own ? own_kp : kst + at;\n"
                + "    const device T* vp = own ? own_vp : vst + at;\n"),
            ("  keys += 4 * inner_k_stride;\n  values += 4 * inner_v_stride;\n", ""),
            ("    const device T* kp = (OWN && i == L) ? own_kp : keys;\n"
                + "    const device T* vp = (OWN && i == L) ? own_vp : values;\n",
             "    const size_t at = (OWN && i == L) ? 0 : nmp_at(i);\n"
                + "    const device T* kp = (OWN && i == L) ? own_kp : kst + at;\n"
                + "    const device T* vp = (OWN && i == L) ? own_vp : vst + at;\n"),
            ("  keys += inner_k_stride;\n  values += inner_v_stride;\n", ""),
        ]
        for (old, new) in swaps {
            guard s.components(separatedBy: old).count == 2 else { return nil }
            s = s.replacingOccurrences(of: old, with: new)
        }
        guard !s.contains("keys_"), !s.contains("values_"), !s.contains("inner_") else { return nil }
        return s
    }()
    private static func pagedSource(own: Bool) -> String? {
        pagedSource.map { s in
            own ? s
                .replacingOccurrences(of: "OWN_KEYS", with:
                    "own_keys + kv_head_idx * own_keys_strides[1] + simd_lid * qk_per_thread")
                .replacingOccurrences(of: "OWN_VALUES", with:
                    "own_values + kv_head_idx * own_values_strides[1] + simd_lid * v_per_thread")
                : s
                .replacingOccurrences(of: "OWN_KEYS", with: "nullptr")
                .replacingOccurrences(of: "OWN_VALUES", with: "nullptr")
        }
    }
    private static let pagedKernel = pagedSource(own: false).map {
        MLXFast.metalKernel(
            name: "nemotron_mtp_draft_sdpa_paged_u4_1b",
            inputNames: ["queries", "storage", "pages", "prm", "fmask", "scale", "fence"], outputNames: ["out"],
            source: $0, ensureRowContiguous: false)
    }
    private static let pagedOwnKernel = pagedSource(own: true).map {
        MLXFast.metalKernel(
            name: "nemotron_mtp_draft_sdpa_paged_own_u4_1b",
            inputNames: ["queries", "storage", "pages", "prm", "fmask", "scale", "fence", "own_keys", "own_values"],
            outputNames: ["out"], source: $0, ensureRowContiguous: false)
    }

    // NEMOTRON_MTP_DRAFT_SDPA_FENCE: each paged launch's twin that also binds the group's live write fence (`live`,
    // last input) and stores its successor as a second output, `fence_out = live + 1` from one thread before the
    // body runs (the gather's read-launch fence). The body text is unchanged.
    static let fenceEnabled = ProcessInfo.processInfo.environment["NEMOTRON_MTP_DRAFT_SDPA_FENCE"] != "0"
    nonisolated(unsafe) private(set) static var fenceVerified = false
    static let fenceStore = "if (thread_position_in_grid.x == 0) {\n  fence_out[0] = live[0] + 1;\n}\n"
    private static let pagedFencedKernel = pagedSource(own: false).map {
        MLXFast.metalKernel(
            name: "nemotron_mtp_draft_sdpa_paged_fe_u4_1b",
            inputNames: ["queries", "storage", "pages", "prm", "fmask", "scale", "fence", "live"],
            outputNames: ["out", "fence_out"], source: fenceStore + $0, ensureRowContiguous: false)
    }
    private static let pagedOwnFencedKernel = pagedSource(own: true).map {
        MLXFast.metalKernel(
            name: "nemotron_mtp_draft_sdpa_paged_own_fe_u4_1b",
            inputNames: ["queries", "storage", "pages", "prm", "fmask", "scale", "fence", "own_keys", "own_values",
                         "live"],
            outputNames: ["out", "fence_out"], source: fenceStore + $0, ensureRowContiguous: false)
    }

    /// `row.directRead()` when the paged launch is qualified and serves its range (BF16, 2 x 128 heads, 16-slot
    /// pages, fewer than 1023 keys so the second draft's own row still stays below 1024), else nil (the caller
    /// keeps `snapshot()` and the contiguous launch).
    static func directRead(_ row: PagedSequenceKV) -> CBv2PagedDirectRead? {
        guard pagedEnabled, pagedVerified, verified, let read = row.directRead() else { return nil }
        return servesPaged(read) ? read : nil
    }

    private static func servesPaged(_ read: CBv2PagedDirectRead) -> Bool {
        read.dtype == .bfloat16 && read.kvHeads == 2 && read.headDim == 128 && read.pageSize == 16
            && (1 ..< 1023).contains(read.count) && read.firstSlot >= 0 && read.firstSlot < 16
            && read.valueOffset > 0 && read.valueOffset < Int(Int32.max)
            && read.storage.dtype == .bfloat16 && read.storage.ndim == 5
            && read.storage.shape == [2, read.storage.dim(1), 2, 16, 128]
            && read.valueOffset == read.storage.dim(1) * 2 * 16 * 128
            && read.pages.dtype == .int32 && read.pages.ndim == 1
            && read.pages.dim(0) == (read.firstSlot + read.count + 15) / 16
    }

    /// The attention of `attend` over the history `read` names (and the `own` row as key `read.count`), read from
    /// the pages in place; the read's back-edge is published on `row`. Nil outside the qualified shapes.
    static func attendPaged(
        queries: MLXArray, read: CBv2PagedDirectRead, row: PagedSequenceKV?,
        own: (keys: MLXArray, values: MLXArray)? = nil, scale: Float, mask: MLXArray
    ) -> MLXArray? {
        guard pagedVerified else { return nil }
        if let row, fenceEnabled, fenceVerified {
            // The read's back-edge from the launch itself: it binds the group's live fence (after an earlier read
            // of the same operands, that read's successor, not `read.writeFence`) and its successor becomes the
            // group's fence.
            let live = row.liveWriteFence
            guard let outputs = pagedOutputs(
                queries: queries, read: read, own: own, scale: scale, mask: mask, live: live) else { return nil }
            row.publishDirectReadFence(outputs[1], after: live)
            return outputs[0]
        }
        guard let out = launchPaged(
            queries: queries, read: read, own: own, scale: scale, mask: mask) else { return nil }
        row?.publishDirectRead(out)
        return out
    }

    private static func launchPaged(
        queries: MLXArray, read: CBv2PagedDirectRead, own: (keys: MLXArray, values: MLXArray)?,
        scale: Float, mask: MLXArray
    ) -> MLXArray? {
        pagedOutputs(queries: queries, read: read, own: own, scale: scale, mask: mask, live: nil)?[0]
    }

    /// The paged launch's outputs: [out], or with `live` (a [1] int32 fence) its fenced twin's [out, live + 1].
    private static func pagedOutputs(
        queries: MLXArray, read: CBv2PagedDirectRead, own: (keys: MLXArray, values: MLXArray)?,
        scale: Float, mask: MLXArray, live: MLXArray?
    ) -> [MLXArray]? {
        let n = read.count + (own == nil ? 0 : 1)
        guard pagedEnabled, Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              servesPaged(read), queries.shape == [1, 32, 1, 128], queries.dtype == .bfloat16,
              (1 ..< 1024).contains(n), mask.shape == [1, 1, 1, n], mask.dtype == .bfloat16,
              live.map({ $0.shape == [1] && $0.dtype == .int32 }) ?? true,
              let kernel = live == nil
                ? (own == nil ? pagedKernel : pagedOwnKernel)
                : (own == nil ? pagedFencedKernel : pagedOwnFencedKernel)
        else { return nil }
        if let own {
            guard own.keys.shape == [1, 2, 1, 128], own.values.shape == [1, 2, 1, 128],
                  own.keys.dtype == .bfloat16, own.values.dtype == .bfloat16,
                  own.keys.strides[3] == 1, own.values.strides[3] == 1
            else { return nil }
        }
        let prm = MLXArray([Int32(read.count), Int32(read.firstSlot), Int32(read.valueOffset)])
        let inputs = [queries, read.storage, read.pages, prm, mask, scaleArray(scale), read.writeFence]
            + (own.map { [$0.keys, $0.values] } ?? []) + (live.map { [$0] } ?? [])
        return kernel(
            inputs, template: [("T", DType.bfloat16), ("D", 128), ("V", 128), ("OWN", own != nil),
                               ("KVH", 2), ("PS", 16)],
            grid: (32 * 1024, 1, 1), threadGroup: (1024, 1, 1),
            outputShapes: [[1, 32, 1, 128]] + (live == nil ? [] : [[1]]),
            outputDTypes: [.bfloat16] + (live == nil ? [] : [.int32]))
    }

    /// Load time, after `qualify`: the paged launch against the contiguous launch on the same keys, bit for bit,
    /// on seeded paged storage laid out as the head cache's segment ([2, pages, 2, 16, 128] BF16): scattered local
    /// page lists, first slots 0 ... 15, key counts across the 32-key, four-key-step and 16-slot page boundaries up to
    /// 1022, query and key scales 1 to 16, masks as the drafts build them plus scattered -inf keys, with and
    /// without the own row. The contiguous launch reads the gathered copy of the same elements. Engages only if
    /// every output is bitwise equal; an MLX error keeps the gather.
    static func qualifyPaged(_ block: NemotronH35MTPAttentionBlock) {
        pagedVerified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron mtp draft sdpa paged: \(message)\n".utf8))
        }
        guard pagedEnabled else { return report("off (setting)") }
        guard verified else { return report("off (the contiguous draft launch is off)") }
        guard pagedKernel != nil, pagedOwnKernel != nil else { return report("off (the launch text moved)") }
        let scale = block.mixer.scale
        let counts = [1, 2, 15, 16, 17, 31, 32, 33, 64, 97, 127, 128, 129, 255, 256, 257, 488, 489, 552, 600, 616,
                      617, 1000, 1021, 1022]
        let inputScales: [(Float, Float)] = [(1, 1), (4, 4), (16, 8)]
        var checked = 0
        do {
            let matches = try withError { () -> Bool in
                for own in [false, true] {
                    for (i, count) in counts.enumerated() {
                        let n = count + (own ? 1 : 0)
                        guard n < 1024 else { continue }
                        let seed = UInt64(0xd5da_0000 + (own ? 4096 : 0) + i)
                        let keys5 = MLXRandom.split(key: MLXRandom.key(seed), into: 7)
                        let (qScale, kvScale) = inputScales[i % inputScales.count]
                        let firstSlot = (i * 5) % 16
                        let pageCount = (firstSlot + count + 15) / 16
                        // A segment with spare pages; the range's pages are a scattered permutation (never page 0).
                        let segmentPages = pageCount + 9
                        let storage = MLXRandom.normal(
                            [2, segmentPages, 2, 16, 128], dtype: .bfloat16, scale: kvScale, key: keys5[0])
                        let order = argSort(MLXRandom.uniform(Float(0) ..< 1, [segmentPages - 1], key: keys5[1]))
                        let local = (order[..<pageCount] + Int32(1)).asType(.int32)
                        let read = CBv2PagedDirectRead(
                            storage: storage, valueOffset: segmentPages * 2 * 16 * 128, pages: local,
                            firstSlot: firstSlot, count: count, kvHeads: 2, headDim: 128, pageSize: 16,
                            dtype: .bfloat16, offset: count, writeFence: MLXArray.zeros([1], dtype: .int32))
                        // The gathered copy of the same elements: [1, 2, count, 128] per plane.
                        func gathered(_ plane: Int) -> MLXArray {
                            take(storage[plane], local, axis: 0)
                                .transposed(1, 0, 2, 3)
                                .reshaped(2, pageCount * 16, 128)[0..., firstSlot ..< (firstSlot + count), 0...]
                                .expandedDimensions(axis: 0)
                        }
                        let keys = MLX.contiguous(gathered(0))
                        let values = MLX.contiguous(gathered(1))
                        let q = MLXRandom.normal([1, 4096], dtype: .bfloat16, scale: qScale, key: keys5[2])
                        let queries = q.reshaped(1, 1, 32, 128).transposed(0, 2, 1, 3)
                        let ownRow: (keys: MLXArray, values: MLXArray)? = own
                            ? (MLXRandom.normal([1, 256], dtype: .bfloat16, scale: kvScale, key: keys5[3])
                                .reshaped(1, 1, 2, 128).transposed(0, 2, 1, 3),
                               MLXRandom.normal([1, 256], dtype: .bfloat16, scale: kvScale, key: keys5[4])
                                .reshaped(1, 1, 2, 128).transposed(0, 2, 1, 3))
                            : nil
                        let tail = min(i % 4, count - 1)
                        let position = MLXArray(Int32(0) ..< Int32(n))
                        var keep = position .< Int32(count - tail)
                        if own { keep = keep .|| (position .== Int32(count)) }
                        if i % 5 == 4 {
                            let drop = MLXRandom.uniform(Float(0) ..< 1, [n], key: keys5[5]) .< MLXArray(Float(0.2))
                            keep = keep .&& (logicalNot(drop) .|| (position .== Int32(0)))
                        }
                        let mask = which(keep, MLXArray(Float(0)), MLXArray(-Float.infinity))
                            .asType(.bfloat16).reshaped(1, 1, 1, n)
                        guard let paged = launchPaged(
                                  queries: queries, read: read, own: ownRow, scale: scale, mask: mask),
                              let expected = attend(
                                  queries: queries, keys: keys, values: values, own: ownRow, scale: scale, mask: mask)
                        else { return false }
                        if !(paged.view(dtype: .uint16) .== expected.view(dtype: .uint16)).all().item(Bool.self) {
                            return false
                        }
                        checked += 1
                    }
                }
                return true
            }
            pagedVerified = matches
        } catch {
            return report("off (\(error))")
        }
        report(pagedVerified
            ? "on (\(checked) cases bitwise equal to the contiguous launch on the gathered pages)"
            : "off (differs from the contiguous launch after \(checked) cases)")
    }
}

// MARK: - Draft attention walking the acceptance packet itself

/// The paged draft attention launches with the device accept walk folded in, so the row pick
/// (`NemotronMTPDraftRowPick`: walk, both masks, the picked fused / query / key / value rows) is not launched.
/// Each launch runs the row pick's integer walk over the same acceptance packet (`a`, the count of leading
/// confirmed drafts), reads the first draft's query from the post-verify query rows at row `a` (the element the
/// row pick copied), and forms each key's mask value from `a` with the masks kernel's expression
/// (`key <= a + firstSpeculated`, and for the second draft's launch also its own row `key == L`) instead of
/// reading the mask array: the same T(0) / T(-inf) values reach the same statements. Everything else is the paged
/// launch's text. The join after the first draft walks the same way (`NemotronHResidualRMSNorm.applyWalk`).
/// Used only after `qualifyWalk` matched the row pick plus the paged launches and the join, bit for bit.
/// NEMOTRON_MTP_DRAFT_WALK_FOLD=0 keeps the row pick.
extension NemotronMTPDraftSDPA {
    static let walkEnabled = ProcessInfo.processInfo.environment["NEMOTRON_MTP_DRAFT_WALK_FOLD"] != "0"
    nonisolated(unsafe) private(set) static var walkVerified = false

    private static func walkSource(own: Bool) -> String? {
        guard var s = pagedSource(own: own) else { return nil }
        let keep = own ? "key <= nmw_last || key == L" : "key <= nmw_last"
        let swaps: [(String, String)] = [
            ("const size_t vbase = size_t(uint(prm[2]));\n",
             "const size_t vbase = size_t(uint(prm[2]));\n"
                + "// The row pick's walk: the count of leading packet[targets + a] == packet[drafts + a], a < count.\n"
                + "int nmw_a = 0;\n"
                + "{\n"
                + "  const int nmw_count = prm[3];\n"
                + "  const int nmw_drafts = prm[4];\n"
                + "  const int nmw_targets = prm[5];\n"
                + "  const int64_t nmw_step = packet_strides[0];\n"
                + "  while (nmw_a < nmw_count && packet[(nmw_targets + nmw_a) * nmw_step] == packet[(nmw_drafts + nmw_a) * nmw_step]) {\n"
                + "    ++nmw_a;\n"
                + "  }\n"
                + "}\n"
                + "const int nmw_last = nmw_a + prm[6];\n"
                + "auto nmw_keep = [&](int key) -> bool { return \(keep); };\n"),
            ("const int mask_kv_seq_stride = fmask_shape[3] > 1 ? int(fmask_strides[3]) : 0;\n", ""),
            // The first draft's query is post-verify row a; the second draft's is its own single row.
            ("+ simd_lid * qk_per_thread * queries_strides[3];\n",
             own ? "+ simd_lid * qk_per_thread * queries_strides[3];\n"
                 : "+ simd_lid * qk_per_thread * queries_strides[3] + nmw_a * queries_strides[2];\n"),
            ("auto fm = fmask + simd_gid * mask_kv_seq_stride;\n", ""),
            ("mk[u] = fm[u * BN * mask_kv_seq_stride];\n",
             "mk[u] = nmw_keep(i + u * BN) ? T(0.0f) : T(-INFINITY);\n"),
            ("fm += 4 * BN * mask_kv_seq_stride;\n", ""),
            ("bool use_key = (fm[0] >= Limits<T>::finite_min);\n",
             "const T nmw_mk = nmw_keep(i) ? T(0.0f) : T(-INFINITY);\nbool use_key = (nmw_mk >= Limits<T>::finite_min);\n"),
            ("score += static_cast<U>(fm[0]);\n", "score += static_cast<U>(nmw_mk);\n"),
            ("fm += BN * mask_kv_seq_stride;\n", ""),
        ]
        for (old, new) in swaps {
            guard s.components(separatedBy: old).count == 2 else { return nil }
            s = s.replacingOccurrences(of: old, with: new)
        }
        guard !s.contains("fmask"), !s.contains("fm["), !s.contains("fm +="), !s.contains("mask_kv_seq_stride")
        else { return nil }
        return s
    }
    private static let walkKernel = walkSource(own: false).map {
        MLXFast.metalKernel(
            name: "nemotron_mtp_draft_sdpa_paged_walk_u4_1b",
            inputNames: ["queries", "storage", "pages", "prm", "packet", "scale", "fence"], outputNames: ["out"],
            source: $0, ensureRowContiguous: false)
    }
    private static let walkOwnKernel = walkSource(own: true).map {
        MLXFast.metalKernel(
            name: "nemotron_mtp_draft_sdpa_paged_own_walk_u4_1b",
            inputNames: ["queries", "storage", "pages", "prm", "packet", "scale", "fence", "own_keys", "own_values"],
            outputNames: ["out"], source: $0, ensureRowContiguous: false)
    }
    // NEMOTRON_MTP_DRAFT_SDPA_FENCE: the walk launches' fenced twins (as the paged launches' above).
    private static let walkFencedKernel = walkSource(own: false).map {
        MLXFast.metalKernel(
            name: "nemotron_mtp_draft_sdpa_paged_walk_fe_u4_1b",
            inputNames: ["queries", "storage", "pages", "prm", "packet", "scale", "fence", "live"],
            outputNames: ["out", "fence_out"], source: fenceStore + $0, ensureRowContiguous: false)
    }
    private static let walkOwnFencedKernel = walkSource(own: true).map {
        MLXFast.metalKernel(
            name: "nemotron_mtp_draft_sdpa_paged_own_walk_fe_u4_1b",
            inputNames: ["queries", "storage", "pages", "prm", "packet", "scale", "fence", "own_keys", "own_values",
                         "live"],
            outputNames: ["out", "fence_out"], source: fenceStore + $0, ensureRowContiguous: false)
    }

    /// Whether the walk launches serve a round: qualified, the walk's packet 1-D int32 holding both ranges, the
    /// query rows [1, 32, W, 128] BF16 with W > count, and `read` a served paged range of at least W keys.
    static func walkServes(queryRows: MLXArray, packet: MLXArray, count: Int, drafts: Int, targets: Int,
                           read: CBv2PagedDirectRead) -> Bool {
        walkEnabled && walkVerified && pagedVerified && servesPaged(read)
            && queryRows.ndim == 4 && queryRows.dim(0) == 1 && queryRows.dim(1) == 32 && queryRows.dim(3) == 128
            && queryRows.dtype == .bfloat16 && count >= 0 && count < queryRows.dim(2) && read.count >= queryRows.dim(2)
            && packet.ndim == 1 && packet.dtype == .int32 && drafts >= 0 && targets >= 0
            && drafts + count <= packet.dim(0) && targets + count <= packet.dim(0)
    }

    /// The first draft's attention (`attendPaged` of the row the walk selects, under its first mask) or, with
    /// `own`, the second draft's (`queries` its own query row, under the second mask), read from the pages in
    /// place; the read's back-edge is published on `row`. `firstSpeculated`: the history position of post-verify
    /// row 0. Nil outside the qualified shapes.
    static func attendPagedWalk(
        queries: MLXArray, packet: MLXArray, count: Int, drafts: Int, targets: Int, firstSpeculated: Int,
        read: CBv2PagedDirectRead, row: PagedSequenceKV?, own: (keys: MLXArray, values: MLXArray)? = nil,
        scale: Float
    ) -> MLXArray? {
        guard walkEnabled, walkVerified else { return nil }
        if let row, fenceEnabled, fenceVerified {
            // As in `attendPaged`: the launch binds the group's live fence and its successor becomes the group's.
            let live = row.liveWriteFence
            guard let outputs = walkOutputs(
                queries: queries, packet: packet, count: count, drafts: drafts, targets: targets,
                firstSpeculated: firstSpeculated, read: read, own: own, scale: scale, live: live)
            else { return nil }
            row.publishDirectReadFence(outputs[1], after: live)
            return outputs[0]
        }
        guard let out = launchPagedWalk(
            queries: queries, packet: packet, count: count, drafts: drafts, targets: targets,
            firstSpeculated: firstSpeculated, read: read, own: own, scale: scale) else { return nil }
        row?.publishDirectRead(out)
        return out
    }

    private static func launchPagedWalk(
        queries: MLXArray, packet: MLXArray, count: Int, drafts: Int, targets: Int, firstSpeculated: Int,
        read: CBv2PagedDirectRead, own: (keys: MLXArray, values: MLXArray)?, scale: Float
    ) -> MLXArray? {
        walkOutputs(queries: queries, packet: packet, count: count, drafts: drafts, targets: targets,
                    firstSpeculated: firstSpeculated, read: read, own: own, scale: scale, live: nil)?[0]
    }

    /// The walk launch's outputs: [out], or with `live` (a [1] int32 fence) its fenced twin's [out, live + 1].
    private static func walkOutputs(
        queries: MLXArray, packet: MLXArray, count: Int, drafts: Int, targets: Int, firstSpeculated: Int,
        read: CBv2PagedDirectRead, own: (keys: MLXArray, values: MLXArray)?, scale: Float, live: MLXArray?
    ) -> [MLXArray]? {
        let n = read.count + (own == nil ? 0 : 1)
        guard walkEnabled, Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              servesPaged(read), (1 ..< 1024).contains(n),
              queries.ndim == 4, queries.dim(0) == 1, queries.dim(1) == 32, queries.dim(3) == 128,
              queries.dtype == .bfloat16, queries.dim(2) > (own == nil ? count : 0),
              packet.ndim == 1, packet.dtype == .int32, count >= 0, drafts >= 0, targets >= 0,
              drafts + count <= packet.dim(0), targets + count <= packet.dim(0),
              firstSpeculated >= 0, firstSpeculated + count < read.count,
              live.map({ $0.shape == [1] && $0.dtype == .int32 }) ?? true,
              let kernel = live == nil
                ? (own == nil ? walkKernel : walkOwnKernel)
                : (own == nil ? walkFencedKernel : walkOwnFencedKernel)
        else { return nil }
        if let own {
            guard queries.dim(2) == 1, own.keys.shape == [1, 2, 1, 128], own.values.shape == [1, 2, 1, 128],
                  own.keys.dtype == .bfloat16, own.values.dtype == .bfloat16,
                  own.keys.strides[3] == 1, own.values.strides[3] == 1
            else { return nil }
        }
        let prm = MLXArray([Int32(read.count), Int32(read.firstSlot), Int32(read.valueOffset), Int32(count),
                            Int32(drafts), Int32(targets), Int32(firstSpeculated)])
        let inputs = [queries, read.storage, read.pages, prm, packet, scaleArray(scale), read.writeFence]
            + (own.map { [$0.keys, $0.values] } ?? []) + (live.map { [$0] } ?? [])
        return kernel(
            inputs, template: [("T", DType.bfloat16), ("D", 128), ("V", 128), ("OWN", own != nil),
                               ("KVH", 2), ("PS", 16)],
            grid: (32 * 1024, 1, 1), threadGroup: (1024, 1, 1),
            outputShapes: [[1, 32, 1, 128]] + (live == nil ? [] : [[1]]),
            outputDTypes: [.bfloat16] + (live == nil ? [] : [.int32]))
    }

    /// Load time, after `qualifyPaged`: each walk launch against the composition it replaces, bit for bit. The
    /// row pick (`NemotronMTPDraftRowPick.pick`, forced; its own masks + gathers when it is off) on the same
    /// packet, then the paged launch with the picked query row and first mask, the paged launch with the
    /// second draft's own query and K/V row under the second mask, and the join of the picked fused row
    /// (`NemotronHResidualRMSNorm.apply`) against `applyWalk` with the head's norm. Every accept count 0...2 of a
    /// depth-2 packet, in both argument spaces (6 elements: constant, 11: device), on seeded paged storage as in
    /// `qualifyPaged` (scattered local pages, first slots 0 ... 15, key counts across the 32-key, four-key-step
    /// and page boundaries), with the post-verify layouts (the Q/K/V rows as transposed views). Engages only if
    /// every output is bitwise equal; an MLX error keeps the row pick. Also compiles every pipeline.
    static func qualifyWalk(_ block: NemotronH35MTPAttentionBlock, norm: RMSNorm) {
        walkVerified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron mtp draft walk fold: \(message)\n".utf8))
        }
        guard walkEnabled else { return report("off (setting)") }
        guard pagedVerified else { return report("off (the paged draft launch is off)") }
        guard walkKernel != nil, walkOwnKernel != nil, NemotronHResidualRMSNorm.walkKernelAvailable
        else { return report("off (the launch text moved)") }
        let scale = block.mixer.scale
        let counts = [3, 17, 33, 97, 129, 488, 600, 1000, 1021]
        let width = 3
        var checked = 0
        func bits(_ a: MLXArray) -> MLXArray { a.view(dtype: a.itemSize == 2 ? .uint16 : .uint32) }
        do {
            let matched = try withError { () -> Bool in
                for (i, count) in counts.enumerated() {
                    let keys7 = MLXRandom.split(key: MLXRandom.key(UInt64(0xd5db_0000 + i)), into: 9)
                    let firstSlot = (i * 5 + 3) % 16
                    let pageCount = (firstSlot + count + 15) / 16
                    let segmentPages = pageCount + 7
                    let storage = MLXRandom.normal(
                        [2, segmentPages, 2, 16, 128], dtype: .bfloat16, scale: Float(1 + i % 3), key: keys7[0])
                    let order = argSort(MLXRandom.uniform(Float(0) ..< 1, [segmentPages - 1], key: keys7[1]))
                    let local = (order[..<pageCount] + Int32(1)).asType(.int32)
                    let read = CBv2PagedDirectRead(
                        storage: storage, valueOffset: segmentPages * 2 * 16 * 128, pages: local,
                        firstSlot: firstSlot, count: count, kvHeads: 2, headDim: 128, pageSize: 16,
                        dtype: .bfloat16, offset: count, writeFence: MLXArray.zeros([1], dtype: .int32))
                    // The post-verify rows as `frontRows` lays them out, and the second draft's own rows.
                    let fused = MLXRandom.normal([1, width, 2688], key: keys7[2]).asType(.bfloat16)
                    let queries = MLXRandom.normal([1, width, 4096], dtype: .bfloat16, scale: Float(1 + i % 4),
                                                   key: keys7[3])
                        .reshaped(1, width, 32, 128).transposed(0, 2, 1, 3)
                    let rowKeys = MLXRandom.normal([1, width, 256], dtype: .bfloat16, key: keys7[4])
                        .reshaped(1, width, 2, 128).transposed(0, 2, 1, 3)
                    let rowValues = MLXRandom.normal([1, width, 256], dtype: .bfloat16, key: keys7[5])
                        .reshaped(1, width, 2, 128).transposed(0, 2, 1, 3)
                    let ownQueries = MLXRandom.normal([1, 4096], dtype: .bfloat16, scale: 2, key: keys7[6])
                        .reshaped(1, 1, 32, 128).transposed(0, 2, 1, 3)
                    let own = (keys: MLXRandom.normal([1, 256], dtype: .bfloat16, key: keys7[7])
                                .reshaped(1, 1, 2, 128).transposed(0, 2, 1, 3),
                               values: MLXRandom.normal([1, 256], dtype: .bfloat16, key: keys7[8])
                                .reshaped(1, 1, 2, 128).transposed(0, 2, 1, 3))
                    let attention = (MLXRandom.normal([1, 1, 2688], key: keys7[2]) * 0.5).asType(.bfloat16)
                    let firstSpeculated = count - width
                    for size in [6, 11] {
                        for accepted in 0 ... 2 {
                            // drafts at 0..<2, targets at 2..<5; the rest is filler.
                            var host = [Int32](repeating: 7, count: size)
                            host[0] = 11; host[1] = 12
                            host[2] = accepted >= 1 ? 11 : 21
                            host[3] = accepted >= 2 ? 12 : 22
                            host[4] = 13
                            let packet = MLXArray(host)
                            let walk = CBv2MTPDeviceAcceptWalk.Operands(
                                packet: packet, count: 2, drafts: 0, targets: 2)
                            // The composition the walk launches replace.
                            let row = MLXArray([Int32(accepted)])
                            let picked = NemotronMTPDraftRowPick.pick(
                                walk: walk, fused: fused, queries: queries, keys: rowKeys, values: rowValues,
                                firstSpeculated: firstSpeculated, length: count, dtype: .bfloat16, force: true)
                            let position = MLXArray(Int32(0) ..< Int32(count + 1))
                            let firstMask = picked?.first ?? which(
                                position[..<count] .<= Int32(accepted + firstSpeculated),
                                MLXArray(Float(0)), MLXArray(-Float.infinity)).asType(.bfloat16)
                                .reshaped(1, 1, 1, count)
                            let secondMask = picked?.second ?? which(
                                (position .<= Int32(accepted + firstSpeculated)) .|| (position .== Int32(count)),
                                MLXArray(Float(0)), MLXArray(-Float.infinity)).asType(.bfloat16)
                                .reshaped(1, 1, 1, count + 1)
                            let rowQuery = picked?.queries ?? take(queries, row, axis: 2)
                            let rowFused = picked?.fused ?? take(fused, row, axis: 1)
                            guard let first = launchPaged(
                                      queries: rowQuery, read: read, own: nil, scale: scale, mask: firstMask),
                                  let second = launchPaged(
                                      queries: ownQueries, read: read, own: own, scale: scale, mask: secondMask),
                                  let joined = NemotronHResidualRMSNorm.apply(rowFused, attention, norm: norm),
                                  let walkedFirst = launchPagedWalk(
                                      queries: queries, packet: packet, count: 2, drafts: 0, targets: 2,
                                      firstSpeculated: firstSpeculated, read: read, own: nil, scale: scale),
                                  let walkedSecond = launchPagedWalk(
                                      queries: ownQueries, packet: packet, count: 2, drafts: 0, targets: 2,
                                      firstSpeculated: firstSpeculated, read: read, own: own, scale: scale),
                                  let walkedJoin = NemotronHResidualRMSNorm.applyWalk(
                                      fused, attention, packet: packet, count: 2, drafts: 0, targets: 2, norm: norm)
                            else { return false }
                            var same = MLXArray(true)
                            for (a, b) in [(walkedFirst, first), (walkedSecond, second), (walkedJoin.0, joined.0),
                                           (walkedJoin.1, joined.1)] {
                                guard a.shape == b.shape, a.dtype == b.dtype else { return false }
                                same = MLX.logicalAnd(same, all(bits(a) .== bits(b)))
                            }
                            if !same.item(Bool.self) { return false }
                            checked += 1
                        }
                    }
                }
                return true
            }
            walkVerified = matched
        } catch {
            return report("off (\(error))")
        }
        report(walkVerified
            ? "on (\(checked) cases: both draft attentions and the join bitwise equal to the row pick + paged launches + join, accept counts 0...2, packets of 6 and 11)"
            : "off (differs after \(checked) cases)")
    }
}

// MARK: - Draft attention publishing its read fence itself

extension NemotronMTPDraftSDPA {
    /// Load time, after `qualifyPaged` and `qualifyWalk`: every fenced twin (NEMOTRON_MTP_DRAFT_SDPA_FENCE) against
    /// its launch, the attention bit for bit and the emitted fence equal to the bound live fence plus one. Seeded
    /// paged storage as in `qualifyPaged` (scattered local pages, first slots, key counts across the 32-key,
    /// four-key-step and page boundaries), with and without the own row; the walk twins, when the walk fold
    /// qualified, on depth-2 packets of 6 and 11 elements at every accept count 0...2. Engages only if every case
    /// matches; an MLX error keeps `publishDirectRead`. Also compiles every fenced pipeline.
    static func qualifyFence(_ block: NemotronH35MTPAttentionBlock) {
        fenceVerified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron mtp draft sdpa fence: \(message)\n".utf8))
        }
        guard fenceEnabled else { return report("off (setting)") }
        guard pagedVerified else { return report("off (the paged draft launch is off)") }
        guard pagedFencedKernel != nil, pagedOwnFencedKernel != nil, walkFencedKernel != nil,
              walkOwnFencedKernel != nil
        else { return report("off (the launch text moved)") }
        let scale = block.mixer.scale
        let counts = [3, 33, 129, 600, 1021]
        var checked = 0
        do {
            let matched = try withError { () -> Bool in
                func same(_ got: [MLXArray]?, _ expected: MLXArray?, live: Int32) -> Bool {
                    guard let got, got.count == 2, let expected, got[0].shape == expected.shape,
                          got[0].dtype == expected.dtype, got[1].shape == [1], got[1].dtype == .int32
                    else { return false }
                    eval(got[0], got[1], expected)
                    return (got[0].view(dtype: .uint16) .== expected.view(dtype: .uint16)).all().item(Bool.self)
                        && got[1].item(Int32.self) == live + 1
                }
                for (i, count) in counts.enumerated() {
                    let keys8 = MLXRandom.split(key: MLXRandom.key(UInt64(0xd5dc_0000 + i)), into: 8)
                    let firstSlot = (i * 7 + 2) % 16
                    let pageCount = (firstSlot + count + 15) / 16
                    let segmentPages = pageCount + 5
                    let storage = MLXRandom.normal(
                        [2, segmentPages, 2, 16, 128], dtype: .bfloat16, scale: Float(1 + i % 3), key: keys8[0])
                    let order = argSort(MLXRandom.uniform(Float(0) ..< 1, [segmentPages - 1], key: keys8[1]))
                    let local = (order[..<pageCount] + Int32(1)).asType(.int32)
                    let read = CBv2PagedDirectRead(
                        storage: storage, valueOffset: segmentPages * 2 * 16 * 128, pages: local,
                        firstSlot: firstSlot, count: count, kvHeads: 2, headDim: 128, pageSize: 16,
                        dtype: .bfloat16, offset: count, writeFence: MLXArray.zeros([1], dtype: .int32))
                    let query = MLXRandom.normal([1, 4096], dtype: .bfloat16, scale: 2, key: keys8[2])
                        .reshaped(1, 1, 32, 128).transposed(0, 2, 1, 3)
                    let queryRows = MLXRandom.normal([1, 3, 4096], dtype: .bfloat16, scale: 2, key: keys8[3])
                        .reshaped(1, 3, 32, 128).transposed(0, 2, 1, 3)
                    let ownRow = (keys: MLXRandom.normal([1, 256], dtype: .bfloat16, key: keys8[4])
                                    .reshaped(1, 1, 2, 128).transposed(0, 2, 1, 3),
                                  values: MLXRandom.normal([1, 256], dtype: .bfloat16, key: keys8[5])
                                    .reshaped(1, 1, 2, 128).transposed(0, 2, 1, 3))
                    for own in [false, true] {
                        let n = count + (own ? 1 : 0)
                        let tail = min(i % 3, count - 1)
                        let position = MLXArray(Int32(0) ..< Int32(n))
                        var keep = position .< Int32(count - tail)
                        if own { keep = keep .|| (position .== Int32(count)) }
                        let mask = which(keep, MLXArray(Float(0)), MLXArray(-Float.infinity))
                            .asType(.bfloat16).reshaped(1, 1, 1, n)
                        let ownKV: (keys: MLXArray, values: MLXArray)? = own ? ownRow : nil
                        let live = Int32(100 + checked)
                        guard same(
                            pagedOutputs(queries: query, read: read, own: ownKV, scale: scale, mask: mask,
                                         live: MLXArray([live])),
                            launchPaged(queries: query, read: read, own: ownKV, scale: scale, mask: mask),
                            live: live)
                        else { return false }
                        checked += 1
                        guard walkVerified else { continue }
                        for size in [6, 11] {
                            for accepted in 0 ... 2 {
                                var host = [Int32](repeating: 7, count: size)
                                host[0] = 11; host[1] = 12
                                host[2] = accepted >= 1 ? 11 : 21
                                host[3] = accepted >= 2 ? 12 : 22
                                host[4] = 13
                                let packet = MLXArray(host)
                                let queries = own ? query : queryRows
                                let walkLive = Int32(100 + checked)
                                guard same(
                                    walkOutputs(queries: queries, packet: packet, count: 2, drafts: 0, targets: 2,
                                                firstSpeculated: count - 3, read: read, own: ownKV, scale: scale,
                                                live: MLXArray([walkLive])),
                                    launchPagedWalk(queries: queries, packet: packet, count: 2, drafts: 0,
                                                    targets: 2, firstSpeculated: count - 3, read: read, own: ownKV,
                                                    scale: scale),
                                    live: walkLive)
                                else { return false }
                                checked += 1
                            }
                        }
                    }
                }
                return true
            }
            fenceVerified = matched
        } catch {
            return report("off (\(error))")
        }
        report(fenceVerified
            ? "on (\(checked) cases: attention bitwise equal to the unfenced launches, emitted fence = live + 1; paged\(walkVerified ? " and walk" : "") forms)"
            : "off (differs after \(checked) cases)")
    }
}

/// The draft head's MoE block norm and router, for `NemotronMTPJoinRouter.qualifyDraft`.
func nemotronDraftTailNormGate(_ head: Module) -> (RMSNorm, NemotronHMoEGate)? {
    guard let block = head.modules().compactMap({ $0 as? NemotronH35MTPExpertBlock }).first,
          type(of: block.norm) == RMSNorm.self else { return nil }
    return (block.norm, block.mixer.gate)
}
