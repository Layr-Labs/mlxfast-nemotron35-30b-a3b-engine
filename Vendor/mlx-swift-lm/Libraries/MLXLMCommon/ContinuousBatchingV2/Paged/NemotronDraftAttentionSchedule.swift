import Foundation
import MLX
import MLXFast
import MLXRandom

/// Schedule of the drafter's paged-attention part kernel (exp35). Two
/// switchable pieces on top of the 4-query-head drafter geometry; neither
/// changes any per-head arithmetic, so every output bit equals the stock
/// HPT 4 / NSG 4 kernel's, and a load-time bitwise qualification against the
/// stock kernel gates them (any mismatch falls back to stock).
///
/// 1. `NEMOTRON_MTP_DRAFT_ATTN_PSG` (default 4 = off; `8` opts in): PSG
///    physical SIMD groups per threadgroup carry the NSG = 4 LOGICAL token
///    chains. Physical group s runs chain s % 4 (tokens lo+s%4, lo+s%4+4, ...)
///    for HPT*4/PSG heads starting at head (s / 4) * HPT*4/PSG. The per-head
///    dot (32 lanes x 4 elements), the online softmax, the four logical states
///    and their ascending merge are those of the stock kernel. Off by default:
///    the M1 read it 20-27 % slower at 9-10 partitions. `hpt2` = the pre-probe
///    (stock kernel at HPT 2, PTOK selection unchanged); `neg` = PSG 8 with a
///    negative control (the second head pair reads the wrong query).
/// 2. `NEMOTRON_MTP_DRAFT_ATTN_QREG` (default 1): each lane keeps its scaled
///    query values in registers (no q_smem staging barrier), and the page of a
///    token is resolved once per page visit instead of once per token.
///    `q` = registers only, `p` = page only, `0` = off, `neg` = unscaled
///    queries (negative control).
public enum NemotronDraftAttentionSchedule {
    struct Variant: Equatable {
        var psg = 4
        var qreg = false
        var hoist = false
        var neg = 0
        var isStock: Bool { psg == 4 && !qreg && !hoist && neg == 0 }
        var tag: String { "p\(psg)q\(qreg ? 1 : 0)h\(hoist ? 1 : 0)n\(neg)" }
    }

    static let environment = ProcessInfo.processInfo.environment
    static let requestedHPT = environment["NEMOTRON_MTP_DRAFT_ATTN_PSG"] == "hpt2" ? 2 : 4
    static let requested: Variant = {
        var v = Variant()
        switch environment["NEMOTRON_MTP_DRAFT_ATTN_PSG"] {
        case "8": v.psg = 8
        case "neg": v.psg = 8; v.neg = 1
        default: v.psg = 4
        }
        switch environment["NEMOTRON_MTP_DRAFT_ATTN_QREG"] ?? "1" {
        case "0": break
        case "q": v.qreg = true
        case "p": v.hoist = true
        case "neg": v.qreg = true; v.hoist = true; v.neg = 2
        default: v.qreg = true; v.hoist = true
        }
        return v
    }()
    static let shadowPath = environment["NEMOTRON_MTP_DRAFT_ATTN_SHADOW"]

    /// The qualified candidate (hpt, variant); nil = stock. Published once,
    /// by the first load's qualification, under a lock.
    private static let stateLock = NSLock()
    nonisolated(unsafe) private static var activeState: (hpt: Int, variant: Variant)?
    nonisolated(unsafe) private static var qualificationDone = false
    static var active: (hpt: Int, variant: Variant)? { stateLock.withLock { activeState } }

    // MARK: kernel

    static let header = """

        namespace x35 {
        using namespace metal;
        template <
            typename T, int D, int S, int GQA, int HPT, int NSG, int PSG, bool QREG, bool HOIST,
            int NEG, int PTOK, bool HAS_WRITE, bool HAS_SOFTCAP, typename Cache>
        inline void draft_part(
            const device T* q, const device T* knew, const device T* vnew,
            thread const Cache& cache, const device int32_t* seqinfo,
            const device float* params, const int kvh, const int maxpart,
            threadgroup float* q_smem, threadgroup float* red_smem,
            device float* partials, device float* meta,
            uint3 tgpig, uint3 tpitg, uint sgitg, uint lane
        ) {
            static_assert(GQA % HPT == 0, "HPT must divide GQA");
            static_assert(PSG % NSG == 0, "PSG must be a multiple of NSG");
            constexpr int ROUNDS = PSG / NSG;
            static_assert(HPT % ROUNDS == 0, "heads must split evenly");
            constexpr int HPS = HPT / ROUNDS; // heads per physical simdgroup
            constexpr int SPLITS = GQA / HPT;
            constexpr int EPT = D / 32;
            constexpr int TG = 32 * PSG;
            constexpr int RSTRIDE = D + 2;

            const int kv_head = tgpig.x / SPLITS;
            const int hgroup = tgpig.x % SPLITS;
            const int b = tgpig.y;
            const int part = tgpig.z;
            const int lin = tpitg.x;
            const device int32_t* info = seqinfo + b * 8;
            const int attend_start = info[0];
            const int attend_len = info[1];
            const int table_len = info[2];
            const int lo = part * PTOK;
            if (lo >= attend_len) {
                return;
            }
            const int hi = min(attend_len, lo + PTOK);
            const float softcap = params[0];
            const float scale = params[1];

            // Fused decode write: the stock single-writer rule, same bytes.
            const size_t tile_src = ((size_t)b * (size_t)kvh + (size_t)kv_head) * D;
            if (HAS_WRITE && hgroup == 0 && part == (attend_len - 1) / PTOK) {
                const auto page = cache.write_page(info[3]);
                const size_t dst = ((size_t)kv_head * S + (size_t)info[4]) * D;
                device T* kdst = const_cast<device T*>(page.key) + dst;
                device T* vdst = const_cast<device T*>(page.value) + dst;
                for (int i = lin; i < D; i += TG) {
                    kdst[i] = knew[tile_src + i];
                    vdst[i] = vnew[tile_src + i];
                }
            }

            // Logical chain and head range of this physical simdgroup.
            const int chain = (int)sgitg % NSG;
            const int hbase = ((int)sgitg / NSG) * HPS;
            const int q_head0 = kv_head * GQA + hgroup * HPT;
            const device T* q_base = q + ((size_t)b * (size_t)(kvh * GQA) + (size_t)q_head0) * D;
            int qhead[HPS];
        #pragma unroll
            for (int g = 0; g < HPS; g++) {
                qhead[g] = (NEG == 1 && (int)sgitg >= NSG) ? ((hbase + g) ^ 1) : (hbase + g);
            }
            float qr[HPS][EPT];
            if (QREG) {
        #pragma unroll
                for (int g = 0; g < HPS; g++) {
        #pragma unroll
                    for (int e = 0; e < EPT; e++) {
                        const float qv = float(q_base[qhead[g] * D + (int)lane * EPT + e]);
                        qr[g][e] = NEG == 2 ? qv : qv * scale;
                    }
                }
            } else {
                for (int i = lin; i < HPT * D; i += TG) {
                    q_smem[i] = float(q_base[i]) * scale;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            float m[HPS];
            float l[HPS];
            float acc[HPS][EPT];
        #pragma unroll
            for (int g = 0; g < HPS; g++) {
                m[g] = -FLT_MAX;
                l[g] = 0.0f;
        #pragma unroll
                for (int e = 0; e < EPT; e++) {
                    acc[g][e] = 0.0f;
                }
            }

            float kf[EPT];
            float vf[EPT];
            int cached = -1;
            cbv2::PagedKVPage<T> pg = {knew, vnew};
            for (int i = lo + chain; i < hi; i += NSG) {
                if (HAS_WRITE && i == attend_len - 1) {
                    cbv2::load_row<T, EPT>(knew + tile_src, lane, kf);
                    cbv2::load_row<T, EPT>(vnew + tile_src, lane, vf);
                } else {
                    const int pos = attend_start + i;
                    const int lpage = pos / S;
                    const int slot = pos % S;
                    if (!HOIST || lpage != cached) {
                        pg = cache.read_page(seqinfo, lpage, table_len);
                        cached = lpage;
                    }
                    const size_t base = ((size_t)kv_head * S + (size_t)slot) * D;
                    cbv2::load_row<T, EPT>(pg.key + base, lane, kf);
                    cbv2::load_row<T, EPT>(pg.value + base, lane, vf);
                }

        #pragma unroll
                for (int g = 0; g < HPS; g++) {
                    float partial = 0.0f;
        #pragma unroll
                    for (int e = 0; e < EPT; e++) {
                        const float qv = QREG ? qr[g][e] : q_smem[qhead[g] * D + (int)lane * EPT + e];
                        partial += qv * kf[e];
                    }
                    float qk = simd_sum(partial);
                    if (HAS_SOFTCAP) {
                        qk = softcap * precise::tanh(qk / softcap);
                    }
                    qk *= M_LOG2E_F;
                    const float m_new = max(m[g], qk);
                    const float corr = exp2(m[g] - m_new);
                    const float p = exp2(qk - m_new);
                    l[g] = l[g] * corr + p;
        #pragma unroll
                    for (int e = 0; e < EPT; e++) {
                        acc[g][e] = acc[g][e] * corr + p * vf[e];
                    }
                    m[g] = m_new;
                }
            }

            // Publish into the LOGICAL record (chain, head): the stock layout.
        #pragma unroll
            for (int g = 0; g < HPS; g++) {
                threadgroup float* mine = red_smem + (size_t)(chain * HPT + hbase + g) * RSTRIDE;
        #pragma unroll
                for (int e = 0; e < EPT; e++) {
                    mine[lane * EPT + e] = acc[g][e];
                }
                if (lane == 0) {
                    mine[D] = m[g];
                    mine[D + 1] = l[g];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Stock merge: head g by physical group g, logical states s ascending.
            for (int g = sgitg; g < HPT; g += PSG) {
                float pm = -FLT_MAX;
                for (int s = 0; s < NSG; s++) {
                    pm = max(pm, red_smem[(size_t)(s * HPT + g) * RSTRIDE + D]);
                }
                float pl = 0.0f;
                for (int s = 0; s < NSG; s++) {
                    const float ms = red_smem[(size_t)(s * HPT + g) * RSTRIDE + D];
                    const float ls = red_smem[(size_t)(s * HPT + g) * RSTRIDE + D + 1];
                    pl += ls * exp2(ms - pm);
                }
                const size_t slot =
                    ((size_t)b * (size_t)(kvh * GQA) + (size_t)(q_head0 + g)) * (size_t)maxpart
                    + (size_t)part;
                device float* prow = partials + slot * D;
                for (int e = (int)lane; e < D; e += 32) {
                    float v = 0.0f;
                    for (int s = 0; s < NSG; s++) {
                        const float ms = red_smem[(size_t)(s * HPT + g) * RSTRIDE + D];
                        v += red_smem[(size_t)(s * HPT + g) * RSTRIDE + e] * exp2(ms - pm);
                    }
                    prow[e] = v;
                }
                if (lane == 0) {
                    meta[slot * 2] = pm;
                    meta[slot * 2 + 1] = pl;
                }
            }
        }
        } // namespace x35

        """

    static func body(bindings: Int) -> String {
        let pointers = (0 ..< bindings).map { "segment\($0)" }.joined(separator: ", ")
        return """
            const device int32_t* record = records + threadgroup_position_in_grid.y * STRIDE;
            const cbv2::PagedSegmentAccessor<T, SEGMENTS> cache{
                {\(pointers)}, value_offsets, record, (size_t)KVH * S * D};
            const uint3 position(threadgroup_position_in_grid.x, record[0], record[1]);
            if (thread_position_in_grid.x == 0 && thread_position_in_grid.y == 0) {
                fence[0] = previous[0] + 1;
            }
            threadgroup float q_smem[QREG ? 4 : HPT * D];
            threadgroup float red_smem[NSG * HPT * (D + 2)];
            x35::draft_part<T, D, S, GQA, HPT, NSG, PSG, QREG, HOIST, NEG, PTOK, HAS_WRITE, HAS_SOFTCAP>(
                q, knew, vnew, cache, seqinfo, params, KVH, record[6],
                q_smem, red_smem, partials, meta,
                position, thread_position_in_threadgroup, simdgroup_index_in_threadgroup,
                thread_index_in_simdgroup);
            """
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var kernels: [String: MLXFast.MLXFastKernel] = [:]

    static func kernel(name: String, bindings: Int, hasWrite: Bool, source: String)
        -> MLXFast.MLXFastKernel
    {
        lock.withLock {
            if let existing = kernels[name] { return existing }
            let made = MLXFast.metalKernel(
                name: name, inputNames: PagedSegmentAttention.inputNames(bindings: bindings),
                outputNames: ["fence"], source: body(bindings: bindings),
                header: source + header, ensureRowContiguous: true,
                mutableInputs: ["partials", "meta"]
                    + (hasWrite ? (0 ..< bindings).map { "segment\($0)" } : []))
            kernels[name] = made
            return made
        }
    }

    // MARK: bitwise comparison

    struct Outputs { let out: MLXArray; let partials: MLXArray; let meta: MLXArray }

    static func bits(_ a: MLXArray) -> [UInt32] {
        a.asType(.float32).asArray(Float.self).map(\.bitPattern)
    }

    /// (partials equal, meta equal, out equal)
    static func compare(_ a: Outputs, _ b: Outputs, check: () throws -> Void) rethrows
        -> (Bool, Bool, Bool)
    {
        eval(a.out, a.partials, a.meta, b.out, b.partials, b.meta)
        try check()
        return (bits(a.partials) == bits(b.partials), bits(a.meta) == bits(b.meta),
                bits(a.out) == bits(b.out))
    }

    private static let shadowLock = NSLock()
    nonisolated(unsafe) private static var shadowCalls = 0
    nonisolated(unsafe) private static var shadowMismatches = 0

    /// Gate-only: every production call also runs the stock kernel and
    /// compares partials, meta and output bit for bit (host sync per call).
    static func shadowRecord(_ stock: Outputs, _ candidate: Outputs, describe: String) {
        guard let path = shadowPath else { return }
        let (p, m, o) = compare(stock, candidate, check: {})
        shadowLock.withLock {
            shadowCalls += 1
            if !(p && m && o) { shadowMismatches += 1 }
            let line = "call=\(shadowCalls) \(describe) partials=\(p) meta=\(m) out=\(o) "
                + "mismatches=\(shadowMismatches)\n"
            if let handle = FileHandle(forWritingAtPath: path) {
                handle.seekToEndOfFile()
                handle.write(line.data(using: .utf8)!)
                try? handle.close()
            } else {
                try? line.write(toFile: path, atomically: true, encoding: .utf8)
            }
        }
    }

    // MARK: load-time qualification

    /// Compares the requested candidate against the stock kernel on seeded
    /// random K/V, queries and new tiles at every partition-boundary length
    /// class, with and without the fused write, at each reachable PTOK, at
    /// B = 1 and B = 3; then checks the candidate's fused write by reading the
    /// written slot back through the stock kernel. Any mismatch keeps stock.
    /// Runs once per process; later loads keep the published result. MLX
    /// errors (a rejected compile or pipeline) are caught and keep stock.
    static func qualifyOnce(row: PagedSequenceKV, group: PagedKVGroup, source: String) {
        guard stateLock.withLock({ !qualificationDone }) else { return }
        let candidateHPT = requestedHPT
        let variant = requested
        var passed = false
        if candidateHPT != 4 || !variant.isStock {
            if variant.neg != 0 {
                passed = true // negative control: engaged unqualified
            } else {
                // An MLX error leaves the failed candidate as the group's
                // fence; restore the last evaluated one so later stock work
                // never consumes (and retries) it.
                let fence = group.writeFence
                do {
                    passed = try withError { box in
                        try qualify(row: row, group: group, source: source,
                                    hpt: candidateHPT, variant: variant) { try box.check() }
                    }
                } catch {
                    log("draft attention schedule \(variant.tag): MLX error \(error); using stock")
                    group.writeFence = fence
                    passed = false
                }
            }
        }
        stateLock.withLock {
            guard !qualificationDone else { return }
            qualificationDone = true
            activeState = passed ? (candidateHPT, variant) : nil
        }
    }

    private static func qualify(row: PagedSequenceKV, group: PagedKVGroup, source: String,
                                hpt candidateHPT: Int, variant: Variant,
                                check: () throws -> Void) throws -> Bool {
        let stockPlan = { (ptok: Int) in
            NemotronDraftAttentionGeometry.Plan(hpt: 4, ptok: ptok) }
        let candidatePlan = { (ptok: Int) in
            NemotronDraftAttentionGeometry.Plan(hpt: candidateHPT, ptok: ptok, schedule: variant) }
        let params = MLXArray([Float(0), Float(1 / sqrt(128.0)), 0, 0, 0, 0, 0, 0])
        let qOne = MLXRandom.normal([1, 32, 128], key: MLXRandom.key(351)).asType(.bfloat16)
        let qThree = MLXRandom.normal([3, 32, 128], key: MLXRandom.key(352)).asType(.bfloat16)
        let kNew = MLXRandom.normal([1, 2, 128], key: MLXRandom.key(353)).asType(.bfloat16)
        let vNew = MLXRandom.normal([1, 2, 128], key: MLXRandom.key(354)).asType(.bfloat16)
        let lengths = [1, 2, 3, 4, 5, 15, 16, 17, 31, 32, 33, 63, 64, 65, 66, 67, 127, 128, 129,
                       255, 256, 257, 491, 512, 555, 556, 557, 620, 1023, 1024, 1025, 1200, 1400]
        let available = row.absoluteOffset
        var cases = 0
        func run(_ q: MLXArray, _ rows: [PagedSegmentDispatchPlan.Row], write: Bool,
                 _ plan: NemotronDraftAttentionGeometry.Plan) -> Outputs {
            PagedSegmentAttention.decodeOutputs(queries: q, newKeys: write ? kNew : nil,
                newValues: write ? vNew : nil, group: group, rows: rows, sinks: nil,
                params: params, softcap: false, source: source, geometry: plan)
        }
        func descriptor(_ length: Int) -> PagedSegmentDispatchPlan.Row {
            PagedSegmentDispatchPlan.Row(pages: row.table,
                info: row.seqInfoRow(attending: (start: 0, length: length)))
        }
        for ptok in [64, 128, 256] {
            for length in lengths where length <= available {
                for write in [false, true] {
                    // A write call stores the new tile at the newest attended
                    // position (the dispatch plan's rule); stock and candidate
                    // store the same bytes there, so both see identical slabs.
                    let target = (page: row.table[(length - 1) / 16], slot: (length - 1) % 16)
                    let rows = write
                        ? [PagedSegmentDispatchPlan.Row(pages: row.table,
                            info: row.seqInfoRow(attending: (start: 0, length: length),
                                writeTarget: target))]
                        : [descriptor(length)]
                    let a = run(qOne, rows, write: write, stockPlan(ptok))
                    let b = run(qOne, rows, write: write, candidatePlan(ptok))
                    cases += 1
                    let (p, m, o) = try compare(a, b, check: check)
                    guard p && m && o else {
                        log("draft attention schedule \(variant.tag) hpt \(candidateHPT): "
                            + "MISMATCH ptok \(ptok) len \(length) write \(write) "
                            + "partials \(p) meta \(m) out \(o); using stock")
                        return false
                    }
                }
            }
            for trio in [[1, 64, 65], [555, 556, 557], [1023, 1024, 1025], [62, 63, 64]]
                where trio.allSatisfy({ $0 <= available }) {
                let rows = trio.map(descriptor)
                cases += 1
                let (p, m, o) = try compare(run(qThree, rows, write: false, stockPlan(ptok)),
                                            run(qThree, rows, write: false, candidatePlan(ptok)),
                                            check: check)
                guard p && m && o else {
                    log("draft attention schedule \(variant.tag): MISMATCH B=3 \(trio); using stock")
                    return false
                }
            }
        }
        // The candidate's fused write: write at the frontier, then attend
        // the written slot from the slab with the stock kernel.
        for ptok in [64, 256] {
            let target = row.prepareDecodeWrite()
            let length = row.absoluteOffset
            let written = run(qOne, [PagedSegmentDispatchPlan.Row(pages: row.table,
                info: row.seqInfoRow(attending: (start: 0, length: length), writeTarget: target))],
                write: true, candidatePlan(ptok))
            eval(written.out, group.writeFence)
            try check()
            let readBack = run(qOne, [descriptor(length)], write: false, stockPlan(ptok))
            cases += 1
            let (p, m, o) = try compare(written, readBack, check: check)
            guard p && m && o else {
                log("draft attention schedule \(variant.tag): fused write MISMATCH; using stock")
                return false
            }
        }
        log("draft attention schedule \(variant.tag) hpt \(candidateHPT): qualified "
            + "bitwise on \(cases) cases")
        return true
    }

    static func log(_ message: String) {
        FileHandle.standardError.write(("[nemotron-draft-attn] " + message + "\n").data(using: .utf8)!)
        if let path = environment["NEMOTRON_MTP_DRAFT_ATTN_LOG"] {
            try? (message + "\n").write(toFile: path, atomically: true, encoding: .utf8)
        }
    }
}
