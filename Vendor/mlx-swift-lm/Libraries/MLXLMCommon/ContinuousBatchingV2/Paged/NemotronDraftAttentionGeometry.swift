import Foundation
import MLX
import MLXRandom

/// Stage cuts in the draft head's per-round chain. Each one removes a
/// dependent dispatch and leaves every draft value bitwise unchanged; the
/// target never reaches these paths.
/// NEMOTRON_MTP_DRAFT_CHAIN_CLEANUP=0 disables both.
public enum NemotronMTPDraftChainCleanup {
    static let environment = ProcessInfo.processInfo.environment
    static let enabled = environment["NEMOTRON_MTP_DRAFT_CHAIN_CLEANUP"] != "0"
    /// NEMOTRON_MTP_DRAFT_RETAINED_SEED_ROW=0: the prestaged first draft
    /// rewrites its seed K/V row again.
    public static let retainedSeedRow = enabled
        && environment["NEMOTRON_MTP_DRAFT_RETAINED_SEED_ROW"] != "0"
    /// NEMOTRON_MTP_DRAFT_HOST_OFFSETS=0: the draft cache advances its
    /// device position offsets per attention call again.
    static let hostPositionOffsets = enabled
        && environment["NEMOTRON_MTP_DRAFT_HOST_OFFSETS"] != "0"
}

/// Only changes launch geometry for the embedded assistant. All other caches
/// and shapes retain the original selectors. NSG stays four so HPT alone
/// cannot change the per-head SIMD or partition reduction tree.
public enum NemotronDraftAttentionGeometry {
    struct Plan {
        let hpt: Int
        let ptok: Int
        var schedule = NemotronDraftAttentionSchedule.Variant()
        var isStock: Bool { hpt == NemotronDraftAttentionGeometry.selectedHPT && schedule.isStock }
        var stockTwin: Plan { Plan(hpt: NemotronDraftAttentionGeometry.selectedHPT, ptok: ptok) }
    }
    static let environment = ProcessInfo.processInfo.environment
    static let enabled = environment["NEMOTRON_MTP_DRAFT_HPT"] != "0"
    // Production choice: HPT4, NSG4, main15 partition target64 policy.
    // NEMOTRON_MTP_DRAFT_HPT=0 restores the original dispatch geometry.
    static let selectedHPT = 4
    static let selectedPTOK = 0
    static func select(enabled requested: Bool, dtype: DType, queryHeads: Int,
                       kvHeads: Int, headDim: Int, pageSize: Int,
                       length: Int, batch: Int) -> Plan? {
        guard requested, enabled, dtype == .bfloat16, queryHeads == 32,
              kvHeads == 2, headDim == 128, pageSize == 16 else { return nil }
        let stock = plan(hpt: selectedHPT, fixed: selectedPTOK, length: length, batch: batch)
        // exp35: the qualified schedule keeps the stock PTOK selection.
        guard let active = NemotronDraftAttentionSchedule.active else { return stock }
        return Plan(hpt: active.hpt, ptok: stock.ptok, schedule: active.variant)
    }
    static func plan(hpt: Int, fixed: Int, length: Int, batch: Int) -> Plan {
        Plan(hpt: hpt, ptok: fixed == 256 ? 256 : PagedAttentionKernel.partitionTokensForDispatch(
            maxAttendLength: length, batch: batch, kvHeads: 2,
            headSplits: 2, pageSize: 16))
    }
    /// Warm bounded part/merge signatures on throwaway state before the final
    /// cache drain. Neither model inputs nor request state are retained.
    public static func warm() throws {
        guard enabled else { return }
        let kind = CBv2LayerKind(attention: .full, headDim: 128, kvHeads: 2, queryHeads: 32)
        let backend = try PagedKVBackend(layerKinds: [kind], config: .init(
            capacityBytes: 8 << 20, dtype: .bfloat16, maxPrefillChunk: 512,
            nominalMaxSequenceLength: 2048, segmentSizeBytes: 8 << 20))
        let states = try backend.makeSequenceState(layerKinds: [kind], promptLength: 0, maxLength: 2048)
        let row = states[0] as! PagedSequenceKV
        let group = backend.pool.group(row.groupKey)
        defer { backend.release(states) }
        while row.absoluteOffset < 1400 {
            let count = min(512, 1400 - row.absoluteOffset)
            // Seeded random K/V: the schedule qualification below reads them.
            let keys = MLXRandom.normal([2, count, 128], key: MLXRandom.key(UInt64(350 + row.absoluteOffset)))
                .asType(.bfloat16)
            let values = MLXRandom.normal([2, count, 128], key: MLXRandom.key(UInt64(360 + row.absoluteOffset)))
                .asType(.bfloat16)
            row.write(keys: keys, values: values)
            eval(group.writeFence)
        }
        NemotronDraftAttentionSchedule.qualifyOnce(row: row, group: group,
            source: backend.pool.kernelSource)
        let q = MLXArray.ones([1, 32, 128], dtype: .bfloat16)
        let k = MLXArray.zeros([1, 2, 128], dtype: .bfloat16)
        let params = MLXArray([Float(1), Float(1 / sqrt(128.0)), 0, 0, 0, 0, 0, 0])
        for ptok in [64, 128, 256] {
            for write in [true, false] {
                let descriptor = PagedSegmentDispatchPlan.Row(pages: row.table,
                    info: row.seqInfoRow(attending: (start: 0, length: 1400),
                        writeTarget: write ? row.prepareDecodeWrite() : nil))
                let out = PagedSegmentAttention.decode(queries: q,
                    newKeys: write ? k : nil, newValues: write ? k : nil,
                    group: group, rows: [descriptor], sinks: nil, params: params,
                    softcap: false, source: backend.pool.kernelSource,
                    geometryOverride: Plan(hpt: selectedHPT, ptok: ptok))
                eval(out, group.writeFence)
                if let active = NemotronDraftAttentionSchedule.active {
                    let warmed = PagedSegmentAttention.decode(queries: q,
                        newKeys: write ? k : nil, newValues: write ? k : nil,
                        group: group, rows: [descriptor], sinks: nil, params: params,
                        softcap: false, source: backend.pool.kernelSource,
                        geometryOverride: Plan(hpt: active.hpt, ptok: ptok, schedule: active.variant))
                    eval(warmed, group.writeFence)
                }
            }
        }
    }
}
