import Foundation
import MLX
import MLXFast

/// A short-lived invocation, never a cache entry. `build` closes over only
/// immutable kernel functions and host dispatch geometry; every request array
/// is in `inputs`. A model can extend the graph before the cache commits fence.
public struct NemotronDeviceSecondAttentionPlan {
    public let inputs: [MLXArray]
    public let key: [Int]
    public let build: @Sendable ([MLXArray]) -> [MLXArray]
}

/// Cache-only metadata for an execution graph that produces Q/K/V itself.
/// `inputs` are [K slab, V slab, table, params, write fence, accepted, config].
/// A third-draft plan appends explicit previous K/V after those seven arrays.
/// Call `build` with [Q, own K, own V] followed by those seven arrays. No
/// placeholder query, activation, projection or K/V array is constructed.
public struct NemotronDeviceSecondFrontPlan {
    public let inputs: [MLXArray]
    public let key: [Int]
    public let build: @Sendable ([MLXArray]) -> [MLXArray]
}

/// Read-only, current-request attention for an assistant's device-picked second
/// draft. The newest K/V comes from explicit inputs, never from a slab write.
/// Private twins retain complete native math, with thread-local descriptors
/// and the single fused store disabled. No metadata kernel/buffer is needed.
enum NemotronDeviceSecondAttention {
    static let enabled = ProcessInfo.processInfo.environment[
        "NEMOTRON_MTP_DEVICE_SECOND_PAGED"] != "0"
    private static let lock = NSLock()
    nonisolated(unsafe) private static var kernels: [String: MLXFast.MLXFastKernel] = [:]

    /// Append private native-math twins whose descriptor lives in each
    /// thread. Numeric statements remain the complete original functions.
    private static func localDescriptorHeader(source: String, preceding: Bool = false) -> String? {
        func function(_ name: String) -> String? {
            guard let body = source.range(of: "inline void " + name + "("),
                  let start = source.range(of: "template <", options: .backwards,
                    range: source.startIndex..<body.lowerBound),
                  let end = source.range(of: "\n}\n", range: body.lowerBound..<source.endIndex)
            else { return nil }
            return String(source[start.lowerBound..<end.upperBound])
        }
        func swap(_ value: inout String, _ old: String, _ new: String) -> Bool {
            guard value.components(separatedBy: old).count == 2 else { return false }
            value = value.replacingOccurrences(of: old, with: new)
            return true
        }
        guard var part = function("paged_attention_part_cached_impl"),
              var merge = function("paged_attention_merge_impl"),
              swap(&part, "paged_attention_part_cached_impl", "nemotron_second_part_local_impl"),
              swap(&part, "const device int32_t* seqinfo,", "const thread int32_t* seqinfo,"),
              swap(&part, "const device int32_t* info = seqinfo + b * 8;",
                "const thread int32_t* info = seqinfo + b * 8;"),
              swap(&part, "if (HAS_WRITE && hgroup == 0 && part == (attend_len - 1) / PTOK) {",
                "if (false && HAS_WRITE && hgroup == 0 && part == (attend_len - 1) / PTOK) {"),
              swap(&merge, "paged_attention_merge_impl", "nemotron_second_merge_local_impl"),
              swap(&merge, "const device int32_t* seqinfo,", "const thread int32_t* seqinfo,")
        else { return nil }
        if preceding {
            // A third draft has two virtual rows: second's K/V precedes its
            // own. Substitute only that row's source address, retaining the
            // original per-position online-softmax math and reduction order.
            guard swap(&part, "const device T* vnew,", "const device T* vnew,\n    const device T* kprevious,\n    const device T* vprevious,"),
                  swap(&part, "load_row<T, EPT>(vnew + tile_src, lane, vf);\n        } else {",
                    "load_row<T, EPT>(vnew + tile_src, lane, vf);\n        } else if (i == attend_len - 2) {\n            load_row<T, EPT>(kprevious + tile_src, lane, kf);\n            load_row<T, EPT>(vprevious + tile_src, lane, vf);\n        } else {")
            else { return nil }
        }
        return source + "\nnamespace cbv2 {\n" + part + merge + "\n}\n"
    }

    private static let localPartBody = """
        const int kvh = kcache_shape[1];
        const int maxp = tables_shape[1];
        const int maxpart = threadgroups_per_grid.z;
        const int n = clamp(int(accepted[0]), 0, int(config[1]) - 1);
        int local_info[8] = {0, config[0] + n + config[3], config[2], 0, 0, 0, 0, 0};
        const cbv2::PagedSlabAccessor<T> cache{kcache, vcache, (size_t)kvh * S * D};
        if (thread_position_in_grid.x == 0 && thread_position_in_grid.y == 0
            && thread_position_in_grid.z == 0) {
            fence[0] = wfence[0] + 1;
        }
        threadgroup float q_smem[HPT * D];
        threadgroup float red_smem[NSG * HPT * (D + 2)];
        const uint3 position(threadgroup_position_in_grid.x, 0, threadgroup_position_in_grid.z);
        cbv2::nemotron_second_part_local_impl<T, D, S, GQA, HPT, NSG, PTOK, true, HAS_SOFTCAP>(
            q, knew, vnew, cache, tables, local_info, params, kvh, maxp, maxpart,
            q_smem, red_smem, partials, meta, position,
            thread_position_in_threadgroup, simdgroup_index_in_threadgroup,
            thread_index_in_simdgroup);
        """

    private static let localMergeBody = """
        const int heads = partials_shape[1];
        const int maxpart = partials_shape[2];
        const int n = clamp(int(accepted[0]), 0, int(config[1]) - 1);
        int local_info[8] = {0, config[0] + n + config[3], config[2], 0, 0, 0, 0, 0};
        const uint3 position(threadgroup_position_in_grid.x, 0, 0);
        cbv2::nemotron_second_merge_local_impl<T, D, PTOK, HAS_SINKS>(
            partials, meta, local_info, sinks, heads, maxpart, out, position,
            thread_index_in_simdgroup);
        """

    private static func partKernel(source: String, hpt: Int, ptok: Int, preceding: Bool = false)
        -> MLXFast.MLXFastKernel?
    {
        let name = "cbv2_nemotron_second_draft_part_bf16_d128_s16_h\(hpt)_n4_t\(ptok)_inline_v2"
            + (preceding ? "_third" : "")
        return lock.withLock {
            if let old = kernels[name] { return old }
            // No descriptor allocation/launch: both passes derive their
            // identical local descriptor directly from explicit device inputs.
            guard let header = localDescriptorHeader(source: source, preceding: preceding) else { return nil }
            guard !preceding || localPartBody.components(separatedBy: "q, knew, vnew, cache,").count == 2
            else { return nil }
            let body = preceding ? localPartBody.replacingOccurrences(
                of: "q, knew, vnew, cache,", with: "q, knew, vnew, kprevious, vprevious, cache,") : localPartBody
            let made = MLXFast.metalKernel(name: name,
                inputNames: ["q", "knew", "vnew", "kcache", "vcache", "tables", "accepted", "config", "params", "wfence"]
                    + (preceding ? ["kprevious", "vprevious"] : []),
                outputNames: ["partials", "meta", "fence"],
                source: body, header: header, ensureRowContiguous: true)
            kernels[name] = made
            return made
        }
    }

    private static func mergeKernel(source: String, hpt: Int, ptok: Int)
        -> MLXFast.MLXFastKernel?
    {
        let name = "cbv2_nemotron_second_draft_merge_bf16_d128_h\(hpt)_t\(ptok)_inline_v2"
        return lock.withLock {
            if let old = kernels[name] { return old }
            guard let header = localDescriptorHeader(source: source) else { return nil }
            let made = MLXFast.metalKernel(name: name,
                inputNames: ["partials", "meta", "accepted", "config", "sinks", "previous"],
                outputNames: ["out"], source: localMergeBody,
                header: header, ensureRowContiguous: true)
            kernels[name] = made
            return made
        }
    }

    static func attend(
        queries: MLXArray, ownKeys: MLXArray, ownValues: MLXArray,
        accepted: MLXArray, trustedOffset: Int, postVerifyRows: Int, ownRowDelta: Int,
        kcache: MLXArray, vcache: MLXArray, tables: MLXArray,
        params: MLXArray, group: PagedKVGroup, source: String, graph: Graph,
        execution: ((NemotronDeviceSecondAttentionPlan) -> [MLXArray])?
    ) -> MLXArray? {
        let maximumLength = trustedOffset + postVerifyRows + ownRowDelta - 1
        let geometry = NemotronDraftAttentionGeometry.select(enabled: true,
            dtype: .bfloat16, queryHeads: 32, kvHeads: 2, headDim: 128,
            pageSize: 16, length: maximumLength, batch: 1)
        let hpt = geometry?.hpt ?? PagedAttentionKernel.headsPerThreadgroup(headDim: 128, gqa: 16)
        let ptok = geometry?.ptok ?? PagedAttentionKernel.partitionTokensForDispatch(
            maxAttendLength: maximumLength, batch: 1, kvHeads: 2,
            headSplits: 16 / hpt, pageSize: 16)
        guard let part = partKernel(source: source, hpt: hpt, ptok: ptok),
              let merge = mergeKernel(source: source, hpt: hpt, ptok: ptok) else { return nil }
        let maxParts = (maximumLength + ptok - 1) / ptok
        let config = MLXArray([Int32(trustedOffset), Int32(postVerifyRows),
            Int32(tables.dim(1)), Int32(ownRowDelta), 0, 0, 0, 0])
        let plan = Graph.plan([queries, ownKeys, ownValues, kcache, vcache,
                tables, params, group.writeFence, accepted, config],
            hpt: hpt, ptok: ptok, maxParts: maxParts, part: part, merge: merge)
        let result = execution.map { $0(plan) } ?? graph.run(plan)
        guard result.count == 2 else { return nil }
        // Pool state changes only here, OUTSIDE the stateless compiled graph.
        group.writeFence = result[1]
        return result[0]
    }

    /// Same native attention builder, but the caller supplies Q/K/V INSIDE
    /// its complete draft graph. This only prepares cache metadata and commits
    /// the returned read fence after execution; no cache mutation is traced.
    static func attendFromFront(
        accepted: MLXArray, trustedOffset: Int, postVerifyRows: Int,
        precedingKeys: MLXArray? = nil, precedingValues: MLXArray? = nil,
        kcache: MLXArray, vcache: MLXArray, tables: MLXArray,
        params: MLXArray, group: PagedKVGroup, source: String,
        execution: (NemotronDeviceSecondFrontPlan) -> [MLXArray]
    ) -> MLXArray? {
        let preceding = precedingKeys != nil && precedingValues != nil
        let ownRowDelta = preceding ? 3 : 2
        let maximumLength = trustedOffset + postVerifyRows + ownRowDelta - 1
        let geometry = NemotronDraftAttentionGeometry.select(enabled: true,
            dtype: .bfloat16, queryHeads: 32, kvHeads: 2, headDim: 128,
            pageSize: 16, length: maximumLength, batch: 1)
        let hpt = geometry?.hpt ?? PagedAttentionKernel.headsPerThreadgroup(headDim: 128, gqa: 16)
        let ptok = geometry?.ptok ?? PagedAttentionKernel.partitionTokensForDispatch(
            maxAttendLength: maximumLength, batch: 1, kvHeads: 2,
            headSplits: 16 / hpt, pageSize: 16)
        if preceding {
            // Keep the original one-token partition choice for EVERY possible
            // accepted count. A boundary-crossing range uses host third draft.
            for n in 0..<postVerifyRows {
                let length = trustedOffset + n + 3
                let exact = NemotronDraftAttentionGeometry.select(enabled: true,
                    dtype: .bfloat16, queryHeads: 32, kvHeads: 2, headDim: 128,
                    pageSize: 16, length: length, batch: 1)
                let exactHPT = exact?.hpt ?? PagedAttentionKernel.headsPerThreadgroup(headDim: 128, gqa: 16)
                let exactPTOK = exact?.ptok ?? PagedAttentionKernel.partitionTokensForDispatch(
                    maxAttendLength: length, batch: 1, kvHeads: 2,
                    headSplits: 16 / exactHPT, pageSize: 16)
                guard exactHPT == hpt, exactPTOK == ptok else { return nil }
            }
        }
        guard let part = partKernel(source: source, hpt: hpt, ptok: ptok, preceding: preceding),
              let merge = mergeKernel(source: source, hpt: hpt, ptok: ptok) else { return nil }
        let maxParts = (maximumLength + ptok - 1) / ptok
        let config = MLXArray([Int32(trustedOffset), Int32(postVerifyRows),
            Int32(tables.dim(1)), Int32(ownRowDelta), 0, 0, 0, 0])
        var metadata = [kcache, vcache, tables, params, group.writeFence, accepted, config]
        if let precedingKeys, let precedingValues { metadata += [precedingKeys, precedingValues] }
        // Graph.plan's builder captures only kernel functions/host geometry.
        // Its input array list is deliberately never captured by that builder.
        let attention = Graph.plan(metadata, hpt: hpt, ptok: ptok,
            maxParts: maxParts, part: part, merge: merge, preceding: preceding)
        let plan = NemotronDeviceSecondFrontPlan(
            inputs: metadata, key: attention.key, build: attention.build)
        let result = execution(plan)
        guard result.count == 2 else { return nil }
        group.writeFence = result[1]
        return result[0]
    }

    /// One bounded compiled entry per request cache, no arrays in the entry.
    /// All mutable/current-request state is supplied as explicit graph inputs.
    final class Graph {
        private struct Entry {
            let key: [Int]
            let function: @Sendable ([MLXArray]) -> [MLXArray]
        }
        private let lock = NSLock()
        private var entry: Entry?
        private static let enabled = ProcessInfo.processInfo.environment[
            "NEMOTRON_MTP_DEVICE_SECOND_GRAPH"] != "0"

        static func plan(_ inputs: [MLXArray], hpt: Int, ptok: Int, maxParts: Int,
                         part: MLXFast.MLXFastKernel, merge: MLXFast.MLXFastKernel,
                         preceding: Bool = false)
            -> NemotronDeviceSecondAttentionPlan
        {
            let build: @Sendable ([MLXArray]) -> [MLXArray] = { arrays in
                let partInputs = [arrays[0], arrays[1], arrays[2], arrays[3], arrays[4],
                        arrays[5], arrays[8], arrays[9], arrays[6], arrays[7]]
                    + (preceding ? [arrays[10], arrays[11]] : [])
                let parts = part(partInputs,
                    template: [("T", DType.bfloat16), ("D", 128), ("S", 16),
                        ("GQA", 16), ("HPT", hpt), ("NSG", 4), ("PTOK", ptok),
                        ("HAS_SOFTCAP", false)],
                    grid: (2 * (16 / hpt) * 128, 1, maxParts), threadGroup: (128, 1, 1),
                    outputShapes: [[1, 32, maxParts, 128], [max(8, 32 * maxParts * 2)], [1]],
                    outputDTypes: [.float32, .float32, .int32])
                let sinks = MLXArray.zeros([32], dtype: .float32)
                let out = merge([parts[0], parts[1], arrays[8], arrays[9], sinks, parts[2]],
                    template: [("T", DType.bfloat16), ("D", 128), ("PTOK", ptok), ("HAS_SINKS", false)],
                    grid: (32 * 32, 1, 1), threadGroup: (32, 1, 1),
                    outputShapes: [[1, 32, 128]], outputDTypes: [.bfloat16])[0]
                return [out, parts[2]]
            }
            return NemotronDeviceSecondAttentionPlan(
                inputs: inputs, key: [hpt, ptok, maxParts, preceding ? 1 : 0], build: build)
        }

        func run(_ plan: NemotronDeviceSecondAttentionPlan) -> [MLXArray] {
            guard Self.enabled else { return plan.build(plan.inputs) }
            var retired: Entry?
            let current = lock.withLock { () -> Entry in
                if let entry, entry.key == plan.key { return entry }
                // No model/state arrays are closed over. Shape changes follow
                // compile's normal signature handling (shapeless stays false).
                let function: @Sendable ([MLXArray]) -> [MLXArray] = compile(plan.build)
                let replacement = Entry(key: plan.key, function: function)
                retired = entry
                entry = replacement
                return replacement
            }
            let result = withExtendedLifetime(retired) { current.function(plan.inputs) }
            return result.count == 2 ? result : plan.build(plan.inputs)
        }
    }

}
