import Foundation
import MLX
import MLXFast

/// Bounded page transfers. A dispatch binds one segment and the request's input
/// or final output, never a concatenation of the pool's native backing.
enum PagedSegmentTransfers {
    private static let writeBody = """
        const int h = int(thread_position_in_grid.y);
        const int d = int(thread_position_in_grid.x);
        const int r = int(thread_position_in_grid.z);
        const int token = records[r * 3];
        const int page = records[r * 3 + 1];
        const int slot = records[r * 3 + 2];
        const int n = keys_shape[1];
        if (page > 0) {
            const size_t source = ((size_t)h * n + token) * D + d;
            const size_t target = (((size_t)page * H + h) * S + slot) * D + d;
            device T* destination = storage;
            destination[target] = keys[source];
            destination[VBASE + target] = values[source];
        }
        if (h == 0 && d == 0 && r == 0) fence[0] = previous[0] + 1;
        """

    /// CBV2_PAGED_STRIDED_WRITE: `writeBody` reading keys/values through
    /// their own element strides, so a transposed or sliced `[H, n, D]` view
    /// is written without the implicit contiguous copy of each input. The
    /// source element for (h, token, d) is the same one the copy would have
    /// placed at ((h * n + token) * D + d); every store is unchanged.
    private static let stridedWriteBody = """
        const int h = int(thread_position_in_grid.y);
        const int d = int(thread_position_in_grid.x);
        const int r = int(thread_position_in_grid.z);
        const int token = records[r * 3];
        const int page = records[r * 3 + 1];
        const int slot = records[r * 3 + 2];
        if (page > 0) {
            const int64_t ks = int64_t(h) * keys_strides[0]
                + int64_t(token) * keys_strides[1] + int64_t(d) * keys_strides[2];
            const int64_t vs = int64_t(h) * values_strides[0]
                + int64_t(token) * values_strides[1] + int64_t(d) * values_strides[2];
            const size_t target = (((size_t)page * H + h) * S + slot) * D + d;
            device T* destination = storage;
            destination[target] = keys[ks];
            destination[VBASE + target] = values[vs];
        }
        if (h == 0 && d == 0 && r == 0) fence[0] = previous[0] + 1;
        """

    private static let readBody = """
        const int h = int(thread_position_in_grid.y);
        const int d = int(thread_position_in_grid.x);
        const int r = int(thread_position_in_grid.z);
        const int token = records[r * 3];
        const int page = records[r * 3 + 1];
        const int slot = records[r * 3 + 2];
        const int count = output_shape[3];
        const size_t source = (((size_t)page * H + h) * S + slot) * D + d;
        const size_t target = ((size_t)h * count + token) * D + d;
        device T* destination = output;
        destination[target] = storage[source];
        destination[(size_t)H * count * D + target] = storage[VBASE + source];
        if (h == 0 && d == 0 && r == 0) fence[0] = previous[0] + 1;
        """

    /// CBV2_PAGED_READ_OUTPUT: `readBody` for a gather whose records all name
    /// one segment, storing into a real kernel output. The grid's depth is the
    /// record count, which is then the token count: every (token, head, lane)
    /// of both planes is stored once, so nothing needs a fill beforehand.
    private static let readOutputBody = """
        const int h = int(thread_position_in_grid.y);
        const int d = int(thread_position_in_grid.x);
        const int r = int(thread_position_in_grid.z);
        const int token = records[r * 3];
        const int page = records[r * 3 + 1];
        const int slot = records[r * 3 + 2];
        const int count = int(threads_per_grid.z);
        const size_t source = (((size_t)page * H + h) * S + slot) * D + d;
        const size_t target = ((size_t)h * count + token) * D + d;
        output[target] = storage[source];
        output[(size_t)H * count * D + target] = storage[VBASE + source];
        if (h == 0 && d == 0 && r == 0) fence[0] = previous[0] + 1;
        """

    /// Depends is a buffer alias, not an encoded Metal dependency. A later
    /// Slice may copy that alias, so force one real GPU consumer of the final
    /// transfer fence first. Its read-after-write barrier covers the hidden
    /// in-place destination writes; only four bytes of new output are needed.
    private static let completionKernel = MLXFast.metalKernel(
        name: "cbv2_segment_transfer_complete", inputNames: ["previous"],
        outputNames: ["witness"], source: "witness[0] = previous[0] + 1;",
        ensureRowContiguous: true)

    private static let lock = NSLock()
    nonisolated(unsafe) private static var kernels: [String: MLXFast.MLXFastKernel] = [:]

    private static func kernel(reading: Bool, dtype: DType) -> MLXFast.MLXFastKernel {
        let key = "cbv2_segment_\(reading ? "read" : "write")_\(dtype)"
        return lock.withLock {
            if let existing = kernels[key] { return existing }
            let made = MLXFast.metalKernel(
                name: key,
                inputNames: reading
                    ? ["storage", "records", "output", "previous"]
                    : ["keys", "values", "storage", "records", "previous"],
                outputNames: ["fence"], source: reading ? readBody : writeBody,
                ensureRowContiguous: true,
                mutableInputs: [reading ? "output" : "storage"])
            kernels[key] = made
            return made
        }
    }

    private static func readOutputKernel(dtype: DType) -> MLXFast.MLXFastKernel {
        let key = "cbv2_segment_read_output_\(dtype)"
        return lock.withLock {
            if let existing = kernels[key] { return existing }
            let made = MLXFast.metalKernel(
                name: key, inputNames: ["storage", "records", "previous"],
                outputNames: ["output", "fence"], source: readOutputBody,
                ensureRowContiguous: true)
            kernels[key] = made
            return made
        }
    }

    private static func stridedWriteKernel(dtype: DType) -> MLXFast.MLXFastKernel {
        let key = "cbv2_segment_write_strided_\(dtype)"
        return lock.withLock {
            if let existing = kernels[key] { return existing }
            let made = MLXFast.metalKernel(
                name: key,
                inputNames: ["keys", "values", "storage", "records", "previous"],
                outputNames: ["fence"], source: stridedWriteBody,
                ensureRowContiguous: false,
                mutableInputs: ["storage"])
            kernels[key] = made
            return made
        }
    }

    private static func records(_ triples: [Int32]) -> MLXArray {
        precondition(triples.count % 3 == 0)
        return MLXArray(triples + Array(repeating: 0, count: max(0, 24 - triples.count)))
    }

    /// Each record is (input/output token index, local page, slot).
    private static func buckets(group: PagedKVGroup, slots: [Int32]) -> [(PagedKVSegment, [Int32])] {
        let layout = group.segmentLayout!
        var records: [Int: [Int32]] = [:]
        for (token, position) in slots.enumerated() {
            precondition(position >= 0)
            let page = position / Int32(group.pageSize)
            precondition(group.isAllocatable(page))
            let index = layout.segmentIndex(page: page)
            records[index, default: []].append(contentsOf: [
                Int32(token), Int32(layout.localPage(page)), position % Int32(group.pageSize)])
        }
        return records.keys.sorted().map { index in
            guard let segment = group.segments[index] else {
                preconditionFailure("transfer names an uncommitted segment")
            }
            return (segment, records[index]!)
        }
    }

    static func write(group: PagedKVGroup, slots: [Int32], keys: MLXArray, values: MLXArray) {
        guard group.writeValidation.validate(keys: keys, values: values, expected: group.dtype) else { return }
        precondition(keys.shape == [group.key.kvHeads, slots.count, group.key.headDim])
        precondition(values.shape == keys.shape)
        let k = keys
        let v = values
        // Strided reads replace the implicit contiguous copies of k and v
        // (one launch each for a transposed or sliced view). Same stores.
        let write = CBv2CriticalPathCuts.stridedSegmentWrite
            ? stridedWriteKernel(dtype: group.dtype) : kernel(reading: false, dtype: group.dtype)
        for (segment, triples) in buckets(group: group, slots: slots) {
            group.writeFence = write(
                [k, v, segment.storage, records(triples), group.writeFence],
                template: [("T", group.dtype), ("H", group.key.kvHeads),
                           ("D", group.key.headDim), ("S", group.pageSize),
                           ("VBASE", segment.valueOffset)],
                grid: (group.key.headDim, group.key.kvHeads, triples.count / 3),
                threadGroup: (min(256, group.key.headDim), 1, 1),
                outputShapes: [[1]], outputDTypes: [.int32])[0]
        }
    }

    static func gather(group: PagedKVGroup, pages: [Int32], firstSlot: Int, count: Int)
        -> (keys: MLXArray, values: MLXArray)
    {
        let (ordered, direct) = combined(group: group, pages: pages, firstSlot: firstSlot, count: count)
        // CBV2_PAGED_READ_OUTPUT: the planes as slice views of the read
        // launch's own output (an Int subscript is an MLX `take`: one gather
        // launch and copy per plane).
        if direct {
            return (ordered[0 ..< 1].squeezed(axis: 0), ordered[1 ..< 2].squeezed(axis: 0))
        }
        return (ordered[0], ordered[1])
    }

    /// Historical owners retain this single output, avoiding two potentially
    /// materializing Slice views. The caller reserves every output/record/fence
    /// before calling and keeps its charge until the captured owner retires.
    /// `publishReadFence: false` requires the owning engine step to block all
    /// successor writes until the private read has completed or drained.
    static func gatherCombined(group: PagedKVGroup, pages: [Int32], firstSlot: Int, count: Int,
                               publishReadFence: Bool = true, stream: StreamOrDevice = .default) -> MLXArray {
        combined(group: group, pages: pages, firstSlot: firstSlot, count: count,
                 publishReadFence: publishReadFence, stream: stream).output
    }

    /// `direct`: the output is the single-segment read launch's own output
    /// (CBV2_PAGED_READ_OUTPUT), not an alias of an in-place destination.
    private static func combined(group: PagedKVGroup, pages: [Int32], firstSlot: Int, count: Int,
                                 publishReadFence: Bool = true, stream: StreamOrDevice = .default)
        -> (output: MLXArray, direct: Bool)
    {
        let h = group.key.kvHeads, d = group.key.headDim
        guard count > 0 else {
            return (MLXArray.zeros([2, 1, h, 0, d], dtype: group.dtype, stream: stream), false)
        }
        precondition(firstSlot >= 0 && firstSlot < group.pageSize)
        precondition(pages.count * group.pageSize >= firstSlot + count)
        let slots = (0 ..< count).map { token -> Int32 in
            let offset = firstSlot + token
            return pages[offset / group.pageSize] * Int32(group.pageSize)
                + Int32(offset % group.pageSize)
        }
        let transfers = buckets(group: group, slots: slots)
        guard CBv2CriticalPathCuts.segmentReadOutput, transfers.count == 1 else {
            return (inPlaceGather(group: group, transfers: transfers, count: count,
                                  publishReadFence: publishReadFence, stream: stream), false)
        }
        let (segment, triples) = transfers[0]
        precondition(triples.count == 3 * count)
        let read = readOutputKernel(dtype: group.dtype)(
            [segment.storage, records(triples), group.writeFence],
            template: [("T", group.dtype), ("H", h), ("D", d),
                       ("S", group.pageSize), ("VBASE", segment.valueOffset)],
            grid: (d, h, count), threadGroup: (min(256, d), 1, 1),
            outputShapes: [[2, 1, h, count, d], [1]], outputDTypes: [group.dtype, .int32],
            stream: stream)
        if CBv2CriticalPathCuts.selfTest {
            selfTestRead(read[0], reference: inPlaceGather(
                group: group, transfers: transfers, count: count, publishReadFence: false,
                stream: stream))
        }
        // The read's own fence orders every later write of the group after it.
        if publishReadFence { group.writeFence = read[1] }
        return (read[0], true)
    }

    /// NEMOTRON_KERNEL_SELFTEST=1 with CBV2_PAGED_READ_OUTPUT: the direct read
    /// must equal the in-place gather bit for bit (one host sync per gather;
    /// self-test only). Stops the process on any difference.
    private static func selfTestRead(_ output: MLXArray, reference: MLXArray) {
        func bits(_ a: MLXArray) -> MLXArray {
            a.view(dtype: a.itemSize == 2 ? .uint16 : (a.itemSize == 4 ? .uint32 : .uint8))
        }
        let same = output.shape == reference.shape && output.dtype == reference.dtype
            && all(bits(output) .== bits(reference)).item(Bool.self)
        guard same else {
            FileHandle.standardError.write(Data(
                "CBv2 paged read output self-test: gather differs\n".utf8))
            preconditionFailure("CBv2 paged read output self-test failed")
        }
    }

    /// Every segment's read stores into one zero-filled destination in place;
    /// a completion launch then consumes the final transfer fence.
    private static func inPlaceGather(
        group: PagedKVGroup, transfers: [(PagedKVSegment, [Int32])], count: Int,
        publishReadFence: Bool, stream: StreamOrDevice
    ) -> MLXArray {
        let h = group.key.kvHeads, d = group.key.headDim
        let output = MLXArray.zeros([2, 1, h, count, d], dtype: group.dtype, stream: stream)
        var fence = group.writeFence
        for (segment, triples) in transfers {
            fence = kernel(reading: true, dtype: group.dtype)(
                [segment.storage, records(triples), output, fence],
                template: [("T", group.dtype), ("H", h), ("D", d),
                           ("S", group.pageSize), ("VBASE", segment.valueOffset)],
                grid: (d, h, triples.count / 3), threadGroup: (min(256, d), 1, 1),
                outputShapes: [[1]], outputDTypes: [.int32], stream: stream)[0]
        }
        fence = completionKernel(
            [fence], grid: (1, 1, 1), threadGroup: (1, 1, 1),
            outputShapes: [[1]], outputDTypes: [.int32], stream: stream)[0]
        // Keep the final destination; no second whole-gather allocation.
        if publishReadFence { group.writeFence = fence }
        // MLX Depends inherits its input primitive stream (ops.cpp); host
        // Int32 record arrays have no primitive or stream to drain. Source
        // storage/fence dependencies cross streams through MLX events.
        return depends(input: output, dependencies: [fence])
    }
}
