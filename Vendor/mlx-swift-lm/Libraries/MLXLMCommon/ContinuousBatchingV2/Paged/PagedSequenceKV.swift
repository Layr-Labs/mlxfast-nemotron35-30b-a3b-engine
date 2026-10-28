// PagedSequenceKV.swift
//
// Per-sequence, per-layer KV state for the paged backend (WS-C).
//
// A `PagedSequenceKV` owns a page TABLE (`[Int32]` physical page ids into
// its group's slabs) plus its absolute position counter. There is no shared
// frontier: joining a batch adds an object, leaving drops it, and nothing a
// batchmate does can move this sequence's positions or evict its pages.
//
// Windowed layers use a RING of pages: token at absolute position `p` lives
// in table slot `(p / pageSize) % ringPages`, slot `p % pageSize`. That is a
// bijection onto `p % (ringPages * pageSize)`, i.e. the ring is a flat
// circular buffer of `ringPages * pageSize` TOKEN slots that happens to be
// paged. Eviction is therefore implicit overwrite keyed to absolute
// positions — the recent end always survives and no window masks over
// shared buffers exist. How far back that end reaches is
// `oldestValidPosition`, measured from the highest position ever WRITTEN
// rather than from the cursor: a rollback retreats the cursor, it does not
// un-write a slot.
//
// The ring is sized (`PagedKVPool.ringPageCount`) to cover the trailing
// window plus one speculative span (`CBv2PagedSpeculation
// .maxSpeculativeSpan`), so an MTP round overwrites only already-evicted
// slots. It does NOT carry a prefill-chunk term, and that is a PROPERTY OF
// THIS FILE, not a free choice made in the pool: both routes into the ring
// gather a chunk's window history BEFORE writing the chunk and concatenate
// the chunk tensor the caller already holds, so the widest range this row is
// ever asked for is `window - 1`. The serving route is
// `PagedLayerCache.prefillKV` calling `write` / `gatherRange` directly; the
// protocol route is this file's `update` (see there for who calls it). `maxWindowExposure` is where that promise is
// written down, and `PagedKVPool.ringPageCount` reads it — widen the
// exposure and the ring grows with it or the pool refuses to build.

import Foundation
import MLX

public final class PagedSequenceKV: CBv2SequenceKV, CBv2PagedSpeculativeRow {
    let pool: PagedKVPool
    let groupKey: PagedKVGroupKey
    /// Pool-issued monotonic identity (never reused, unlike a heap address).
    let serial: UInt64
    /// Sliding window in tokens (nil == full attention).
    public let windowSize: Int?
    /// Ring capacity in pages for windowed layers (nil for full).
    let ringPages: Int?
    /// Worst-case pages this sequence may allocate (reserved at admission;
    let reservedPages: Int
    /// Hard bound on `absoluteOffset` (from the request's maxLength).
    public let maxLength: Int

    /// Physical page ids. Full: index == logical page. Windowed: index ==
    private(set) var table: [Int32] = []
    /// Bumped whenever `table` changes — lets the layer cache reuse device
    private(set) var tableVersion: Int = 0

    public private(set) var absoluteOffset: Int = 0
    /// Highest absolute position this row has ever WRITTEN. Equal to
    private(set) var writtenHighWater: Int = 0
    /// Bumped by every write or storage adoption, never by rollback. A
    public private(set) var writeGeneration: UInt64 = 0
    /// Absolute position before which nothing was ever written here
    private(set) var baseOffset: Int = 0
    /// Absolute position through which this row's storage was adopted EXACT
    private(set) var frozenHighWater: Int = 0

    /// Absolute frontier as of `beginSpeculativeWrite()`; `nil` outside a
    private(set) var speculativeBase: Int?

    private var released = false

    init(
        pool: PagedKVPool, kind: CBv2LayerKind, groupKey: PagedKVGroupKey,
        maxLength: Int, reservedPages: Int
    ) {
        self.pool = pool
        self.groupKey = groupKey
        self.serial = pool.nextRowSerial()
        switch kind.attention {
        case .full:
            self.windowSize = nil
            self.ringPages = nil
        case .slidingWindow(let window):
            self.windowSize = window
            self.ringPages = PagedKVPool.ringPageCount(window: window, config: pool.config)
        }
        self.maxLength = maxLength
        self.reservedPages = reservedPages
    }

    deinit {
        assert(released, "[PagedSequenceKV] leaked without release — pages still allocated")
    }

    // MARK: - CBv2SequenceKV

    /// Positions this row can still be asked to GATHER, ending at
    public var retainedCount: Int {
        let written = absoluteOffset - baseOffset
        guard let window = windowSize else { return written }
        return min(written, Self.maxWindowExposure(window: window))
    }

    /// The widest range a windowed row of `window` can ask the ring for.
    static func maxWindowExposure(window: Int) -> Int { window }

    public var byteCount: Int {
        table.count * pool.group(groupKey).pageBytes
    }

    /// Append `keys`/`values` and return the KV a chunk of `n` queries
    public func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        guard pool.writeValidation.validate(keys: keys, values: values, expected: groupKey.dtype)
        else { return (keys, values) }
        var k = keys
        var v = values
        if k.ndim == 4 {
            precondition(k.dim(0) == 1, "per-sequence update requires batch dim 1")
            k = k.squeezed(axis: 0)
            v = v.squeezed(axis: 0)
        }
        guard let window = windowSize, k.dim(1) > 0 else {
            write(keys: k, values: v)
            return attendableViews()
        }
        // Pre-write history: at most `window - 1` positions, which is the
        // ONLY demand this path makes of the ring.
        let historyStart = max(baseOffset, absoluteOffset - (window - 1))
        let (historyKeys, historyValues) = gatherRange(
            start: historyStart, count: absoluteOffset - historyStart)
        write(keys: k, values: v)
        // The chunk lands in the slab under the native group dtype, so hand back the
        // values the slab will hold — otherwise this chunk is scored at a
        // precision no later decode over the same tokens can reproduce.
        let chunkKeys = k.expandedDimensions(axis: 0)
        let chunkValues = v.expandedDimensions(axis: 0)
        guard historyKeys.dim(2) > 0 else { return (chunkKeys, chunkValues) }
        return (
            concatenated([historyKeys, chunkKeys], axis: 2),
            concatenated([historyValues, chunkValues], axis: 2)
        )
    }

    public func snapshot() -> (keys: MLXArray, values: MLXArray, offset: Int) {
        let (k, v) = attendableViews()
        return (k, v, absoluteOffset)
    }

    // MARK: - Speculative (MTP) transaction

    /// Positions this row can write past its confirmed frontier before a
    public var speculativeHeadroom: Int {
        guard let window = windowSize, let ring = ringPages else { return .max }
        return ring * pool.config.pageSize - window
    }

    /// Eligibility is DERIVED from `speculativeHeadroom` — never from the
    public var supportsSpeculativeWrites: Bool {
        speculativeHeadroom >= CBv2PagedSpeculation.maxSpeculativeSpan
    }

    public func beginSpeculativeWrite() {
        precondition(
            speculativeBase == nil,
            "[PagedSequenceKV] beginSpeculativeWrite while already armed")
        speculativeBase = absoluteOffset
    }

    public func commitSpeculativeWrite() {
        guard speculativeBase != nil else { return }
        speculativeBase = nil
        // Pages `rollback` took out of the table were only QUEUED: the
        // round's gathers are lazy and still name those physical pages, so
        // handing them back before the round closes lets another row
        // allocate and rewrite them underneath a live capture. This is the
        // first moment no capture can reach them.
        pool.drainDeferredFrees()
    }

    /// Trap before a transaction writes further past its base than the ring
    private func guardSpeculativeSpan(adding n: Int) {
        guard let base = speculativeBase else { return }
        let span = absoluteOffset + n - base
        precondition(
            span <= speculativeHeadroom,
            "[PagedSequenceKV] speculative span \(span) exceeds headroom "
                + "\(speculativeHeadroom) — rollback would not be value-exact")
        guard let ring = ringPages else { return }
        let ringTokens = ring * pool.config.pageSize
        let oldestAttendable = absoluteOffset - retainedCount
        precondition(
            absoluteOffset + n - ringTokens <= oldestAttendable,
            "[PagedSequenceKV] speculative write to \(absoluteOffset + n - 1) would evict "
                + "position \(absoluteOffset + n - 1 - ringTokens), still attendable from "
                + "\(retainedCount) plus the round. The ring holds "
                + "PagedSequenceKV.maxWindowExposure plus one speculative span and nothing "
                + "more — re-size the ring (PagedKVPool.ringPageCount) or shorten the round.")
    }

    public func rollback(_ n: Int) {
        precondition(n >= 0 && n <= absoluteOffset - baseOffset, "rollback past written tokens")
        precondition(
            absoluteOffset - n >= frozenHighWater,
            "rollback cannot cross the frozen high-water")
        if let base = speculativeBase {
            // Confirmed history is NOT recoverable by rewinding the counter:
            // the round's writes already overwrote the ring slots behind the
            // base, and a full row's pages past its frontier are queued for
            // release. Crossing the base would re-expose those slots as if
            // they still held their original positions.
            precondition(
                absoluteOffset - n >= base,
                "[PagedSequenceKV] rollback into confirmed history "
                    + "(\(absoluteOffset) - \(n) < speculative base \(base))")
        }
        guard n > 0 else { return }
        absoluteOffset -= n
        if ringPages == nil {
            // Full layer: release pages past the new frontier. Freed pages
            // may hold stale bytes; they are only ever re-read by their NEXT
            // owner after that owner writes them (all reads are bounded by
            // the owner's absoluteOffset), so no scrubbing pass is needed.
            let keepPages = (absoluteOffset + pool.config.pageSize - 1) / pool.config.pageSize
            if keepPages < table.count {
                if speculativeBase == nil {
                    pool.freePages(group: groupKey, pages: table[keepPages...].reversed())
                } else {
                    // Inside a round the release is DEFERRED to commit; see
                    // `commitSpeculativeWrite`. The pages stay out of the
                    // free list until then, so `ensurePage` will draw fresh
                    // ones — which cannot breach the reservation, because a
                    // transaction never writes after its rollback (the engine
                    // rolls back at finalize, EngineLoopV2+MTPFinalize).
                    pool.deferFreePages(
                        group: groupKey, pages: table[keepPages...].reversed())
                }
                table.removeSubrange(keepPages...)
                tableVersion += 1
            }
        }
        // Windowed rings keep their pages; the rolled-back slots are
        // overwritten before they can re-enter any attendable range.
    }

    /// Advance over the `n` slots just past the frontier WITHOUT writing:
    func readvanceRetained(_ n: Int, writtenAt generation: UInt64) -> Bool {
        guard n > 0, writeGeneration == generation, ringPages == nil,
            absoluteOffset >= frozenHighWater,
            absoluteOffset + n <= writtenHighWater, absoluteOffset + n <= maxLength,
            (absoluteOffset + n - 1) / pool.config.pageSize < table.count
        else { return false }
        guardSpeculativeSpan(adding: n)
        absoluteOffset += n
        return true
    }

    // MARK: - Writes

    /// Append `keys`/`values` (`[kvHeads, n, headDim]`) at the current
    func write(keys: MLXArray, values: MLXArray) {
        guard pool.writeValidation.validate(keys: keys, values: values, expected: groupKey.dtype)
        else { return }
        let n = keys.dim(1)
        guard n > 0 else { return }
        if absoluteOffset + n <= frozenHighWater {
            // Frozen replay (`adoptFrozen`): storage below M is the adopted
            // prefix and must stay byte-exact, so the cursor advances and
            // nothing is written.
            absoluteOffset += n
            writtenHighWater = max(writtenHighWater, absoluteOffset)
            writeGeneration &+= 1
            return
        }
        let slots = reserveWrite(n)
        pool.writeTokens(group: groupKey, slots: slots, keys: keys, values: values)

        absoluteOffset += n
        writtenHighWater = max(writtenHighWater, absoluteOffset)
        writeGeneration &+= 1
        if CBv2CriticalPathCuts.selfTest, CBv2CriticalPathCuts.stridedSegmentWrite, windowSize == nil {
            selfTestReadBack(start: absoluteOffset - n, keys: keys, values: values)
        }
    }

    /// `write`'s checks and page reservation for `n` tokens at the frontier (past any frozen region): the
    private func reserveWrite(_ n: Int) -> [Int32] {
        precondition(
            absoluteOffset >= frozenHighWater,
            "frozen replay chunk crosses M (\(absoluteOffset) + \(n) vs \(frozenHighWater)) "
                + "— the plan's clampedChunk must split at M")
        precondition(
            absoluteOffset + n <= maxLength,
            "write past maxLength (\(absoluteOffset) + \(n) > \(maxLength))")
        if windowSize != nil {
            precondition(
                n <= pool.config.maxPrefillChunk,
                "windowed update of \(n) tokens exceeds maxPrefillChunk "
                    + "\(pool.config.maxPrefillChunk) — the ring cannot hold it")
        }
        guardSpeculativeSpan(adding: n)
        let s = pool.config.pageSize

        // Physical destination (page * pageSize + slot) per token.
        var slots = [Int32]()
        slots.reserveCapacity(n)
        for i in 0 ..< n {
            let pos = absoluteOffset + i
            let physical = ensurePage(logicalPage: pos / s)
            slots.append(physical * Int32(s) + Int32(pos % s))
        }
        return slots
    }

    /// Reserve the destination for ONE decode token and advance the
    func prepareDecodeWrite() -> (page: Int32, slot: Int) {
        precondition(
            absoluteOffset + 1 <= maxLength,
            "write past maxLength (\(absoluteOffset) + 1 > \(maxLength))")
        precondition(
            absoluteOffset >= frozenHighWater, "decode write inside the frozen region")
        guardSpeculativeSpan(adding: 1)
        let s = pool.config.pageSize
        let page = ensurePage(logicalPage: absoluteOffset / s)
        let slot = absoluteOffset % s
        absoluteOffset += 1
        writtenHighWater = max(writtenHighWater, absoluteOffset)
        writeGeneration &+= 1
        return (page, slot)
    }

    private func ensurePage(logicalPage lp: Int) -> Int32 {
        let slotIndex: Int
        if let ring = ringPages {
            slotIndex = lp % ring
        } else {
            slotIndex = lp
        }
        while table.count <= slotIndex {
            precondition(
                table.count < reservedPages,
                "[PagedSequenceKV] allocation past reservation (\(reservedPages) pages) — "
                    + "admission accounting bug")
            table.append(pool.allocatePage(group: groupKey))
            tableVersion += 1
        }
        return table[slotIndex]
    }

    // MARK: - Reads

    /// Gathered `[1, kvHeads, retainedCount, headDim]` views over the
    func attendableViews() -> (keys: MLXArray, values: MLXArray) {
        let retained = retainedCount
        let start = absoluteOffset - retained
        return gatherRange(start: start, count: retained)
    }

    /// Oldest absolute position this row still physically holds.
    var oldestValidPosition: Int {
        guard let ring = ringPages else { return baseOffset }
        return max(baseOffset, writtenHighWater - ring * pool.config.pageSize)
    }

    /// Gather an arbitrary written range (used by prefill fallback and
    func gatherRange(start: Int, count: Int) -> (keys: MLXArray, values: MLXArray) {
        guard count > 0 else {
            return pool.gather(group: groupKey, pages: [], firstSlot: 0, count: 0)
        }
        precondition(start >= baseOffset && start + count <= absoluteOffset)
        precondition(
            start >= oldestValidPosition,
            "gather of evicted range: \(start) is older than this row's oldest resident "
                + "position \(oldestValidPosition) (window \(windowSize ?? 0), ring "
                + "\(ringPages ?? 0) pages, written high water \(writtenHighWater), frontier "
                + "\(absoluteOffset))")
        let s = pool.config.pageSize
        let lpFirst = start / s
        let lpLast = (start + count - 1) / s
        var pages: [Int32] = []
        pages.reserveCapacity(lpLast - lpFirst + 1)
        for lp in lpFirst ... lpLast {
            let idx = ringPages.map { lp % $0 } ?? lp
            pages.append(table[idx])
        }
        return pool.gather(
            group: groupKey, pages: pages, firstSlot: start % s, count: count)
    }

    /// NEMOTRON_KERNEL_SELFTEST=1 with CBV2_PAGED_STRIDED_WRITE: read the
    private func selfTestReadBack(start: Int, keys: MLXArray, values: MLXArray) {
        let back = gatherRange(start: start, count: keys.dim(1))
        func bits(_ a: MLXArray) -> MLXArray {
            a.view(dtype: a.itemSize == 2 ? .uint16 : (a.itemSize == 4 ? .uint32 : .uint8))
        }
        let same = all(bits(back.keys.reshaped(keys.shape)) .== bits(keys))
            .&& all(bits(back.values.reshaped(values.shape)) .== bits(values))
        guard same.item(Bool.self) else {
            FileHandle.standardError.write(Data(
                "CBv2 strided segment write self-test: read-back differs at \(start)\n".utf8))
            preconditionFailure("CBv2 strided segment write self-test failed")
        }
    }

    /// Modular table length the decode kernel divides by to resolve a
    var decodeTableLength: Int { ringPages ?? table.count }

    /// This row's `seqinfo` entry for a decode dispatch attending `range`.
    func seqInfoRow(
        attending range: (start: Int, length: Int),
        writeTarget: (page: Int32, slot: Int)? = nil
    ) -> PagedAttentionKernel.SeqInfoRow {
        PagedAttentionKernel.SeqInfoRow(
            attendStart: range.start,
            attendLength: range.length,
            tableLength: decodeTableLength,
            writePage: writeTarget?.page ?? 0,
            writeSlot: writeTarget?.slot ?? 0)
    }

    /// Kernel-facing row descriptor for decode: the absolute range the
    var decodeAttendRange: (start: Int, length: Int) {
        let qPos = absoluteOffset - 1
        precondition(qPos >= baseOffset, "decode before any token was written")
        var start = baseOffset
        if let window = windowSize {
            start = max(start, qPos - window + 1)
        }
        return (start, qPos - start + 1)
    }

    // MARK: - Lifecycle

    /// Advance the position counter without writing storage. Only valid on
    public func fastForward(to offset: Int) {
        precondition(windowSize != nil, "fastForward is only for windowed layers")
        precondition(
            table.isEmpty && absoluteOffset == 0 && baseOffset == 0,
            "fastForward requires a fresh state")
        precondition(offset >= 0 && offset <= maxLength)
        absoluteOffset = offset
        baseOffset = offset
        writtenHighWater = offset
        writeGeneration &+= 1
    }

    // MARK: - Page-native prefix sharing

    /// Generation-checked handles for an immutable, page-aligned range of a
    func prefixPageHandles(tokens range: Range<Int>) -> [PagedKVPageHandle]? {
        guard windowSize == nil, !released else { return nil }
        let pageSize = pool.config.pageSize
        guard range.lowerBound >= 0, range.upperBound <= writtenHighWater,
            range.lowerBound % pageSize == 0, range.upperBound % pageSize == 0
        else { return nil }
        let pageRange = (range.lowerBound / pageSize) ..< (range.upperBound / pageSize)
        guard pageRange.upperBound <= table.count else { return nil }
        return table[pageRange].map {
            pool.currentHandle(group: groupKey, page: $0)
        }
    }

    /// Install pages whose retains were acquired transactionally by the pool.
    func adoptSharedPages(
        _ handles: [PagedKVPageHandle], storedThrough: Int, cursor: Int,
        frozenThrough: Int
    ) {
        precondition(windowSize == nil, "shared-page adoption is full-attention only")
        precondition(
            table.isEmpty && absoluteOffset == 0 && baseOffset == 0,
            "shared-page adoption requires a fresh row")
        let pageSize = pool.config.pageSize
        precondition(storedThrough == handles.count * pageSize)
        precondition(storedThrough <= maxLength && handles.count <= reservedPages)
        precondition(cursor >= 0 && cursor <= storedThrough)
        precondition(frozenThrough == 0 || frozenThrough == storedThrough)
        precondition(frozenThrough == 0 ? cursor == storedThrough : cursor <= frozenThrough)
        for handle in handles {
            precondition(handle.group == groupKey && pool.isValid(handle))
        }
        table = handles.map(\.page)
        tableVersion += 1
        absoluteOffset = cursor
        writtenHighWater = storedThrough
        writeGeneration &+= 1
        frozenHighWater = frozenThrough
    }

    /// Imported pages have one exclusive owner. Their final page may be
    func adoptExclusiveCheckpointPages(_ pages: [Int32], storedThrough: Int) {
        precondition(windowSize == nil && !released)
        precondition(table.isEmpty && absoluteOffset == 0 && baseOffset == 0)
        precondition(storedThrough > 0 && storedThrough <= maxLength)
        precondition(pages.count == (storedThrough - 1) / pool.config.pageSize + 1)
        precondition(pages.count <= reservedPages)
        table = pages
        tableVersion += 1
        absoluteOffset = storedThrough
        writtenHighWater = storedThrough
        writeGeneration &+= 1
    }

    /// A private complete-checkpoint ring was filled at its absolute slots.
    func adoptHistoricalWindowPages(_ pages: [Int32], retainedStart: Int, storedThrough: Int) {
        precondition(windowSize != nil && !released)
        precondition(table.isEmpty && absoluteOffset == 0 && baseOffset == 0)
        precondition(storedThrough > 0 && storedThrough <= maxLength)
        precondition(retainedStart == max(0, storedThrough - windowSize!))
        precondition(pages.count == reservedPages && pages.count <= ringPages!)
        table = pages
        tableVersion += 1
        baseOffset = retainedStart
        absoluteOffset = storedThrough
        writtenHighWater = storedThrough
        writeGeneration &+= 1
    }

    /// A row prepared off-lock but never published owns no pages/reservation.
    func discardUninstalledCheckpointRow() {
        precondition(table.isEmpty && absoluteOffset == 0 && !released)
        released = true
    }

    /// Adopt `[0, M)` as immutable storage with the logical cursor at C.
    func adoptFrozen(keys: MLXArray, values: MLXArray, replayStart: Int) {
        precondition(windowSize == nil, "frozen adoption is only for full-attention rows")
        precondition(
            table.isEmpty && absoluteOffset == 0 && baseOffset == 0,
            "frozen adoption requires a fresh row")
        let m = keys.dim(1)
        precondition(replayStart >= 0 && replayStart <= m, "frozen adoption needs 0 <= C <= M")
        write(keys: keys, values: values)
        frozenHighWater = m
        absoluteOffset = replayStart
    }

    /// Return every page to the pool and release the reservation.
    func releaseStorage() {
        guard !released else { return }
        released = true
        if speculativeBase != nil {
            // Released mid-round: the row finished in flight, so the deferred
            // release fence tears it down and `commitSpeculativeWrite` never
            // runs. Draining here is what keeps its queued pages from being
            // orphaned, and it is safe under the RECYCLE INVARIANT above —
            // releases only happen between host-synced steps, so no lazy
            // capture can still name a queued page.
            speculativeBase = nil
            pool.drainDeferredFrees()
        }
        // Tail-first release gives equally recent cached pages leaf-biased
        // eviction: divergent suffix pages re-enter the LRU before ancestors.
        pool.freePages(group: groupKey, pages: table.reversed())
        table.removeAll()
        tableVersion += 1
        pool.unreserve([groupKey: reservedPages])
    }
}

// MARK: - Direct reads (the MTP draft attention)

/// The operands of one single-segment read of a row's attendable range: what `snapshot()`'s gather reads,
public struct CBv2PagedDirectRead {
    /// The segment's `[2, pages, kvHeads, pageSize, headDim]` storage holding every page of the range.
    public let storage: MLXArray
    public let valueOffset: Int
    /// `[pageCount]` int32: the segment-local page of each logical page of the range, in order.
    public let pages: MLXArray
    public let firstSlot: Int
    public let count: Int
    public let kvHeads: Int
    public let headDim: Int
    public let pageSize: Int
    public let dtype: DType
    /// The row's `absoluteOffset` when the operands were taken.
    public let offset: Int
    /// The group's write fence when the operands were taken: bind it to the reading launch so the read orders
    public let writeFence: MLXArray

    public init(storage: MLXArray, valueOffset: Int, pages: MLXArray, firstSlot: Int, count: Int, kvHeads: Int,
                headDim: Int, pageSize: Int, dtype: DType, offset: Int, writeFence: MLXArray) {
        self.storage = storage
        self.valueOffset = valueOffset
        self.pages = pages
        self.firstSlot = firstSlot
        self.count = count
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.pageSize = pageSize
        self.dtype = dtype
        self.offset = offset
        self.writeFence = writeFence
    }
}

extension PagedSequenceKV {
    /// The attendable range's direct-read operands, or nil where `snapshot()` would not be one single-segment read
    public func directRead() -> CBv2PagedDirectRead? {
        let count = retainedCount
        let start = absoluteOffset - count
        guard count > 0, start >= baseOffset, start >= oldestValidPosition else { return nil }
        let group = pool.group(groupKey)
        guard let layout = group.segmentLayout else { return nil }
        let s = pool.config.pageSize
        guard s == group.pageSize else { return nil }
        let lpFirst = start / s
        let lpLast = (start + count - 1) / s
        var local: [Int32] = []
        local.reserveCapacity(lpLast - lpFirst + 1)
        var segmentIndex: Int? = nil
        for lp in lpFirst ... lpLast {
            let idx = ringPages.map { lp % $0 } ?? lp
            guard idx < table.count else { return nil }
            let page = table[idx]
            guard group.isAllocatable(page) else { return nil }
            let index = layout.segmentIndex(page: page)
            if let segmentIndex, segmentIndex != index { return nil }
            segmentIndex = index
            local.append(Int32(layout.localPage(page)))
        }
        guard let segmentIndex, let segment = group.segments[segmentIndex] else { return nil }
        return CBv2PagedDirectRead(
            storage: segment.storage, valueOffset: segment.valueOffset, pages: MLXArray(local),
            firstSlot: start % s, count: count, kvHeads: groupKey.kvHeads, headDim: groupKey.headDim,
            pageSize: s, dtype: group.dtype, offset: absoluteOffset, writeFence: group.writeFence)
    }

    /// After a launch that read `directRead()`'s storage, fold one element of its output into the group's write
    public func publishDirectRead(_ output: MLXArray) {
        guard output.size > 0 else { return }
        let group = pool.group(groupKey)
        let probe = output.reshaped(-1)[0]
        group.writeFence = group.writeFence + probe.asType(group.writeFence.dtype) * 0
    }

    /// The group's write fence as it stands now. A launch that reads `directRead()`'s storage binds it and emits
    public var liveWriteFence: MLXArray { pool.group(groupKey).writeFence }

    /// `publishDirectRead` for a reading launch that bound `live` (`liveWriteFence` when it was built) and stores
    public func publishDirectReadFence(_ fence: MLXArray, after live: MLXArray) {
        let group = pool.group(groupKey)
        precondition(
            group.writeFence === live,
            "[PagedSequenceKV] direct read fence published after the group's fence moved")
        group.writeFence = fence
    }
}

// MARK: - Direct writes (the MTP post-verify rows)

/// Where a row's next `count` tokens go when a launch stores their K/V itself instead of the segment write
public struct CBv2PagedDirectWrite {
    /// The segment's `[2, pages, kvHeads, pageSize, headDim]` storage holding every destination page.
    public let storage: MLXArray
    public let valueOffset: Int
    public let pages: [Int32]
    public let slots: [Int32]
    public let kvHeads: Int
    public let headDim: Int
    public let pageSize: Int
    public let dtype: DType
    /// The group's write fence at the reservation.
    public let writeFence: MLXArray
    public var count: Int { pages.count }
    /// The row's frontier and write generation at the reservation (`commitDirectWrite` requires both).
    let offset: Int
    let generation: UInt64
}

extension PagedSequenceKV {
    /// Reserve the destination of `n` tokens at the frontier with `write`'s checks and page reservation, without
    public func directWrite(_ n: Int) -> CBv2PagedDirectWrite? {
        guard n > 0, !pool.writeValidation.isFaulted, absoluteOffset >= frozenHighWater else { return nil }
        let group = pool.group(groupKey)
        guard let layout = group.segmentLayout, group.pageSize == pool.config.pageSize else { return nil }
        let reserved = reserveWrite(n)
        var pages: [Int32] = []
        var slots: [Int32] = []
        var segmentIndex: Int? = nil
        for position in reserved {
            let page = position / Int32(group.pageSize)
            guard group.isAllocatable(page) else { return nil }
            let index = layout.segmentIndex(page: page)
            if let segmentIndex, segmentIndex != index { return nil }
            segmentIndex = index
            pages.append(Int32(layout.localPage(page)))
            slots.append(position % Int32(group.pageSize))
        }
        guard let segmentIndex, let segment = group.segments[segmentIndex] else { return nil }
        return CBv2PagedDirectWrite(
            storage: segment.storage, valueOffset: segment.valueOffset, pages: pages, slots: slots,
            kvHeads: groupKey.kvHeads, headDim: groupKey.headDim, pageSize: group.pageSize, dtype: group.dtype,
            writeFence: group.writeFence, offset: absoluteOffset, generation: writeGeneration)
    }

    /// After the launch that stored `write`'s tokens: what `write` does once its dispatch is encoded. The
    public func commitDirectWrite(_ write: CBv2PagedDirectWrite, fence: MLXArray) {
        let group = pool.group(groupKey)
        precondition(
            write.offset == absoluteOffset && write.generation == writeGeneration
                && group.writeFence === write.writeFence,
            "[PagedSequenceKV] direct write committed after the row or the group's fence moved")
        absoluteOffset += write.count
        writtenHighWater = max(writtenHighWater, absoluteOffset)
        writeGeneration &+= 1
        group.writeFence = fence
    }
}
