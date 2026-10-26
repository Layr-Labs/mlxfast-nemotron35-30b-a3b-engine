// FullSequenceKV.swift
//
// ContinuousBatchingV2 per-sequence KV storage for FULL attention layers.
//
// One instance owns the K/V for ONE sequence at ONE layer. There is no shared
// batch frontier, no left padding, and no batch-wide trim: joining or leaving
// a batch never touches this object (batch membership is just list membership
// in `CBv2LayerCache.rows`).

import Cmlx
import Foundation
import MLX

/// Counters for the v2 core runtime's own host-interaction points.
///
/// The engine step loop must never force a host sync (`.item()`, `asArray`,
/// blocking `eval`) and must never rebuild per-row metadata arrays from host
/// integers outside membership changes. Any CBv2 core code that *does* touch
/// the host goes through these counters so tests can assert the step loop is
/// clean (see `CBv2CoreTests`). This deliberately does not instrument MLX
/// itself — only our own sync points.
public enum CBv2CoreInstrumentation {
    private static let lock = NSLock()

    nonisolated(unsafe) private static var _hostSyncs = 0
    nonisolated(unsafe) private static var _positionOffsetsHostRebuilds = 0

    /// Number of host syncs performed by CBv2 core code.
    public static var hostSyncs: Int {
        lock.lock()
        defer { lock.unlock() }
        return _hostSyncs
    }

    /// Number of times a layer cache rebuilt `positionOffsets` from host
    /// integers. Must only ever increase on batch membership changes.
    public static var positionOffsetsHostRebuilds: Int {
        lock.lock()
        defer { lock.unlock() }
        return _positionOffsetsHostRebuilds
    }

    /// Gate for `recordHostSync`: the engine's finalize readbacks count
    /// only while a test has switched counting on. Off (the default) the
    /// step path pays one static bool read per readback and never touches
    /// the lock — the same pattern as the engine's signposter gate. Tests
    /// flip it on before stepping and reset it afterwards.
    nonisolated(unsafe) static var countingEnabled = false

    static func recordHostSync() {
        guard countingEnabled else { return }
        lock.lock()
        defer { lock.unlock() }
        _hostSyncs += 1
    }

    static func recordPositionOffsetsHostRebuild() {
        lock.lock()
        defer { lock.unlock() }
        _positionOffsetsHostRebuilds += 1
    }
}

/// Internal hook so `CBv2LayerCache` can hand a row's lazily-mutated storage
/// arrays to the engine loop's `asyncEval` (graph/metadata hygiene: no
/// unconsumed lazy chain may grow O(steps) — DAR-325).
protocol CBv2InnerStateProviding {
    func cbv2InnerState() -> [MLXArray]
}

/// `CBv2SequenceKV` for full (non-windowed) attention.
///
/// Storage is one contiguous `[1, kvHeads, capacity, headDim]` buffer per
/// K and V, grown by doubling (initial capacity = promptLength + 256, capped
/// at `maxLength`). Appends are slice assignments — `mlx_slice_update`
/// donates the input buffer when refcount permits, so an append is O(n), not
/// O(cache). `update` returns temporal-order zero-copy strided views
/// `[..., 0..<retained, :]`; MLX SDPA accepts strided K/V.
public final class CBv2FullSequenceKV: CBv2SequenceKV, CBv2InnerStateProviding {

    /// Extra slots allocated beyond the prompt so the first decode steps
    /// don't immediately grow the buffer.
    static let initialSlack = 256

    /// Set by the Nemotron request prefill (the seed of the scored decode)
    /// around its forward, only after its load-time raw-word check of
    /// `seedFirstWriteCheck`: a sequence's first write then builds the
    /// storage as the update followed by the zero slack in one concatenate,
    /// instead of a zero storage that a slice update rewrites whole. Same
    /// stored values, same shape; the storage stays fresh and row-contiguous.
    nonisolated(unsafe) public static var seedFirstWrite = false
    nonisolated(unsafe) private static var seedFirstWriteReported = false

    public private(set) var absoluteOffset: Int = 0
    public var retainedCount: Int { absoluteOffset }

    /// Hard cap on this sequence's length; growth beyond it is an engine
    /// admission bug and traps.
    public let maxLength: Int

    let kvHeads: Int
    let headDim: Int

    private var keys: MLXArray?
    private var values: MLXArray?
    private var capacity: Int

    /// - Parameters:
    ///   - promptLength: expected prompt length, used to size the initial
    ///     allocation (`promptLength + 256`, capped at `maxLength`).
    ///   - maxLength: maximum total tokens this sequence may ever hold.
    ///   - kvHeads/headDim: from the layer's `CBv2LayerKind`; validated
    ///     against the arrays passed to `update`.
    public init(promptLength: Int, maxLength: Int, kvHeads: Int, headDim: Int) {
        precondition(maxLength > 0, "CBv2FullSequenceKV: maxLength must be > 0")
        precondition(
            promptLength <= maxLength,
            "CBv2FullSequenceKV: promptLength \(promptLength) exceeds maxLength \(maxLength)")
        self.maxLength = maxLength
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.capacity = min(maxLength, max(1, promptLength + Self.initialSlack))
    }

    public var byteCount: Int {
        (keys?.nbytes ?? 0) + (values?.nbytes ?? 0)
    }

    /// Transfer an exclusively owned, fully authenticated native destination.
    /// No prefix copy or lazy assignment may retain the staging buffers.
    init(
        restoredKeys: MLXArray, restoredValues: MLXArray, offset: Int,
        maxLength: Int, kvHeads: Int, headDim: Int
    ) throws {
        let shape = [1, kvHeads, maxLength, headDim]
        guard maxLength > 0, offset > 0, offset <= maxLength,
            restoredKeys.shape == shape, restoredValues.shape == shape,
            restoredKeys.dtype == restoredValues.dtype
        else { throw CBv2CompleteCheckpointError.incompatibleCheckpoint }
        self.maxLength = maxLength
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.capacity = maxLength
        self.absoluteOffset = offset
        self.keys = restoredKeys
        self.values = restoredValues
    }

    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let n = newKeys.dim(2)
        precondition(newKeys.dim(0) == 1 && newValues.dim(0) == 1,
            "CBv2FullSequenceKV holds ONE sequence; got batch \(newKeys.dim(0))")
        precondition(newKeys.dim(1) == kvHeads,
            "CBv2FullSequenceKV: kvHeads mismatch (\(newKeys.dim(1)) != \(kvHeads))")
        precondition(newValues.dim(2) == n,
            "CBv2FullSequenceKV: keys/values token count mismatch")
        precondition(
            absoluteOffset + n <= maxLength,
            "CBv2FullSequenceKV: append past maxLength (\(absoluteOffset) + \(n) > \(maxLength)) — admission bug"
        )

        if keys == nil, Self.seedFirstWrite, absoluteOffset == 0,
           newKeys.ndim == 4, newValues.ndim == 4, newKeys.dim(0) == 1, newKeys.dim(1) == kvHeads,
           newValues.dim(1) == kvHeads, newKeys.dtype.size == 2, newValues.dtype.size == 2
        {
            let first = min(maxLength, max(capacity, n))
            if first > n {
                if !Self.seedFirstWriteReported {
                    Self.seedFirstWriteReported = true
                    FileHandle.standardError.write(Data(
                        "nemotron seedx kv: engaged (first write \(n) rows into \(first))\n".utf8))
                }
                capacity = first
                keys = Self.firstWrite(newKeys, capacity: first)
                values = Self.firstWrite(newValues, capacity: first)
                absoluteOffset = n
                return (
                    keys![.ellipsis, ..<absoluteOffset, 0...],
                    values![.ellipsis, ..<absoluteOffset, 0...]
                )
            }
        }

        ensureCapacity(absoluteOffset + n, keyTemplate: newKeys, valueTemplate: newValues)

        keys = Self.write(keys!, newKeys, at: absoluteOffset)
        values = Self.write(values!, newValues, at: absoluteOffset)
        absoluteOffset += n

        return (
            keys![.ellipsis, ..<absoluteOffset, 0...],
            values![.ellipsis, ..<absoluteOffset, 0...]
        )
    }

    /// CBV2_KV_IN_PLACE_APPEND: the storage a producer kernel may write the
    /// next `n` tokens' K/V into directly (`[1, kvHeads, capacity, headDim]`,
    /// row-contiguous), with the append position and capacity. nil unless
    /// the buffers exist, are evaluated, are this row's only owners of their
    /// bytes (no view or snapshot can observe the write), match `dtype` and
    /// `headDim`, and the append fits without growth. Nothing changes until
    /// `commitInPlaceAppend`.
    func inPlaceAppendTarget(count n: Int, dtype: DType, headDim d: Int)
        -> (keys: MLXArray, values: MLXArray, offset: Int, capacity: Int)?
    {
        guard CBv2CriticalPathCuts.kvInPlaceAppend, n > 0, let keys, let values,
            d == headDim, keys.dtype == dtype, values.dtype == dtype,
            keys.shape == [1, kvHeads, capacity, headDim], values.shape == keys.shape,
            absoluteOffset + n <= capacity, absoluteOffset + n <= maxLength
        else { return nil }
        // Exclusive, materialized, row-contiguous storage only: an in-place
        // write is then visible to nothing that predates it.
        guard let kInfo = try? keys.evaluatedBufferInfo(), kInfo.isUnique, kInfo.isRowContiguous,
            let vInfo = try? values.evaluatedBufferInfo(), vInfo.isUnique, vInfo.isRowContiguous
        else { return nil }
        return (keys, values, absoluteOffset, capacity)
    }

    /// After a kernel wrote tokens `[offset, offset + n)` into the buffers
    /// `inPlaceAppendTarget` returned (`written` is any output of that
    /// kernel): advance exactly as `update` advances and return the same
    /// views, each ordered after the write by a dependency edge on `written`.
    /// The Depends nodes are built on the views (Slice primitives on the
    /// caller's stream), so the edge stays on that stream. Later rounds read
    /// the buffers only after this round's readback.
    func commitInPlaceAppend(count n: Int, written: MLXArray) -> (MLXArray, MLXArray) {
        precondition(keys != nil && values != nil && absoluteOffset + n <= capacity,
            "CBv2FullSequenceKV: in-place append without a target")
        absoluteOffset += n
        let k = keys![.ellipsis, ..<absoluteOffset, 0...]
        let v = values![.ellipsis, ..<absoluteOffset, 0...]
        return (
            depends(input: k, dependencies: [written]),
            depends(input: v, dependencies: [written])
        )
    }

    public func snapshot() -> (keys: MLXArray, values: MLXArray, offset: Int) {
        guard let keys, let values else {
            return (
                MLXArray.zeros([1, kvHeads, 0, headDim], dtype: .float16),
                MLXArray.zeros([1, kvHeads, 0, headDim], dtype: .float16),
                absoluteOffset
            )
        }
        return (
            keys[.ellipsis, ..<absoluteOffset, 0...],
            values[.ellipsis, ..<absoluteOffset, 0...],
            absoluteOffset
        )
    }

    /// Plain rollback is already value-exact (see `rollback`: the offset
    /// decrement makes the un-confirmed tail structurally unreachable and
    /// the confirmed prefix is untouched), so speculative begin/commit are
    /// the contract's default no-ops.
    public var supportsSpeculativeWrites: Bool { true }

    /// Rollback the last `n` tokens (speculative rejection). The un-confirmed
    /// tail is structurally unreachable afterwards: every view this class
    /// hands out is sliced to `..<absoluteOffset`, and the tail slots are
    /// overwritten by the next `update` before they can ever be exposed —
    /// so no zeroing pass is needed.
    public func rollback(_ n: Int) {
        precondition(n >= 0, "CBv2FullSequenceKV.rollback: negative n")
        precondition(
            n <= absoluteOffset,
            "CBv2FullSequenceKV.rollback(\(n)) exceeds retained \(absoluteOffset)")
        absoluteOffset -= n
    }

    func cbv2InnerState() -> [MLXArray] {
        [keys, values].compactMap { $0 }
    }

    // MARK: - Private

    /// `storage[..., offset ..< offset + n, :] = update` as one slice update
    /// of the 4-D update as given. The subscript form drops the update's
    /// leading singleton and reshapes it back, which copies a strided update
    /// (the transposed K/V projection views) before the slice write; here the
    /// slice write reads the strided view directly. The stored values are the
    /// same. `NEMOTRON_KV_DIRECT_WRITE=0` keeps the subscript form.
    static let directWrite =
        ProcessInfo.processInfo.environment["NEMOTRON_KV_DIRECT_WRITE"] != "0"

    private static func write(_ storage: MLXArray, _ update: MLXArray, at offset: Int) -> MLXArray {
        guard directWrite, storage.ndim == 4, update.ndim == 4,
              update.dim(0) == storage.dim(0), update.dim(1) == storage.dim(1),
              update.dim(3) == storage.dim(3), update.dtype == storage.dtype
        else {
            var result = storage
            result[.ellipsis, offset ..< (offset + update.dim(2)), 0...] = update
            return result
        }
        let starts: [Int32] = [0, 0, Int32(offset), 0]
        let stops: [Int32] = [
            Int32(storage.dim(0)), Int32(storage.dim(1)), Int32(offset + update.dim(2)),
            Int32(storage.dim(3)),
        ]
        let strides: [Int32] = [1, 1, 1, 1]
        var result = mlx_array_new()
        let rc = mlx_slice_update(
            &result, storage.ctx, update.ctx, starts, starts.count, stops, stops.count,
            strides, strides.count, StreamOrDevice.default.ctx)
        precondition(rc == 0, "CBv2FullSequenceKV: mlx_slice_update failed with rc=\(rc)")
        return MLXArray(result)
    }

    /// `[update, zeros]` along the token axis as one concatenate: the update
    /// rows are copied once and only the slack rows are written as zeros. The
    /// slack is a host 16-bit zero word viewed as the K/V dtype and broadcast
    /// (no buffer, no launch of its own; `MLXArray.zeros` would fill one), so
    /// the concatenate's two copies are the only launches. 16-bit dtypes only.
    private static func firstWrite(_ update: MLXArray, capacity: Int) -> MLXArray {
        let slack = broadcast(
            MLXArray(UInt16(0)).view(dtype: update.dtype),
            to: [update.dim(0), update.dim(1), capacity - update.dim(2), update.dim(3)])
        return concatenated([update, slack], axis: 2)
    }

    /// For the load-time check: the seed's first-write storage and the stock
    /// one (a zero storage of `capacity` rows, then `write` at offset 0).
    public static func seedFirstWriteCheck(_ update: MLXArray, capacity: Int) -> (MLXArray, MLXArray) {
        let stock = write(
            MLXArray.zeros([update.dim(0), update.dim(1), capacity, update.dim(3)], dtype: update.dtype),
            update, at: 0)
        return (firstWrite(update, capacity: capacity), stock)
    }

    private func ensureCapacity(_ needed: Int, keyTemplate: MLXArray, valueTemplate: MLXArray) {
        if keys == nil {
            capacity = min(maxLength, max(capacity, needed))
            keys = MLXArray.zeros(
                [1, kvHeads, capacity, keyTemplate.dim(3)], dtype: keyTemplate.dtype)
            values = MLXArray.zeros(
                [1, kvHeads, capacity, valueTemplate.dim(3)], dtype: valueTemplate.dtype)
            return
        }
        guard needed > capacity else { return }

        // Grow by doubling, capped at maxLength. The concat copies the old
        // buffer once per doubling — amortized O(1) per appended token.
        let newCapacity = min(maxLength, max(capacity * 2, needed))
        let growth = newCapacity - capacity
        keys = concatenated(
            [keys!, MLXArray.zeros([1, kvHeads, growth, keys!.dim(3)], dtype: keys!.dtype)],
            axis: 2)
        values = concatenated(
            [values!, MLXArray.zeros([1, kvHeads, growth, values!.dim(3)], dtype: values!.dtype)],
            axis: 2)
        capacity = newCapacity
    }
}
