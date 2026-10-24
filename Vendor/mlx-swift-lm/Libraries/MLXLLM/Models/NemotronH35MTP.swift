import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Drafts score only the first half of the target head's rows; the target
/// still scores every emitted token over the full vocabulary. Ids below 1000
/// are control tokens (both EOS ids included), 1000..<1256 the raw bytes, and
/// every later id is a BPE merge in rank order, so the first half keeps all
/// control and byte ids and the earliest merges. A row prefix of a row-major
/// quantized matrix is a strided view (no copy) that runs the same qmv kernel,
/// so each kept logit equals the full head's. Held outside the module tree so
/// it registers no parameters.
private final class NemotronMTPDraftHead {
    let target: NemotronHModel

    init(target: NemotronHModel) { self.target = target }

    lazy var prefix: QuantizedLinear? = {
        guard let head = self.target.lmHead as? QuantizedLinear else { return nil }
        let rows = 0..<(head.weight.dim(0) / 2)
        return QuantizedLinear(
            weight: head.weight[rows], bias: head.bias?[rows],
            scales: head.scales[rows], biases: head.biases?[rows],
            groupSize: head.groupSize, bits: head.bits, mode: head.mode)
    }()

    func logits(_ hidden: MLXArray) -> MLXArray {
        prefix?.callAsFunction(hidden) ?? target.logits(hidden)
    }

    /// Multi-row twin for the post-verify head rows: the same prefix view
    /// through the multi-row qmv, so each row's argmax matches the one-token
    /// draft's over the same rows. The full-head chain is kept as fallback.
    func rowLogits(_ rows: MLXArray) -> MLXArray {
        if let prefix { return nemotronMTPLinearRows(rows, prefix) }
        return target.lmHead.map { nemotronMTPHeadRows(rows, $0) } ?? target.logits(rows)
    }
}

/// Request-local embedded assistant with native paged attention. Each draft
/// retains the ordinary one-token shape; the target verifies every proposal.
/// Trusted shifted prompt history primes the assistant before its first
/// proposal and is preserved by the durable prefix checkpoint codec.
public final class NemotronH35MTPAssistant: Module, CBv2MTPRequestStatefulDrafter, @unchecked Sendable {
    let module: NemotronH35MTPModule
    let target: NemotronH35Model
    private let draftHead: NemotronMTPDraftHead
    private let kvOnlyHistory: Bool
    private let primePromptHistory: Bool
    private let draftLimit: Int
    public var mtpTargetIdentity: ObjectIdentifier? { ObjectIdentifier(target) }
    public var requiredVerificationMode: CBv2MTPVerificationMode? {
        ProcessInfo.processInfo.environment["DARKBLOOM_NEMOTRON35_MTP_CAPTURE_VERIFY"] != "0"
            ? .rectangularExact : .serialTarget
    }
    public var maximumDraftTokens: Int? { draftLimit }
    public var maximumSpeculativeBatch: Int? { 1 }
    public var supportsTargetPrefixAcceptance: Bool { true }
    public var prefersBatchedRectangularAttention: Bool { true }
    public var requestStateBytesPerToken: Int {
        let a = target.configuration
        // KV may promote to FP32; trusted carry/backlog and graph roots are
        // charged conservatively in addition to both attention buffer families.
        let logical = 2 * a.numKeyValueHeads
            * (a.headDim ?? a.hiddenSize / a.numAttentionHeads) * 4
            + 4 * a.hiddenSize * 4 + 16
        // The segmented pool carries one bounded segment of rounding/poison
        // slack. Admission rounds at 256 tokens, so this floor prepays that
        // physical minimum even for tiny test or future model geometries.
        return max(logical, 20 * 1024)
    }
    public var requestStateTokenGranularity: Int { 256 }
    public var requestStateTokenAllocationPadding: Int { draftLimit + 1 }

    init(target: NemotronH35Model, maximumDraftTokens: Int? = nil,
         primePromptHistory: Bool = true) {
        self.target = target
        self.draftHead = NemotronMTPDraftHead(target: target)
        self.module = NemotronH35MTPModule(target.configuration)
        self.kvOnlyHistory = ProcessInfo.processInfo.environment["DARKBLOOM_NEMOTRON35_MTP_KV_ONLY_HISTORY"] != "0"
        self.primePromptHistory = primePromptHistory
        let environment = ProcessInfo.processInfo.environment
        let requested = maximumDraftTokens
            ?? Int(environment["DARKBLOOM_NEMOTRON35_MTP_MAX_DRAFT_TOKENS"] ?? "") ?? 7
        self.draftLimit = (1...7).contains(requested) ? requested : 7
    }

    final class State: CBv2MTPRequestState {
        let owner: ObjectIdentifier
        let normalBacklogLimit: Int
        var paged: PagedState?
        var committedInputCount = 0
        var stagedInputCount = 0
        var released = false
        var started = false
        var baseOffset = 0
        var trustedOffset = 0
        var pendingTokens: [MLXArray] = []
        var pendingHidden: [MLXArray] = []
        var frontier: MLXArray?
        var savedTokens: [MLXArray] = []
        var savedHidden: [MLXArray] = []
        var savedFrontier: MLXArray?
        var roots: [MLXArray] = []
        var prestagedHidden: MLXArray?
        var prestagedResult: (tokens: MLXArray, hidden: MLXArray)?
        // The engine can build the first verify before the seed consumer has
        // replied. Keep sv out of that round and its finalize boundary.
        var canPrestageFirstDraft = false
        /// Post-verify head rows over (target token r, target hidden r).
        var speculation: (ids: MLXArray, out: MLXArray, keys: MLXArray, values: MLXArray)?
        /// One device-selected expert/head output, but the full attention K/V.
        var selectedSpeculation: (ids: MLXArray, out: MLXArray, keys: MLXArray, values: MLXArray, width: Int)?
        /// Natural acceptance from the existing packet readback, never a new sync.
        var selectedNaturalAcceptance: Int?
        /// The accepted row of `speculation` after finalize (next first draft).
        var nextFirstDraft: (ids: MLXArray, out: MLXArray, keys: MLXArray, values: MLXArray)?
        init(owner: ObjectIdentifier, normalBacklogLimit: Int) {
            self.owner = owner
            self.normalBacklogLimit = normalBacklogLimit
        }
        var cache: PagedLayerCache {
            guard let paged else {
                preconditionFailure("Nemotron MTP paged state was not configured")
            }
            return paged.cache
        }
        var cacheOffset: Int { paged?.row.absoluteOffset ?? 0 }
        var cacheSnapshot: [MLXArray] {
            guard let snapshot = paged?.row.snapshot() else { return [] }
            return [snapshot.keys, snapshot.values]
        }
        var arrays: [MLXArray] {
            (paged?.cache.innerState() ?? [])
                + pendingTokens + pendingHidden + savedTokens + savedHidden
                + [frontier, savedFrontier].compactMap { $0 } + roots
        }
        var materializedBytes: Int {
            let arraysBytes = arrays.reduce(0) { total, array in
                // A frontier or queued slice can retain its complete parent.
                // Inspect without evaluating or waiting. Shared buffers are
                // conservatively charged per root: metadata has no identity
                // with which to safely deduplicate them.
                let bytes: Int
                do {
                    bytes = max(array.nbytes, try array.evaluatedBufferInfo()?.allocatedBytes ?? 0)
                } catch {
                    return Int.max
                }
                let (sum, overflow) = total.addingReportingOverflow(bytes)
                return overflow ? Int.max : sum
            }
            let storage = paged?.backend.bytesWired ?? 0
            let (total, overflow) = arraysBytes.addingReportingOverflow(storage)
            return overflow ? Int.max : total
        }
        var hasPendingPrefillForCostAccounting: Bool {
            // Initial head allocation/JIT and catch-up after target-only work
            // are preparation, not the steady per-token draft cost.
            !started || pendingTokens.reduce(0) { $0 + $1.dim(1) } > normalBacklogLimit
        }
    }
    final class PagedState {
        let backend: PagedKVBackend
        let row: PagedSequenceKV
        let cache: PagedLayerCache
        let maximumSequenceLength: Int
        private var released = false

        init(kind: CBv2LayerKind, dtype: DType, maximumSequenceLength: Int) throws {
            self.maximumSequenceLength = maximumSequenceLength
            let elementBytes = dtype == .float32 ? 4 : 2
            let pageTokens = CBv2PagedDefaults.pageSize
            let pages = (maximumSequenceLength + pageTokens - 1) / pageTokens
            let pageBytes = 2 * kind.kvHeads * kind.headDim * pageTokens * elementBytes
            // One poison page is unavailable to requests; retain another page
            // of rounding slack so the request's admitted bound is realizable.
            let capacity = max(pageBytes * (pages + 2), 1) + 4 * 1024 * 1024
            backend = try PagedKVBackend(layerKinds: [kind], config: .init(
                capacityBytes: capacity, dtype: dtype, maxPrefillChunk: 512,
                nominalMaxSequenceLength: maximumSequenceLength,
                segmentSizeBytes: min(capacity, 4 * 1024 * 1024),
                layerDTypes: [dtype]))
            let state = try backend.makeSequenceState(
                layerKinds: [kind], promptLength: 0, maxLength: maximumSequenceLength)
            row = state[0] as! PagedSequenceKV
            cache = backend.makeLayerCaches()[0]
            cache.setRows([row])
        }

        func release() {
            guard !released else { return }
            cache.setRows([])
            backend.release([row])
            released = true
        }

        deinit { release() }
    }
    private final class Prepared: CBv2MTPPreparedCapture {}
    private func state(_ value: any CBv2MTPRequestState) -> State {
        guard let s = value as? State, s.owner == ObjectIdentifier(self), !s.released else {
            preconditionFailure("Nemotron MTP foreign or released request state")
        }
        return s
    }
    public func makeRequestState() -> any CBv2MTPRequestState {
        State(owner: ObjectIdentifier(self), normalBacklogLimit: draftLimit)
    }
    public func configureRequestState(
        _ requestState: any CBv2MTPRequestState, maximumSequenceLength: Int
    ) throws {
        let s = state(requestState)
        guard maximumSequenceLength > 0 else {
            throw CBv2KVError.backendIneligible(
                reason: "Nemotron MTP requires a positive sequence bound")
        }
        let (bound, overflow) = maximumSequenceLength.addingReportingOverflow(
            requestStateTokenAllocationPadding)
        guard !overflow else {
            throw CBv2KVError.backendIneligible(
                reason: "Nemotron MTP sequence bound overflows allocation padding")
        }
        if let paged = s.paged {
            guard paged.maximumSequenceLength >= bound else {
                throw CBv2KVError.backendIneligible(
                    reason: "Nemotron MTP state bound \(paged.maximumSequenceLength) is smaller "
                        + "than requested \(bound)")
            }
            return
        }
        let a = target.configuration
        let kind = CBv2LayerKind(
            attention: .full, headDim: a.headDim ?? a.hiddenSize / a.numAttentionHeads,
            kvHeads: a.numKeyValueHeads, queryHeads: a.numAttentionHeads,
            modelLayerIndex: 0)
        let hiddenDType = target.mtpEmbedding(MLXArray([Int32(0)]).reshaped([1, 1])).dtype
        let dtype = module.projectedKVDType(hiddenDType: hiddenDType)
        s.paged = try PagedState(kind: kind, dtype: dtype, maximumSequenceLength: bound)
    }
    public func prepare(rows: [CBv2MTPRowCapture]) -> CBv2MTPPreparedCapture { Prepared() }
    public func draftStep(tokens: MLXArray, hidden: MLXArray, prepared: CBv2MTPPreparedCapture) -> (tokens: MLXArray, hidden: MLXArray) {
        preconditionFailure("Nemotron MTP requires request-owned state")
    }
    public func observeCommittedTarget(_ observation: CBv2MTPCommittedTargetObservation, requestState: any CBv2MTPRequestState) {
        let s = state(requestState)
        if s.prestagedResult != nil { discardRound(requestState: requestState) }
        s.nextFirstDraft = nil
        precondition(s.stagedInputCount == 0)
        let n = observation.tokens.dim(1)
        precondition(observation.tokens.shape == [1, n] && observation.hidden.shape == [1, n, target.configuration.hiddenSize])
        guard n > 0 else { return }
        // Keep every known (next token, previous target hidden) pair, including
        // the bridge between prefill chunks. The final frontier pairs with the
        // next observed token or the seed supplied to draftStep.
        if s.started || primePromptHistory {
            if let previous = s.frontier {
                s.pendingTokens.append(observation.tokens[0..., 0..<1])
                s.pendingHidden.append(previous)
            }
            if n > 1 {
                s.pendingTokens.append(observation.tokens[0..., 1..<n])
                s.pendingHidden.append(observation.hidden[0..., 0..<(n - 1), 0...])
            }
        }
        s.frontier = observation.hidden[0..., (n - 1)..<n, 0...]
        s.committedInputCount += n
    }
    public func draftStep(tokens: MLXArray, hidden: MLXArray, shortlist: MLXArray?, requestState: any CBv2MTPRequestState) -> (tokens: MLXArray, hidden: MLXArray) {
        let s = state(requestState)
        if let prestaged = s.prestagedResult {
            // The first call after a prestage is the next round's first draft:
            // hand it out only for the exact carry it was built from.
            let carry = s.prestagedHidden
            s.prestagedResult = nil
            s.prestagedHidden = nil
            if s.stagedInputCount == 1, let carry, carry === hidden { return prestaged }
            discardRound(requestState: requestState)
        }
        s.nextFirstDraft = nil
        precondition(s.paged != nil, "Nemotron MTP state requires admitted sequence bound")
        precondition(s.stagedInputCount < draftLimit && tokens.shape == [1, 1] && hidden.shape == [1, 1, target.configuration.hiddenSize])
        let first = s.stagedInputCount == 0
        if first {
            s.baseOffset = s.cacheOffset
            s.paged!.row.beginSpeculativeWrite()
            s.savedTokens = s.pendingTokens
            s.savedHidden = s.pendingHidden
            s.savedFrontier = s.frontier
        }
        // Canonical fixed-width priming makes checkpoint restore independent
        // of the target scheduler's original prefill chunk boundaries. This
        // matters for quantized projections whose selected graph may depend
        // on row count. It also bounds temporary fused/K/V tensors.
        if primePromptHistory && !s.started && !s.pendingTokens.isEmpty {
            let ids = concatenated(s.pendingTokens, axis: 1)
            let rows = concatenated(s.pendingHidden, axis: 1)
            for start in stride(from: 0, to: ids.dim(1), by: 512) {
                let end = min(start + 512, ids.dim(1))
                let primed = module.appendTrustedKV(
                    hidden: rows[0..., start..<end, 0...],
                    embedding: target.mtpEmbedding(ids[0..., start..<end]), cache: s.cache)
                asyncEval(primed)
                s.roots = primed
            }
        } else {
            // Post-prefill accepted target rows retain one-token arithmetic.
            for (ids, rows) in zip(s.pendingTokens, s.pendingHidden) {
                for i in 0..<ids.dim(1) {
                    let hiddenRow = rows[0..., i..<(i + 1), 0...]
                    let embedding = target.mtpEmbedding(ids[0..., i..<(i + 1)])
                    if kvOnlyHistory {
                        s.roots += module.appendTrustedKV(
                            hidden: hiddenRow, embedding: embedding, cache: s.cache)
                    } else {
                        s.roots.append(module.pagedForward(
                            hidden: hiddenRow, embedding: embedding, cache: s.cache))
                    }
                }
            }
            // Start the trusted-history replay while the seed step is built.
            if !s.pendingTokens.isEmpty { asyncEval(s.roots) }
        }
        s.pendingTokens = []
        s.pendingHidden = []
        s.frontier = nil
        let out = module.pagedForward(
            hidden: hidden, embedding: target.mtpEmbedding(tokens), cache: s.cache)
        let ids = argMax(draftHead.logits(out)[0..., -1, 0...], axis: -1).asType(.int32)
        s.stagedInputCount += 1
        if first { s.trustedOffset = s.cacheOffset }
        s.roots.append(contentsOf: [out, ids])
        return (ids, out)
    }

    public func evaluationTargets(for requestState: any CBv2MTPRequestState) -> [MLXArray] { state(requestState).arrays }
    public func finalizeRound(requestState: any CBv2MTPRequestState, confirmedInputTokens: Int,
        committedDraftTokens: MLXArray, committedTargetHidden: MLXArray) {
        let s = state(requestState)
        precondition(s.stagedInputCount > 0 && (1...(s.stagedInputCount + 1)).contains(confirmedInputTokens))
        let n = committedDraftTokens.dim(1)
        precondition(n <= s.stagedInputCount && n <= confirmedInputTokens - 1 && committedTargetHidden.shape == [1, n, target.configuration.hiddenSize])
        // Only the first seed used authoritative target hidden. Discard every
        // chained KV row even on full acceptance, then replay accepted inputs
        // using the verifier's true hidden states before the next round.
        let speculation = s.speculation.flatMap {
            $0.ids.dim(1) == s.stagedInputCount + 1 && n < $0.ids.dim(1) ? $0 : nil
        }
        let selected = s.selectedSpeculation.flatMap {
            $0.width == s.stagedInputCount + 1 && n < $0.width ? $0 : nil
        }
        // EOS, length or a common-width clamp can shorten the committed prefix.
        // Keep its exact K/V, but reuse the selected draft only at its own row.
        let reuseSelected = selected != nil && s.selectedNaturalAcceptance == n
        if speculation != nil || selected != nil {
            // Post-verify speculation already appended rows (target token r,
            // target hidden r) after the trusted seed row. Exact-greedy
            // acceptance makes rows 0..<n the accepted replay; keep them.
            s.paged!.row.rollback(s.cacheOffset - (s.trustedOffset + n))
        } else {
            s.paged!.row.rollback(s.cacheOffset - s.trustedOffset)
            if n > 0 {
                s.pendingTokens = [committedDraftTokens]
                s.pendingHidden = [committedTargetHidden]
            }
        }
        s.committedInputCount += confirmedInputTokens
        s.canPrestageFirstDraft = s.started
        s.started = true
        s.paged!.row.commitSpeculativeWrite()
        clearRound(s)
        // Row n pairs the next seed (target token n) with the carry hidden:
        // it is the next round's first draft, already computed.
        if let selected, reuseSelected {
            s.nextFirstDraft = (
                ids: selected.ids.reshaped([1]), out: selected.out,
                keys: selected.keys[0..., 0..., n ..< (n + 1), 0...],
                values: selected.values[0..., 0..., n ..< (n + 1), 0...])
        } else {
            s.nextFirstDraft = speculation.map { spec in
                (ids: spec.ids[0..., n], out: spec.out[0..., n ..< (n + 1), 0...],
                 keys: spec.keys[0..., 0..., n ..< (n + 1), 0...],
                 values: spec.values[0..., 0..., n ..< (n + 1), 0...])
            }
        }
    }
    public func discardRound(requestState: any CBv2MTPRequestState) {
        let s = state(requestState)
        guard s.stagedInputCount > 0 else { return }
        s.paged!.row.rollback(s.cacheOffset - s.baseOffset)
        s.pendingTokens = s.savedTokens
        s.pendingHidden = s.savedHidden
        s.frontier = s.savedFrontier
        s.paged!.row.commitSpeculativeWrite()
        clearRound(s)
    }
    private func clearRound(_ s: State) {
        s.prestagedHidden = nil; s.prestagedResult = nil; s.speculation = nil
        s.selectedSpeculation = nil; s.selectedNaturalAcceptance = nil
        s.stagedInputCount = 0
        s.savedTokens = []; s.savedHidden = []; s.savedFrontier = nil; s.roots = []
    }
    public func releaseRequestState(_ requestState: any CBv2MTPRequestState) {
        guard let s = requestState as? State, s.owner == ObjectIdentifier(self), !s.released else { return }
        clearRound(s)
        s.nextFirstDraft = nil
        s.paged?.release(); s.paged = nil
        s.pendingTokens = []; s.pendingHidden = []; s.frontier = nil
        s.committedInputCount = 0; s.released = true
    }
}

extension NemotronH35MTPAssistant: CBv2MTPSelectedRowSpeculating {
    /// Called from finalize once the next round's carry is final, while the
    /// GPU is idle after the acceptance readback. Builds and submits exactly
    /// the first draft the next round would build (same ops, same inputs:
    /// trusted replay of accepted rows + one head step). `draftStep` returns
    /// it only for this carry `hidden` object; any other next transition
    /// discards it exactly like a rejected round.
    public func prestageFirstDraft(
        token: Int, hidden: MLXArray, requestState: any CBv2MTPRequestState
    ) {
        let s = state(requestState)
        let next = s.nextFirstDraft
        s.nextFirstDraft = nil
        guard s.canPrestageFirstDraft, s.started, s.stagedInputCount == 0,
              s.prestagedResult == nil, s.paged != nil,
              hidden.shape == [1, 1, target.configuration.hiddenSize]
        else { return }
        if let next, s.pendingTokens.isEmpty {
            // Post-verify speculation computed this draft (and the accepted
            // replay) already: open the round and re-append its trusted K/V
            // row inside the window, exactly where the first draft writes it.
            s.baseOffset = s.cacheOffset
            s.paged!.row.beginSpeculativeWrite()
            s.savedTokens = []; s.savedHidden = []; s.savedFrontier = s.frontier
            s.frontier = nil
            s.roots = s.cache.appendTrusted(keys: next.keys, values: next.values)
                + [next.ids, next.out]
            s.stagedInputCount = 1
            s.trustedOffset = s.cacheOffset
            asyncEval(s.roots)
            s.prestagedHidden = hidden
            s.prestagedResult = (tokens: next.ids, hidden: next.out)
            return
        }
        let result = draftStep(
            tokens: MLXArray([Int32(token)]).reshaped([1, 1]), hidden: hidden,
            shortlist: nil, requestState: requestState)
        asyncEval([result.tokens, result.hidden] + s.arrays)
        s.prestagedHidden = hidden
        s.prestagedResult = result
    }

    /// Called right after the round's verify graph is submitted (before the
    /// acceptance readback): one causal head forward over the k+1 rows
    /// (target token r, target hidden r). Whatever the acceptance a turns out
    /// to be, rows 0..<a are the accepted replay and row a is the next first
    /// draft, so finalize keeps them and nothing waits on the host. Draft
    /// values only; target work is unaffected.
    public func speculateAfterVerify(
        targetTokens: MLXArray, targetHidden: MLXArray, requestState: any CBv2MTPRequestState
    ) {
        buildPostVerifySpeculation(
            draftTokens: nil, targetTokens: targetTokens, targetHidden: targetHidden,
            requestState: requestState)
    }

    public func speculateSelectedRowAfterVerify(
        draftTokens: MLXArray, targetTokens: MLXArray, targetHidden: MLXArray,
        requestState: any CBv2MTPRequestState
    ) {
        buildPostVerifySpeculation(
            draftTokens: draftTokens, targetTokens: targetTokens, targetHidden: targetHidden,
            requestState: requestState)
    }

    public func reconcileSelectedRowSpeculation(
        acceptedDraftTokens: Int, requestState: any CBv2MTPRequestState
    ) {
        let s = state(requestState)
        if s.selectedSpeculation != nil { s.selectedNaturalAcceptance = acceptedDraftTokens }
    }

    private func buildPostVerifySpeculation(
        draftTokens: MLXArray?, targetTokens: MLXArray, targetHidden: MLXArray,
        requestState: any CBv2MTPRequestState
    ) {
        let s = state(requestState)
        let width = s.stagedInputCount + 1
        guard s.started, kvOnlyHistory, s.paged != nil, s.stagedInputCount > 0, s.speculation == nil,
              s.selectedSpeculation == nil,
              s.prestagedResult == nil, targetTokens.shape == [1, width],
              targetHidden.shape == [1, width, target.configuration.hiddenSize]
        else { return }
        let selectedRow: MLXArray?
        if let draftTokens {
            guard draftTokens.shape == [1, width - 1], draftTokens.dtype == .int32,
                  targetTokens.dtype == .int32 else { return }
            // Products of 0/1 equality flags remain one exactly until the first
            // mismatch. Their integer sum is the same prefix as the host walk.
            // Do not count isolated matches after a rejection or read a scalar.
            let matches = (draftTokens .== targetTokens[0..., 0 ..< (width - 1)]).asType(.int32)
            selectedRow = cumprod(matches, axis: 1).sum(axis: 1).asType(.int32)
        } else {
            selectedRow = nil
        }
        // Chained draft rows are never trusted; drop them before appending.
        s.paged!.row.rollback(s.cacheOffset - s.trustedOffset)
        // Attend the rows one query column at a time through the decode
        // kernel the one-token draft already uses (the verify column path):
        // each row is then bitwise a one-token draft step, and the kernel
        // choice does not depend on the KV length.
        let cache = s.cache
        if NemotronH35MTPSpeculation.columnAttention {
            cache.mtpSerializesRectangularAttention = true
            cache.mtpBatchesRectangularAttention = prefersBatchedRectangularAttention
        }
        defer {
            cache.mtpSerializesRectangularAttention = false
            cache.mtpBatchesRectangularAttention = false
        }
        let rows = module.pagedForwardRows(
            hidden: targetHidden, embedding: target.mtpEmbedding(targetTokens), cache: s.cache,
            selectedRow: selectedRow)
        let logits = draftHead.rowLogits(rows.output)
        let ids = argMax(logits, axis: -1).asType(.int32)
        if selectedRow != nil {
            s.selectedSpeculation = (ids: ids, out: rows.output, keys: rows.keys,
                                    values: rows.values, width: width)
        } else {
            s.speculation = (ids: ids, out: rows.output, keys: rows.keys, values: rows.values)
        }
        let roots = [ids, rows.output] + s.cache.innerState()
        s.roots += roots
        asyncEval(roots)
    }
}

enum NemotronH35MTPSpeculation {
    /// DARKBLOOM_NEMOTRON35_MTP_SV_COLUMN_ATTENTION=0 attends post-verify rows
    /// through the prompt-chunk SDPA path instead (KV-length-dependent kernels).
    static let columnAttention = ProcessInfo.processInfo.environment[
        "DARKBLOOM_NEMOTRON35_MTP_SV_COLUMN_ATTENTION"] != "0"
    /// DARKBLOOM_NEMOTRON35_MTP_SV_WARMUP=0 skips the load-time warm-up below.
    static let warmUp = ProcessInfo.processInfo.environment[
        "DARKBLOOM_NEMOTRON35_MTP_SV_WARMUP"] != "0"
}

extension NemotronH35MTPAssistant {
    /// Load-time JIT of the post-verify speculation path at every row width a
    /// round can request (2...4 = depth 1...3, tail-clamped rounds included):
    /// the head's attention rows, device prefix selection, one-row expert/LM
    /// head tail and argmax, finalize's
    /// row slices and prestage's re-append + handout. Runs the real entry
    /// points on throwaway request state with zero inputs; values are
    /// discarded. The one-token draft and the target paths are left to the
    /// general warm-up.
    func warmPostVerifySpeculation() {
        guard NemotronH35MTPSpeculation.warmUp, kvOnlyHistory,
              CBv2MTPFirstDraftPrestage.speculateAfterVerify
        else { return }
        let a = target.configuration
        let headDim = a.headDim ?? a.hiddenSize / a.numAttentionHeads
        let hiddenDType = target.mtpEmbedding(MLXArray([Int32(0)]).reshaped([1, 1])).dtype
        let kvDType = module.projectedKVDType(hiddenDType: hiddenDType)
        func kv(_ n: Int) -> MLXArray {
            MLXArray.zeros([1, a.numKeyValueHeads, n, headDim], dtype: kvDType)
        }
        // A decode-sized trusted history (kernel choice does not depend on it).
        let history = 600
        for width in 2 ... min(draftLimit + 1, 4) {
            let requestState = makeRequestState()
            defer { releaseRequestState(requestState) }
            do {
                try configureRequestState(requestState, maximumSequenceLength: history + 2 * width)
            } catch { return }
            let s = state(requestState)
            eval(s.cache.appendTrusted(keys: kv(history), values: kv(history)))
            s.started = true
            // Open a round as draftStep does: trusted seed row, then the
            // chained draft rows that speculation rolls back.
            s.baseOffset = s.cacheOffset
            s.paged!.row.beginSpeculativeWrite()
            s.roots = s.cache.appendTrusted(keys: kv(1), values: kv(1))
            s.trustedOffset = s.cacheOffset
            if width > 2 { s.roots += s.cache.appendTrusted(keys: kv(width - 2), values: kv(width - 2)) }
            s.stagedInputCount = width - 1
            // The same views the engine hands over: a row of the flat int32
            // acceptance packet and a row block of the verify hidden.
            let drafts = width - 1
            let packet = MLXArray.zeros([drafts + width], dtype: .int32)
            let targets = packet[drafts ..< (drafts + width)].reshaped([1, width])
            let hidden = MLXArray.zeros([1, width, a.hiddenSize], dtype: hiddenDType)
            speculateSelectedRowAfterVerify(
                draftTokens: packet[0 ..< drafts].reshaped([1, drafts]),
                targetTokens: targets[0 ..< 1], targetHidden: hidden[0 ..< 1],
                requestState: requestState)
            eval(evaluationTargets(for: requestState))
            reconcileSelectedRowSpeculation(acceptedDraftTokens: drafts, requestState: requestState)
            finalizeRound(
                requestState: requestState, confirmedInputTokens: drafts + 1,
                committedDraftTokens: MLXArray.zeros([1, drafts], dtype: .int32),
                committedTargetHidden: hidden[0..., 0 ..< drafts, 0...])
            let carry = hidden[0..., drafts ..< (drafts + 1), 0...]
            prestageFirstDraft(token: 0, hidden: carry, requestState: requestState)
            let first = draftStep(
                tokens: MLXArray.zeros([1, 1], dtype: .int32), hidden: carry, shortlist: nil,
                requestState: requestState)
            eval([first.tokens, first.hidden] + evaluationTargets(for: requestState))
            discardRound(requestState: requestState)
        }
    }
}
