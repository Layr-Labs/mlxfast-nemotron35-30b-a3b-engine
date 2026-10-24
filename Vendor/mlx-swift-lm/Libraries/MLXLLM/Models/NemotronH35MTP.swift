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

    /// Load-time screen plan (input-independent): the selected K-groups and
    /// the shared head arrays it reads in place. nil = stock argmax.
    private struct Screen {
        let weight: MLXArray, scales: MLXArray, biases: MLXArray
        let groups: MLXArray
        let groupCount: Int, chunk: Int, keep: Int
    }
    private var screen: Screen?
    private var checkSteps = 0, checkMisses = 0, checkRescoreMismatches = 0

    /// Greedy draft id [1] int32 for hidden [1, 1, K]: the screen when it
    /// qualified at load and the shapes match, else the stock argmax.
    func greedy(_ hidden: MLXArray) -> MLXArray {
        guard let screen, hidden.shape == [1, 1, NemotronHeadScreen.width],
              hidden.dtype == .bfloat16, StreamOrDevice.default == .gpu
        else { return argMax(logits(hidden)[0..., -1, 0...], axis: -1).asType(.int32) }
        let result = NemotronHeadScreen.run(
            hidden, weight: screen.weight, scales: screen.scales, biases: screen.biases,
            groups: screen.groups, groupCount: screen.groupCount, chunk: screen.chunk,
            keep: screen.keep)
        if NemotronHeadScreen.check { checkAgainstStock(hidden, result, screen) }
        return result.id
    }

    /// NEMOTRON_MTP_HEAD_SCREEN_CHECK=1 only: synchronous comparison with the
    /// stock argmax, plus the stock argmax's rank under the partial score
    /// (global and within 512/1024/2048/4096-row chunks) for offline tuning.
    private func checkAgainstStock(
        _ hidden: MLXArray, _ result: NemotronHeadScreen.Result, _ screen: Screen
    ) {
        let full = logits(hidden)[0, -1].asType(.float32)
        let stock = argMax(full, axis: -1).asType(.int32)
        eval(full, stock, result.id, result.scores, result.candidateIds, result.candidateValues)
        let fullValues = full.asArray(Float.self)
        let scores = result.scores.asArray(Float.self)
        let ids = result.candidateIds.asArray(Int32.self)
        let values = result.candidateValues.asArray(Float.self)
        let stockId = Int(stock.item(Int32.self)), screenId = Int(result.id.item(Int32.self))
        for (id, value) in zip(ids, values) where fullValues[Int(id)].bitPattern != value.bitPattern {
            checkRescoreMismatches += 1
        }
        let target = scores[stockId]
        func rank(_ range: Range<Int>) -> Int {
            var n = 0
            for i in range where scores[i] > target || (scores[i] == target && i < stockId) { n += 1 }
            return n
        }
        let chunkRanks = [512, 1024, 2048, 4096].map { size -> Int in
            let start = stockId / size * size
            return rank(start..<(start + size))
        }
        checkSteps += 1
        let kept = ids.contains(Int32(stockId))
        if stockId != screenId { checkMisses += 1 }
        let line = "HSCHK step \(checkSteps) stock \(stockId) screen \(screenId) kept \(kept ? 1 : 0)"
            + " rank \(rank(0..<NemotronHeadScreen.rows)) chunk \(chunkRanks.map(String.init).joined(separator: " "))"
            + " stockv \(fullValues[stockId]) screenv \(fullValues[screenId])"
            + " misses \(checkMisses) rescore_mismatch \(checkRescoreMismatches)\n"
        FileHandle.standardError.write(Data(line.utf8))
    }

    /// Choose the screen groups from the weights alone and qualify the
    /// rescore against the stock head bitwise, once, at load. gamma is the
    /// head input's final RMSNorm weight.
    func prepareScreen(gamma: MLXArray) {
        guard NemotronHeadScreen.enabled else { return }
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron mtp head screen: \(message)\n".utf8))
        }
        let environment = ProcessInfo.processInfo.environment
        let groupCount = Int(environment["NEMOTRON_MTP_HEAD_SCREEN_GROUPS"] ?? "") ?? 16
        let chunk = Int(environment["NEMOTRON_MTP_HEAD_SCREEN_CHUNK"] ?? "") ?? 1024
        let keep = Int(environment["NEMOTRON_MTP_HEAD_SCREEN_KEEP"] ?? "") ?? 32
        let k = NemotronHeadScreen.width, rows = NemotronHeadScreen.rows
        guard (2...16).contains(groupCount), groupCount % 2 == 0,
              [512, 1024, 2048].contains(chunk), keep % 4 == 0, (4...min(chunk / 8, 64)).contains(keep),
              Device.defaultDevice().deviceType == .gpu,
              let head = target.lmHead as? QuantizedLinear, type(of: head) == QuantizedLinear.self,
              head.mode == .affine, head.bits == 4, head.groupSize == 64, head.bias == nil,
              head.weight.dtype == .uint32,
              head.weight.shape == [NemotronHeadScreen.headRows, NemotronHeadScreen.wordsPerRow],
              head.scales.dtype == .bfloat16,
              head.scales.shape == [NemotronHeadScreen.headRows, NemotronHeadScreen.groupsPerRow],
              let biases = head.biases, biases.dtype == .bfloat16, biases.shape == head.scales.shape,
              let prefix, prefix.weight.dim(0) == rows,
              gamma.shape == [k]
        else { return report("off (head, shape or setting does not qualify)") }
        // Weight-only importance per K-group: sum over the group's columns of
        // gamma[c]^2 * ||W[:, c]||^2 over the draft rows.
        var columns = MLXArray.zeros([k], dtype: .float32)
        for start in stride(from: 0, to: rows, by: 8192) {
            let r = start..<(start + 8192)
            let d = dequantized(head.weight[r], scales: head.scales[r], biases: biases[r],
                                groupSize: 64, bits: 4).asType(.float32)
            columns = columns + (d * d).sum(axis: 0)
            eval(columns)
        }
        let g = gamma.asType(.float32)
        let importance = (columns * g * g).reshaped(NemotronHeadScreen.groupsPerRow, 64)
            .sum(axis: 1).asArray(Float.self)
        // Whole aligned 64-byte pairs (groups 2p, 2p + 1; rows are 1344 =
        // 21 * 64 bytes): the screen then reads 8 sectors of each row.
        let pairs = (0..<(NemotronHeadScreen.groupsPerRow / 2)).sorted {
            let a = importance[2 * $0] + importance[2 * $0 + 1]
            let b = importance[2 * $1] + importance[2 * $1 + 1]
            return a > b || (a == b && $0 < $1)
        }
        let selected = pairs.prefix(groupCount / 2).flatMap { [2 * $0, 2 * $0 + 1] }.sorted()
        let plan = Screen(
            weight: head.weight, scales: head.scales, biases: biases,
            groups: MLXArray(selected.map { UInt32($0) }),
            groupCount: groupCount, chunk: chunk, keep: keep)
        // Qualification on fixed synthetic inputs: every kept value must be
        // bitwise the stock prefix logit, and the pick the max kept value.
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        var agree = 0
        for _ in 0..<4 {
            let t = (0..<k).map { _ -> Float in
                state = state &* 6364136223846793005 &+ 1442695040888963407
                return (Float(state >> 40) / Float(1 << 24) - 0.5) * 3.4641016
            }
            let x = (MLXArray(t) * g).asType(.bfloat16).reshaped(1, 1, k)
            let result = NemotronHeadScreen.run(
                x, weight: plan.weight, scales: plan.scales, biases: plan.biases,
                groups: plan.groups, groupCount: groupCount, chunk: chunk, keep: keep)
            let full = prefix(x)[0, -1].asType(.float32)
            let stock = argMax(full, axis: -1).asType(.int32)
            eval(full, stock, result.id, result.candidateIds, result.candidateValues)
            let fullValues = full.asArray(Float.self)
            let ids = result.candidateIds.asArray(Int32.self)
            let values = result.candidateValues.asArray(Float.self)
            let id = Int(result.id.item(Int32.self))
            var best = 0
            for i in ids.indices
            where values[i] > values[best] || (values[i] == values[best] && ids[i] < ids[best]) { best = i }
            guard Set(ids).count == ids.count, ids.allSatisfy({ (0..<Int32(rows)).contains($0) }),
                  zip(ids, values).allSatisfy({ fullValues[Int($0)].bitPattern == $1.bitPattern }),
                  id == Int(ids[best])
            else { return report("off (rescore is not bitwise the stock head)") }
            if id == Int(stock.item(Int32.self)) { agree += 1 }
        }
        screen = plan
        report("on, \(groupCount) of 42 groups \(selected), keep \(keep) per \(chunk) rows,"
            + " qualified (synthetic agreement \(agree)/4)")
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

    /// Load-time preparation of the draft-only head screen (see
    /// NemotronHeadScreen); weights-only, before the shape warm-up.
    func prepareDraftHeadScreen() {
        draftHead.prepareScreen(
            gamma: (module.layers[1] as! NemotronH35MTPExpertBlock).finalNorm.weight)
    }

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
        /// The second draft chained on the prestaged first one; handed out
        /// only for the first draft's own `hidden` output object.
        var prestagedSecond: (tokens: MLXArray, hidden: MLXArray, parent: MLXArray)?
        // The engine can build the first verify before the seed consumer has
        // replied. Keep sv out of that round and its finalize boundary.
        var canPrestageFirstDraft = false
        /// Post-verify attention/KV only; finalize selects the one expert row.
        var speculation: (attention: MLXArray, keys: MLXArray, values: MLXArray)?
        /// The single expert/head continuation selected at the existing host boundary.
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
            // One request owns this pool. Reserve one complete size-class
            // segment, including its poison page, instead of evaluating and
            // synchronizing each binary-remainder segment separately.
            var storagePages = 1
            while storagePages < pages {
                let (next, overflow) = storagePages.multipliedReportingOverflow(by: 2)
                guard !overflow else {
                    throw CBv2KVError.backendIneligible(reason: "MTP segment bound overflow")
                }
                storagePages = next
            }
            let (storageLength, lengthOverflow) = storagePages.multipliedReportingOverflow(by: pageTokens)
            let (withPoison, pageOverflow) = storagePages.addingReportingOverflow(1)
            let (storageBytes, byteOverflow) = withPoison.multipliedReportingOverflow(by: pageBytes)
            guard !lengthOverflow, !pageOverflow, !byteOverflow else {
                throw CBv2KVError.backendIneligible(reason: "MTP backing geometry overflow")
            }
            let capacity = try Memory.allocationFootprintUpperBound(byteCount: storageBytes)
            backend = try PagedKVBackend(layerKinds: [kind], config: .init(
                capacityBytes: capacity, dtype: dtype, maxPrefillChunk: 512,
                nominalMaxSequenceLength: storageLength,
                segmentSizeBytes: storageBytes,
                layerDTypes: [dtype]))
            let state = try backend.makeSequenceState(
                layerKinds: [kind], promptLength: 0, maxLength: storageLength)
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
        if s.prestagedResult != nil || s.prestagedSecond != nil {
            discardRound(requestState: requestState)
        }
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
            let staged = s.prestagedSecond == nil ? 1 : 2
            if s.stagedInputCount == staged, let carry, carry === hidden { return prestaged }
            discardRound(requestState: requestState)
        } else if let second = s.prestagedSecond {
            // The second call chains on the first draft's own output.
            s.prestagedSecond = nil
            if s.stagedInputCount == 2, second.parent === hidden {
                return (second.tokens, second.hidden)
            }
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
        let ids = draftHead.greedy(out)
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
            $0.attention.dim(1) == s.stagedInputCount + 1 && n < $0.attention.dim(1) ? $0 : nil
        }
        if speculation != nil {
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
        // The engine has already read and reconciled exact target acceptance.
        // Use the ACTUALLY committed prefix, including stop/length truncation.
        // Slice a view with this host index: no device prefix scan or gather,
        // no new readback, and no expert/head work for discarded rows.
        // Build the ordinary one-token tail lazily. The existing prestage
        // submits it only when the engine has a continuing carry.
        s.nextFirstDraft = speculation.map { spec in
            let out = module.finishPagedAttentionRow(
                spec.attention[0..., n ..< (n + 1), 0...])
            let ids = draftHead.greedy(out)
            return (ids: ids, out: out,
                    keys: spec.keys[0..., 0..., n ..< (n + 1), 0...],
                    values: spec.values[0..., 0..., n ..< (n + 1), 0...])
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
        s.prestagedHidden = nil; s.prestagedResult = nil; s.prestagedSecond = nil
        s.speculation = nil
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

extension NemotronH35MTPAssistant: CBv2MTPFirstDraftPrestaging {
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
            // Attention/KV are already submitted; the selected one-token
            // expert/head tail is lazy. Re-append its trusted K/V and submit
            // the tail at this existing boundary inside the decode window.
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

    /// Also chains the next round's second draft on the prestaged first one
    /// while the GPU runs it, instead of at round build. Same ops on the same
    /// inputs as the second `draftStep`; only the submission moves earlier.
    public func prestageDrafts(
        token: Int, hidden: MLXArray, requestState: any CBv2MTPRequestState, depth: Int
    ) {
        prestageFirstDraft(token: token, hidden: hidden, requestState: requestState)
        let s = state(requestState)
        guard depth >= 2, draftLimit >= 2, let first = s.prestagedResult,
              s.prestagedSecond == nil, s.stagedInputCount == 1
        else { return }
        let out = module.pagedForward(
            hidden: first.hidden,
            embedding: target.mtpEmbedding(first.tokens.reshaped([1, 1])), cache: s.cache)
        let ids = draftHead.greedy(out)
        s.stagedInputCount += 1
        s.roots.append(contentsOf: [out, ids])
        asyncEval([ids, out] + s.arrays)
        s.prestagedSecond = (tokens: ids, hidden: out, parent: first.hidden)
    }

    /// Called right after the round's verify graph is submitted (before the
    /// acceptance readback): causal attention/KV over all k+1 row pairs.
    /// Finalize retains the committed prefix and selects one row for the
    /// ordinary one-token expert/head tail. This preserves attention overlap
    /// without computing unused expert/head rows. Target work is unaffected.
    public func speculateAfterVerify(
        targetTokens: MLXArray, targetHidden: MLXArray, requestState: any CBv2MTPRequestState
    ) {
        let s = state(requestState)
        let width = s.stagedInputCount + 1
        guard s.started, kvOnlyHistory, s.paged != nil, s.stagedInputCount > 0, s.speculation == nil,
              s.prestagedResult == nil, s.prestagedSecond == nil, targetTokens.shape == [1, width],
              targetHidden.shape == [1, width, target.configuration.hiddenSize]
        else { return }
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
        let rows = module.pagedAttentionRows(
            hidden: targetHidden, embedding: target.mtpEmbedding(targetTokens), cache: s.cache)
        s.speculation = (attention: rows.output, keys: rows.keys, values: rows.values)
        let roots = [rows.output] + s.cache.innerState()
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
    /// A private warm-up may replace one synthetic width with the same
    /// post-verify entry over a real target verification graph. This guard
    /// has the original warm-up policy and width cap, not a timed-phase gate.
    func canWarmPostVerifySpeculation(width: Int) -> Bool {
        NemotronH35MTPSpeculation.warmUp && kvOnlyHistory
            && CBv2MTPFirstDraftPrestage.speculateAfterVerify
            && (2 ... min(draftLimit + 1, 4)).contains(width)
    }

    /// Warm the remaining promoted synthetic widths exactly once each.
    /// A width already attempted by the private general warm-up is omitted;
    /// no extra request/cache transaction is added for row offsets.
    func warmPostVerifySpeculation(skippingWidth: Int? = nil) {

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
            if width == skippingWidth { continue }
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
            speculateAfterVerify(
                targetTokens: targets[0 ..< 1], targetHidden: hidden[0 ..< 1],
                requestState: requestState)
            eval(evaluationTargets(for: requestState))
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
