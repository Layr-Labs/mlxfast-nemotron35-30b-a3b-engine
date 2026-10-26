// EngineLoopV2+MTPFinalize.swift
//
// Finalize-time target-authoritative acceptance, streaming, and KV rollback.

import Foundation
import MLX

extension EngineLoopV2 {
    /// Minimum target top-K probability mass (parts-per-million) at the
    /// carry position before the next draft may score only the shortlist
    /// rows. Below this the shortlist would too often miss the token the
    /// target actually produces next, costing acceptance for no byte win —
    /// the round simply reads the full head instead. 0.90 keeps greedy
    /// production traffic (mass typically ≥0.99 at K=256) shortlisted while
    /// flat/uncertain positions fall back.
    static let mtpShortlistMassThresholdPPM: Int32 = 900_000

    /// Runs at the step's existing host-sync boundary after ordinary sampled
    /// rows finalize and before deferred KV releases.
    func finalizeMTPRound(_ step: CBv2InFlightStep) {
        guard let mtp, let round = step.mtpRound else { return }

        // Plain prompt/decode observations may own request-state arrays that
        // the just-fenced target graph referenced. Restore or release them
        // only now, never at cancellation time while evaluation is live.
        for observation in round.committedObservationRows {
            if step.discard.contains(observation.id)
                || scheduler.record(for: observation.id) == nil
            {
                round.deferredAssistantReleases.append(
                    observation.assistantState)
            } else {
                mtp.restoreAssistantState(
                    observation.assistantState, for: observation.id)
            }
        }

        // The ordinary finalize loop has confirmed each seed row's bonus.
        if let seedHidden = round.seedHidden {
            let seedPolicyTopTwo =
                round.seedPolicyTopTwoValues?.asArray(Float.self)
            if seedPolicyTopTwo != nil { CBv2CoreInstrumentation.recordHostSync() }
            for (id, decodeIndex) in round.seedRows {
                guard !step.discard.contains(id),
                    let rec = scheduler.record(for: id)
                else { continue }
                let margin = seedPolicyTopTwo.map { values in
                    let base = decodeIndex * 2
                    return Double(values[base] - values[base + 1])
                }
                mtp.storeCarry(
                    id: id, token: rec.tokens.last!,
                    hidden: seedHidden[decodeIndex ..< (decodeIndex + 1), 0..., 0...],
                    previousTopTwoMargin: margin,
                    tokensCount: rec.tokens.count,
                    kvOffset: rec.numComputedTokens)
                round.finalizedSeedIDs.insert(id)
            }
        }

        // A completed prompt's sampled token was confirmed by the ordinary
        // finalize; its predicting hidden is the next round's carry.
        for (id, hidden) in round.prefillCarries {
            guard !step.discard.contains(id), let rec = scheduler.record(for: id),
                rec.generatedTokenCount < rec.request.maxTokens,
                let token = rec.tokens.last
            else { continue }
            mtp.storeCarry(
                id: id, token: token, hidden: hidden,
                tokensCount: rec.tokens.count, kvOffset: rec.numComputedTokens)
        }

        guard let verify = round.verify else { return }
        let k = verify.k
        // Host readbacks of the MTP round, each counted: an MTP-round
        // target verification adds one blocking eval per column at launch
        // (`EngineLoopV2+MTPTargetVerification`); a logprob segment adds
        // three readbacks (`CBv2Logprobs.assemble`); a round whose capture
        // could not be fenced adds one blocking eval (`CBv2MTPCaptureFence`
        // fallback in `EngineLoopV2+MTPExecution`).
        CBv2MTPReadbackSpin.awaitAvailable(verify.acceptancePacket)
        let host = verify.acceptancePacket.asArray(Int32.self)
        CBv2CoreInstrumentation.recordHostSync()
        let policyTopTwoHost: [Float]?
        if CBv2MTPPolicyPacket.enabled {
            policyTopTwoHost = verify.policyTopTwoValues.map { values in
                host.suffix(values.size).map { Float(bitPattern: UInt32(bitPattern: $0)) }
            }
        } else {
            policyTopTwoHost = verify.policyTopTwoValues?.asArray(Float.self)
            if policyTopTwoHost != nil { CBv2CoreInstrumentation.recordHostSync() }
        }
        let draftCount = verify.rows.count * k
        let targetWidth = 1 + k
        var anyRejected = false

        struct RowOutcome {
            let batchIndex: Int
            let metadata: CBv2MTPRoundInFlight.VerifyRow
            let rec: CBv2ScheduledRequest
            let targets: [Int]
            let accepted: Int
        }

        // Resolve each row's natural target-authoritative prefix, then choose
        // one committed width for the rectangular step.
        var outcomes: [RowOutcome] = []
        outcomes.reserveCapacity(verify.rows.count)
        var commonEmitted = targetWidth

        for (batchIndex, metadata) in verify.rows.enumerated() {
            let id = metadata.id
            if step.discard.contains(id) || scheduler.record(for: id) == nil {
                if let evaluations = verify.recurrentEvaluations[id] {
                    do {
                        for evaluation in evaluations.reversed() { try evaluation.rollback() }
                    } catch {
                        preconditionFailure(
                            "CBv2 recurrent MTP discard rollback failed for \(id): \(error)")
                    }
                }
                if let state = metadata.assistantState {
                    round.deferredAssistantReleases.append(state)
                }
                anyRejected = true
                continue
            }
            let rec = scheduler.record(for: id)!
            let drafts = (0 ..< k).map { Int(host[batchIndex * k + $0]) }
            let targets = (0 ..< targetWidth).map {
                Int(host[draftCount + batchIndex * targetWidth + $0])
            }

            var accepted = 0
            // mtpDeviceAcceptedDrafts repeats this walk on device.
            while accepted < k, targets[accepted] == drafts[accepted] { accepted += 1 }
            var naturalEmitted = accepted + 1
            naturalEmitted = min(
                naturalEmitted,
                rec.request.maxTokens - rec.generatedTokenCount)
            if let stopIndex = targets[..<naturalEmitted].firstIndex(where: {
                rec.request.stopTokens.contains($0)
            }) {
                naturalEmitted = stopIndex + 1
            }
            commonEmitted = min(commonEmitted, naturalEmitted)
            outcomes.append(
                RowOutcome(
                    batchIndex: batchIndex,
                    metadata: metadata,
                    rec: rec,
                    targets: targets,
                    accepted: accepted))
        }

        round.finalizedVerifyIDs = Set(outcomes.map { $0.metadata.id })
        round.claimedSeedCostNanos = mtp.claimPendingSeedCost(
            decodeRowBucket: mtp.planDecodeRowBucket,
            finalizedVerifyIDs: round.finalizedVerifyIDs,
            measurement: step.mtpMeasurement)

        if !outcomes.isEmpty {
            let stepAccepted = outcomes.map { min($0.accepted, commonEmitted) }.min() ?? 0
            let observedDrafts =
                commonEmitted <= stepAccepted
                ? commonEmitted : min(k, stepAccepted + 1)
            mtp.recordStepAcceptance(
                drafted: k,
                accepted: stepAccepted,
                observedDrafts: observedDrafts,
                decodeRowBucket: mtp.planDecodeRowBucket,
                measurement: step.mtpMeasurement)
        }

        for outcome in outcomes {
            let batchIndex = outcome.batchIndex
            let metadata = outcome.metadata
            let id = metadata.id
            let rec = outcome.rec
            let accepted = outcome.accepted
            let emitted = Array(outcome.targets.prefix(commonEmitted))

            // Confirm in order with the same stop and length semantics as the
            // ordinary finalize loop.
            let detokenizer = detokenizers[id]
            let hasStopStrings = !rec.request.stopStrings.isEmpty
            var kept: [Int] = []
            var textPieces: [String] = []
            var finishReason: CBv2FinishReason?
            for token in emitted {
                scheduler.recordSampled(id: id, token: token)
                kept.append(token)
                if rec.request.stopTokens.contains(token) {
                    finishReason = .stop
                    break
                }
                if hasStopStrings {
                    textPieces.append(detokenizer?.push([token]) ?? "")
                    if detokenizer?.matchedStopString == true {
                        finishReason = .stop
                        break
                    }
                }
                if rec.generatedTokenCount >= rec.request.maxTokens {
                    finishReason = .length
                    break
                }
            }

            // Correct KV and scheduler state before any terminal release.
            let confirmed = kept.count
            round.committedVerifyTokenCount += kept.filter {
                !rec.request.stopTokens.contains($0)
            }.count
            for packet in verify.diagnostics where packet.requestID == id {
                let drafts = (0..<k).map { Int(host[batchIndex * k + $0]) }
                packet.reconcile(
                    accepted: accepted, confirmed: confirmed, drafts: drafts, targets: outcome.targets)
            }
            let rejected = (1 + k) - confirmed
            // Head work first. The assistant's finalize and the next round's
            // first draft read only the accepted draft inputs and the
            // verifier's hidden rows, never the target's committed recurrent
            // state or KV, so submitting the draft before the target-side
            // commit below (strict-prefix replay graphs on a partial
            // acceptance, KV rollback, bookkeeping) lets the GPU run it while
            // the host finishes. The assistant still receives only accepted
            // draft inputs and their trusted target hidden rows; every
            // speculative head suffix is discarded.
            let committedDraftCount = min(accepted, max(0, confirmed - 1))
            if verify.promptLookup {
                mtp.observeLookupRound(id, drafted: k, accepted: accepted)
                // No head step ran: queue the committed window as trusted
                // target history (seed, accepted drafts) with its verifier
                // hidden rows, like any other target-only observation.
                if let state = metadata.assistantState,
                    let observer = mtp.drafter as? any CBv2MTPHostDraftObserving
                {
                    (mtp.drafter as? any CBv2MTPCommittedTokenObserving)?.observeCommittedTokens(
                        kept, requestState: state)
                    let inputs = 1 + committedDraftCount
                    observer.observeHostDraftedRound(
                        seedHidden: verify.lookupSeedHidden[batchIndex],
                        tokens: MLXArray(
                            verify.lookupWindows[batchIndex].prefix(inputs).map { Int32($0) }
                        ).reshaped([1, inputs]),
                        hidden: verify.lastHidden[
                            batchIndex ..< batchIndex + 1, 0 ..< inputs, 0...],
                        requestState: state)
                }
            } else if let stateful = mtp.drafter as? any CBv2MTPRequestStatefulDrafter,
                let state = metadata.assistantState
            {
                (stateful as? any CBv2MTPCommittedTokenObserving)?.observeCommittedTokens(
                    kept, requestState: state)
                (stateful as? any CBv2MTPDeviceAcceptedPrestaging)?.noteHostAcceptedDrafts(
                    accepted, requestState: state)
                stateful.finalizeRound(
                    requestState: state,
                    confirmedInputTokens: 1 + committedDraftCount,
                    committedDraftTokens: verify.draftIDs[
                        batchIndex ..< batchIndex + 1,
                        0 ..< committedDraftCount],
                    committedTargetHidden: verify.lastHidden[
                        batchIndex ..< batchIndex + 1,
                        0 ..< committedDraftCount,
                        0...])
            }
            var carryColumn = 0
            var carryHidden: MLXArray?
            var carryShortlist: MLXArray?
            if finishReason == nil {
                carryColumn = CBv2MTPHiddenIndex.carryColumn(
                    targetOutputIndex: confirmed - 1, draftDepth: k)
                // Shortlist coverage gate: hand the accepted position's
                // target top-K ids to the next draft only when their
                // probability mass (packet tail, parts-per-million) clears
                // the threshold; otherwise the draft falls back to the full
                // head for that round.
                if let shortlistIDs = verify.shortlistIDs {
                    let massBase = draftCount + verify.rows.count * targetWidth
                    let mass = host[massBase + batchIndex * targetWidth + carryColumn]
                    if mass >= Self.mtpShortlistMassThresholdPPM {
                        carryShortlist = shortlistIDs[batchIndex, carryColumn]
                    }
                }
                let hidden = verify.lastHidden[
                    batchIndex ..< (batchIndex + 1),
                    carryColumn ..< (carryColumn + 1), 0...]
                carryHidden = hidden
                // The acceptance packet was just read back, so the GPU is idle
                // while the host finishes finalize and plans the next step:
                // let the drafter submit the next round's first draft now.
                // A row whose next round verifies prompt-lookup drafts runs
                // no head step then; leave its first draft unstaged.
                let fixedDepth = mtp.config.fixedDraftTokens ?? 0
                if CBv2MTPFirstDraftPrestage.enabled, carryShortlist == nil,
                    let prestager = mtp.drafter as? any CBv2MTPFirstDraftPrestaging,
                    let state = metadata.assistantState,
                    !mtp.wouldLookup(
                        rec,
                        planDepth: min(fixedDepth, rec.request.maxTokens - rec.generatedTokenCount - 1))
                {
                    // A fixed depth plans the next round with exactly this
                    // many drafts whenever enough tokens remain for it.
                    let fixed = fixedDepth
                    let depth = CBv2MTPFirstDraftPrestage.chained
                        && rec.request.maxTokens - rec.generatedTokenCount >= fixed ? fixed : 1
                    CBv2MTPSubmitMerge.begin()
                    prestager.prestageDrafts(
                        token: kept[confirmed - 1], hidden: hidden, requestState: state,
                        depth: depth)
                }
            }
            if let evaluations = verify.recurrentEvaluations[id] {
                if evaluations.count == 1, evaluations[0].isCaptured {
                    // Capture-verify: one transaction spans the window. The
                    // accepted prefix commits by device-side selection of the
                    // captured state at position `confirmed`; zero confirmed
                    // tokens restore the pre-verify snapshot by rollback.
                    do {
                        if confirmed > 0 {
                            try evaluations[0].commit(keepPositions: confirmed)
                        } else {
                            try evaluations[0].rollback()
                        }
                    } catch {
                        preconditionFailure(
                            "CBv2 captured MTP finalization failed for \(id): \(error)")
                    }
                } else {
                    precondition(
                        evaluations.count == 1 + k,
                        "CBv2 recurrent MTP verification generation count mismatch")
                    do {
                        for evaluation in evaluations.suffix(rejected).reversed() {
                            try evaluation.rollback()
                        }
                        for evaluation in evaluations.prefix(confirmed) {
                            try evaluation.commit()
                        }
                    } catch {
                        preconditionFailure(
                            "CBv2 recurrent MTP finalization failed for \(id): \(error)")
                    }
                }
            }
            if rejected > 0 {
                for sequence in metadata.storageRows { sequence.rollback(rejected) }
                anyRejected = true
            }
            for sequence in metadata.storageRows { sequence.commitSpeculativeWrite() }
            if rejected > 0 {
                scheduler.discardPendingSamples(id: id, count: rejected)
                scheduler.rollbackComputed(id: id, tokens: rejected)
            }
            // The speculative suffix is now reconciled in BOTH the page tables
            // and scheduler record. Publish only the accepted frontier; doing
            // this before rollback would make a rejected block cache-visible.
            let launchedEnd = step.computedRanges[id]?.upperBound ?? rec.numComputedTokens
            publishFinalizedResidentBlocks(
                requestID: id,
                safeComputedEnd: min(launchedEnd, rec.numComputedTokens))

            if hasStopStrings {
                stream(for: id)?.emit(
                    .delta(text: textPieces.joined(), tokens: kept, logprobs: nil))
            } else {
                let stream = stream(for: id)
                stream?.reserveEmission()
                let endsWithStopToken = finishReason == .stop
                let pushTokens = endsWithStopToken ? Array(kept.dropLast()) : kept
                let allTokens = kept
                detokQueue.async {
                    let text = pushTokens.isEmpty ? "" : (detokenizer?.push(pushTokens) ?? "")
                    stream?.emit(
                        .delta(text: text, tokens: allTokens, logprobs: nil),
                        consumingReservation: true)
                }
            }

            let observedAccepted = min(accepted, confirmed)
            // Per-request timing: this verify row confirmed at the step's
            // readback-done instant (already read by `finalize`).
            rec.recordStepParticipation(step: step, batchRows: step.tokenProducingRows)
            rec.recordMTPRound(drafted: k, accepted: observedAccepted)
            if confirmed > 0 {
                rec.timing.decodeSteps &+= 1
                decodeRowsTotal = Self.saturatingAdd(decodeRowsTotal, 1)
            }
            // Acceptance/rollback audit record (observability): every value is
            // already on the host at this boundary. The scheduler fields are
            // read AFTER recordSampled/rollbackComputed above, so the record
            // states the row's post-round accounting — the boundary invariant
            // a consumer checks is
            // `numComputedAfter == tokensCountAfter - 1`.
            mtp.recordRound(
                drafted: k, accepted: observedAccepted, emitted: confirmed,
                audit: CBv2MTPRoundAuditRecord(
                    requestID: id.raw,
                    k: k,
                    draftTokens: Array(
                        host[batchIndex * k ..< (batchIndex + 1) * k].map(Int.init)),
                    targetTokens: outcome.targets,
                    accepted: accepted,
                    confirmed: confirmed,
                    rejected: rejected,
                    tokensCountAfter: rec.tokens.count,
                    numComputedAfter: rec.numComputedTokens,
                    generatedAfter: rec.generatedTokenCount,
                    finishReason: finishReason.map { String(describing: $0) }))
            let rejectionObserved = accepted < k && confirmed > accepted
            let acceptanceTruncated =
                !rejectionObserved && confirmed <= accepted && confirmed < k
            mtp.observeRequestAcceptance(
                id: id,
                draftedDepth: k,
                acceptedDepth: observedAccepted,
                rejectionObserved: rejectionObserved,
                endedByTruncation: acceptanceTruncated)

            if let finishReason {
                if let state = metadata.assistantState {
                    mtp.releaseDetachedAssistantState(state)
                }
                finishRequest(id, reason: finishReason)
            } else {
                if let state = metadata.assistantState {
                    mtp.restoreAssistantState(state, for: id)
                }
                // No inline deadline check: an MTP row that just confirmed
                // tokens is making progress and its decode lease is refreshed
                // in `refreshProgressLeases` (run at the end of the enclosing
                // `finalize`, after this round). Lease expiry is evaluated
                // centrally in `processLeaseExpiry` — identical typed-terminal
                // semantics to the ordinary decode path.
                // The carry and its first draft were prepared above.
                let hiddenColumn = carryColumn
                let previousTopTwoMargin: Double? = policyTopTwoHost.map { values in
                    let base =
                        (batchIndex * targetWidth + hiddenColumn) * 2
                    return Double(values[base] - values[base + 1])
                }
                mtp.storeCarry(
                    id: id, token: kept[confirmed - 1],
                    hidden: carryHidden!,
                    shortlist: carryShortlist,
                    previousTopTwoMargin: previousTopTwoMargin,
                    // A lookup round's observation already ends on this
                    // carry's hidden.
                    needsHistoryTransition: mtp.tracksPersistentHistory && !verify.promptLookup,
                    tokensCount: rec.tokens.count,
                    kvOffset: rec.numComputedTokens)
            }
        }

        // Verify rows emitted tokens without passing through the sampler;
        // drop their configured row state so the next ordinary sample
        // reconfigures from confirmed history (exact penalty counts and
        // per-request RNG step indices).
        if !outcomes.isEmpty {
            sampler.mtpRoundDidCommit(requestIDs: outcomes.map { $0.metadata.id })
        }

        // Rejected suffixes advanced eager device offsets past host truth.
        if anyRejected {
            eagerCompositionStale = true
        }
    }
}

/// Optional request-stateful drafter capability: submit the next round's
/// first draft from finalize, as soon as its carry (token, target hidden) is
/// final. The drafter must hand that draft out only to the next first
/// `draftStep` for this exact carry `hidden` object and otherwise discard it
/// (as a rejected round) before any other state transition. Draft values are
/// unchanged; only their submission moves earlier.
public protocol CBv2MTPFirstDraftPrestaging: AnyObject {
    func prestageFirstDraft(
        token: Int, hidden: MLXArray, requestState: any CBv2MTPRequestState)
    /// Right after a round's verify graph is submitted: `targetTokens`
    /// [1, k+1] are the verifier's per-position tokens (the accept walk's
    /// targets) and `targetHidden` [1, k+1, H] its hidden rows. The drafter
    /// may queue acceptance-independent head work on them.
    func speculateAfterVerify(
        targetTokens: MLXArray, targetHidden: MLXArray, requestState: any CBv2MTPRequestState)
    /// `prestageFirstDraft` plus, when `depth` > 1, the drafts the next
    /// round would chain on it. The same hand-out and discard rules apply to
    /// every prestaged draft.
    func prestageDrafts(
        token: Int, hidden: MLXArray, requestState: any CBv2MTPRequestState, depth: Int)
}

extension CBv2MTPFirstDraftPrestaging {
    public func prestageDrafts(
        token: Int, hidden: MLXArray, requestState: any CBv2MTPRequestState, depth: Int
    ) {
        prestageFirstDraft(token: token, hidden: hidden, requestState: requestState)
    }
}

/// Request-stateful drafter capability for rounds whose drafts came from the
/// host (prompt lookup): no draft step ran, so the committed window [seed,
/// accepted drafts] and its verifier hidden rows are queued as trusted
/// history. `seedHidden` is the carry hidden that predicted the seed; it
/// pairs with the seed unless the queued history already ends on it.
public protocol CBv2MTPHostDraftObserving: AnyObject {
    func observeHostDraftedRound(
        seedHidden: MLXArray, tokens: MLXArray, hidden: MLXArray,
        requestState: any CBv2MTPRequestState)
}

/// Optional request-stateful drafter capability: the target tokens a request
/// commits, on the host, in order (the prompt's chunks at prefill, then each
/// round's confirmed tokens at finalize). Drafters may use them only to choose
/// how they draft; the target still verifies every proposal.
public protocol CBv2MTPCommittedTokenObserving: AnyObject {
    func observeCommittedTokens(_ tokens: [Int], requestState: any CBv2MTPRequestState)
}

/// Optional: queue the next round's first draft right behind the verify, for
/// the row the device picks from the acceptance packet, so the head does not
/// wait for the host's finalize. The host still decides: finalize reports its
/// own accept count, and the drafter adopts the queued draft only when that
/// count selects the same row (otherwise it builds it as before).
public protocol CBv2MTPDeviceAcceptedPrestaging: CBv2MTPFirstDraftPrestaging {
    /// After `speculateAfterVerify`. acceptedDrafts: [1] int32, the leading
    /// draft matches in the acceptance packet.
    func speculateNextFirstDraft(acceptedDrafts: MLXArray, requestState: any CBv2MTPRequestState)
    /// Before `finalizeRound`: the host walk's accept count.
    func noteHostAcceptedDrafts(_ accepted: Int, requestState: any CBv2MTPRequestState)
}

/// The acceptance readback polls the packet's completion event instead of
/// sleeping in the event wait, so the engine thread resumes as soon as the
/// verify signals rather than after a thread wake-up. Bounded: after the cap it
/// falls back to the blocking read. Only when the waiting changes, never what
/// is read. CBV2_MTP_SPIN_READBACK=0 keeps the blocking wait.
enum CBv2MTPReadbackSpin {
    static let enabled =
        ProcessInfo.processInfo.environment["CBV2_MTP_SPIN_READBACK"] != "0"
    static let capSeconds = 0.05

    static func awaitAvailable(_ packet: MLXArray) {
        guard enabled else { return }
        let start = CFAbsoluteTimeGetCurrent()
        while true {
            // An inspection error ends the poll; the blocking read reports it.
            do {
                if try packet.evaluatedBufferInfo() != nil { return }
            } catch {
                return
            }
            if CFAbsoluteTimeGetCurrent() - start > capSeconds { return }
        }
    }
}

public enum CBv2MTPFirstDraftPrestage {
    /// CBV2_MTP_PRESTAGE_FIRST_DRAFT=0 restores submission at round build.
    public static let enabled =
        ProcessInfo.processInfo.environment["CBV2_MTP_PRESTAGE_FIRST_DRAFT"] != "0"
    /// CBV2_MTP_SPECULATE_AFTER_VERIFY=0 disables the post-verify head rows.
    public static let speculateAfterVerify = enabled
        && ProcessInfo.processInfo.environment["CBV2_MTP_SPECULATE_AFTER_VERIFY"] != "0"
    /// NEMOTRON_MTP_PRESTAGE_SECOND=0 prestages only the first draft.
    public static let chained = enabled
        && ProcessInfo.processInfo.environment["NEMOTRON_MTP_PRESTAGE_SECOND"] != "0"
    /// CBV2_MTP_DEVICE_ACCEPTED_FIRST_DRAFT=0 restores host-selected tail submission.
    public static let deviceAccepted = speculateAfterVerify
        && ProcessInfo.processInfo.environment["CBV2_MTP_DEVICE_ACCEPTED_FIRST_DRAFT"] != "0"
}

/// Head-chain submissions grouped into fewer `asyncEval` calls (evaluation
/// order only): finalize's prestaged drafts and the round-build draft steps
/// ride the verify's first submission, so the host reaches that submission
public enum CBv2MTPSubmitMerge {
    public static let enabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_MERGE_SUBMITS"] != "0"
    nonisolated(unsafe) private static var held: [MLXArray]? = nil

    /// Start holding submissions (engine thread only).
    public static func begin() {
        if enabled && held == nil { held = [] }
    }

    /// `asyncEval(arrays)`, or hold them for the next `take`/`flush`.
    public static func submit(_ arrays: [MLXArray]) {
        if held != nil { held!.append(contentsOf: arrays) } else { asyncEval(arrays) }
    }

    /// The held arrays (empty when none); holding ends.
    public static func take() -> [MLXArray] {
        defer { held = nil }
        return held ?? []
    }

    /// Submit the held arrays on their own; holding ends.
    public static func flush() {
        let arrays = take()
        if !arrays.isEmpty { asyncEval(arrays) }
    }
}

extension EngineLoopV2 {
    /// Queue the drafter's acceptance-independent work behind a submitted
    /// verify. It depends on the verify outputs, so it runs after them; it is
    /// a separate submission, so the acceptance readback does not wait on it.
    func mtpSpeculateAfterVerify(
        _ verify: CBv2MTPRoundInFlight.Verify, driver mtp: CBv2MTPRoundDriver
    ) {
        guard CBv2MTPFirstDraftPrestage.speculateAfterVerify, !verify.promptLookup,
            let speculator = mtp.drafter as? any CBv2MTPFirstDraftPrestaging
        else { return }
        let batch = verify.rows.count
        let width = verify.k + 1
        let draftCount = batch * verify.k
        let targets = verify.acceptancePacket[draftCount ..< (draftCount + batch * width)]
            .reshaped([batch, width])
        CBv2MTPSubmitMerge.begin()
        defer { CBv2MTPSubmitMerge.flush() }
        for (index, row) in verify.rows.enumerated() {
            guard let state = row.assistantState else { continue }
            speculator.speculateAfterVerify(
                targetTokens: targets[index ..< (index + 1)],
                targetHidden: verify.lastHidden[index ..< (index + 1)],
                requestState: state)
            if CBv2MTPFirstDraftPrestage.deviceAccepted, verify.k > 0,
                let deviceAccepted = speculator as? any CBv2MTPDeviceAcceptedPrestaging
            {
                deviceAccepted.speculateNextFirstDraft(
                    acceptedDrafts: Self.mtpDeviceAcceptedDrafts(verify, row: index),
                    requestState: state)
            }
        }
    }

    /// The finalize accept walk on device, from the acceptance packet's own
    /// draft and target ids: [1] int32. Must stay identical to the host walk
    /// in `finalizeMTPRound`.
    static func mtpDeviceAcceptedDrafts(_ verify: CBv2MTPRoundInFlight.Verify, row: Int) -> MLXArray {
        let k = verify.k, width = k + 1, draftCount = verify.rows.count * k
        let packet = verify.acceptancePacket
        func original() -> MLXArray {
            let drafts = packet[(row * k) ..< (row * k + k)]
            let targets = packet[(draftCount + row * width) ..< (draftCount + row * width + k)]
            let matches = (targets .== drafts).asType(.int32)
            return cumprod(matches, axis: 0).sum(axis: 0, keepDims: true).asType(.int32)
        }
        guard let walked = CBv2MTPDeviceAcceptWalk.apply(
            packet, count: k, drafts: row * k, targets: draftCount + row * width)
        else { return original() }
        if CBv2CriticalPathCuts.selfTest {
            let expected = original()
            eval(walked, expected)
            precondition(
                walked.shape == expected.shape && walked.dtype == expected.dtype
                    && walked.item(Int32.self) == expected.item(Int32.self),
                "CBV2_MTP_ACCEPT_WALK: device walk differs from the stock ops")
        }
        return walked
    }
}

/// CBV2_MTP_ACCEPT_WALK: the device accept walk of one row (the number of
/// leading draft ids its targets confirm) as one launch instead of four
/// (Equal, AsType, cumprod scan, sum). Integer-only: the count of leading
/// `targets[a] == drafts[a]`, which is what the cumulative product of the
/// 0/1 matches sums to, returned as the same [1] int32.
public enum CBv2MTPDeviceAcceptWalk {
    private static let kernel = MLXFast.metalKernel(
        name: "cbv2_mtp_device_accept_walk",
        inputNames: ["packet", "bounds"], outputNames: ["accepted"],
        source: """
            if (thread_position_in_grid.x == 0) {
                int count = bounds[0];
                int drafts = bounds[1];
                int targets = bounds[2];
                int a = 0;
                while (a < count && packet[targets + a] == packet[drafts + a]) {
                    ++a;
                }
                accepted[0] = a;
            }
            """)

    /// nil (nothing built) when the switch is off or the packet is not the
    /// 1-D int32 acceptance packet holding both ranges.
    static func apply(_ packet: MLXArray, count: Int, drafts: Int, targets: Int) -> MLXArray? {
        guard CBv2CriticalPathCuts.acceptWalkKernel, packet.ndim == 1, packet.dtype == .int32,
              count > 0, drafts >= 0, targets >= 0,
              drafts + count <= packet.dim(0), targets + count <= packet.dim(0)
        else { return nil }
        return kernel(
            [packet, MLXArray([Int32(count), Int32(drafts), Int32(targets)])],
            grid: (1, 1, 1), threadGroup: (1, 1, 1),
            outputShapes: [[1]], outputDTypes: [.int32])[0]
    }

    /// Compile the pipelines at load so no JIT lands in a timed window: one
    /// packet under eight elements (constant argument space) and the depth-2
    /// packet's eleven (device argument space).
    public static func warm() {
        guard CBv2CriticalPathCuts.acceptWalkKernel, CBv2MTPFirstDraftPrestage.deviceAccepted
        else { return }
        for size in [6, 11] {
            if let walked = apply(MLXArray([Int32](repeating: 0, count: size)),
                                  count: 2, drafts: 0, targets: 2) {
                eval(walked)
            }
        }
    }
}
