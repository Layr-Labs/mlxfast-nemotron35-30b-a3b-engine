// Batched target verification with the ordinary one-token Mamba recurrence.
// Projections are built once over the window. Every confirmed-prefix state is
// staged for the engine's existing transactional commit/rollback mechanism.
import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// NEMOTRON_MTP_SSM_PENDING_FOLD: a strict-prefix commit replays every Mamba
/// layer's SSM state over the accepted steps, and the grouped replay reads and
/// writes each layer's FP32 state. The replayed state is an array the commit
/// only builds; this table maps it to its operands (an evaluated anchor state
/// and the accepted steps' conv/dt tape). When the next compact verify of that
/// layer receives it as its initial state, the fused SSM launch runs those
/// steps from the anchor before its window (`NemotronHCompactSSMGated
/// .pendingKernel`, the replay kernel's per-step statements) and the replay
/// array is never evaluated. Every other reader of the committed state still
/// evaluates the replay. Strict prefixes in a row extend the pending steps up
/// to `strictSteps`. Admitted at load only after the folded launch matched replay
/// + launch bitwise for every step count (`qualify`).
final class NemotronMTPPendingSSM {
    struct Pending {
        let anchor: MLXArray, conv: MLXArray, dt: MLXArray
        let steps: Int
    }
    /// The largest folded step count compiled and checked at load (the anchor skip folds up to `anchorAt + 2`).
    static let maxSteps = 11
    /// Strict-prefix chains without the anchor skip (NEMOTRON_MTP_SSM_PENDING_STEPS, 1 ... 8).
    static let strictSteps = min(max(Int(
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_SSM_PENDING_STEPS"] ?? "") ?? 8, 1), 8)
    static let baseSteps = min(4, strictSteps)
    static let requested =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_SSM_PENDING_FOLD"] != "0"
    static let warmsSteps =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_WARMUP_PENDING"] != "0"
    /// NEMOTRON_MTP_SSM_ANCHOR_SKIP (see NemotronMTPSSMAnchor); =0 keeps the window's state store.
    static let anchorRequested =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_SSM_ANCHOR_SKIP"] != "0"
    /// A window launch with at least this many pending steps stores the state after them
    /// (NEMOTRON_MTP_SSM_ANCHOR_AT, 1 ... maxSteps - 2).
    static let anchorAt = min(max(Int(
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_SSM_ANCHOR_AT"] ?? "") ?? 6, 1), maxSteps - 2)
    nonisolated(unsafe) static var qualified = false
    nonisolated(unsafe) static var anchorQualified = false
    nonisolated(unsafe) static var limit = 0
    static var enabled: Bool { requested && qualified }
    static var anchorSkip: Bool { enabled && anchorRequested && anchorQualified }

    private final class Entry {
        weak var state: MLXArray?
        let pending: Pending
        init(_ state: MLXArray, _ pending: Pending) {
            self.state = state
            self.pending = pending
        }
    }
    private static let lock = NSLock()
    nonisolated(unsafe) private static var entries: [ObjectIdentifier: Entry] = [:]

    /// `state` is the replay of `pending.steps` steps of `pending` from its anchor.
    static func register(_ state: MLXArray, _ pending: Pending) {
        guard enabled, (1 ... limit).contains(pending.steps) else { return }
        lock.withLock {
            entries = entries.filter { $0.value.state != nil }
            entries[ObjectIdentifier(state)] = Entry(state, pending)
        }
    }

    /// The pending replay `state` stands for, if it is one (removed: a second
    /// reader evaluates the replay).
    static func take(_ state: MLXArray) -> Pending? {
        guard enabled else { return nil }
        return lock.withLock {
            guard let entry = entries.removeValue(forKey: ObjectIdentifier(state)),
                  entry.state === state else { return nil }
            return entry.pending
        }
    }

    /// Load time: compile the folded launch for every step count and admit it
    /// only if its normed rows, prepared activation and final state equal the
    /// replay followed by the plain fused launch, bitwise, on seeded operands.
    static func qualify(_ mixer: NemotronHMamba2Mixer) {
        guard requested else { return }
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron mtp ssm pending fold: \(message)\n".utf8))
        }
        let (heads, dim, groups, stateDim) =
            (mixer.numHeads, mixer.headDim, mixer.numGroups, mixer.ssmStateSize)
        var seed: UInt64 = 0x9E37_79B9_7F4A_7C15
        func values(_ count: Int, _ low: Float, _ high: Float) -> [Float] {
            (0 ..< count).map { _ -> Float in
                seed = seed &* 6364136223846793005 &+ 1442695040888963407
                return low + (high - low) * Float(seed >> 40) / Float(1 << 24)
            }
        }
        func bits(_ a: MLXArray, _ b: MLXArray) -> Bool {
            let dtype: DType = a.dtype == .float32 ? .uint32 : .uint16
            return (a.view(dtype: dtype) .== b.view(dtype: dtype)).all().item(Bool.self)
        }
        let convDim = mixer.convDim
        let projectedWidth = mixer.intermediateSize + convDim + heads
        var passed = 0
        for steps in 1 ... maxSteps {
            let pconv = MLXArray(values(steps * convDim, -2, 2), [1, steps, convDim]).asType(.bfloat16)
            let pdt = MLXArray(values(steps * heads, 0.01, 0.5), [1, steps, heads])
            let conv = MLXArray(values(3 * convDim, -2, 2), [1, 3, convDim]).asType(.bfloat16)
            let dt = MLXArray(values(3 * heads, 0.01, 0.5), [1, 3, heads])
            let anchor = MLXArray(values(heads * dim * stateDim, -0.5, 0.5), [1, heads, dim, stateDim])
            let projected = MLXArray(values(3 * projectedWidth, -3, 3), [1, 3, projectedWidth])
                .asType(.bfloat16)
            let replayed = nemotronMTPSSMCompact(
                convolution: pconv, ALog: mixer.aLog, D: mixer.D, dt: pdt, state: anchor,
                heads: heads, headDim: dim, groups: groups, stateDim: stateDim,
                steps: steps, writeOutput: false).finalState
            guard let reference = mixer.norm.compactSSMPreparedRows(
                    convolution: conv, ALog: mixer.aLog, D: mixer.D, dt: dt, state: replayed,
                    projected: projected, heads: heads, headDim: dim, groups: groups,
                    stateDim: stateDim),
                  let folded = mixer.norm.compactSSMPreparedRows(
                    convolution: conv, ALog: mixer.aLog, D: mixer.D, dt: dt, state: anchor,
                    projected: projected, heads: heads, headDim: dim, groups: groups,
                    stateDim: stateDim, pending: (pconv, pdt, steps))
            else {
                if steps <= baseSteps { return report("off (fused launch unavailable)") }
                break
            }
            eval(reference.0, reference.1, reference.2, folded.0, folded.1, folded.2)
            guard bits(reference.0, folded.0), bits(reference.1, folded.1), bits(reference.2, folded.2)
            else {
                if steps <= baseSteps { return report("off (folded launch differs at \(steps) pending steps)") }
                break
            }
            passed = steps
        }
        limit = min(passed, strictSteps)
        qualified = true
        report("on (1...\(passed) pending steps bitwise equal to replay + launch)")
        NemotronMTPSSMAnchor.qualify(mixer, passed: passed)
    }
}

/// NEMOTRON_MTP_SSM_ANCHOR_SKIP: the compact verify window's fused launch
/// (NemotronHCompactSSMGated) without its store of the state after the
/// window, which a strict prefix never reads. Every acceptance, full or
/// strict, commits the replay of its accepted steps from an anchor state
/// and registers it with NemotronMTPPendingSSM, so the next window's launch
/// folds those steps (compile-time `P`, the pending prelude). A launch whose
/// pending count is at least `NemotronMTPPendingSSM.anchorAt` stores the
/// state after its pending steps (`anchor` form; always a committed state),
/// which anchors the next tape; below that it stores no state (`none` form)
/// and the tape grows by the accepted steps (at most `anchorAt + 2`). The
/// state stays FP32 in registers through the fold, so a stored-then-reloaded
/// state and a folded one are the same bits; `qualify` checks every form and
/// whole acceptance sequences against the window-store launches before admitting it.
enum NemotronMTPSSMAnchor {
    enum Store { case none, anchor }

    static let windowStore = "        if (t == L - 1) {\n"
        + "            state_out[(n * Dh + d_idx) * Ds + s_idx] = static_cast<U>(state);\n"
        + "        }\n"
    static let windowLoop = "\nfor (int t = 0; t < L; ++t) {\n"
    static let anchorStore = """
        for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * ds_idx + i;
            if constexpr (Ds % 32 != 0) { if (s_idx >= Ds) continue; }
            state_out[(n * Dh + d_idx) * Ds + s_idx] = static_cast<U>(prior[i]);
        }

        """

    /// NEMOTRON_MTP_SSM_LOADS_FIRST=0 keeps the stores where the record text has them. Otherwise no device load
    /// of the window loop follows a device store of the same thread: each step's output (the same
    /// `static_cast<T>(acc + x_ * D[h_idx])`) is kept in `nss_y[t]` and lane 0 stores the window's outputs after
    /// the loop, and the anchor form copies the state after the pending steps to `nss_a` and stores it after the
    /// loop (the same `static_cast<U>` of the same values, to the same addresses). The window loop is fully
    /// unrolled so `nss_y` is indexed statically. `qualify` checks these launches against the window-store
    /// launches, which keep the record text.
    static let loadsFirst = ProcessInfo.processInfo.environment["NEMOTRON_MTP_SSM_LOADS_FIRST"] != "0"
    static let windowOut = "    acc = simd_sum(acc);\n"
        + "    if (thread_index_in_simdgroup == 0) {\n"
        + "        out[(t * H + n) * Dh + d_idx] = static_cast<T>(acc + x_ * D[h_idx]);\n"
        + "    }\n}\n"
    static let tailStart = "\nthreadgroup uint gr_last;\n"
    private static func loadsFirstSource(_ source: String, anchor: Bool) -> String? {
        guard source.components(separatedBy: windowOut).count == 2,
              source.components(separatedBy: windowLoop).count == 2,
              source.components(separatedBy: tailStart).count == 2 else { return nil }
        var s = source.replacingOccurrences(
            of: windowOut, with: "    acc = simd_sum(acc);\n    nss_y[t] = static_cast<T>(acc + x_ * D[h_idx]);\n}\n")
        var after = """
            if (thread_index_in_simdgroup == 0) {
                #pragma clang loop unroll(full)
                for (int t = 0; t < L; ++t) {
                    out[(t * H + n) * Dh + d_idx] = nss_y[t];
                }
            }

            """
        if anchor {
            guard s.components(separatedBy: anchorStore).count == 2 else { return nil }
            s = s.replacingOccurrences(of: anchorStore, with: """
                float nss_a[n_per_t];
                for (int i = 0; i < n_per_t; ++i) {
                    nss_a[i] = prior[i];
                }

                """)
            after += """
                for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * ds_idx + i;
                    if constexpr (Ds % 32 != 0) { if (s_idx >= Ds) continue; }
                    state_out[(n * Dh + d_idx) * Ds + s_idx] = static_cast<U>(nss_a[i]);
                }

                """
        }
        s = s.replacingOccurrences(
            of: windowLoop, with: "\nT nss_y[L];\n#pragma clang loop unroll(full)\nfor (int t = 0; t < L; ++t) {\n")
        return s.replacingOccurrences(of: tailStart, with: "\n" + after + tailStart)
    }

    /// Load-time geometry tune of these launches (NEMOTRON_SSM_TUNE): 8 d's (8 SIMD groups) per thread group, the
    /// incumbent, or 4 d's (`threadGroup` (32, 4, 1), 1024 thread groups, same grid): every SIMD group runs the same
    /// per-(h, d) text, and a norm group's last thread group is its 128th arrival instead of its 64th (the one count
    /// in the text, `arrival8` -> `arrival4`); its threads 0 ... 127 run the same tail text on the same stored
    /// outputs. `tune` checks the 4-d launches against the 8-d launches and adopts them only when
    /// `NemotronTuneAdoption.decide` admits them (at least 3% faster by median and faster in at least 75% of the
    static let tuneEnabled = ProcessInfo.processInfo.environment["NEMOTRON_SSM_TUNE"] == "1"
    static let tuneForced = Int(ProcessInfo.processInfo.environment["NEMOTRON_SSM_FORCE"] ?? "")
    /// Soft cap on the tune's load time: the timing does not start past it.
    static let tuneBudgetSeconds = 1.2 * nemotronWindowTuneBudgetScale
    nonisolated(unsafe) static var tg4Adopted = false
    static let arrival8 = "gr_last = prev == uint((512 / Dh) * (Dh / 8)) - 1u ? 1u : 0u;"
    static let arrival4 = "gr_last = prev == uint((512 / Dh) * (Dh / 4)) - 1u ? 1u : 0u;"

    /// Tail form 0 record, 1 shuffle, 2 one barrier (as NemotronHCompactSSMGated's kernels).
    private static func make(form: Int, pending: Bool, store: Store, tg4: Bool) -> MLXFast.MLXFastKernel? {
        source(form: form, pending: pending, store: store, tg4: tg4).map {
            launchKernel(name(form: form, pending: pending, store: store, tg4: tg4), $0, pending: pending, store: store)
        }
    }
    /// A launch of this family's inputs and outputs (`pending`, `store`) with `source`.
    static func launchKernel(_ name: String, _ source: String, pending: Bool, store: Store) -> MLXFast.MLXFastKernel {
        MLXFast.metalKernel(
            name: name,
            inputNames: ["conv", "A_log", "D", "dt", "state_in", "proj", "weight", "identity",
                         "eps", "counter"] + (pending ? ["pconv", "pdt"] : []),
            outputNames: store == .anchor ? ["out", "state_out", "normed", "prepared"] : ["out", "normed", "prepared"],
            source: source, mutableInputs: ["counter"])
    }
    static func name(form: Int, pending: Bool, store: Store, tg4: Bool) -> String {
        "nemotron_mtp_ssm_compact_gated_prepared" + (pending ? "_pending" : "")
            + (store == .anchor ? "_anchor" : "_nostore") + ["", "_sh", "_sh1b"][form] + (tg4 ? "_tg4" : "")
    }
    /// The text of the launch `make` builds, or nil when a statement it rewrites moved.
    static func source(form: Int, pending: Bool, store: Store, tg4: Bool) -> String? {
        let base = pending ? NemotronHCompactSSMGated.pendingSource : NemotronHCompactSSMGated.source
        guard base.components(separatedBy: windowStore).count == 2,
              base.components(separatedBy: windowLoop).count == 2,
              store == .none || pending else { return nil }
        var source = base.replacingOccurrences(of: windowStore, with: "")
        if store == .anchor {
            source = source.replacingOccurrences(of: windowLoop, with: "\n" + anchorStore + windowLoop)
        }
        if loadsFirst, let s = loadsFirstSource(source, anchor: store == .anchor) {
            source = s
        }
        guard var tailed = form == 2 ? NemotronHCompactSSMGated.rowsTail(source)
            : form == 1 ? NemotronHCompactSSMGated.shuffledTail(source) : source else { return nil }
        if tg4 {
            guard tailed.components(separatedBy: arrival8).count == 2 else { return nil }
            tailed = tailed.replacingOccurrences(of: arrival8, with: arrival4)
        }
        return tailed
    }

    /// [tg4][form][pending][anchor]
    private static let kernels: [[[[MLXFast.MLXFastKernel?]]]] = [false, true].map { tg4 in
        (0 ..< 3).map { form in
            [false, true].map { pending in
                [Store.none, .anchor].map { make(form: form, pending: pending, store: $0, tg4: tg4) }
            }
        }
    }

    /// The tail form `kernel` launches after the load-time checks.
    static var form: Int {
        NemotronHCompactSSMGated.rowsTailQualified && NemotronHCompactSSMGated.rowsKernel != nil ? 2
            : NemotronHCompactSSMGated.shuffleQualified && NemotronHCompactSSMGated.shuffleKernel != nil ? 1 : 0
    }
    /// The form `compactSSMPreparedRows` launches by default after the load-time checks.
    static func kernel(pending: Bool, store: Store, tg4: Bool = false) -> MLXFast.MLXFastKernel? {
        kernels[tg4 ? 1 : 0][form][pending ? 1 : 0][store == .anchor ? 1 : 0]
    }

    /// The pending count of the latest anchor-skip window (warm-up bookkeeping).
    nonisolated(unsafe) static var lastPending = 0

    /// Warm-up: the acceptance length for the round after one whose windows folded `pending` steps and
    /// committed `keep`, so that the anchor-skip windows reach every pending count 0 ... anchorAt + 2
    /// (the first step of a shortest path to a count not yet in `seen`); nil once every count was reached.
    static func warmKeep(pending: Int, keep: Int, seen: inout Set<Int>) -> Int? {
        let at = NemotronMTPPendingSSM.anchorAt
        func after(_ count: Int, _ keep: Int) -> Int { (count >= at ? 0 : count) + keep }
        seen.insert(pending)
        let next = after(pending, keep)
        let missing = !seen.contains(next)
        seen.insert(next)
        guard missing || seen.count < at + 3 else { return nil }
        var first: [Int: Int] = [:]
        var queue: [Int] = []
        for keep in 1 ... 3 where first[after(next, keep)] == nil {
            first[after(next, keep)] = keep
            queue.append(after(next, keep))
        }
        var index = 0
        while index < queue.count {
            let count = queue[index]
            index += 1
            if !seen.contains(count) { return first[count] }
            for keep in 1 ... 3 where first[after(count, keep)] == nil {
                first[after(count, keep)] = first[count]
                queue.append(after(count, keep))
            }
        }
        return 1
    }

    /// Load time, after the pending fold qualified 1 ... `passed` steps. With operands seeded as in
    /// NemotronMTPPendingSSM.qualify and `mixer`'s parameters, for every pending count 0 ... anchorAt + 2:
    /// the `none` and `anchor` launches against the record launch (normed rows, prepared activation),
    /// the anchor store against the replay of the pending steps, and the record launch's window state
    /// against the replays the anchor skip commits instead; then whole acceptance sequences (full and
    /// strict, every pending count) run under both policies: every window's normed rows and prepared
    /// activation, and the committed state replayed from either representation, bitwise equal.
    /// Admitted only if every case matches; otherwise the window keeps its store.
    static func qualify(_ mixer: NemotronHMamba2Mixer, passed: Int) {
        NemotronMTPPendingSSM.anchorQualified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron mtp ssm anchor skip: \(message)\n".utf8))
        }
        let at = NemotronMTPPendingSSM.anchorAt
        let top = at + 2
        guard NemotronMTPPendingSSM.anchorRequested else { return report("off (setting)") }
        guard passed >= top else { return report("off (pending fold qualified to \(passed) steps, needs \(top))") }
        guard kernel(pending: false, store: .none) != nil, kernel(pending: true, store: .none) != nil,
              kernel(pending: true, store: .anchor) != nil else { return report("off (kernel source)") }
        let (heads, dim, groups, stateDim) =
            (mixer.numHeads, mixer.headDim, mixer.numGroups, mixer.ssmStateSize)
        let convDim = mixer.convDim
        let projectedWidth = mixer.intermediateSize + convDim + heads
        var seed: UInt64 = 0xA5C3_7E21_9B4D_0F17
        func values(_ count: Int, _ low: Float, _ high: Float) -> [Float] {
            (0 ..< count).map { _ -> Float in
                seed = seed &* 6364136223846793005 &+ 1442695040888963407
                return low + (high - low) * Float(seed >> 40) / Float(1 << 24)
            }
        }
        func bits(_ a: MLXArray, _ b: MLXArray) -> Bool {
            guard a.shape == b.shape, a.dtype == b.dtype else { return false }
            let dtype: DType = a.dtype == .float32 ? .uint32 : .uint16
            return (a.view(dtype: dtype) .== b.view(dtype: dtype)).all().item(Bool.self)
        }
        typealias Tape = (conv: MLXArray, dt: MLXArray, steps: Int)
        func window(_ scale: Float) -> (conv: MLXArray, dt: MLXArray, projected: MLXArray) {
            (MLXArray(values(3 * convDim, -2 * scale, 2 * scale), [1, 3, convDim]).asType(.bfloat16),
             MLXArray(values(3 * heads, 0.01, 0.5), [1, 3, heads]),
             MLXArray(values(3 * projectedWidth, -3 * scale, 3 * scale), [1, 3, projectedWidth]).asType(.bfloat16))
        }
        func replay(_ tape: Tape?, _ state: MLXArray) -> MLXArray {
            guard let tape else { return state }
            return nemotronMTPSSMCompact(
                convolution: tape.conv, ALog: mixer.aLog, D: mixer.D, dt: tape.dt, state: state,
                heads: heads, headDim: dim, groups: groups, stateDim: stateDim,
                steps: tape.steps, writeOutput: false).finalState
        }
        func extend(_ tape: Tape?, _ w: (conv: MLXArray, dt: MLXArray, projected: MLXArray), _ keep: Int) -> Tape {
            let accepted = (w.conv[0..., ..<keep, 0...], w.dt[0..., ..<keep, 0...])
            guard let tape else { return (accepted.0, accepted.1, keep) }
            return (concatenated([tape.conv, accepted.0], axis: 1), concatenated([tape.dt, accepted.1], axis: 1),
                    tape.steps + keep)
        }
        // One fused launch per evaluation: the launches share the arrival counters.
        func record(_ w: (conv: MLXArray, dt: MLXArray, projected: MLXArray), _ state: MLXArray, _ tape: Tape?)
            -> (MLXArray, MLXArray, MLXArray)? {
            guard let r = mixer.norm.compactSSMPreparedRows(
                convolution: w.conv, ALog: mixer.aLog, D: mixer.D, dt: w.dt, state: state,
                projected: w.projected, heads: heads, headDim: dim, groups: groups, stateDim: stateDim,
                pending: tape) else { return nil }
            eval(r.0, r.1, r.2)
            return r
        }
        func skipping(_ w: (conv: MLXArray, dt: MLXArray, projected: MLXArray), _ state: MLXArray, _ tape: Tape?,
                      _ store: Store) -> (MLXArray, MLXArray, MLXArray?)? {
            guard let r = mixer.norm.compactSSMAnchorRows(
                convolution: w.conv, ALog: mixer.aLog, D: mixer.D, dt: w.dt, state: state,
                projected: w.projected, heads: heads, headDim: dim, groups: groups, stateDim: stateDim,
                pending: tape, store: store, tg4: false) else { return nil }
            eval([r.0, r.1] + (r.2.map { [$0] } ?? []))
            return r
        }
        var cases = 0
        var failure: String?
        // Per pending count.
        perCount: for steps in 0 ... top {
            for scale: Float in [1, 4] {
                let anchor = MLXArray(values(heads * dim * stateDim, -0.5, 0.5), [1, heads, dim, stateDim])
                let tape: Tape? = steps == 0 ? nil : (
                    MLXArray(values(steps * convDim, -2 * scale, 2 * scale), [1, steps, convDim]).asType(.bfloat16),
                    MLXArray(values(steps * heads, 0.01, 0.5), [1, steps, heads]), steps)
                let w = window(scale)
                guard let reference = record(w, anchor, tape), let none = skipping(w, anchor, tape, .none) else {
                    failure = "launch unavailable at \(steps) pending steps"; break perCount
                }
                var same = bits(reference.0, none.0) && bits(reference.1, none.1)
                let after = replay(tape, anchor)
                if steps > 0 {
                    guard let stored = skipping(w, anchor, tape, .anchor), let state = stored.2 else {
                        failure = "anchor launch unavailable at \(steps) pending steps"; break perCount
                    }
                    same = same && bits(reference.0, stored.0) && bits(reference.1, stored.1) && bits(state, after)
                    // A full acceptance after an anchor store: three steps from the stored state.
                    same = same && bits(reference.2, replay(extend(nil, w, 3), state))
                }
                // A full acceptance below the threshold: the tape grows by the window.
                if steps + 3 <= NemotronMTPPendingSSM.maxSteps {
                    same = same && bits(reference.2, replay(extend(tape, w, 3), anchor))
                }
                // Strict prefixes: the longer tape from the old anchor and the short one from the state after it.
                for keep in 1 ... 2 where steps > 0 && steps + keep <= NemotronMTPPendingSSM.maxSteps {
                    same = same && bits(replay(extend(tape, w, keep), anchor), replay(extend(nil, w, keep), after))
                }
                cases += 1
                if !same { failure = "differs at \(steps) pending steps"; break perCount }
            }
        }
        // Whole acceptance sequences under both policies (window store: the window's state on full acceptance,
        // strict chains up to strictSteps; anchor skip: every acceptance pending, the store at `at`).
        if failure == nil {
            var sequences: [[Int]] = []
            var steer: [Int] = [2]
            var seen = Set<Int>()
            var pendingCount = 0
            while steer.count < 6 * (top + 1), let keep = warmKeep(pending: pendingCount, keep: steer.last!, seen: &seen) {
                pendingCount = (pendingCount >= at ? 0 : pendingCount) + steer.last!
                steer.append(keep)
            }
            sequences.append(steer + [3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2])
            sequences.append((0 ..< 24).map { _ in values(1, 0, 1)[0] < 0.6 ? 3 : (values(1, 0, 1)[0] < 0.5 ? 1 : 2) })
            sequence: for (index, keeps) in sequences.enumerated() {
                let scale: Float = index == 0 ? 1 : 3
                var record11: (anchor: MLXArray, tape: Tape?) =
                    (MLXArray(values(heads * dim * stateDim, -0.5, 0.5), [1, heads, dim, stateDim]), nil)
                var skip: (anchor: MLXArray, tape: Tape?) = (record11.anchor, nil)
                for keep in keeps {
                    let w = window(scale)
                    let writes = (skip.tape?.steps ?? 0) >= at
                    guard let a = record(w, record11.anchor, record11.tape),
                          let b = skipping(w, skip.anchor, skip.tape, writes ? .anchor : .none),
                          !writes || b.2 != nil else {
                        failure = "sequence launch unavailable"; break sequence
                    }
                    var same = bits(a.0, b.0) && bits(a.1, b.1)
                    if keep == 3 {
                        record11 = (a.2, nil)
                    } else {
                        let grown = extend(record11.tape, w, keep)
                        record11 = grown.steps <= NemotronMTPPendingSSM.strictSteps
                            ? (record11.anchor, grown) : (replay(grown, record11.anchor), nil)
                    }
                    skip = writes ? (b.2!, extend(nil, w, keep)) : (skip.anchor, extend(skip.tape, w, keep))
                    same = same && (skip.tape?.steps ?? 0) <= top
                        && bits(replay(record11.tape, record11.anchor), replay(skip.tape, skip.anchor))
                    cases += 1
                    if !same { failure = "acceptance sequence \(index) differs"; break sequence }
                }
            }
        }
        if let failure { return report("off (\(failure))") }
        NemotronMTPPendingSSM.limit = top
        NemotronMTPPendingSSM.anchorQualified = true
        report("on (state stored only by windows with >= \(at) pending steps, up to \(top) pending; \(cases) cases: "
            + "normed rows, prepared activation and committed states bitwise equal to the window-store launches)")
    }

    /// Load time, after `qualify` (the 4-d launches only replace anchor-skip launches): with operands seeded as
    /// in `qualify` and `mixer`'s parameters, the 4-d launches against the 8-d launches. For every pending count
    /// 0 ... anchorAt + 2, at two scales: the `none` launch, and with pending steps the `anchor` launch
    /// (normalized rows, prepared activation, stored state). Then two whole acceptance sequences (one steered
    /// through every pending count, one random) run the anchor-skip policy once per geometry, each run from its
    /// own outputs: every window's normalized rows and prepared activation, every stored anchor and the committed
    /// state replayed from either run. Returns the cases run and the first miss (nil: every case bitwise equal).
    static func checkTG4(_ mixer: NemotronHMamba2Mixer) -> (cases: Int, failure: String?) {
        let at = NemotronMTPPendingSSM.anchorAt
        let top = at + 2
        let (heads, dim, groups, stateDim) =
            (mixer.numHeads, mixer.headDim, mixer.numGroups, mixer.ssmStateSize)
        let convDim = mixer.convDim
        let projectedWidth = mixer.intermediateSize + convDim + heads
        var seed: UInt64 = 0x3C6E_F372_FE94_F82B
        func values(_ count: Int, _ low: Float, _ high: Float) -> [Float] {
            (0 ..< count).map { _ -> Float in
                seed = seed &* 6364136223846793005 &+ 1442695040888963407
                return low + (high - low) * Float(seed >> 40) / Float(1 << 24)
            }
        }
        func bits(_ a: MLXArray, _ b: MLXArray) -> Bool {
            guard a.shape == b.shape, a.dtype == b.dtype else { return false }
            let dtype: DType = a.dtype == .float32 ? .uint32 : .uint16
            return (a.view(dtype: dtype) .== b.view(dtype: dtype)).all().item(Bool.self)
        }
        typealias Tape = (conv: MLXArray, dt: MLXArray, steps: Int)
        typealias Window = (conv: MLXArray, dt: MLXArray, projected: MLXArray)
        func window(_ scale: Float) -> Window {
            (MLXArray(values(3 * convDim, -2 * scale, 2 * scale), [1, 3, convDim]).asType(.bfloat16),
             MLXArray(values(3 * heads, 0.01, 0.5), [1, 3, heads]),
             MLXArray(values(3 * projectedWidth, -3 * scale, 3 * scale), [1, 3, projectedWidth]).asType(.bfloat16))
        }
        func replay(_ tape: Tape?, _ state: MLXArray) -> MLXArray {
            guard let tape else { return state }
            return nemotronMTPSSMCompact(
                convolution: tape.conv, ALog: mixer.aLog, D: mixer.D, dt: tape.dt, state: state,
                heads: heads, headDim: dim, groups: groups, stateDim: stateDim,
                steps: tape.steps, writeOutput: false).finalState
        }
        func extend(_ tape: Tape?, _ w: Window, _ keep: Int) -> Tape {
            let accepted = (w.conv[0..., ..<keep, 0...], w.dt[0..., ..<keep, 0...])
            guard let tape else { return (accepted.0, accepted.1, keep) }
            return (concatenated([tape.conv, accepted.0], axis: 1), concatenated([tape.dt, accepted.1], axis: 1),
                    tape.steps + keep)
        }
        // One fused launch per evaluation: the launches share the arrival counters.
        func launch(_ w: Window, _ state: MLXArray, _ tape: Tape?, _ store: Store, _ quad: Bool)
            -> (MLXArray, MLXArray, MLXArray?)? {
            guard let r = mixer.norm.compactSSMAnchorRows(
                convolution: w.conv, ALog: mixer.aLog, D: mixer.D, dt: w.dt, state: state,
                projected: w.projected, heads: heads, headDim: dim, groups: groups, stateDim: stateDim,
                pending: tape, store: store, tg4: quad) else { return nil }
            eval([r.0, r.1] + (r.2.map { [$0] } ?? []))
            return r
        }
        func same(_ a: (MLXArray, MLXArray, MLXArray?), _ b: (MLXArray, MLXArray, MLXArray?)) -> Bool {
            guard bits(a.0, b.0), bits(a.1, b.1) else { return false }
            if let x = a.2, let y = b.2 { return bits(x, y) }
            return a.2 == nil && b.2 == nil
        }
        var cases = 0
        var failure: String?
        perCount: for steps in 0 ... top {
            for scale: Float in [1, 4] {
                let anchor = MLXArray(values(heads * dim * stateDim, -0.5, 0.5), [1, heads, dim, stateDim])
                let tape: Tape? = steps == 0 ? nil : (
                    MLXArray(values(steps * convDim, -2 * scale, 2 * scale), [1, steps, convDim]).asType(.bfloat16),
                    MLXArray(values(steps * heads, 0.01, 0.5), [1, steps, heads]), steps)
                let w = window(scale)
                for store in steps == 0 ? [Store.none] : [.none, .anchor] {
                    guard let a = launch(w, anchor, tape, store, false), let b = launch(w, anchor, tape, store, true)
                    else { failure = "launch unavailable at \(steps) pending steps"; break perCount }
                    cases += 1
                    if !same(a, b) { failure = "differs at \(steps) pending steps"; break perCount }
                }
            }
        }
        if failure == nil {
            var sequences: [[Int]] = []
            var steer: [Int] = [2]
            var seen = Set<Int>()
            var pendingCount = 0
            while steer.count < 6 * (top + 1), let keep = warmKeep(pending: pendingCount, keep: steer.last!, seen: &seen) {
                pendingCount = (pendingCount >= at ? 0 : pendingCount) + steer.last!
                steer.append(keep)
            }
            sequences.append(steer + [3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2])
            sequences.append((0 ..< 24).map { _ in values(1, 0, 1)[0] < 0.6 ? 3 : (values(1, 0, 1)[0] < 0.5 ? 1 : 2) })
            sequence: for (index, keeps) in sequences.enumerated() {
                let scale: Float = index == 0 ? 1 : 3
                let start = MLXArray(values(heads * dim * stateDim, -0.5, 0.5), [1, heads, dim, stateDim])
                var runs: [(anchor: MLXArray, tape: Tape?)] = [(start, nil), (start, nil)]
                for keep in keeps {
                    let w = window(scale)
                    var outputs: [(MLXArray, MLXArray, MLXArray?)] = []
                    for g in 0 ..< 2 {
                        let writes = (runs[g].tape?.steps ?? 0) >= at
                        guard let r = launch(w, runs[g].anchor, runs[g].tape, writes ? .anchor : .none, g == 1),
                              !writes || r.2 != nil else { failure = "sequence launch unavailable"; break sequence }
                        outputs.append(r)
                        runs[g] = writes ? (r.2!, extend(nil, w, keep)) : (runs[g].anchor, extend(runs[g].tape, w, keep))
                    }
                    cases += 1
                    guard same(outputs[0], outputs[1]), (runs[1].tape?.steps ?? 0) <= top,
                          bits(replay(runs[0].tape, runs[0].anchor), replay(runs[1].tape, runs[1].anchor))
                    else { failure = "acceptance sequence \(index) differs"; break sequence }
                }
            }
        }
        return (cases, failure)
    }

    /// Load time, after the anchor-skip check and with the GPU warm, before any verify graph is traced: the 8-d and
    /// 4-d launches over every Mamba layer's parameters with seeded operands (per layer a `none` launch after 2
    /// pending steps and an `anchor` launch after `anchorAt`), first compared bit for bit (normed rows, prepared
    /// activation, stored state of every launch), then timed round robin (`NemotronTuneAdoption.order`, median) as
    /// one chain per sample (the launches share the arrival counters, so they run one after another). When
    /// `NemotronTuneAdoption.decide` admits the 4-d launches (or they are forced), `checkTG4` runs on the first Mamba
    /// layer (every pending count, both forms, two
    /// acceptance sequences) and the 4-d launches are adopted only if it passes.
    static func tune(_ model: Module) {
        tg4Adopted = false
        let t0 = DispatchTime.now().uptimeNanoseconds
        func elapsed() -> Double { Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9 }
        func report(_ message: String) {
            FileHandle.standardError.write(Data(
                "nemotron ssm tune: \(message) (\(String(format: "%.2f", elapsed())) s)\n".utf8))
        }
        guard tuneEnabled else { return report("off (setting)") }
        guard NemotronMTPPendingSSM.anchorSkip else { return report("off (anchor skip off)") }
        let mixers = model.modules().compactMap { $0 as? NemotronHMamba2Mixer }
        guard let first = mixers.first else { return report("off (no Mamba layer)") }
        guard kernel(pending: false, store: .none, tg4: true) != nil, kernel(pending: true, store: .none, tg4: true) != nil,
              kernel(pending: true, store: .anchor, tg4: true) != nil else { return report("off (kernel source)") }
        var cases = 0
        var failure: String?
        let at = NemotronMTPPendingSSM.anchorAt
        let (heads, dim, groups, stateDim) = (first.numHeads, first.headDim, first.numGroups, first.ssmStateSize)
        let convDim = first.convDim
        let projectedWidth = first.intermediateSize + convDim + heads
        let chain = 2 * mixers.count
        var medians: [Double] = []
        var verdict: NemotronTuneAdoption.Verdict? = nil
        var reps = 0
        var compared = 0
        if failure == nil, elapsed() >= tuneBudgetSeconds { failure = "timing skipped (load budget)" }
        if failure == nil {
            do {
                try withError {
                    let keys = MLXRandom.split(key: MLXRandom.key(0x55A7), into: 8)
                    let conv = MLXRandom.uniform(Float(-2) ..< 2, [1, 3, convDim], key: keys[0]).asType(.bfloat16)
                    let dt = MLXRandom.uniform(Float(0.01) ..< 0.5, [1, 3, heads], key: keys[1])
                    let projected = MLXRandom.uniform(Float(-3) ..< 3, [1, 3, projectedWidth], key: keys[2])
                        .asType(.bfloat16)
                    let tapes: [(conv: MLXArray, dt: MLXArray, steps: Int)] = [2, at].enumerated().map { i, steps in
                        (MLXRandom.uniform(Float(-2) ..< 2, [1, steps, convDim], key: keys[3 + 2 * i]).asType(.bfloat16),
                         MLXRandom.uniform(Float(0.01) ..< 0.5, [1, steps, heads], key: keys[4 + 2 * i]), steps)
                    }
                    let states = mixers.indices.map {
                        MLXRandom.uniform(Float(-0.5) ..< 0.5, [1, heads, dim, stateDim],
                                          key: MLXRandom.key(UInt64(0x55B0 + $0)))
                    }
                    eval([conv, dt, projected] + tapes.flatMap { [$0.conv, $0.dt] } + states)
                    func run(_ quad: Bool) -> [MLXArray]? {
                        var outs: [MLXArray] = []
                        for (l, mixer) in mixers.enumerated() {
                            for (t, store) in [Store.none, .anchor].enumerated() {
                                guard let r = mixer.norm.compactSSMAnchorRows(
                                    convolution: conv, ALog: mixer.aLog, D: mixer.D, dt: dt, state: states[l],
                                    projected: projected, heads: heads, headDim: dim, groups: groups,
                                    stateDim: stateDim, pending: tapes[t], store: store, tg4: quad) else { return nil }
                                outs += [r.0, r.1] + (r.2.map { [$0] } ?? [])
                            }
                        }
                        return outs
                    }
                    guard let a = run(false), let b = run(true), a.count == b.count else {
                        failure = "timed launch unavailable"
                        return
                    }
                    eval(a)
                    eval(b)
                    var ok = MLXArray(true)
                    for (x, y) in zip(a, b) {
                        let dtype: DType = x.dtype == .float32 ? .uint32 : .uint16
                        ok = ok .&& (x.view(dtype: dtype) .== y.view(dtype: dtype)).all()
                    }
                    guard ok.item(Bool.self) else {
                        failure = "differs on the timed launches"
                        return
                    }
                    compared = chain
                    func round(_ order: [Int]) -> [Double?] {
                        order.map { idx -> Double? in
                            guard let outs = run(idx == 1) else { return nil }
                            let begin = DispatchTime.now().uptimeNanoseconds
                            eval(outs)
                            Stream.gpu.synchronize()
                            return Double(DispatchTime.now().uptimeNanoseconds - begin) / 1e6 / Double(chain)
                        }
                    }
                    var wall = 0.0
                    for _ in 0 ..< 2 {
                        let begin = DispatchTime.now().uptimeNanoseconds
                        _ = round([0, 1])
                        wall = Double(DispatchTime.now().uptimeNanoseconds - begin) / 1e6
                    }
                    reps = NemotronTuneAdoption.cycles(target: 120, wall: wall, cap: 61)
                    var samples: [[Double]] = [[], []]
                    for rep in 0 ..< reps {
                        let order = NemotronTuneAdoption.order(2, cycle: rep)
                        for (i, ms) in zip(order, round(order)) {
                            if let ms { samples[i].append(ms) }
                        }
                    }
                    guard samples.allSatisfy({ $0.count == reps }) else {
                        failure = "timing incomplete"
                        return
                    }
                    verdict = NemotronTuneAdoption.decide(samples)
                    medians = verdict?.medians ?? []
                }
            } catch {
                failure = "error: \(error)"
            }
        }
        var why = "kept"
        if failure == nil, medians.count == 2 {
            let wanted: Bool
            if let f = tuneForced {
                wanted = f == 1
                why = f == 1 ? "forced" : f == 0 ? "forced V0" : "forced \(f) not qualified, kept"
            } else {
                wanted = verdict?.adopted == 1
                if wanted { why = "adopted" }
            }
            // The whole check before an adoption.
            if wanted {
                do {
                    (cases, failure) = try withError { checkTG4(first) }
                } catch {
                    failure = "error: \(error)"
                }
                if failure == nil {
                    tg4Adopted = true
                } else {
                    why = "kept"
                }
            }
        }
        func ms(_ i: Int) -> String { i < medians.count ? String(format: "%.4f", medians[i]) : "n/a" }
        let rule = verdict?.rule({ _ in "tg4" }) ?? "no candidate timed"
        report("\(why) \(tg4Adopted ? "tg4" : "V0") \(ms(tg4Adopted ? 1 : 0)) ms vs V0 (tg8) \(ms(0)) ms per launch"
            + " (\(rule); median of \(reps), \(chain) chained launches/sample on \(mixers.count) layers;"
            + " \(failure == nil ? 1 : 0)/1 bitwise over \(compared) timed launches"
            + (cases > 0 ? " and \(cases) cases on layer 0" : "")
            + (medians.count == 2 ? "; [tg4 \(ms(1)) \(verdict?.won(1) ?? "n/a")]" : "")
            + (failure.map { "; dropped: tg4 \($0)" } ?? "") + ")")
    }
}

extension NemotronHRMSNormGated {
    /// NEMOTRON_MTP_SSM_ANCHOR_SKIP: `compactSSMPreparedRows` (same guards, operands, tail form and
    /// pending fold) without the store of the state after the window; `.anchor` stores the state after
    /// the pending steps instead. Returns (normalized, prepared, that state for `.anchor`) or nil.
    /// `tg4` (default: the geometry `NemotronMTPSSMAnchor.tune` adopted) runs 4 d's per thread group.
    func compactSSMAnchorRows(
        convolution: MLXArray, ALog: MLXArray, D: MLXArray, dt: MLXArray,
        state: MLXArray, projected: MLXArray, heads: Int, headDim: Int, groups: Int,
        stateDim: Int, pending: (conv: MLXArray, dt: MLXArray, steps: Int)?, store: NemotronMTPSSMAnchor.Store,
        tg4: Bool? = nil, tail32: Bool? = nil, tc: Bool? = nil
    ) -> (MLXArray, MLXArray, MLXArray?)? {
        let quad = tg4 ?? NemotronMTPSSMAnchor.tg4Adopted
        if let pending {
            guard (1 ... NemotronMTPPendingSSM.maxSteps).contains(pending.steps),
                  pending.conv.shape == [1, pending.steps, 6144], pending.conv.dtype == .bfloat16,
                  pending.dt.shape == [1, pending.steps, 64], pending.dt.dtype == .float32
            else { return nil }
        }
        guard store == .none || pending != nil,
              NemotronHCompactSSMGated.enabled, NemotronMTPExecution.batchedNorm,
              NemotronMultiRowQMV.enabled, NemotronMultiRowQMV.narrowEnabled,
              Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              type(of: self) == NemotronHRMSNormGated.self,
              groupSize == 512, heads == 64, headDim == 64, groups == 8, stateDim == 128,
              convolution.shape == [1, 3, 6144], convolution.dtype == .bfloat16,
              dt.dtype == .float32, dt.shape == [1, 3, 64],
              state.dtype == .float32, state.shape == [1, 64, 64, 128],
              projected.shape == [1, 3, 10304], projected.dtype == .bfloat16,
              weight.shape == [4096], weight.dtype == .bfloat16,
              let counter = NemotronHCompactSSMGated.counter,
              let kernel = ((tail32 ?? NemotronSSMTail32.admits(tg4: quad))
                  ? NemotronSSMTail32.kernel(pending: pending != nil, store: store, tg4: quad) : nil)
                  ?? NemotronMTPSSMAnchor.kernel(pending: pending != nil, store: store, tg4: quad)
        else { return nil }
        // Whether `kernel` is the 32-bit twin (the text `NemotronSSMTailTC`'s twin of this launch is built from).
        let o32 = (tail32 ?? NemotronSSMTail32.admits(tg4: quad))
            && NemotronSSMTail32.kernel(pending: pending != nil, store: store, tg4: quad) != nil
        var inputs: [any ScalarOrArray] = [
            convolution, ALog, D, dt, state, projected, weight, NemotronHGatedRMSKernel.identity, eps, counter]
        var template: [(String, any KernelTemplateArg)] = [
            ("T", DType.bfloat16), ("U", DType.float32),
            ("Dh", 64), ("Ds", 128), ("H", 64), ("G", 8), ("L", 3),
            ("CD", 6144), ("BO", 4096), ("CO", 5120), ("PS", 10304)]
        if let pending {
            inputs += [pending.conv, pending.dt]
            template.append(("P", pending.steps))
        }
        let stores = store == .anchor
        // NemotronSSMTailTC: this launch's twin with eps and the identity scale as template constants.
        let constants = (tc ?? NemotronSSMTailTC.admits(tg4: quad, eps: eps))
            ? NemotronSSMTailTC.kernel(pending: pending != nil, store: store, tg4: quad, o32: o32)
                .flatMap { twin in NemotronSSMTailTC.template(eps: eps).map { (twin, $0) } } : nil
        let outputs = (constants?.0 ?? kernel)(
            inputs, template: template + (constants?.1 ?? []),
            grid: (32, 64, 64), threadGroup: (32, quad ? 4 : 8, 1),
            outputShapes: [[1, 3, 64, 64]] + (stores ? [[1, 64, 64, 128]] : []) + [[1, 3, 4096], [3, 8, 17 * 32]],
            outputDTypes: [.bfloat16] + (stores ? [.float32] : []) + [.bfloat16, .float32])
        return stores ? (outputs[2], outputs[3], outputs[1]) : (outputs[1], outputs[2], nil)
    }
}

/// Cache the complete numerical Mamba verify window, with explicit current
/// conv/SSM inputs and explicit replay-tape outputs. No evaluation object,
/// transaction, prefix-replay closure or request identity enters this graph.
final class NemotronH35MTPCompactGraphCache {
    private struct Entry {
        let identity: [ObjectIdentifier]
        let function: @Sendable ([MLXArray]) -> [MLXArray]
    }
    private static let enabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_COMPILED_MAMBA_WINDOW"] != "0"
    private let lock = NSLock()
    /// [plain, consuming a prepared in_proj activation] x [unjoined, joined]
    /// x pending steps (0 ... NemotronMTPPendingSSM.maxSteps) x [window store,
    /// no store, anchor store] (NemotronMTPSSMAnchor): separate graphs.
    private var entries: [Int: Entry] = [:]
    private var joinTimed = false
    private var joinChecked = false
    private static let joinCheck = ProcessInfo.processInfo.environment["NEMOTRON_MTP_MAMBA_JOIN_CHECK"] == "1"
    /// Timing records are not written in this build (no file output).
    private static let joinTiming: String? = nil

    func supportsJoined(_ hidden: MLXArray, owner: NemotronHMamba2Mixer) -> Bool {
        Self.enabled && NemotronMTPExecution.batchedM1 && NemotronMTPExecution.batchedNorm
            && Device.defaultDevice().deviceType == .gpu && StreamOrDevice.default == .gpu
            && hidden.shape == [1, 3, 2688] && hidden.dtype == .bfloat16
            && Self.nativeIdentity(owner) != nil
    }

    func callAsFunction(
        _ hidden: MLXArray, convState: MLXArray, initialSSM: MLXArray,
        prepared: MLXArray? = nil, pending: NemotronMTPPendingSSM.Pending? = nil,
        store: NemotronMTPSSMAnchor.Store? = nil, owner: NemotronHMamba2Mixer,
        residual: MLXArray? = nil, nextNorm: RMSNorm? = nil
    ) -> [MLXArray] {
        let fold = pending.map { (conv: $0.conv, dt: $0.dt, steps: $0.steps) }
        let joins = residual != nil && nextNorm != nil
        func finish(_ values: [MLXArray], _ lhs: MLXArray?) -> [MLXArray] {
            guard joins, let lhs, let nextNorm else { return values }
            guard let joined = NemotronHResidualRMSNorm.applyPreparedRows(
                lhs, values[0], norm: nextNorm) else {
                preconditionFailure("Qualified Mamba compiled join became ineligible")
            }
            return [joined.0] + Array(values[1...4]) + [joined.1, joined.2]
        }
        guard Self.enabled, NemotronMTPExecution.batchedM1, NemotronMTPExecution.batchedNorm,
              Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              hidden.ndim == 3, hidden.dim(0) == 1, (2...4).contains(hidden.dim(1)),
              hidden.dim(2) == 2688, hidden.dtype == .bfloat16,
              convState.shape == [1, 3, 6144], convState.dtype == .bfloat16,
              initialSSM.shape == [1, 64, 64, 128], initialSSM.dtype == .float32,
              var identity = Self.nativeIdentity(owner)
        else {
            return finish(owner.mtpCompactArrays(
                hidden, convState: convState, initialSSM: initialSSM, prepared: prepared,
                pending: fold, store: store), residual)
        }
        if joins, let nextNorm {
            identity += [ObjectIdentifier(nextNorm), ObjectIdentifier(nextNorm.weight)]
        }
        let usesPrepared = prepared != nil
        let pendingSteps = pending?.steps ?? 0
        let slot = (usesPrepared ? 1 : 0) + (joins ? 2 : 0) + 4 * pendingSteps
            + 4 * (NemotronMTPPendingSSM.maxSteps + 1) * (store.map { $0 == .anchor ? 2 : 1 } ?? 0)
        var retired: Entry?
        let current = lock.withLock { () -> Entry in
            if let entry = entries[slot], entry.identity == identity { return entry }
            let state = owner.innerState() + (joins ? [nextNorm!.weight] : [])
            let function: @Sendable ([MLXArray]) -> [MLXArray] = compile(inputs: [state]) {
                [weak owner] arrays in
                guard let owner else { preconditionFailure("Released Nemotron MTP Mamba-window owner") }
                let base = pendingSteps > 0 ? 5 : 3
                let values = owner.mtpCompactArrays(
                    arrays[0], convState: arrays[1], initialSSM: arrays[2],
                    prepared: usesPrepared ? arrays[base] : nil,
                    pending: pendingSteps > 0 ? (arrays[3], arrays[4], pendingSteps) : nil, store: store)
                guard joins, let nextNorm else { return values }
                guard let joined = NemotronHResidualRMSNorm.applyPreparedRows(
                    arrays[base + (usesPrepared ? 1 : 0)], values[0], norm: nextNorm) else {
                    preconditionFailure("Qualified Mamba compiled join became ineligible")
                }
                return [joined.0] + Array(values[1...4]) + [joined.1, joined.2]
            }
            let replacement = Entry(identity: identity, function: function)
            retired = entries[slot]
            entries[slot] = replacement
            return replacement
        }
        let result = withExtendedLifetime(retired) {
            current.function([hidden, convState, initialSSM]
                + (pending.map { [$0.conv, $0.dt] } ?? []) + (prepared.map { [$0] } ?? [])
                + (joins ? [residual!] : []))
        }
        guard result.count == (joins ? 7 : 5) else {
            // Retain any enclosing MLX error while avoiding empty indexing.
            return finish(owner.mtpCompactArrays(
                hidden, convState: convState, initialSSM: initialSSM, prepared: prepared,
                pending: fold, store: store), residual)
        }
        guard NemotronMoELeader.selfTest else { return result }
        // Reference: in_proj prepares its activation from `hidden` itself.
        let expected = finish(owner.mtpCompactArrays(
            hidden, convState: convState, initialSSM: initialSSM, pending: fold, store: store), residual)
        return zip(result, expected).enumerated().map {
            joins && $0.offset == 6 ? $0.element.0
                : nemotronBitwiseOrNaN($0.element.0, $0.element.1)
        }
    }

    func checkAndBenchmarkJoin(_ hidden: MLXArray, convState: MLXArray, initialSSM: MLXArray,
                              prepared: MLXArray?, residual: MLXArray, nextNorm: RMSNorm,
                              owner: NemotronHMamba2Mixer, layerIndex: Int, result: [MLXArray]) {
        let check = Self.joinCheck
        let directory = Self.joinTiming
        guard check || directory != nil else { return }
        func make(_ joined: Bool) -> ([MLXArray], (MLXArray, MLXArray, MLXArray)) {
            if joined {
                precondition(supportsJoined(hidden, owner: owner)
                    && NemotronHResidualRMSNorm.plumbingAvailable
                    && residual.shape == hidden.shape && residual.dtype == .bfloat16
                    && type(of: nextNorm) == RMSNorm.self
                    && nextNorm.weight.shape == [2688] && nextNorm.weight.dtype == .bfloat16)
            }
            let y = self(hidden, convState: convState, initialSSM: initialSSM,
                prepared: prepared, owner: owner,
                residual: joined ? residual : nil, nextNorm: joined ? nextNorm : nil)
            if joined { return (y, (y[0], y[5], y[6])) }
            let j = NemotronHResidualRMSNorm.applyPreparedRows(residual, y[0], norm: nextNorm)!
            return (y, j)
        }
        func roots(_ r: ([MLXArray], (MLXArray, MLXArray, MLXArray))) -> [MLXArray] {
            [r.1.0] + Array(r.0[1...4]) + [r.1.1, r.1.2]
        }
        if check {
            let expected = roots(make(false))
            for index in 0..<6 {
                let dtype: DType = result[index].dtype == .bfloat16 ? .uint16 : .uint32
                precondition((result[index].view(dtype: dtype) .== expected[index].view(dtype: dtype))
                    .all().item(Bool.self), "Mamba compiled join output/tape bit mismatch")
            }
            let got = result[6].reshaped([3, 11, 9, 32]).view(dtype: .uint32)
            let want = expected[6].reshaped([3, 11, 9, 32]).view(dtype: .uint32)
            precondition(MLX.logicalAnd(
                (got[0..., 0..<10, 0..., 0...] .== want[0..., 0..<10, 0..., 0...]).all(),
                (got[0..., 10, 0..., 0..<16] .== want[0..., 10, 0..., 0..<16]).all()).item(Bool.self),
                "Mamba compiled join live prepared bits mismatch")
            lock.withLock {
                if !joinChecked {
                    fputs("nemotron compiled Mamba join: residual, norm, live prepared and all replay tape bits exact\n", stderr)
                    joinChecked = true
                }
            }
        }
        guard let directory, lock.withLock({ () -> Bool in
            if joinTimed { return false }; joinTimed = true; return true
        }) else { return }
        eval([hidden, convState, initialSSM, residual] + (prepared.map { [$0] } ?? []))
        for i in 0..<16 { eval(roots(make(i % 2 == 0))) }
        var blocks: [[String: Any]] = []
        for block in 0..<13 {
            var measurements: [String: Any] = ["block": block]
            for joined in (block % 2 == 0 ? [false, true] : [true, false]) {
                var times: [Double] = []
                for _ in 0..<64 {
                    let start = DispatchTime.now().uptimeNanoseconds
                    let y = make(joined)
                    let elapsed = DispatchTime.now().uptimeNanoseconds - start
                    times.append(Double(elapsed) / 1000.0)
                    eval(roots(y))
                }
                measurements[joined ? "candidate_us" : "frontier_us"] = times
            }
            blocks.append(measurements)
        }
        let record: [String: Any] = ["layer": layerIndex, "blocks": blocks,
            "scope": "host graph construction only; GPU eval excluded",
            "prepared": prepared != nil, "rows": hidden.shape]
        let data = try! JSONSerialization.data(withJSONObject: record, options: [.sortedKeys])
        try! data.write(to: URL(fileURLWithPath: directory)
            .appendingPathComponent("layer-\(layerIndex).json"), options: .atomic)
    }

    private static func nativeIdentity(_ owner: NemotronHMamba2Mixer) -> [ObjectIdentifier]? {
        func linear(_ value: Linear) -> Bool {
            type(of: value) == Linear.self || type(of: value) == QuantizedLinear.self
        }
        guard type(of: owner) == NemotronHMamba2Mixer.self,
              owner.numHeads == 64, owner.headDim == 64, owner.ssmStateSize == 128,
              owner.numGroups == 8, owner.convKernelSize == 4, owner.convDim == 6144,
              type(of: owner.conv1d) == Conv1d.self,
              type(of: owner.norm) == NemotronHRMSNormGated.self,
              linear(owner.inProj), linear(owner.outProj) else { return nil }
        let modules: [Module] = [owner, owner.inProj, owner.conv1d, owner.norm, owner.outProj]
        return modules.map(ObjectIdentifier.init)
    }
}

extension NemotronHMamba2Mixer {
    func cbv2ForwardCapturedJoined(
        _ hidden: MLXArray, modelLayerIndex: Int,
        recurrentState: [CBv2RecurrentStateEvaluation], prepared: MLXArray?,
        residual: MLXArray, nextNorm: RMSNorm
    ) -> (MLXArray, MLXArray, MLXArray)? {
        guard nemotronMTPCompactSSM, recurrentState.count == 1,
              mtpCompactGraphCache.supportsJoined(hidden, owner: self),
              NemotronHResidualRMSNorm.plumbingAvailable,
              residual.shape == hidden.shape, residual.dtype == .bfloat16,
              type(of: nextNorm) == RMSNorm.self,
              nextNorm.weight.shape == [2688], nextNorm.weight.dtype == .bfloat16 else { return nil }
        let evaluation = recurrentState[0]
        let initial = evaluation.inputState(modelLayerIndex: modelLayerIndex)
        let conv = initial?.conv ?? MLXArray.zeros([1, 3, 6144], dtype: hidden.dtype)
        let ssm = initial?.ssm ?? MLXArray.zeros([1, 64, 64, 128], dtype: .float32)
        let y = mtpCompactForward(hidden, modelLayerIndex: modelLayerIndex,
            evaluation: evaluation, convState: conv, initialSSM: ssm,
            prepared: prepared, residual: residual, nextNorm: nextNorm)
        return (y[0], y[5], y[6])
    }

    func cbv2ForwardCaptured(
        _ hiddenStates: MLXArray, modelLayerIndex: Int,
        recurrentState: [CBv2RecurrentStateEvaluation], prepared: MLXArray? = nil
    ) -> MLXArray {
        let batch = hiddenStates.dim(0), length = hiddenStates.dim(1)
        precondition(batch == 1 && recurrentState.count == 1 && (1...8).contains(length))
        let evaluation = recurrentState[0]
        let initial = evaluation.inputState(modelLayerIndex: modelLayerIndex)
        let convState = initial?.conv ?? MLXArray.zeros(
            [batch, max(0, convKernelSize - 1), convDim], dtype: hiddenStates.dtype)
        let ssmState = initial?.ssm ?? MLXArray.zeros(
            [batch, numHeads, headDim, ssmStateSize], dtype: .float32)
        if nemotronMTPCompactSSM && length > 1 {
            return mtpCompactForward(hiddenStates, modelLayerIndex: modelLayerIndex,
                evaluation: evaluation, convState: convState, initialSSM: ssmState,
                prepared: prepared)[0]
        }
        let arrays = mtpCapturedArrays(hiddenStates, convState: convState, initialSSM: ssmState)
        do {
            try evaluation.stageCaptured(modelLayerIndex: modelLayerIndex,
                conv: arrays[1], ssm: arrays[2], positions: length)
        } catch {
            preconditionFailure("Nemotron MTP captured-state stage failed: \(error)")
        }
        return arrays[0]
    }

    /// Compact verify: the window kernel writes only the final SSM state and
    /// the window's y; a strict accepted prefix replays the same kernel for
    /// `keep` steps from the same input state at commit (bitwise equal to the
    /// captured per-position state). Conv states are views of `padded`.
    func mtpCompactForward(
        _ hiddenStates: MLXArray, modelLayerIndex: Int,
        evaluation: CBv2RecurrentStateEvaluation, convState: MLXArray, initialSSM: MLXArray,
        prepared: MLXArray? = nil, residual: MLXArray? = nil, nextNorm: RMSNorm? = nil
    ) -> [MLXArray] {
        let length = hiddenStates.dim(1)
        // NEMOTRON_MTP_SSM_PENDING_FOLD: a committed strict-prefix replay
        // arrives as its anchor and tape; the window's fused launch runs it.
        let pending = length == 3 ? NemotronMTPPendingSSM.take(initialSSM) : nil
        let start = pending?.anchor ?? initialSSM
        // NEMOTRON_MTP_SSM_ANCHOR_SKIP: no state store after the window; the state after the pending
        // steps is stored once they reach anchorAt (see NemotronMTPSSMAnchor).
        let anchorSkip = length == 3 && NemotronMTPPendingSSM.anchorSkip
        let writes = anchorSkip && (pending?.steps ?? 0) >= NemotronMTPPendingSSM.anchorAt
        let arrays = mtpCompactGraphCache(
            hiddenStates, convState: convState, initialSSM: start, prepared: prepared,
            pending: pending, store: anchorSkip ? (writes ? .anchor : NemotronMTPSSMAnchor.Store.none) : nil,
            owner: self, residual: residual, nextNorm: nextNorm)
        if anchorSkip {
            NemotronMTPSSMAnchor.lastPending = pending?.steps ?? 0
            stageAnchorReplay(evaluation, modelLayerIndex: modelLayerIndex, length: length,
                window: (arrays[1], arrays[2], arrays[3]), anchor: writes ? arrays[4] : start,
                carried: writes ? nil : pending)
            return arrays
        }
        if let residual, let nextNorm, pending == nil {
            mtpCompactGraphCache.checkAndBenchmarkJoin(hiddenStates, convState: convState,
                initialSSM: initialSSM, prepared: prepared, residual: residual, nextNorm: nextNorm,
                owner: self, layerIndex: modelLayerIndex, result: arrays)
        }
        let (padded, convolution, dt, finalSSM) = (arrays[1], arrays[2], arrays[3], arrays[4])
        let (heads, dim, groups, state) = (numHeads, headDim, numGroups, ssmStateSize)
        let (logA, skip) = (self.aLog, self.D)
        let history = convKernelSize - 1
        let tape = [padded, convolution, dt]
        let tapeBytes = tape.reduce(0) { $0 + $1.nbytes }
        // NEMOTRON_MTP_SSM_REPLAY_GROUP: replay with this forward's other
        // Mamba layers in grouped launches (same per-layer statements).
        let grouped = pending == nil ? NemotronMTPReplayGroup.join(evaluation, .init(
            convolution: convolution, dt: dt, aLog: logA, D: skip, state: initialSSM,
            heads: heads, headDim: dim, groups: groups, stateDim: state)) : nil
        let pendingTape = pending.map { [$0.conv, $0.dt] } ?? []
        let pendingBytes = pendingTape.reduce(0) { $0 + $1.nbytes }
        do {
            try evaluation.stagePrefixReplay(
                modelLayerIndex: modelLayerIndex, positions: length,
                finalConv: padded[0..., length ..< (length + history), 0...],
                finalSSM: finalSSM,
                materializedByteCount: tapeBytes + finalSSM.nbytes,
                evaluationRoots: tape,
                strictReplayRetainedByteCount: tapeBytes + start.nbytes + pendingBytes,
                strictReplayRetainedRoots: tape + [start] + pendingTape,
                replay: { keep in
                    let ssm: MLXArray
                    if let pending {
                        // The window started `pending.steps` steps past the
                        // anchor: replay those and the accepted ones from it.
                        let steps = pending.steps + keep
                        let tapeConv = concatenated([pending.conv, convolution[0..., ..<keep, 0...]], axis: 1)
                        let tapeDt = concatenated([pending.dt, dt[0..., ..<keep, 0...]], axis: 1)
                        ssm = nemotronMTPSSMCompact(
                            convolution: tapeConv, ALog: logA, D: skip, dt: tapeDt,
                            state: pending.anchor, heads: heads, headDim: dim, groups: groups,
                            stateDim: state, steps: steps, writeOutput: false).finalState
                        NemotronMTPPendingSSM.register(ssm, .init(
                            anchor: pending.anchor, conv: tapeConv, dt: tapeDt, steps: steps))
                    } else {
                        ssm = grouped.map { $0.group.finalState(slot: $0.slot, keep: keep) }
                            ?? nemotronMTPSSMCompact(
                            convolution: convolution, ALog: logA, D: skip, dt: dt,
                            state: initialSSM, heads: heads, headDim: dim, groups: groups,
                            stateDim: state, steps: keep, writeOutput: false).finalState
                        NemotronMTPPendingSSM.register(ssm, .init(
                            anchor: initialSSM, conv: convolution[0..., ..<keep, 0...],
                            dt: dt[0..., ..<keep, 0...], steps: keep))
                    }
                    return CBv2RecurrentLayerState(
                        conv: padded[0..., keep ..< (keep + history), 0...], ssm: ssm)
                })
        } catch {
            preconditionFailure("Nemotron MTP compact recurrent stage failed: \(error)")
        }
        return arrays
    }

    /// NEMOTRON_MTP_SSM_ANCHOR_SKIP commit: an accepted prefix of `keep` window steps (full or strict) is
    /// the compact replay, from `anchor`, of the `carried` pending steps the window's launch did not store
    /// followed by the accepted ones, registered so the next window's launch folds it. A full acceptance
    /// is built on first read like a strict prefix (`defersFullAcceptance`).
    private func stageAnchorReplay(
        _ evaluation: CBv2RecurrentStateEvaluation, modelLayerIndex: Int, length: Int,
        window: (padded: MLXArray, convolution: MLXArray, dt: MLXArray), anchor: MLXArray,
        carried: NemotronMTPPendingSSM.Pending?
    ) {
        let (heads, dim, groups, state) = (numHeads, headDim, numGroups, ssmStateSize)
        let (logA, skip) = (self.aLog, self.D)
        let history = convKernelSize - 1
        let tape = [window.padded, window.convolution, window.dt]
        let retained = tape + [anchor] + (carried.map { [$0.conv, $0.dt] } ?? [])
        let retainedBytes = retained.reduce(0) { $0 + $1.nbytes }
        func committed(_ keep: Int) -> CBv2RecurrentLayerState {
            var conv = window.convolution[0..., ..<keep, 0...]
            var dt = window.dt[0..., ..<keep, 0...]
            if let carried {
                conv = concatenated([carried.conv, conv], axis: 1)
                dt = concatenated([carried.dt, dt], axis: 1)
            }
            let steps = (carried?.steps ?? 0) + keep
            let ssm = nemotronMTPSSMCompact(
                convolution: conv, ALog: logA, D: skip, dt: dt, state: anchor,
                heads: heads, headDim: dim, groups: groups, stateDim: state,
                steps: steps, writeOutput: false).finalState
            NemotronMTPPendingSSM.register(ssm, .init(anchor: anchor, conv: conv, dt: dt, steps: steps))
            return CBv2RecurrentLayerState(conv: window.padded[0..., keep ..< (keep + history), 0...], ssm: ssm)
        }
        do {
            try evaluation.stagePrefixReplay(
                modelLayerIndex: modelLayerIndex, positions: length,
                finalConv: window.padded[0..., length ..< (length + history), 0...],
                finalSSM: anchor,
                materializedByteCount: tape.reduce(0) { $0 + $1.nbytes } + anchor.nbytes,
                evaluationRoots: tape,
                strictReplayRetainedByteCount: retainedBytes,
                strictReplayRetainedRoots: retained,
                fullAcceptanceRetainedByteCount: retainedBytes,
                fullAcceptanceRetainedRoots: retained,
                fullAcceptance: { committed(length) },
                defersFullAcceptance: true,
                replay: committed)
        } catch {
            preconditionFailure("Nemotron MTP anchor-skip recurrent stage failed: \(error)")
        }
    }

    /// Original numerical graph, with request state supplied as dynamic arrays.
    /// Return its replay tape explicitly; staging/commit stay outside compile.
    /// `inputPrepared`: in_proj's K=2688 activation, prepared by the preceding
    /// join from these exact BF16 rows (otherwise prepared here).
    func mtpCompactArrays(
        _ hiddenStates: MLXArray, convState: MLXArray, initialSSM: MLXArray,
        prepared inputPrepared: MLXArray? = nil,
        pending: (conv: MLXArray, dt: MLXArray, steps: Int)? = nil,
        store: NemotronMTPSSMAnchor.Store? = nil
    ) -> [MLXArray] {
        let batch = hiddenStates.dim(0), length = hiddenStates.dim(1)
        // The window's conv, padded history and timestep come from in_proj's
        // own epilogue when the shapes allow it (same values, one launch fewer).
        var fusedWindow: (MLXArray, MLXArray, MLXArray)? = nil
        let projected: MLXArray
        if batch == 1, hiddenStates.ndim == 3, convKernelSize == 4, conv1d.groups == convDim,
           conv1d.stride == 1, conv1d.dilation == 1, conv1d.padding == 0,
           let convBias = conv1d.bias, let q = inProj as? QuantizedLinear,
           type(of: q) == QuantizedLinear.self,
           let fused = NemotronMambaInProjConvDt.apply(
               hiddenStates.reshaped(length, hiddenStates.dim(2)), q, prepared: inputPrepared,
               oldState: convState, convWeight: conv1d.weight, convBias: convBias,
               dtBias: dtBias, timeStepLimit: timeStepLimit,
               convStart: intermediateSize, convDim: convDim, heads: numHeads) {
            projected = fused.projected.reshaped(batch, length, -1)
            fusedWindow = (fused.convolution, fused.padded, fused.dt)
        } else {
            projected = nemotronMTPLinearRows(hiddenStates, inProj, prepared: inputPrepared)
        }
        let parts = split(projected, indices: [intermediateSize, intermediateSize + convDim], axis: -1)
        let gate = parts[0], input = parts[1]
        let rawDt = parts[2].reshaped(batch, length, numHeads)
        func separateWindow() -> (MLXArray, MLXArray, MLXArray) {
            packedConvDt(convState, input, rawDt) ?? {
                let (convolution, padded) = packedConv(convState, input)
                return (convolution, padded, nemotronPrefillComputeDt(rawDt, dtBias, timeStepLimit))
            }()
        }
        var (convolution, padded, dt) = fusedWindow ?? separateWindow()
        if fusedWindow != nil, NemotronMambaInProjConvDt.check {
            let reference = separateWindow()
            convolution = nemotronBitwiseOrNaN(convolution, reference.0)
            padded = nemotronBitwiseOrNaN(padded, reference.1)
            dt = nemotronBitwiseOrNaN(dt, reference.2)
        }
        let (heads, dim, groups, state) = (numHeads, headDim, numGroups, ssmStateSize)
        let (logA, skip) = (self.aLog, self.D)
        // NEMOTRON_MTP_SSM_ANCHOR_SKIP (`store`): the last output is the next
        // tape's anchor instead of the window's final state: the state after
        // the pending steps (`.anchor`) or `initialSSM` itself (`.none`).
        func anchorOutput(_ begin: MLXArray, _ final: MLXArray) -> MLXArray {
            store.map { $0 == .anchor ? begin : initialSSM } ?? final
        }
        func fusedRows() -> (MLXArray, MLXArray, MLXArray)? {
            guard let store else {
                return norm.compactSSMPreparedRows(
                    convolution: convolution, ALog: logA, D: skip, dt: dt, state: initialSSM,
                    projected: projected, heads: heads, headDim: dim, groups: groups, stateDim: state,
                    pending: pending)
            }
            return norm.compactSSMAnchorRows(
                convolution: convolution, ALog: logA, D: skip, dt: dt, state: initialSSM,
                projected: projected, heads: heads, headDim: dim, groups: groups, stateDim: state,
                pending: pending, store: store).map { ($0.0, $0.1, $0.2 ?? initialSSM) }
        }
        // NEMOTRON_SSM_GATED_FUSED: the compact SSM window, the gated RMS norm
        // and out_proj's prepared activation in one launch (see
        // NemotronHCompactSSMGated); the same normed rows, prepared activation
        // and final state as the two launches below.
        if length == 3, NemotronMTPExecution.batchedNorm,
           let q = outProj as? QuantizedLinear, q.mode == .affine,
           q.bits == 4, q.groupSize == 64, q.bias == nil,
           q.weight.shape == [2688, 512], q.weight.dtype == .uint32,
           q.scales.shape == [2688, 64], q.scales.dtype == .bfloat16,
           q.biases?.shape == [2688, 64], q.biases?.dtype == .bfloat16,
           var fused = fusedRows() {
            if NemotronMoELeader.selfTest {
                // Reference: the two launches (compact SSM, then the gated
                // norm with the prepared output) on the same operands.
                let begin = replayedPending(pending, initialSSM)
                let reference = nemotronMTPSSMCompact(
                    convolution: convolution, ALog: logA, D: skip, dt: dt,
                    state: begin,
                    heads: heads, headDim: dim, groups: groups, stateDim: state,
                    steps: length, writeOutput: true)
                if let pair = norm.preparedRows(
                    reference.output.reshaped(batch, length, heads * dim), gate: gate) {
                    fused = (nemotronBitwiseOrNaN(fused.0, pair.0),
                             nemotronBitwiseOrNaN(fused.1, pair.1),
                             nemotronBitwiseOrNaN(fused.2, anchorOutput(begin, reference.finalState)))
                }
            }
            return [nemotronMTPLinearRows(fused.0, outProj, prepared: fused.1),
                    padded, convolution, dt, fused.2]
        }
        let begin = replayedPending(pending, initialSSM)
        let result = nemotronMTPSSMCompact(
            convolution: convolution, ALog: logA, D: skip, dt: dt,
            state: begin,
            heads: heads, headDim: dim, groups: groups, stateDim: state,
            steps: length, writeOutput: true)
        let y = result.output.reshaped(batch, length, heads * dim)
        let normalized: MLXArray
        let prepared: MLXArray?
        if NemotronMTPExecution.batchedNorm,
           let q = outProj as? QuantizedLinear, q.mode == .affine,
           q.bits == 4, q.groupSize == 64, q.bias == nil,
           q.weight.shape == [2688, 512], q.weight.dtype == .uint32,
           q.scales.shape == [2688, 64], q.scales.dtype == .bfloat16,
           q.biases?.shape == [2688, 64], q.biases?.dtype == .bfloat16,
           let pair = norm.preparedRows(y, gate: gate) {
            normalized = pair.0
            prepared = pair.1
        } else {
            normalized = NemotronMTPExecution.batchedNorm
                ? norm(y, gate: gate)
                : concatenated((0..<length).map { position in
                norm(y[0..., position..<(position + 1), 0...],
                     gate: gate[0..., position..<(position + 1), 0...])
                }, axis: 1)
            prepared = nil
        }
        return [nemotronMTPLinearRows(normalized, outProj, prepared: prepared),
                padded, convolution, dt, anchorOutput(begin, result.finalState)]
    }

    /// The window's start state: `anchor` itself, or `anchor` advanced over
    /// the pending tape by the compact replay (outside the fused launch).
    func replayedPending(
        _ pending: (conv: MLXArray, dt: MLXArray, steps: Int)?, _ anchor: MLXArray
    ) -> MLXArray {
        guard let pending else { return anchor }
        return nemotronMTPSSMCompact(
            convolution: pending.conv, ALog: aLog, D: D, dt: pending.dt, state: anchor,
            heads: numHeads, headDim: headDim, groups: numGroups, stateDim: ssmStateSize,
            steps: pending.steps, writeOutput: false).finalState
    }

    func mtpCapturedArrays(_ hiddenStates: MLXArray, convState: MLXArray,
                           initialSSM: MLXArray) -> [MLXArray] {
        let batch = hiddenStates.dim(0), length = hiddenStates.dim(1)
        var ssmState = initialSSM
        let projected = nemotronMTPLinearRows(hiddenStates, inProj)
        let parts = split(projected, indices: [intermediateSize, intermediateSize + convDim], axis: -1)
        let gate = parts[0], input = parts[1], dt = parts[2]
        // One causal depthwise convolution covers the complete window. Each
        // position still owns its exact history slice and ordinary M=1 SSM
        // update, so rejecting a suffix cannot advance persistent recurrence.
        let (convolution, padded) = packedConv(convState, input)
        // Per-position conv histories as one strided view of `padded`
        // (window p = rows p+1 ..< p+convKernelSize). Pure data movement:
        // identical values to stacking the per-position slices.
        let convStack = asStrided(
            padded, [length, convKernelSize - 1, convDim],
            strides: [convDim, convDim, 1], offset: convDim)
        var outputs: [MLXArray] = []
        let capturedSSM: MLXArray
        if NemotronMTPExecution.windowSSM && length > 1 {
            let values = split(convolution,
                indices: [intermediateSize, intermediateSize + numGroups * ssmStateSize], axis: -1)
            let result = nemotronMTPSSMWindow(
                hiddenStates: values[0].reshaped(batch, length, numHeads, headDim),
                ALog: aLog, B: values[1].reshaped(batch, length, numGroups, ssmStateSize),
                C: values[2].reshaped(batch, length, numGroups, ssmStateSize), D: D,
                dt: dt.reshaped(batch, length, numHeads), dtBias: dtBias,
                state: ssmState, timeStepLimit: timeStepLimit)
            capturedSSM = result.capturedStates
            if NemotronMTPExecution.batchedNorm {
                // [1, L, H, Dh] is contiguous: one reshape equals the
                // concatenation of the per-position flattened slices.
                outputs = [result.output.reshaped(batch, length, -1)]
            } else {
                outputs = (0..<length).map {
                    result.output[0..., $0..<($0 + 1), 0..., 0...].flattened(start: 2)
                }
            }
        } else {
            var ssmStates: [MLXArray] = []
            for position in 0..<length {
                let values = split(convolution[0..., position..<(position + 1), 0...],
                    indices: [intermediateSize, intermediateSize + numGroups * ssmStateSize], axis: -1)
                let (y, next) = ssmUpdate(
                    hiddenStates: values[0].reshaped(batch, 1, numHeads, headDim),
                    ALog: aLog, B: values[1].reshaped(batch, 1, numGroups, ssmStateSize),
                    C: values[2].reshaped(batch, 1, numGroups, ssmStateSize), D: D,
                    dt: dt[0..., position..<(position + 1), 0...].reshaped(batch, 1, numHeads),
                    dtBias: dtBias, state: ssmState, timeStepLimit: timeStepLimit, mask: nil)
                ssmState = next.asType(.float32)
                ssmStates.append(ssmState)
                outputs.append(y.asType(hiddenStates.dtype).flattened(start: 2))
            }
            capturedSSM = concatenated(ssmStates, axis: 0)
        }
        let normalized: MLXArray
        if NemotronMTPExecution.batchedNorm {
            normalized = norm(
                outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 1), gate: gate)
        } else {
            normalized = concatenated(outputs.enumerated().map { position, y in
                norm(y, gate: gate[0..., position..<(position + 1), 0...])
            }, axis: 1)
        }
        return [nemotronMTPLinearRows(normalized, outProj), convStack, capturedSSM]
    }
}


extension NemotronH35Model: CBv2RecurrentCaptureMTPForwardable {
    public func cbv2ForwardWithHiddenCaptured(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        precondition(positionIds == nil)
        if let fused = cbv2HiddenCapturedFused(
            tokens, caches: caches, recurrentState: recurrentState) {
            let output = lmHead.map { nemotronMTPHeadRows(fused.hidden, $0, prepared: fused.prepared) }
                ?? nemotronMTPMapRows(fused.hidden) { logits($0) }
            return (output, fused.hidden)
        }
        let hidden = cbv2Hidden(tokens, caches: caches, recurrentState: recurrentState, captureMTP: true)
        let output = lmHead.map { nemotronMTPHeadRows(hidden, $0) }
            ?? nemotronMTPMapRows(hidden) { logits($0) }
        return (output, hidden)
    }
}

extension NemotronH35Model {
    /// Compile, at load time, every SSM verify-kernel instantiation an MTP
    /// round can use (window widths 2...8; compact replay prefixes 1...7), so
    /// a JIT compile never lands inside a timed decode window. Dummy inputs
    /// use the real parameter dtypes, hence the exact runtime instantiations.
    func warmMTPSSMKernels(activationDType: DType) {
        guard let mixer = modules().lazy.compactMap({ $0 as? NemotronHMamba2Mixer }).first
        else { return }
        let history = mixer.convKernelSize - 1
        let (heads, dim, groups, stateDim) =
            (mixer.numHeads, mixer.headDim, mixer.numGroups, mixer.ssmStateSize)
        let normInput = MLXArray.zeros([1, 1, heads * dim], dtype: activationDType)
        var roots: [MLXArray] = [mixer.norm(normInput, gate: normInput)]
        // NEMOTRON_SSM_GATED_FUSED: the zeroed arrival counters and the
        // fused launch's pipeline, here at load (zero operands leave the
        // counters at zero).
        NemotronHCompactSSMGated.prepareCounter()
        if let fused = mixer.norm.compactSSMPreparedRows(
            convolution: MLXArray.zeros([1, 3, mixer.convDim], dtype: .bfloat16),
            ALog: mixer.aLog, D: mixer.D,
            dt: MLXArray.zeros([1, 3, heads], dtype: .float32),
            state: MLXArray.zeros([1, heads, dim, stateDim], dtype: .float32),
            projected: MLXArray.zeros(
                [1, 3, mixer.intermediateSize + mixer.convDim + heads], dtype: .bfloat16),
            heads: heads, headDim: dim, groups: groups, stateDim: stateDim) {
            eval(fused.0, fused.1, fused.2)
        }
        NemotronHCompactSSMGated.qualifyShuffle(mixer)
        NemotronHCompactSSMGated.qualifyRowsTail(mixer)
        // NEMOTRON_MTP_SSM_PENDING_FOLD: compiles and checks every folded launch.
        NemotronMTPPendingSSM.qualify(mixer)
        for width in 2...8 {
            let padded = MLXArray.zeros([1, history + width, mixer.convDim], dtype: activationDType)
            let convolution = silu(mixer.conv1d(padded))
            let rawDt = MLXArray.zeros([1, width, heads], dtype: activationDType)
            let state = MLXArray.zeros([1, heads, dim, stateDim], dtype: .float32)
            if nemotronMTPCompactSSM {
                let dt = nemotronPrefillComputeDt(rawDt, mixer.dtBias, mixer.timeStepLimit)
                let window = nemotronMTPSSMCompact(
                    convolution: convolution, ALog: mixer.aLog, D: mixer.D, dt: dt,
                    state: state, heads: heads, headDim: dim, groups: groups,
                    stateDim: stateDim, steps: width, writeOutput: true)
                let replay = nemotronMTPSSMCompact(
                    convolution: convolution, ALog: mixer.aLog, D: mixer.D, dt: dt,
                    state: state, heads: heads, headDim: dim, groups: groups,
                    stateDim: stateDim, steps: width - 1, writeOutput: false)
                roots += [window.output, window.finalState, replay.finalState]
            } else if NemotronMTPExecution.windowSSM {
                let values = split(convolution,
                    indices: [mixer.intermediateSize, mixer.intermediateSize + groups * stateDim],
                    axis: -1)
                let window = nemotronMTPSSMWindow(
                    hiddenStates: values[0].reshaped(1, width, heads, dim),
                    ALog: mixer.aLog, B: values[1].reshaped(1, width, groups, stateDim),
                    C: values[2].reshaped(1, width, groups, stateDim), D: mixer.D,
                    dt: rawDt, dtBias: mixer.dtBias, state: state,
                    timeStepLimit: mixer.timeStepLimit)
                roots += [window.output, window.capturedStates]
            }
        }
        if nemotronMTPCompactSSM {
            let mambaLayers = modules().lazy.filter { $0 is NemotronHMamba2Mixer }.count
            roots += NemotronMTPReplayGroupKernel.warm(
                layerCount: mambaLayers,
                convolution: silu(mixer.conv1d(
                    MLXArray.zeros([1, history + 8, mixer.convDim], dtype: activationDType))),
                aLog: mixer.aLog, D: mixer.D,
                heads: heads, headDim: dim, groups: groups, stateDim: stateDim)
        }
        let packed = mixer.packedConv(
            MLXArray.zeros([1, history, mixer.convDim], dtype: activationDType),
            MLXArray.zeros([1, 1, mixer.convDim], dtype: activationDType))
        roots += [packed.convolution, packed.padded]
        if !roots.isEmpty { eval(roots) }
    }
}
