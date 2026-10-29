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
        /// Smaller group set for the upper vocabulary half (nil: half head).
        let upperGroups: MLXArray?
        let upperGroupCount: Int
        /// Packed selected scales/biases (`NemotronHeadScreen.packScalesBiases`),
        /// used once its partial scores matched the gathered ones bitwise.
        var packed: MLXArray? = nil
        /// Full packed records (`NemotronHeadScreen.packRecords`), adopted the
        /// same way; when set, `packed` is not built.
        var records: MLXArray? = nil
        /// 2-bit screen records (`NemotronHeadScreen.packRecords2`), adopted
        /// once the plan's rescore and pick checks pass on them; when set,
        /// neither `records` nor `packed` is built.
        var records2: MLXArray? = nil
        /// Ticket counter of the fused select + final launch
        /// (`NemotronHeadScreen.makeCounter`), set once its ids matched the
        /// two-launch path bitwise.
        var counter: MLXArray? = nil
        /// The 2-bit partial pass with the upper rows two per pass, set once
        /// its scores matched the one-row record kernel's bitwise.
        var paired = false
        /// The fused select with one barrier fewer (`NemotronHeadScreen`'s B1
        /// select), set once its ids and kept values matched the record's.
        var b1 = false
        /// The B1 select without the ticket (`NemotronHeadScreen.noTicketEnabled`),
        /// set once its pick, kept rows and chunk bests matched the fused B1 select.
        var noTicket = false
        var bestOnly = false
    }
    private var screen: Screen?
    private var checkSteps = 0, checkMisses = 0, checkRescoreMismatches = 0

    /// The plan's paired 2-bit partial pass for `NemotronDraftTune` (nil: not in use).
    var screenTunePlan: NemotronDraftTune.ScreenPlan? {
        guard let screen, let records2 = screen.records2, screen.paired else { return nil }
        return .init(records2: records2, groups: screen.groups, groupCount: screen.groupCount,
                     upperGroups: screen.upperGroups, upperGroupCount: screen.upperGroupCount)
    }

    /// Qualified immutable screen backing is outside the registered module
    /// tree. Include it in the existing decode lease, but never the mutable
    /// fused-selection ticket counter or any request tensor.
    var decodeResidencyArrays: [MLXArray] {
        guard let screen else { return [] }
        return [screen.groups] + [screen.upperGroups, screen.packed, screen.records, screen.records2]
            .compactMap { $0 }
    }

    /// Greedy draft id [1] int32 for hidden [1, 1, K]: the screen when it
    /// qualified at load and the shapes match, else the stock argmax.
    /// `upper`: also screen the upper vocabulary half (when the plan has it).
    func greedy(_ hidden: MLXArray, upper: Bool = false) -> MLXArray {
        guard let screen, hidden.shape == [1, 1, NemotronHeadScreen.width],
              hidden.dtype == .bfloat16, StreamOrDevice.default == .gpu
        else { return argMax(logits(hidden)[0..., -1, 0...], axis: -1).asType(.int32) }
        // `NemotronDraftTune` (screen): the paired 2-bit partial pass at the geometry adopted at load.
        let tuned = screen.records2 != nil && screen.paired ? NemotronDraftTune.adopted[.screen] : nil
        if screen.bestOnly, !NemotronHeadScreen.check, let records2 = screen.records2,
           let result = NemotronHeadScreen.runBestOnly(hidden,
            weight: screen.weight, scales: screen.scales, biases: screen.biases,
            records2: records2, groups: screen.groups, groupCount: screen.groupCount,
            chunk: screen.chunk, keep: screen.keep, upperGroups: upper ? screen.upperGroups : nil,
            upperGroupCount: upper ? screen.upperGroupCount : 0, paired: screen.paired,
            rowsPerSimdgroup: tuned?.rps ?? 8, simdgroups: tuned?.nsg ?? 2, tunedPartial: tuned != nil) {
            NemotronScreenPicks.register(result.id, .init(values: result.bestValues, ids: result.bestIds))
            return result.id
        }
        let result = NemotronHeadScreen.run(
            hidden, weight: screen.weight, scales: screen.scales, biases: screen.biases,
            groups: screen.groups, groupCount: screen.groupCount, chunk: screen.chunk,
            keep: screen.keep, upperGroups: upper ? screen.upperGroups : nil,
            upperGroupCount: upper ? screen.upperGroupCount : 0, packed: screen.packed,
            records: screen.records, records2: screen.records2,
            counter: screen.counter, paired: screen.paired, rowsPerSimdgroup: tuned?.rps ?? 8,
            simdgroups: tuned?.nsg ?? 2, b1: screen.b1, noTicket: screen.noTicket, tunedPartial: tuned != nil)
        if screen.noTicket, let values = result.bestValues, let ids = result.bestIds {
            NemotronScreenPicks.register(result.id, .init(values: values, ids: ids))
        }
        if NemotronHeadScreen.check && !(upper && screen.upperGroups != nil) {
            checkAgainstStock(hidden, result, screen)
        }
        return result.id
    }

    private func qualifyBestOnly(_ plan: Screen, gamma: MLXArray) -> Bool {
        func report(_ text: String) {
            FileHandle.standardError.write(Data("nemotron draft bests only: \(text)\n".utf8))
        }
        guard NemotronHeadScreen.bestOnlyRequested, plan.noTicket, plan.b1,
              let records2 = plan.records2, let prefix, let head = target.lmHead as? QuantizedLinear else {
            report("off (setting or no admitted plan)")
            return false
        }
        var cases = 0
        do {
            let equal = try withError { () -> Bool in
                var state: UInt64 = 0x082E_FA98_EC4E_6C89
                var checks: [MLXArray] = []
                for round in 0 ..< 12 {
                    let values = (0 ..< 2688).map { _ -> Float in
                        state = state &* 6364136223846793005 &+ 1442695040888963407
                        return (Float(state >> 40) / Float(1 << 24) - 0.5) * 3.4641016
                    }
                    var x = (MLXArray(values) * gamma.asType(.float32) * Float(1 + round % 3))
                        .asType(.bfloat16).reshaped(1, 1, 2688)
                    if round == 6 { x = MLXArray.zeros([1, 1, 2688], dtype: .bfloat16) }
                    if round == 7 { x = MLXArray([Float](repeating: -0.0, count: 2688))
                        .asType(.bfloat16).reshaped(1, 1, 2688) }
                    if round == 8 {
                        let padded = concatenated([MLXArray.zeros([1], dtype: .bfloat16), x.reshaped(2688)], axis: 0)
                        x = padded[1 ..< 2689].reshaped(1, 1, 2688)
                    }
                    if round >= 9 {
                        var special = values
                        special[(round * 223) % 2688] = round == 9 ? Float.infinity
                            : (round == 10 ? -Float.infinity : Float.nan)
                        x = MLXArray(special).asType(.bfloat16).reshaped(1, 1, 2688)
                    }
                    for full in plan.upperGroups == nil ? [false] : [false, true] {
                        let old = NemotronHeadScreen.run(x,
                            weight: plan.weight, scales: plan.scales, biases: plan.biases,
                            groups: plan.groups, groupCount: plan.groupCount, chunk: plan.chunk, keep: plan.keep,
                            upperGroups: full ? plan.upperGroups : nil, upperGroupCount: full ? plan.upperGroupCount : 0,
                            records2: records2, paired: plan.paired, b1: true, noTicket: true)
                        guard let new = NemotronHeadScreen.runBestOnly(x,
                            weight: plan.weight, scales: plan.scales, biases: plan.biases,
                            records2: records2, groups: plan.groups, groupCount: plan.groupCount,
                            chunk: plan.chunk, keep: plan.keep, upperGroups: full ? plan.upperGroups : nil,
                            upperGroupCount: full ? plan.upperGroupCount : 0, paired: plan.paired),
                              let bestValues = old.bestValues, let bestIds = old.bestIds else { return false }
                        checks.append((old.scores.view(dtype: .uint32) .== new.scores.view(dtype: .uint32)).all()
                            .&& (bestValues.view(dtype: .uint32) .== new.bestValues.view(dtype: .uint32)).all()
                            .&& (bestIds .== new.bestIds).all() .&& (old.id .== new.id).all())
                        if round < 9 {
                            let logits = (full ? head(x) : prefix(x))[0, -1].asType(.float32)
                            let expected = take(logits, bestIds, axis: 0)
                            checks.append((expected.view(dtype: .uint32) .== new.bestValues.view(dtype: .uint32)).all())
                        }
                        cases += 1
                    }
                }
                eval(checks)
                return !checks.isEmpty && checks.allSatisfy { $0.item(Bool.self) }
            }
            report(equal ? "on (\(cases) partial/best/pick cases bitwise equal; candidate stores omitted)"
                         : "off (source or output words differ)")
            return equal
        } catch {
            report("off (\(error))")
            return false
        }
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
        // NEMOTRON_MTP_HEAD_SCREEN_BITS=4 keeps the 4-bit screen (12 groups,
        // upper 8, keep 16 per 1024 rows); the default 2-bit records read 16 /
        // 6 groups and keep 12 rows per 512-row chunk for the exact rescore.
        let bits = Int(environment["NEMOTRON_MTP_HEAD_SCREEN_BITS"] ?? "") ?? 2
        let groupCount = Int(environment["NEMOTRON_MTP_HEAD_SCREEN_GROUPS"] ?? "") ?? (bits == 2 ? 16 : 12)
        let chunk = Int(environment["NEMOTRON_MTP_HEAD_SCREEN_CHUNK"] ?? "") ?? (bits == 2 ? 512 : 1024)
        let keep = Int(environment["NEMOTRON_MTP_HEAD_SCREEN_KEEP"] ?? "") ?? (bits == 2 ? 12 : 16)
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
        // gamma[c]^2 * ||W[:, c]||^2 over the scored rows.
        let g = gamma.asType(.float32)
        func selectGroups(_ rowRange: Range<Int>, count: Int) -> [Int] {
            var columns = MLXArray.zeros([k], dtype: .float32)
            for start in stride(from: rowRange.lowerBound, to: rowRange.upperBound, by: 8192) {
                let r = start..<(start + 8192)
                let d = dequantized(head.weight[r], scales: head.scales[r], biases: biases[r],
                                    groupSize: 64, bits: 4).asType(.float32)
                columns = columns + (d * d).sum(axis: 0)
                eval(columns)
            }
            let importance = (columns * g * g).reshaped(NemotronHeadScreen.groupsPerRow, 64)
                .sum(axis: 1).asArray(Float.self)
            // Whole aligned 64-byte pairs (groups 2p, 2p + 1; rows are 1344 =
            // 21 * 64 bytes): the screen then reads 8 sectors of each row.
            let pairs = (0..<(NemotronHeadScreen.groupsPerRow / 2)).sorted {
                let a = importance[2 * $0] + importance[2 * $0 + 1]
                let b = importance[2 * $1] + importance[2 * $1 + 1]
                return a > b || (a == b && $0 < $1)
            }
            return pairs.prefix(count / 2).flatMap { [2 * $0, 2 * $0 + 1] }.sorted()
        }
        let selected = selectGroups(0..<rows, count: groupCount)
        // Drafts also screen the upper vocabulary half with a smaller group
        // set (NEMOTRON_MTP_HEAD_SCREEN_UPPER_GROUPS, 0 keeps the half head):
        // the later BPE merges carry a prompt-dependent share of the target's
        // tokens, and a draft that cannot propose them is always rejected.
        let upperCount = Int(environment["NEMOTRON_MTP_HEAD_SCREEN_UPPER_GROUPS"] ?? "") ?? (bits == 2 ? 6 : 8)
        let upperSelection: [Int]? = (2...16).contains(upperCount) && upperCount % 2 == 0
            ? selectGroups(rows..<NemotronHeadScreen.headRows, count: upperCount) : nil
        // Qualification on fixed synthetic inputs: every kept value must be
        // bitwise the stock head logit over the scored rows, and the pick the
        // max kept value. An upper-half plan that does not qualify falls back
        // to the half-head plan, which is qualified the same way.
        func qualify(_ upper: [Int]?) -> (plan: Screen, agree: Int)? {
            let plan = Screen(
                weight: head.weight, scales: head.scales, biases: biases,
                groups: MLXArray(selected.map { UInt32($0) }),
                groupCount: groupCount, chunk: chunk, keep: keep,
                upperGroups: upper.map { MLXArray($0.map { UInt32($0) }) },
                upperGroupCount: upper?.count ?? 0)
            let scoredRows = upper == nil ? rows : NemotronHeadScreen.headRows
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
                    groups: plan.groups, groupCount: groupCount, chunk: chunk, keep: keep,
                    upperGroups: plan.upperGroups, upperGroupCount: plan.upperGroupCount)
                let full = (upper == nil ? prefix(x) : head(x))[0, -1].asType(.float32)
                let stock = argMax(full, axis: -1).asType(.int32)
                eval(full, stock, result.id, result.candidateIds, result.candidateValues)
                let fullValues = full.asArray(Float.self)
                let ids = result.candidateIds.asArray(Int32.self)
                let values = result.candidateValues.asArray(Float.self)
                let id = Int(result.id.item(Int32.self))
                var best = 0
                for i in ids.indices
                where values[i] > values[best] || (values[i] == values[best] && ids[i] < ids[best]) { best = i }
                guard Set(ids).count == ids.count,
                      ids.allSatisfy({ (0..<Int32(scoredRows)).contains($0) }),
                      zip(ids, values).allSatisfy({ fullValues[Int($0)].bitPattern == $1.bitPattern }),
                      id == Int(ids[best])
                else { return nil }
                if id == Int(stock.item(Int32.self)) { agree += 1 }
            }
            return (plan, agree)
        }
        var upper = upperSelection
        var qualified = qualify(upper)
        if qualified == nil, upper != nil {
            report("upper half off (rescore is not bitwise the stock head)")
            upper = nil
            qualified = qualify(nil)
        }
        guard let chosen = qualified else {
            return report("off (rescore is not bitwise the stock head)")
        }
        // Packed screen records: load-time copies of the head's own bytes for
        // the screen's partial pass only (the target and the rescore read the
        // head itself). The full record (selected words, scales and biases) is
        // tried first, then scales/biases alone, else the gathered screen
        // stays. Each is adopted only if its partial scores and picks equal
        // the gathered kernel's bitwise on the same synthetic inputs, for the
        // plan and (when it screens the upper half) its half-head variant.
        var plan = chosen.plan
        var packedNotes: [String] = []
        func screened(_ x: MLXArray, _ s: Screen, upper: Bool) -> NemotronHeadScreen.Result {
            NemotronHeadScreen.run(
                x, weight: s.weight, scales: s.scales, biases: s.biases,
                groups: s.groups, groupCount: groupCount, chunk: chunk, keep: keep,
                upperGroups: upper ? s.upperGroups : nil, upperGroupCount: upper ? s.upperGroupCount : 0,
                packed: s.packed, records: s.records)
        }
        func adopt(_ name: String, _ build: (inout Screen) -> MLXArray) {
            // MLX errors (a kernel that fails to build) throw here instead of
            // stopping the load; the next layout down then stays in use.
            var candidate = chosen.plan
            do {
                let verified = try withError { () -> MLXArray? in
                    let copy = build(&candidate)
                    eval(copy)
                    var state: UInt64 = 0x2545_F491_4F6C_DD1D
                    for _ in 0..<4 {
                        let t = (0..<k).map { _ -> Float in
                            state = state &* 6364136223846793005 &+ 1442695040888963407
                            return (Float(state >> 40) / Float(1 << 24) - 0.5) * 3.4641016
                        }
                        let x = (MLXArray(t) * g).asType(.bfloat16).reshaped(1, 1, k)
                        for upper in chosen.plan.upperGroups == nil ? [false] : [true, false] {
                            let a = screened(x, chosen.plan, upper: upper)
                            let b = screened(x, candidate, upper: upper)
                            let equal = MLX.logicalAnd(
                                (a.scores.view(dtype: .uint32) .== b.scores.view(dtype: .uint32)).all(),
                                MLX.logicalAnd(
                                    (a.candidateIds .== b.candidateIds).all(), (a.id .== b.id).all()))
                            if !equal.item(Bool.self) { return nil }
                        }
                    }
                    return copy
                }
                if let verified {
                    plan = candidate
                    packedNotes.append("\(name) on (\(verified.nbytes >> 10) KiB, bitwise equal)")
                } else {
                    packedNotes.append("\(name) off (partial scores differ)")
                }
            } catch {
                packedNotes.append("\(name) off (\(error))")
            }
        }
        // 2-bit screen records (NEMOTRON_MTP_HEAD_SCREEN_BITS=2): adopted when,
        // on fixed synthetic inputs and for the plan and its half-head variant,
        // every kept value is bitwise the stock head logit over the scored rows,
        // the kept ids are distinct and in range, and the pick is the best kept
        // value (lowest id on ties); the 4-bit layouts are then not built.
        if bits == 2 {
            do {
                let verified = try withError { () -> MLXArray? in
                    let records2 = NemotronHeadScreen.packRecords2(
                        weight: head.weight, scales: head.scales, biases: biases, groups: selected,
                        upperGroups: upper)
                    eval(records2)
                    let candidate = chosen.plan
                    var state: UInt64 = 0x3C6E_F372_FE94_F82B
                    for _ in 0..<4 {
                        let t = (0..<k).map { _ -> Float in
                            state = state &* 6364136223846793005 &+ 1442695040888963407
                            return (Float(state >> 40) / Float(1 << 24) - 0.5) * 3.4641016
                        }
                        let x = (MLXArray(t) * g).asType(.bfloat16).reshaped(1, 1, k)
                        for half in candidate.upperGroups == nil ? [false] : [true, false] {
                            let result = NemotronHeadScreen.run(
                                x, weight: candidate.weight, scales: candidate.scales,
                                biases: candidate.biases, groups: candidate.groups,
                                groupCount: groupCount, chunk: chunk, keep: keep,
                                upperGroups: half ? candidate.upperGroups : nil,
                                upperGroupCount: half ? candidate.upperGroupCount : 0,
                                records2: records2)
                            let scoredRows = half ? NemotronHeadScreen.headRows : rows
                            let full = (half ? head(x) : prefix(x))[0, -1].asType(.float32)
                            eval(full, result.id, result.candidateIds, result.candidateValues)
                            let fullValues = full.asArray(Float.self)
                            let ids = result.candidateIds.asArray(Int32.self)
                            let values = result.candidateValues.asArray(Float.self)
                            let id = Int(result.id.item(Int32.self))
                            var best = 0
                            for i in ids.indices
                            where values[i] > values[best] || (values[i] == values[best] && ids[i] < ids[best]) {
                                best = i
                            }
                            guard Set(ids).count == ids.count,
                                  ids.allSatisfy({ (0..<Int32(scoredRows)).contains($0) }),
                                  zip(ids, values).allSatisfy({ fullValues[Int($0)].bitPattern == $1.bitPattern }),
                                  id == Int(ids[best])
                            else { return nil }
                        }
                    }
                    return records2
                }
                if let verified {
                    plan.records2 = verified
                    packedNotes.append("2-bit records on (\(verified.nbytes >> 10) KiB, rescore bitwise)")
                } else {
                    packedNotes.append("2-bit records off (rescore check failed)")
                }
            } catch {
                packedNotes.append("2-bit records off (\(error))")
            }
        }
        if plan.records2 == nil, NemotronHeadScreen.packedRecords {
            adopt("full records") { candidate in
                let records = NemotronHeadScreen.packRecords(
                    weight: head.weight, scales: head.scales, biases: biases, groups: selected,
                    upperGroups: upper)
                candidate.records = records
                return records
            }
        }
        if plan.records2 == nil, plan.records == nil, NemotronHeadScreen.packedScalesBiases,
           (upper?.count ?? 0) <= groupCount {
            adopt("scales/biases") { candidate in
                let packed = NemotronHeadScreen.packScalesBiases(
                    scales: head.scales, biases: biases, groups: selected,
                    groupCount: groupCount, upperGroups: upper)
                candidate.packed = packed
                return packed
            }
        }
        // Fused select + final: adopted only if, on synthetic inputs and for
        // each half the plan screens, its id and candidates equal the
        // two-launch path's bitwise, one launch at a time and again with all
        // of them queued in one evaluation on the shared counter, which must
        // read 0 afterwards.
        var fusedNote = "off"
        if NemotronHeadScreen.fusedFinal {
            func same(_ a: NemotronHeadScreen.Result, _ b: NemotronHeadScreen.Result) -> Bool {
                MLX.logicalAnd(
                    (a.id .== b.id).all(),
                    MLX.logicalAnd(
                        (a.candidateIds .== b.candidateIds).all(),
                        (a.candidateValues.view(dtype: .uint32)
                            .== b.candidateValues.view(dtype: .uint32)).all())).item(Bool.self)
            }
            let verified: MLXArray?
            do {
                verified = try withError { () -> MLXArray? in
                    let counter = NemotronHeadScreen.makeCounter()
                    var state: UInt64 = 0x6A09_E667_F3BC_C909
                    var expected: [NemotronHeadScreen.Result] = []
                    var queued: [NemotronHeadScreen.Result] = []
                    for _ in 0..<4 {
                        let t = (0..<k).map { _ -> Float in
                            state = state &* 6364136223846793005 &+ 1442695040888963407
                            return (Float(state >> 40) / Float(1 << 24) - 0.5) * 3.4641016
                        }
                        let x = (MLXArray(t) * g).asType(.bfloat16).reshaped(1, 1, k)
                        for half in plan.upperGroups == nil ? [false] : [false, true] {
                            func screened(_ counter: MLXArray?) -> NemotronHeadScreen.Result {
                                NemotronHeadScreen.run(
                                    x, weight: plan.weight, scales: plan.scales, biases: plan.biases,
                                    groups: plan.groups, groupCount: groupCount, chunk: chunk,
                                    keep: keep, upperGroups: half ? plan.upperGroups : nil,
                                    upperGroupCount: half ? plan.upperGroupCount : 0,
                                    packed: plan.packed, records: plan.records, records2: plan.records2,
                                    counter: counter)
                            }
                            let reference = screened(nil)
                            if !same(reference, screened(counter)) { return nil }
                            expected.append(reference)
                            queued.append(screened(counter))
                        }
                    }
                    eval(queued.flatMap { [$0.id, $0.candidateIds, $0.candidateValues] })
                    for (a, b) in zip(expected, queued) where !same(a, b) { return nil }
                    return counter.item(UInt32.self) == 0 ? counter : nil
                }
            } catch {
                verified = nil
                fusedNote = "off (\(error))"
            }
            if let verified {
                plan.counter = verified
                fusedNote = "on (bitwise equal)"
            } else if fusedNote == "off" {
                fusedNote = "off (fused id differs)"
            }
        }
        // Paired upper rows (NEMOTRON_MTP_HEAD_SCREEN_PAIRED=0 keeps one row per
        // pass): adopted only if, on synthetic inputs for each half the plan
        // screens, plus one input read through a view whose data is not
        // 16-byte aligned (the scalar activation loads), its scores, kept ids
        // and pick equal the record kernel's bitwise.
        var pairedNote: String? = nil
        if let records2 = plan.records2, NemotronHeadScreen.pairedEnabled {
            let halves = plan.upperGroups == nil ? [false] : [true, false]
            if !NemotronHeadScreen.pairable(
                groupCount: groupCount, upperGroupCount: plan.upperGroups == nil ? nil : plan.upperGroupCount,
                rowsPerSimdgroup: 8) {
                pairedNote = "off (groups or source do not qualify)"
            } else {
                var cases = 0
                do {
                    let equal = try withError { () -> Bool in
                        var state: UInt64 = 0xBB67_AE85_84CA_A73B
                        var inputs: [MLXArray] = []
                        for _ in 0..<4 {
                            let t = (0..<k).map { _ -> Float in
                                state = state &* 6364136223846793005 &+ 1442695040888963407
                                return (Float(state >> 40) / Float(1 << 24) - 0.5) * 3.4641016
                            }
                            inputs.append((MLXArray(t) * g).asType(.bfloat16).reshaped(1, 1, k))
                        }
                        let padded = concatenated(
                            [MLXArray.zeros([1], dtype: .bfloat16), inputs[0].reshaped(k)], axis: 0)
                        inputs.append(padded[1 ..< (k + 1)].reshaped(1, 1, k))
                        for x in inputs {
                            for half in halves {
                                func screened(_ paired: Bool) -> NemotronHeadScreen.Result {
                                    NemotronHeadScreen.run(
                                        x, weight: plan.weight, scales: plan.scales, biases: plan.biases,
                                        groups: plan.groups, groupCount: groupCount, chunk: chunk,
                                        keep: keep, upperGroups: half ? plan.upperGroups : nil,
                                        upperGroupCount: half ? plan.upperGroupCount : 0,
                                        records2: records2, paired: paired)
                                }
                                let a = screened(false), b = screened(true)
                                let same = MLX.logicalAnd(
                                    (a.scores.view(dtype: .uint32) .== b.scores.view(dtype: .uint32)).all(),
                                    MLX.logicalAnd(
                                        (a.candidateIds .== b.candidateIds).all(), (a.id .== b.id).all()))
                                if !same.item(Bool.self) { return false }
                                cases += 1
                            }
                        }
                        return true
                    }
                    if equal {
                        plan.paired = true
                        pairedNote = "on (upper rows two per pass, scores bitwise equal to the record kernel in \(cases) cases)"
                    } else {
                        pairedNote = "off (scores differ)"
                    }
                } catch {
                    pairedNote = "off (\(error))"
                }
            }
        }
        // B1 select (NEMOTRON_MTP_SCREEN_SELECT_B1=0 keeps the record fused
        // select): adopted only if, on synthetic inputs through the head for
        // each half the plan screens, its id, kept ids and kept values equal the
        // record fused select's bitwise, one launch at a time and again with all
        // of them queued in one evaluation on the shared counter, which must
        // read 0 afterwards.
        var b1Note: String? = nil
        if let counter = plan.counter, NemotronHeadScreen.b1Enabled {
            if !NemotronHeadScreen.b1Available(keep: keep) {
                b1Note = "off (source or keep does not qualify)"
            } else {
                var cases = 0
                do {
                    let equal = try withError { () -> Bool in
                        func same(_ a: NemotronHeadScreen.Result, _ b: NemotronHeadScreen.Result) -> Bool {
                            MLX.logicalAnd(
                                (a.id .== b.id).all(),
                                MLX.logicalAnd(
                                    (a.candidateIds .== b.candidateIds).all(),
                                    (a.candidateValues.view(dtype: .uint32)
                                        .== b.candidateValues.view(dtype: .uint32)).all())).item(Bool.self)
                        }
                        var state: UInt64 = 0x510E_527F_ADE6_82D1
                        var expected: [NemotronHeadScreen.Result] = []
                        var queued: [NemotronHeadScreen.Result] = []
                        for round in 0..<6 {
                            let t = (0..<k).map { _ -> Float in
                                state = state &* 6364136223846793005 &+ 1442695040888963407
                                return (Float(state >> 40) / Float(1 << 24) - 0.5) * 3.4641016
                            }
                            let x = (MLXArray(t) * g * Float(1 + round % 3)).asType(.bfloat16).reshaped(1, 1, k)
                            for half in plan.upperGroups == nil ? [false] : [false, true] {
                                func screened(_ b1: Bool) -> NemotronHeadScreen.Result {
                                    NemotronHeadScreen.run(
                                        x, weight: plan.weight, scales: plan.scales, biases: plan.biases,
                                        groups: plan.groups, groupCount: groupCount, chunk: chunk,
                                        keep: keep, upperGroups: half ? plan.upperGroups : nil,
                                        upperGroupCount: half ? plan.upperGroupCount : 0,
                                        packed: plan.packed, records: plan.records, records2: plan.records2,
                                        counter: counter, paired: plan.paired, b1: b1)
                                }
                                let reference = screened(false)
                                if !same(reference, screened(true)) { return false }
                                expected.append(reference)
                                queued.append(screened(true))
                                cases += 1
                            }
                        }
                        eval(queued.flatMap { [$0.id, $0.candidateIds, $0.candidateValues] })
                        for (a, b) in zip(expected, queued) where !same(a, b) { return false }
                        return counter.item(UInt32.self) == 0
                    }
                    if equal {
                        plan.b1 = true
                        b1Note = "on (ids, kept ids and kept values bitwise equal to the record fused select in \(cases) cases)"
                    } else {
                        b1Note = "off (differs from the record fused select)"
                    }
                } catch {
                    b1Note = "off (\(error))"
                }
            }
        }
        // Ticket-free select (NEMOTRON_MTP_SCREEN_NO_TICKET=0 keeps the fused
        // B1 select): adopted only if, on synthetic inputs through the head for
        // each half the plan screens, its pick, kept ids, kept values and chunk
        // bests equal the fused B1 select's bitwise, one launch at a time and
        // again with all of them queued in one evaluation (the fused counter must
        // read 0 afterwards).
        var noTicketNote: String? = nil
        if plan.b1, let counter = plan.counter, NemotronHeadScreen.noTicketEnabled {
            if !NemotronHeadScreen.noTicketAvailable(keep: keep) {
                noTicketNote = "off (source or keep does not qualify)"
            } else {
                var cases = 0
                do {
                    let equal = try withError { () -> Bool in
                        func same(_ a: NemotronHeadScreen.Result, _ b: NemotronHeadScreen.Result) -> MLXArray {
                            guard let av = a.bestValues, let ai = a.bestIds, let bv = b.bestValues, let bi = b.bestIds
                            else { return MLXArray(false) }
                            return (a.id .== b.id).all()
                                .&& (a.candidateIds .== b.candidateIds).all()
                                .&& (a.candidateValues.view(dtype: .uint32) .== b.candidateValues.view(dtype: .uint32)).all()
                                .&& (av.view(dtype: .uint32) .== bv.view(dtype: .uint32)).all()
                                .&& (ai .== bi).all()
                        }
                        var state: UInt64 = 0x9B05_688C_2B3E_6C1F
                        var expected: [NemotronHeadScreen.Result] = []
                        var queued: [NemotronHeadScreen.Result] = []
                        for round in 0..<6 {
                            let t = (0..<k).map { _ -> Float in
                                state = state &* 6364136223846793005 &+ 1442695040888963407
                                return (Float(state >> 40) / Float(1 << 24) - 0.5) * 3.4641016
                            }
                            let x = (MLXArray(t) * g * Float(1 + round % 3)).asType(.bfloat16).reshaped(1, 1, k)
                            for half in plan.upperGroups == nil ? [false] : [false, true] {
                                func screened(_ noTicket: Bool) -> NemotronHeadScreen.Result {
                                    NemotronHeadScreen.run(
                                        x, weight: plan.weight, scales: plan.scales, biases: plan.biases,
                                        groups: plan.groups, groupCount: groupCount, chunk: chunk,
                                        keep: keep, upperGroups: half ? plan.upperGroups : nil,
                                        upperGroupCount: half ? plan.upperGroupCount : 0,
                                        packed: plan.packed, records: plan.records, records2: plan.records2,
                                        counter: counter, paired: plan.paired, b1: true, noTicket: noTicket)
                                }
                                let reference = screened(false)
                                if !same(reference, screened(true)).item(Bool.self) { return false }
                                expected.append(reference)
                                queued.append(screened(true))
                                cases += 1
                            }
                        }
                        eval(queued.flatMap { [$0.id, $0.candidateIds, $0.candidateValues] + [$0.bestValues!, $0.bestIds!] })
                        for (a, b) in zip(expected, queued) where !same(a, b).item(Bool.self) { return false }
                        return counter.item(UInt32.self) == 0
                    }
                    if equal {
                        plan.noTicket = true
                        NemotronScreenPicks.chunkCounts = plan.upperGroups == nil
                            ? [NemotronHeadScreen.rows / chunk]
                            : [NemotronHeadScreen.rows / chunk, NemotronHeadScreen.headRows / chunk]
                        noTicketNote = "on (pick, kept ids, kept values and chunk bests bitwise equal to the fused B1 select in \(cases) cases)"
                    } else {
                        noTicketNote = "off (differs from the fused B1 select)"
                    }
                } catch {
                    noTicketNote = "off (\(error))"
                }
            }
        } else if NemotronHeadScreen.noTicketEnabled {
            noTicketNote = "off (no fused B1 select to replace)"
        } else {
            noTicketNote = "off (setting)"
        }
        // 32-bit select twin (NEMOTRON_MTP_SCREEN_SELECT_O32=0 keeps the B1 select's offsets): checked against the
        // B1 select on the plan's head, chunk and keep (`NemotronHeadScreen.qualifySelectOffset32`).
        let selectOffset32Note = plan.noTicket
            ? NemotronHeadScreen.qualifySelectOffset32(
                weight: plan.weight, scales: plan.scales, biases: plan.biases, chunk: chunk, keep: keep,
                rowCounts: plan.upperGroups == nil ? [rows] : [rows, NemotronHeadScreen.headRows], gamma: g,
                screen: { x, half, on in
                    NemotronHeadScreen.run(
                        x, weight: plan.weight, scales: plan.scales, biases: plan.biases,
                        groups: plan.groups, groupCount: groupCount, chunk: chunk,
                        keep: keep, upperGroups: half ? plan.upperGroups : nil,
                        upperGroupCount: half ? plan.upperGroupCount : 0,
                        packed: plan.packed, records: plan.records, records2: plan.records2,
                        counter: plan.counter, paired: plan.paired, b1: true, noTicket: true, selectOffset32: on)
                })
            : "off (no ticket-free select)"
        // 32-bit twin of the paired partial on full-head screens (NEMOTRON_MTP_SCREEN_PARTIAL_O32=0 keeps the record
        // addresses): checked against the paired partial on the plan's records (`qualifyPartialOffset32`).
        let partialOffset32Note: String
        if plan.paired, let records2 = plan.records2, let upperGroups = plan.upperGroups {
            partialOffset32Note = NemotronHeadScreen.qualifyPartialOffset32(
                records2: records2, groups: plan.groups, groupCount: groupCount, upperGroups: upperGroups,
                upperGroupCount: plan.upperGroupCount, gamma: g,
                screen: { x, on in
                    NemotronHeadScreen.run(
                        x, weight: plan.weight, scales: plan.scales, biases: plan.biases,
                        groups: plan.groups, groupCount: groupCount, chunk: chunk, keep: keep,
                        upperGroups: plan.upperGroups, upperGroupCount: plan.upperGroupCount,
                        packed: plan.packed, records: plan.records, records2: plan.records2,
                        counter: plan.counter, paired: plan.paired, b1: plan.b1, noTicket: plan.noTicket,
                        partialOffset32: on)
                })
        } else {
            partialOffset32Note = "off (no paired full-head 2-bit screen)"
        }
        plan.bestOnly = qualifyBestOnly(plan, gamma: gamma)
        screen = plan
        // Requests switch between the full and the half-head screen; compile
        // the half-head variant (on the same packed and fused path) here as
        // well, so neither builds inside a timed window.
        if plan.upperGroups != nil {
            let x = (MLXArray.ones([k]) * g).asType(.bfloat16).reshaped(1, 1, k)
            eval(NemotronHeadScreen.run(
                x, weight: plan.weight, scales: plan.scales, biases: plan.biases,
                groups: plan.groups, groupCount: groupCount, chunk: chunk, keep: keep,
                packed: plan.packed, records: plan.records, records2: plan.records2,
                counter: plan.counter, paired: plan.paired, b1: plan.b1, noTicket: plan.noTicket).id)
        }
        report("on, \(plan.records2 == nil ? 4 : 2)-bit, \(groupCount) of 42 groups \(selected), keep \(keep) per \(chunk) rows,"
            + " upper half \(upper.map { "\($0.count) groups \($0)" } ?? "off"),"
            + " packed screen \(packedNotes.isEmpty ? "off" : packedNotes.joined(separator: ", ")), fused final \(fusedNote),"
            + " qualified (synthetic agreement \(chosen.agree)/4)")
        if let pairedNote {
            FileHandle.standardError.write(Data("nemotron mtp head screen paired: \(pairedNote)\n".utf8))
        }
        if let b1Note {
            FileHandle.standardError.write(Data("nemotron mtp head screen select b1: \(b1Note)\n".utf8))
        }
        if let noTicketNote {
            FileHandle.standardError.write(Data("nemotron mtp head screen no ticket: \(noTicketNote)\n".utf8))
        }
        FileHandle.standardError.write(Data("nemotron mtp head screen select o32: \(selectOffset32Note)\n".utf8))
        FileHandle.standardError.write(Data("nemotron mtp head screen partial o32: \(partialOffset32Note)\n".utf8))
    }

    /// A request's drafts screen the upper vocabulary half while it committed
    /// an upper-half id within this many committed tokens; separately per draft
    /// position. Break-even per draft: one draft's upper screen costs ~0.4% of a
    /// round with packed records (~0.01 tokens' worth at ~2.45 tokens per round;
    /// both drafts' together measured ~0.8%). An upper-half hit is worth
    /// ~h (1 + a2) ~ 1.08 accepted tokens for the first draft and ~h a1 ~ 0.48
    /// for the second (h ~0.6, a1 ~0.8, a2 ~0.8), so the first pays above an
    /// upper-id rate of ~0.9% (one per ~110 tokens) and the second above ~2.0%
    /// (one per ~49). NEMOTRON_MTP_HEAD_SCREEN_UPPER_WINDOW_FIRST / _LATER set
    /// them; NEMOTRON_MTP_HEAD_SCREEN_UPPER_WINDOW sets both; 0 screens always.
    private static func window(_ key: String, _ fallback: Int) -> Int {
        let environment = ProcessInfo.processInfo.environment
        return Int(environment[key] ?? "")
            ?? Int(environment["NEMOTRON_MTP_HEAD_SCREEN_UPPER_WINDOW"] ?? "") ?? fallback
    }
    static let upperWindowFirst = window("NEMOTRON_MTP_HEAD_SCREEN_UPPER_WINDOW_FIRST", 110)
    static let upperWindowLater = window("NEMOTRON_MTP_HEAD_SCREEN_UPPER_WINDOW_LATER", 49)
    static func usesUpper(tokensSinceUpper: Int, firstDraft: Bool) -> Bool {
        let window = firstDraft ? upperWindowFirst : upperWindowLater
        return window == 0 || tokensSinceUpper < window
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
public final class NemotronH35MTPAssistant: Module, CBv2MTPRequestStatefulDrafter, CBv2MTPMemoryPreparing, @unchecked Sendable {
    private var decodeResidency: AnyObject?
    func prepareDecodeResidency() {
        if #available(macOS 15.0, *) {
            var head = NemotronSBDict.withoutDictBankMetadata(NemotronZSkip.withoutCopiedDraftBanks(module), module: module)
                + draftHead.decodeResidencyArrays + NemotronSBDict.draftArrays
            var extras = NemotronZSkip.draftCopyArrays + NemotronSBDict.allArrays + NemotronSharedDownZSkip.copyArrays
            // With `NemotronMTPFirstRoundDraft`, the head's routed banks (fc1 bank words with their pair
            // dictionary, the fc2 copy) get their own set, requested after the extras; the head's block skips
            // its routed experts until that request returns.
            var late: [MLXArray] = []
            if NemotronMTPFirstRoundDraft.enabled {
                let block = module.layers[1] as! NemotronH35MTPExpertBlock
                let fc1 = block.mixer.switchMLP.fc1
                late = fc1.parameters().flattened().filter { $0.0 == "weight" }.map { $0.1 }
                    + (NemotronSBDict.draftDict(fc1).map { [$0.index, $0.table] } ?? [])
                    + NemotronZSkip.draftCopyArrays
                let moved = Set(late.map(ObjectIdentifier.init))
                head = head.filter { !moved.contains(ObjectIdentifier($0)) }
                extras = extras.filter { !moved.contains(ObjectIdentifier($0)) }
            }
            decodeResidency = NemotronDecodeResidency(
                head: head,
                layers: NemotronZSkip.decodeCopyLayers(target), owners: NemotronZSkip.decodeCopyOwners(target),
                target: NemotronDecodeResidency.targetGroups(target),
                extras: extras, late: late)
            if let residency = decodeResidency as? NemotronDecodeResidency {
                NemotronDecodeResidency.current = residency
                CBv2EngineInitHooks.onInit = {
                    if #available(macOS 15.0, *) { NemotronDecodeResidency.current?.earlyBegin() }
                }
            }
        }
    }
    public func beginMTPMemoryPreparation() -> (any CBv2MTPMemoryLease)? {
        if #available(macOS 15.0, *), let p = decodeResidency as? NemotronDecodeResidency {
            return p.begin()
        }
        return nil
    }

    let module: NemotronH35MTPModule
    let target: NemotronH35Model
    private let draftHead: NemotronMTPDraftHead
    private let kvOnlyHistory: Bool
    private let primePromptHistory: Bool
    private let draftLimit: Int
    /// The head cache's K/V dtype, probed once from the loaded weights.
    private let headKVDType = NemotronHLoadedMemo<DType>()
    public var mtpTargetIdentity: ObjectIdentifier? { ObjectIdentifier(target) }
    private static let captureVerify =
        ProcessInfo.processInfo.environment["DARKBLOOM_NEMOTRON35_MTP_CAPTURE_VERIFY"] != "0"
    public var requiredVerificationMode: CBv2MTPVerificationMode? {
        Self.captureVerify ? .rectangularExact : .serialTarget
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

    /// Load-time geometry tune of the draft chain's kernels (`NemotronDraftTune`), after the screen plan, the draft
    /// dictionaries and `NemotronU2`, before any draft graph is traced.
    func tuneDraftKernels() {
        NemotronDraftTune.tune(module, screen: draftHead.screenTunePlan)
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
        /// Post-verify rows: fused residuals, queries and the K/V appended to
        /// the head cache; the device-picked row's attention reads them.
        var speculation: (fused: MLXArray, queries: MLXArray, keys: MLXArray, values: MLXArray,
                          width: Int)?
        /// The head row's `writeGeneration` right after `speculation`'s rows
        /// were written; set only there, read only with a `nextFirstDraft`
        /// that finalize built from that same speculation.
        var speculationWriteGeneration: UInt64?
        /// The single expert/head continuation selected at the existing host boundary.
        var nextFirstDraft: (ids: MLXArray, out: MLXArray, keys: MLXArray, values: MLXArray)?
        /// The next first draft queued behind the verify for the device-picked
        /// row, and the host walk's accept count that decides whether to adopt it.
        var deviceFirstDraft: (ids: MLXArray, out: MLXArray, keys: MLXArray, values: MLXArray)?
        var hostAcceptedDrafts: Int?
        /// Committed tokens since the last one in the upper vocabulary half
        /// (prompt tail included); Int.max when none has been seen.
        var tokensSinceUpper = Int.max
        /// Whether a round's first draft (`firstDraft`) or a later one screens
        /// the upper vocabulary half.
        func screensUpper(firstDraft: Bool) -> Bool {
            NemotronMTPDraftHead.usesUpper(tokensSinceUpper: tokensSinceUpper, firstDraft: firstDraft)
        }
        /// The cache-free inputs of the second draft on `deviceFirstDraft`
        /// (`pagedDraftInputs`), queued with it, and the same inputs carried
        /// with `nextFirstDraft` once finalize adopts that first draft.
        var deviceSecondInputs: (inputs: [MLXArray], from: MLXArray)?
        var nextSecondInputs: (inputs: [MLXArray], from: MLXArray)?
        /// The whole second draft on `deviceFirstDraft` (masked attention over
        /// the head cache's rows as the device-picked count keeps them), with
        /// its own K/V row; carried with `nextFirstDraft` once adopted.
        var deviceSecondDraft: (ids: MLXArray, out: MLXArray, keys: MLXArray, values: MLXArray,
                                from: MLXArray)?
        var nextSecondDraft: (ids: MLXArray, out: MLXArray, keys: MLXArray, values: MLXArray,
                              from: MLXArray)?
        var deferredSecondRow: (keys: MLXArray, values: MLXArray)?
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
            let base: [MLXArray] = (paged?.cache.innerState() ?? [])
                + pendingTokens + pendingHidden + savedTokens + savedHidden
                + [frontier, savedFrontier].compactMap { $0 } + roots
            guard let row = deferredSecondRow else { return base }
            return base + [row.keys, row.values]
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
            cache.nemotronDraftAttention = true
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
        let dtype = headKVDType.value {
            let hiddenDType = target.mtpEmbedding(MLXArray([Int32(0)]).reshaped([1, 1])).dtype
            return module.projectedKVDType(hiddenDType: hiddenDType)
        }
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
        s.nextFirstDraft = nil; s.nextSecondInputs = nil; s.nextSecondDraft = nil
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
        s.nextFirstDraft = nil; s.nextSecondInputs = nil; s.nextSecondDraft = nil
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
        if let row = s.deferredSecondRow {
            s.deferredSecondRow = nil
            s.roots += s.cache.appendTrusted(keys: row.keys, values: row.values)
        }
        // Before the request's first verify (`NemotronMTPFirstRoundDraft`).
        NemotronMTPFirstRoundDraft.firstRound = !s.started
        let out = module.pagedForward(
            hidden: hidden, embedding: target.mtpEmbedding(tokens), cache: s.cache)
        NemotronMTPFirstRoundDraft.firstRound = false
        let ids = draftHead.greedy(out, upper: s.screensUpper(firstDraft: first))
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
        // Adopt only when the host's accepted prefix is the device-selected row.
        let deviceFirstDraft = s.hostAcceptedDrafts == n ? s.deviceFirstDraft : nil
        // The post-verify rows carry only the device-picked row's attention;
        // any other count takes the ordinary replay path below.
        let speculation = s.speculation.flatMap {
            $0.width == s.stagedInputCount + 1 && n < $0.width && deviceFirstDraft != nil ? $0 : nil
        }
        let deviceSecondInputs = deviceFirstDraft == nil ? nil : s.deviceSecondInputs
        let deviceSecondDraft = deviceFirstDraft == nil ? nil : s.deviceSecondDraft
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
        s.nextSecondInputs = speculation == nil ? nil : deviceSecondInputs
        s.nextSecondDraft = speculation == nil ? nil : deviceSecondDraft
        s.nextFirstDraft = speculation == nil ? nil : deviceFirstDraft
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
        s.deviceFirstDraft = nil; s.hostAcceptedDrafts = nil; s.deviceSecondInputs = nil
        s.deviceSecondDraft = nil; s.deferredSecondRow = nil
    }
    public func releaseRequestState(_ requestState: any CBv2MTPRequestState) {
        guard let s = requestState as? State, s.owner == ObjectIdentifier(self), !s.released else { return }
        clearRound(s)
        s.nextFirstDraft = nil; s.nextSecondInputs = nil; s.nextSecondDraft = nil
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
        s.nextFirstDraft = nil; s.nextSecondInputs = nil; s.nextSecondDraft = nil
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
            // `next` exists only from finalize's speculation branch, whose
            // rollback stopped exactly at this row: `speculateAfterVerify`
            // appended every post-verify row, so row n's K/V (`next.keys`,
            // `next.values`) are the bytes here, and any cache write since
            // moves the row's write generation. Step over them unless the
            // rollback released their page.
            let retained = NemotronMTPDraftChainCleanup.retainedSeedRow
                ? s.speculationWriteGeneration.flatMap { s.cache.readvanceTrusted(1, writtenAt: $0) }
                : nil
            s.roots = (retained ?? s.cache.appendTrusted(keys: next.keys, values: next.values))
                + [next.ids, next.out]
            s.stagedInputCount = 1
            s.trustedOffset = s.cacheOffset
            CBv2MTPSubmitMerge.submit(s.roots)
            s.prestagedHidden = hidden
            s.prestagedResult = (tokens: next.ids, hidden: next.out)
            return
        }
        let result = draftStep(
            tokens: MLXArray([Int32(token)]).reshaped([1, 1]), hidden: hidden,
            shortlist: nil, requestState: requestState)
        CBv2MTPSubmitMerge.submit([result.tokens, result.hidden] + s.arrays)
        s.prestagedHidden = hidden
        s.prestagedResult = result
    }

    /// Also chains the next round's second draft on the prestaged first one
    /// while the GPU runs it, instead of at round build. Same ops on the same
    /// inputs as the second `draftStep`; only the submission moves earlier.
    public func prestageDrafts(
        token: Int, hidden: MLXArray, requestState: any CBv2MTPRequestState, depth: Int
    ) {
        let secondInputs = state(requestState).nextSecondInputs
        let secondDraft = state(requestState).nextSecondDraft
        prestageFirstDraft(token: token, hidden: hidden, requestState: requestState)
        let s = state(requestState)
        guard depth >= 2, draftLimit >= 2, let first = s.prestagedResult,
              s.prestagedSecond == nil, s.stagedInputCount == 1
        else { return }
        let out: MLXArray, ids: MLXArray
        if let secondDraft, secondDraft.from === first.hidden {
            // The whole second draft already ran behind the verify for this
            // row (`speculateNextFirstDraft`); only its K/V row is appended, at
            // the position the round-build step would write it.
            if NemotronH35MTPDeferSecondRow.enabled {
                s.deferredSecondRow = (keys: secondDraft.keys, values: secondDraft.values)
            } else {
                s.roots += s.cache.appendTrusted(keys: secondDraft.keys, values: secondDraft.values)
            }
            out = secondDraft.out
            ids = secondDraft.ids
        } else {
            // On the adopted device-picked first draft, its cache-free part is
            // already queued (`speculateNextFirstDraft`): same inputs, same ops.
            if let secondInputs, secondInputs.from === first.hidden {
                out = module.pagedForward(inputs: secondInputs.inputs, cache: s.cache)
            } else {
                out = module.pagedForward(
                    hidden: first.hidden,
                    embedding: target.mtpEmbedding(first.tokens.reshaped([1, 1])), cache: s.cache)
            }
            ids = draftHead.greedy(out, upper: s.screensUpper(firstDraft: false))
        }
        s.stagedInputCount += 1
        s.roots.append(contentsOf: [out, ids])
        CBv2MTPSubmitMerge.submit([ids, out] + s.arrays)
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
        s.deferredSecondRow = nil
        s.paged!.row.rollback(s.cacheOffset - s.trustedOffset)
        // Every row's K/V goes to the head cache; only the row the device
        // picks is attended (`speculateNextFirstDraft`). The front norms read
        // the tokens' embedding rows themselves once qualified
        // (NEMOTRON_MTP_FRONT_EMBED), else the embedding-rows launch feeds them.
        // The Q/K/V launch stores the K/V rows into the head cache itself when it can
        // (`NemotronMTPFrontRowsKVDirect`); otherwise they are appended here.
        let rows = target.mtpEmbeddingOperands(targetTokens).flatMap {
            module.frontRows(hidden: targetHidden, operands: $0, cache: s.cache)
        } ?? module.frontRows(
            hidden: targetHidden, embedding: target.mtpEmbedding(targetTokens), cache: s.cache)
        let written = rows.written ?? s.cache.appendTrusted(keys: rows.keys, values: rows.values)
        s.speculation = (fused: rows.fused, queries: rows.queries, keys: rows.keys,
                         values: rows.values, width: width)
        s.speculationWriteGeneration = s.paged!.row.writeGeneration
        let roots = [rows.fused, rows.queries] + written + s.cache.innerState()
        s.roots += roots
        CBv2MTPSubmitMerge.submit(roots)
    }
}

extension NemotronH35MTPAssistant: CBv2MTPDeviceAcceptedPrestaging {
    /// Queue the next round's first draft for the device-picked row right
    /// behind `speculateAfterVerify`: the same one-token expert/head tail and
    /// K/V row the host path takes from `speculation` at finalize, gathered
    /// with the device index instead of a host one.
    public func speculateNextFirstDraft(
        acceptedDrafts: MLXArray, requestState: any CBv2MTPRequestState
    ) {
        queueNextFirstDraft(acceptedDrafts: acceptedDrafts, walk: nil, requestState: requestState)
    }

    /// With the walk's operands one launch walks the packet and yields the
    /// masks and the picked row (`NemotronMTPDraftRowPick`); `acceptedDrafts`
    /// then stays unlaunched.
    public func speculateNextFirstDraft(
        acceptedDrafts: MLXArray, walk: CBv2MTPDeviceAcceptWalk.Operands,
        requestState: any CBv2MTPRequestState
    ) {
        queueNextFirstDraft(acceptedDrafts: acceptedDrafts, walk: walk, requestState: requestState)
    }

    private func queueNextFirstDraft(
        acceptedDrafts: MLXArray, walk: CBv2MTPDeviceAcceptWalk.Operands?,
        requestState: any CBv2MTPRequestState
    ) {
        let s = state(requestState)
        guard let spec = s.speculation, acceptedDrafts.shape == [1],
              spec.width == s.stagedInputCount + 1, let paged = s.paged
        else { return }
        let row = acceptedDrafts.asType(.int32)
        if let walk, queueWalkedFirstDraft(
            spec: (fused: spec.fused, queries: spec.queries, keys: spec.keys, values: spec.values, width: spec.width),
            row: row, walk: walk, paged: paged, state: s) {
            return
        }
        // The picked row attends the trusted prefix and post-verify rows 0...row: the head cache pages in place
        // once the paged draft attention qualified (`NemotronMTPDraftSDPA.directRead`; no gather is built),
        // else the snapshot's gathered copy.
        let direct = NemotronMTPDraftSDPA.directRead(paged.row)
        var rowHistory = direct == nil ? paged.row.snapshot() : nil
        let rowLength = direct?.count ?? rowHistory!.keys.dim(2)
        let rowDType = direct?.dtype ?? rowHistory!.keys.dtype
        let rowOffset = direct?.offset ?? rowHistory!.offset
        let rowFirst = rowLength - spec.width
        let picked = walk.flatMap {
            NemotronMTPDraftRowPick.pick(
                walk: $0, fused: spec.fused, queries: spec.queries, keys: spec.keys,
                values: spec.values, firstSpeculated: rowFirst, length: rowLength,
                dtype: rowDType)
        }
        let fusedMasks = picked.map { ($0.first, $0.second) } ?? NemotronMTPDraftMasks.masks(
            row: row, firstSpeculated: rowFirst, length: rowLength, dtype: rowDType)
        let rowMask: MLXArray
        if let fusedMasks {
            rowMask = fusedMasks.0
        } else {
            let rowKeep = MLXArray(Int32(0) ..< Int32(rowLength)) .<= (row + Int32(rowFirst))
            rowMask = which(rowKeep, MLXArray(Float(0)), MLXArray(-Float.infinity))
                .asType(rowDType).reshaped(1, 1, 1, rowLength)
        }
        let rowFused = picked?.fused ?? take(spec.fused, row, axis: 1)
        let rowQueries = picked?.queries ?? take(spec.queries, row, axis: 2)
        let out: MLXArray
        if let direct, let read = module.finishMaskedRow(
            fused: rowFused, queries: rowQueries, read: direct, row: paged.row, mask: rowMask) {
            out = read
        } else {
            let history = rowHistory ?? paged.row.snapshot()
            rowHistory = history
            out = module.finishMaskedRow(
                fused: rowFused, queries: rowQueries, keys: history.keys, values: history.values, mask: rowMask)
        }
        let ids = draftHead.greedy(out, upper: s.screensUpper(firstDraft: true))
        let keys = picked?.keys ?? take(spec.keys, row, axis: 2)
        let values = picked?.values ?? take(spec.values, row, axis: 2)
        s.deviceFirstDraft = (ids: ids, out: out, keys: keys, values: values)
        // The seed K/V gathers stay lazy: `prestageFirstDraft` reads them only
        // when `readvanceTrusted` cannot re-expose the stored row.
        let seedKV = NemotronMTPLazySeedKV.enabled ? [] : [keys, values]
        // The second draft's cache-free part on this first draft, so after the
        // readback only its attention and tail remain to build.
        guard NemotronH35MTPDeviceSecondFront.enabled, draftLimit >= 2 else {
            CBv2MTPSubmitMerge.submit(pickRoots(ids) + [out] + seedKV)
            return
        }
        let draftToken = ids.reshaped([1, 1])
        // With the ticket-free screen the front norms form the pick from its
        // chunk bests themselves (NEMOTRON_MTP_SCREEN_NO_TICKET).
        let inputs = target.mtpEmbeddingOperands(draftToken).flatMap { operands in
            NemotronScreenPicks.bests(of: ids).flatMap {
                module.pagedDraftInputs(hidden: out, operands: operands, bests: $0)
            } ?? module.pagedDraftInputs(hidden: out, operands: operands)
        } ?? module.pagedDraftInputs(hidden: out, embedding: target.mtpEmbedding(draftToken))
        s.deviceSecondInputs = (inputs: inputs, from: out)
        var roots = pickRoots(ids) + [out] + seedKV + inputs
        // The rest of the second draft too: its attention reads the head cache,
        // which still holds every post-verify row (they sit last). Attend over
        // the trusted prefix, post-verify rows 0...n for the device-picked n,
        // and the draft's own row; mask out the rows past n.
        if NemotronH35MTPDeviceSecondDraft.enabled, let paged = s.paged {
            // Nothing is appended to the head cache since `rowHistory` was
            // gathered (or the pages were read in place), so its views are this
            // snapshot; gather again only if the row moved.
            let reuse = paged.row.absoluteOffset == rowOffset && NemotronMTPDraftMasks.reuseSnapshot
            let direct2 = reuse && rowHistory == nil ? direct : (reuse ? nil : NemotronMTPDraftSDPA.directRead(paged.row))
            var history = direct2 == nil ? (reuse ? rowHistory : nil) ?? paged.row.snapshot() : nil
            let length = direct2?.count ?? history!.keys.dim(2)
            let historyDType = direct2?.dtype ?? history!.keys.dtype
            let firstSpeculated = length - spec.width
            let own = module.draftKeyValues(inputs: inputs)
            let ownKeys = own.keys.asType(historyDType)
            let ownValues = own.values.asType(historyDType)
            let mask: MLXArray
            if let fusedMasks, length == rowLength, firstSpeculated == rowFirst,
               historyDType == rowDType {
                mask = fusedMasks.1
            } else {
                let position = MLXArray(Int32(0) ..< Int32(length + 1))
                let keep = (position .<= (row + Int32(firstSpeculated))) .|| (position .== Int32(length))
                mask = which(keep, MLXArray(Float(0)), MLXArray(-Float.infinity))
                    .asType(historyDType).reshaped(1, 1, 1, length + 1)
            }
            let out2: MLXArray
            if let direct2, let read = module.maskedSecondDraft(
                inputs: inputs, read: direct2, row: paged.row, own: (keys: ownKeys, values: ownValues), mask: mask) {
                out2 = read
            } else {
                let h = history ?? paged.row.snapshot()
                history = h
                out2 = module.maskedSecondDraft(
                    inputs: inputs, historyKeys: h.keys, historyValues: h.values,
                    own: (keys: ownKeys, values: ownValues), mask: mask)
            }
            let ids2 = draftHead.greedy(out2, upper: s.screensUpper(firstDraft: false))
            s.deviceSecondDraft = (ids: ids2, out: out2, keys: ownKeys, values: ownValues, from: out)
            roots += pickRoots(ids2) + [out2, ownKeys, ownValues]
        }
        CBv2MTPSubmitMerge.submit(roots)
    }

    /// The submission roots for a draft pick: a ticket-free screen's pick as its chunk bests, so the final kernel
    /// runs only where the pick itself is read, outside the speculative draft submission (`NemotronScreenPicks`).
    private func pickRoots(_ pick: MLXArray) -> [MLXArray] {
        NemotronScreenPicks.bests(of: pick).map { [$0.values, $0.ids] } ?? [pick]
    }

    /// `queueNextFirstDraft` with the device accept walk folded into its consumers, once
    /// `NemotronMTPDraftSDPA.qualifyWalk` matched (NEMOTRON_MTP_DRAFT_WALK_FOLD): the first draft's attention
    /// reads the walked row's query and masks from the walk, its join reads the walked fused row, and the second
    /// draft's attention masks from the walk, so the row pick (`NemotronMTPDraftRowPick`) is not launched for
    /// them. The same drafts, K/V rows and submissions as the record path otherwise; the seed K/V rows stay lazy
    /// outputs of the row pick, which launches only if `prestageFirstDraft` cannot re-expose the stored row. False,
    /// with nothing built, unless the head cache pages serve the first draft in place.
    private func queueWalkedFirstDraft(
        spec: (fused: MLXArray, queries: MLXArray, keys: MLXArray, values: MLXArray, width: Int),
        row: MLXArray, walk: CBv2MTPDeviceAcceptWalk.Operands, paged: PagedState, state s: State
    ) -> Bool {
        guard let direct = NemotronMTPDraftSDPA.directRead(paged.row),
              let out = module.finishWalkedRow(
                  fusedRows: spec.fused, queryRows: spec.queries, walk: walk,
                  firstSpeculated: direct.count - spec.width, read: direct, row: paged.row)
        else { return false }
        let ids = draftHead.greedy(out, upper: s.screensUpper(firstDraft: true))
        // The seed K/V rows as the record path takes them: the row pick's key and value rows (else the gathers of
        // `row`). Lazy: the pick launches only when `prestageFirstDraft` reads them.
        let seedRows = NemotronMTPDraftRowPick.pick(
            walk: walk, fused: spec.fused, queries: spec.queries, keys: spec.keys, values: spec.values,
            firstSpeculated: direct.count - spec.width, length: direct.count, dtype: direct.dtype)
        let keys = seedRows?.keys ?? take(spec.keys, row, axis: 2)
        let values = seedRows?.values ?? take(spec.values, row, axis: 2)
        s.deviceFirstDraft = (ids: ids, out: out, keys: keys, values: values)
        let seedKV = NemotronMTPLazySeedKV.enabled ? [] : [keys, values]
        guard NemotronH35MTPDeviceSecondFront.enabled, draftLimit >= 2 else {
            CBv2MTPSubmitMerge.submit(pickRoots(ids) + [out] + seedKV)
            return true
        }
        let draftToken = ids.reshaped([1, 1])
        // With the ticket-free screen the front norms form the pick from its
        // chunk bests themselves (NEMOTRON_MTP_SCREEN_NO_TICKET).
        let inputs = target.mtpEmbeddingOperands(draftToken).flatMap { operands in
            NemotronScreenPicks.bests(of: ids).flatMap {
                module.pagedDraftInputs(hidden: out, operands: operands, bests: $0)
            } ?? module.pagedDraftInputs(hidden: out, operands: operands)
        } ?? module.pagedDraftInputs(hidden: out, embedding: target.mtpEmbedding(draftToken))
        s.deviceSecondInputs = (inputs: inputs, from: out)
        var roots = pickRoots(ids) + [out] + seedKV + inputs
        if NemotronH35MTPDeviceSecondDraft.enabled {
            // Nothing is appended to the head cache since `direct` was taken, so the second draft reads the same
            // pages; read them again only if the row moved.
            let reuse = paged.row.absoluteOffset == direct.offset && NemotronMTPDraftMasks.reuseSnapshot
            let direct2 = reuse ? direct : NemotronMTPDraftSDPA.directRead(paged.row)
            var history = direct2 == nil ? paged.row.snapshot() : nil
            let length = direct2?.count ?? history!.keys.dim(2)
            let historyDType = direct2?.dtype ?? history!.keys.dtype
            let firstSpeculated = length - spec.width
            let own = module.draftKeyValues(inputs: inputs)
            let ownKeys = own.keys.asType(historyDType)
            let ownValues = own.values.asType(historyDType)
            var out2 = direct2.flatMap {
                module.maskedSecondDraft(
                    inputs: inputs, read: $0, row: paged.row, own: (keys: ownKeys, values: ownValues), walk: walk,
                    firstSpeculated: firstSpeculated)
            }
            if out2 == nil {
                // The record path's mask from `row` (the walk launch then runs).
                let position = MLXArray(Int32(0) ..< Int32(length + 1))
                let keep = (position .<= (row + Int32(firstSpeculated))) .|| (position .== Int32(length))
                let mask = which(keep, MLXArray(Float(0)), MLXArray(-Float.infinity))
                    .asType(historyDType).reshaped(1, 1, 1, length + 1)
                if let direct2, let read = module.maskedSecondDraft(
                    inputs: inputs, read: direct2, row: paged.row, own: (keys: ownKeys, values: ownValues),
                    mask: mask) {
                    out2 = read
                } else {
                    let h = history ?? paged.row.snapshot()
                    history = h
                    out2 = module.maskedSecondDraft(
                        inputs: inputs, historyKeys: h.keys, historyValues: h.values,
                        own: (keys: ownKeys, values: ownValues), mask: mask)
                }
            }
            let ids2 = draftHead.greedy(out2!, upper: s.screensUpper(firstDraft: false))
            s.deviceSecondDraft = (ids: ids2, out: out2!, keys: ownKeys, values: ownValues, from: out)
            roots += pickRoots(ids2) + [out2!, ownKeys, ownValues]
        }
        CBv2MTPSubmitMerge.submit(roots)
        return true
    }

    public func noteHostAcceptedDrafts(_ accepted: Int, requestState: any CBv2MTPRequestState) {
        state(requestState).hostAcceptedDrafts = accepted
    }
}

extension NemotronH35MTPAssistant: CBv2MTPCommittedTokenObserving {
    /// Tracks how recently the request committed an upper-half token, which
    /// decides whether its drafts screen the upper vocabulary half.
    public func observeCommittedTokens(
        _ tokens: [Int], requestState: any CBv2MTPRequestState
    ) {
        let s = state(requestState)
        for token in tokens {
            if token >= NemotronHeadScreen.rows {
                s.tokensSinceUpper = 0
            } else if s.tokensSinceUpper != Int.max {
                s.tokensSinceUpper += 1
            }
        }
    }
}

/// NEMOTRON_MTP_LAZY_SEED_KV=0 submits the device-picked first draft's seed
/// K/V gathers with its ids and hidden again.
enum NemotronMTPLazySeedKV {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_MTP_LAZY_SEED_KV"] != "0"
}

/// NEMOTRON_MTP_DEVICE_SECOND_FRONT=0 builds the whole second draft after
/// the readback again.
/// NEMOTRON_MTP_DEVICE_SECOND_DRAFT=0 keeps the second draft's attention and
/// tail after the readback.
enum NemotronH35MTPDeviceSecondDraft {
    static let enabled = NemotronH35MTPDeviceSecondFront.enabled
        && ProcessInfo.processInfo.environment["NEMOTRON_MTP_DEVICE_SECOND_DRAFT"] != "0"
}

/// The two additive masks of `speculateNextFirstDraft` from one launch:
/// `first[i]` keeps `i <= row + firstSpeculated` over the snapshot's `length`
/// keys, `second[i]` the same over `length + 1` keys plus the draft's own
/// row `i == length`. Kept entries are +0 and the others -inf in the history
/// dtype, the values the integer compare, `which(keep, 0, -inf)` (FP32) and
/// the cast produce. NEMOTRON_MTP_DRAFT_MASKS=0 builds them with those ops.
enum NemotronMTPDraftMasks {
    static let enabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_DRAFT_MASKS"] != "0"
    /// NEMOTRON_MTP_DRAFT_SNAPSHOT_REUSE=0 gathers the head cache again for
    /// the second draft.
    static let reuseSnapshot =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_DRAFT_SNAPSHOT_REUSE"] != "0"
    private static let kernel = MLXFast.metalKernel(
        name: "nemotron_mtp_draft_masks",
        inputNames: ["row", "params"], outputNames: ["first", "second"],
        source: """
            const int i = int(thread_position_in_grid.x);
            const int last = row[0] + params[0];
            const int n = params[1];
            if (i < n) {
                first[i] = i <= last ? T(0.0f) : T(-INFINITY);
            }
            if (i <= n) {
                second[i] = (i <= last || i == n) ? T(0.0f) : T(-INFINITY);
            }
            """)

    /// row: [1] int32 on the device. Returns ([1, 1, 1, length], [1, 1, 1, length + 1]).
    static func masks(row: MLXArray, firstSpeculated: Int, length: Int, dtype: DType)
        -> (MLXArray, MLXArray)?
    {
        guard enabled, row.shape == [1], row.dtype == .int32, length > 0,
              dtype == .bfloat16 || dtype == .float16 || dtype == .float32 else { return nil }
        let params = MLXArray([Int32(firstSpeculated), Int32(length)])
        let threads = ((length + 1 + 255) / 256) * 256
        let out = kernel(
            [row, params], template: [("T", dtype)],
            grid: (threads, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1, 1, 1, length], [1, 1, 1, length + 1]],
            outputDTypes: [dtype, dtype])
        return (out[0], out[1])
    }
}

/// The device accept walk, both masks of `NemotronMTPDraftMasks` and the
/// walked row of the post-verify fused residuals, queries, keys and values
/// from one launch, instead of the walk launch, the masks launch and four
/// `take` gathers. The walk is the engine's integer loop on the same packet
/// (`CBv2MTPDeviceAcceptWalk`), the masks are the masks kernel's statements
/// and every row element is a copy. Engaged only once `qualify` matched the
/// separate launches bit for bit. NEMOTRON_MTP_DRAFT_ROW_PICK=0 keeps them.
enum NemotronMTPDraftRowPick {
    static let enabled = NemotronMTPDraftMasks.enabled
        && ProcessInfo.processInfo.environment["NEMOTRON_MTP_DRAFT_ROW_PICK"] != "0"
    nonisolated(unsafe) static var verified = false
    private static let kernel = MLXFast.metalKernel(
        name: "nemotron_mtp_draft_row_pick",
        inputNames: ["packet", "params", "fused", "queries", "keys", "values"],
        outputNames: ["first", "second", "fused_row", "query_row", "key_row", "value_row"],
        source: source, ensureRowContiguous: false)
    private static let source = """
            const int i = int(thread_position_in_grid.x);
            const int count = params[0];
            const int drafts = params[1];
            const int targets = params[2];
            const int64_t step = packet_strides[0];
            int a = 0;
            while (a < count && packet[(targets + a) * step] == packet[(drafts + a) * step]) {
                ++a;
            }
            const int last = a + params[3];
            const int n = params[4];
            if (i < n) {
                first[i] = i <= last ? T(0.0f) : T(-INFINITY);
            }
            if (i <= n) {
                second[i] = (i <= last || i == n) ? T(0.0f) : T(-INFINITY);
            }
            if (i < FW) {
                fused_row[i] = fused[a * fused_strides[1] + i * fused_strides[2]];
            }
            if (i < QH * HD) {
                query_row[i] = queries[(i / HD) * queries_strides[1] + a * queries_strides[2]
                    + (i % HD) * queries_strides[3]];
            }
            if (i < KH * HD) {
                key_row[i] = keys[(i / HD) * keys_strides[1] + a * keys_strides[2]
                    + (i % HD) * keys_strides[3]];
                value_row[i] = values[(i / HD) * values_strides[1] + a * values_strides[2]
                    + (i % HD) * values_strides[3]];
            }
            """

    /// NEMOTRON_MTP_ROW_PICK_LOADS_FIRST=0 keeps the record's load / store order. Otherwise each thread reads
    /// its elements of the walked row (fused, query, key and value; the record's addresses, or element 0 where
    /// the record reads none) right after the walk, before its mask stores, and stores them after the masks
    /// (`loadsFirstSource`): no device load follows a device store of the same thread. Every load, value and
    /// address is the record's. Used once `qualify` found it bitwise equal to the record launch.
    static let loadsFirstRequested =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_ROW_PICK_LOADS_FIRST"] != "0"
    nonisolated(unsafe) static var loadsFirstVerified = false
    private static let loadsFirstSource: String? = {
        let anchor = "const int last = a + params[3];\n"
        let copies = "if (i < FW) {\n    fused_row[i] = fused[a * fused_strides[1] + i * fused_strides[2]];\n}\n"
        guard source.components(separatedBy: anchor).count == 2,
              let start = source.range(of: copies), source[start.lowerBound...].hasSuffix("    + (i % HD) * values_strides[3]];\n}")
        else { return nil }
        let loads = """
            const bool nrp_f = i < FW;
            const bool nrp_q = i < QH * HD;
            const bool nrp_k = i < KH * HD;
            const auto nrp_fv = fused[nrp_f ? a * fused_strides[1] + i * fused_strides[2] : 0];
            const auto nrp_qv = queries[nrp_q ? (i / HD) * queries_strides[1] + a * queries_strides[2]
                + (i % HD) * queries_strides[3] : 0];
            const auto nrp_kv = keys[nrp_k ? (i / HD) * keys_strides[1] + a * keys_strides[2]
                + (i % HD) * keys_strides[3] : 0];
            const auto nrp_vv = values[nrp_k ? (i / HD) * values_strides[1] + a * values_strides[2]
                + (i % HD) * values_strides[3] : 0];

            """
        let stores = """
            if (nrp_f) {
                fused_row[i] = nrp_fv;
            }
            if (nrp_q) {
                query_row[i] = nrp_qv;
            }
            if (nrp_k) {
                key_row[i] = nrp_kv;
                value_row[i] = nrp_vv;
            }
            """
        return String(source[..<start.lowerBound]).replacingOccurrences(of: anchor, with: loads + anchor) + stores
    }()
    private static let loadsFirstKernel = loadsFirstSource.map {
        MLXFast.metalKernel(
            name: "nemotron_mtp_draft_row_pick_lf",
            inputNames: ["packet", "params", "fused", "queries", "keys", "values"],
            outputNames: ["first", "second", "fused_row", "query_row", "key_row", "value_row"],
            source: $0, ensureRowContiguous: false)
    }

    /// For the row `walk` selects: (first mask [1, 1, 1, length], second mask
    /// [1, 1, 1, length + 1] as `NemotronMTPDraftMasks.masks`, fused [1, 1, F],
    /// queries [1, QH, 1, D], keys and values [1, KH, 1, D] as `take` along
    /// the row axis). fused [1, W, F], queries [1, QH, W, D] and keys / values
    /// [1, KH, W, D] are read through their strides; nil outside the shapes.
    static func pick(
        walk: CBv2MTPDeviceAcceptWalk.Operands, fused: MLXArray, queries: MLXArray,
        keys: MLXArray, values: MLXArray, firstSpeculated: Int, length: Int, dtype: DType,
        force: Bool = false, loadsFirst: Bool? = nil
    ) -> (first: MLXArray, second: MLXArray, fused: MLXArray, queries: MLXArray,
          keys: MLXArray, values: MLXArray)?
    {
        guard enabled, force || verified, length > 0,
              dtype == .bfloat16 || dtype == .float16 || dtype == .float32,
              walk.packet.ndim == 1, walk.packet.dtype == .int32,
              fused.ndim == 3, fused.dim(0) == 1, walk.count < fused.dim(1),
              queries.ndim == 4, queries.dim(0) == 1, queries.dim(2) == fused.dim(1),
              keys.ndim == 4, keys.dim(0) == 1, keys.dim(2) == fused.dim(1),
              values.shape == keys.shape, queries.dim(3) == keys.dim(3)
        else { return nil }
        let headDim = keys.dim(3)
        let params = MLXArray([Int32(walk.count), Int32(walk.drafts), Int32(walk.targets),
                               Int32(firstSpeculated), Int32(length)])
        let widest = max(length + 1, fused.dim(2), queries.dim(1) * headDim, keys.dim(1) * headDim)
        // The loads-first twin once verified (`loadsFirst` names the form for the load-time check).
        let chosen = (loadsFirst ?? (loadsFirstRequested && loadsFirstVerified)) ? loadsFirstKernel ?? kernel : kernel
        let out = chosen(
            [walk.packet, params, fused, queries, keys, values],
            template: [("T", dtype), ("FW", fused.dim(2)), ("QH", queries.dim(1)),
                       ("KH", keys.dim(1)), ("HD", headDim)],
            grid: ((widest + 255) / 256 * 256, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1, 1, 1, length], [1, 1, 1, length + 1], [1, 1, fused.dim(2)],
                           [1, queries.dim(1), 1, headDim], [1, keys.dim(1), 1, headDim],
                           [1, keys.dim(1), 1, headDim]],
            outputDTypes: [dtype, dtype, fused.dtype, queries.dtype, keys.dtype, values.dtype])
        return (out[0], out[1], out[2], out[3], out[4], out[5])
    }

    /// Load time: the launch against the masks launch and the four `take`
    /// gathers of the row it walks to, bit for bit, for every accept count of
    /// a depth-2 packet in both argument spaces (6 elements: constant, 11:
    /// device), on seeded rows in the post-verify layouts (Q/K/V transposed
    /// views). Engages only if all match; also compiles both pipelines.
    static func qualify() {
        verified = false
        guard enabled else { return }
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron mtp draft row pick: \(message)\n".utf8))
        }
        func bits(_ a: MLXArray) -> MLXArray {
            a.view(dtype: a.itemSize == 2 ? .uint16 : (a.itemSize == 4 ? .uint32 : .uint8))
        }
        func seeded(_ shape: [Int], _ seed: UInt64) -> MLXArray {
            MLXRandom.normal(shape, key: MLXRandom.key(seed)).asType(.bfloat16)
        }
        let width = 3, length = 600
        let fused = seeded([1, width, 2688], 0x7069_636b)
        let queries = seeded([1, width, 4096], 0x7069_636c)
            .reshaped(1, width, 32, 128).transposed(0, 2, 1, 3)
        let keys = seeded([1, width, 256], 0x7069_636d)
            .reshaped(1, width, 2, 128).transposed(0, 2, 1, 3)
        let values = seeded([1, width, 256], 0x7069_636e)
            .reshaped(1, width, 2, 128).transposed(0, 2, 1, 3)
        do {
            let matched = try withError { () -> Bool in
                var ok = MLXArray(true)
                for size in [6, 11] {
                    for accepted in 0 ... 2 {
                        // drafts at 0..<2, targets at 2..<5; the rest is filler.
                        var host = [Int32](repeating: 7, count: size)
                        host[0] = 11; host[1] = 12
                        host[2] = accepted >= 1 ? 11 : 21
                        host[3] = accepted >= 2 ? 12 : 22
                        host[4] = 13
                        let walk = CBv2MTPDeviceAcceptWalk.Operands(
                            packet: MLXArray(host), count: 2, drafts: 0, targets: 2)
                        let row = MLXArray([Int32(accepted)])
                        guard let picked = pick(
                                walk: walk, fused: fused, queries: queries, keys: keys,
                                values: values, firstSpeculated: length - width, length: length,
                                dtype: .bfloat16, force: true),
                              let masks = NemotronMTPDraftMasks.masks(
                                row: row, firstSpeculated: length - width, length: length,
                                dtype: .bfloat16)
                        else { return false }
                        let expected = [masks.0, masks.1, take(fused, row, axis: 1),
                                        take(queries, row, axis: 2), take(keys, row, axis: 2),
                                        take(values, row, axis: 2)]
                        let produced = [picked.first, picked.second, picked.fused, picked.queries,
                                        picked.keys, picked.values]
                        for (a, b) in zip(produced, expected) {
                            guard a.shape == b.shape, a.dtype == b.dtype else { return false }
                            ok = MLX.logicalAnd(ok, all(bits(a) .== bits(b)))
                        }
                        eval([ok] + produced)
                    }
                }
                return ok.item(Bool.self)
            }
            verified = matched
        } catch {
            verified = false
        }
        report(verified ? "on (bitwise equal, accept counts 0...2)" : "off (differs or failed)")
        qualifyLoadsFirst()
    }

    /// Load time, after `qualify`: the loads-first launch against the record launch, all six outputs bit for bit,
    /// for every accept count of a depth-2 packet in both argument spaces, mask lengths below and above the row
    /// widths (so threads past every row read element 0), on seeded rows in the post-verify layouts with zeros,
    /// -0, infinities and NaN among their values. Engages only if every case matches. One stderr line.
    private static func qualifyLoadsFirst() {
        loadsFirstVerified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron mtp draft row pick loads first: \(message)\n".utf8))
        }
        guard loadsFirstRequested else { return report("off (setting)") }
        guard verified, loadsFirstKernel != nil else { return report("off (record launch off or source moved)") }
        func bits(_ a: MLXArray) -> MLXArray {
            a.view(dtype: a.itemSize == 2 ? .uint16 : (a.itemSize == 4 ? .uint32 : .uint8))
        }
        let specials = MLXArray([Float(0), -0.0, .infinity, -.infinity, .nan, 1e-30, -3.5, 65504])
        func seeded(_ shape: [Int], _ seed: UInt64) -> MLXArray {
            let count = shape.reduce(1, *)
            let values = MLXRandom.normal([count], key: MLXRandom.key(seed))
            return concatenated([specials, values[8...]]).reshaped(shape).asType(.bfloat16)
        }
        let width = 3
        let fused = seeded([1, width, 2688], 0x6c66_0001)
        let queries = seeded([1, width, 4096], 0x6c66_0002).reshaped(1, width, 32, 128).transposed(0, 2, 1, 3)
        let keys = seeded([1, width, 256], 0x6c66_0003).reshaped(1, width, 2, 128).transposed(0, 2, 1, 3)
        let values = seeded([1, width, 256], 0x6c66_0004).reshaped(1, width, 2, 128).transposed(0, 2, 1, 3)
        var cases = 0
        do {
            let matched = try withError { () -> Bool in
                var ok = MLXArray(true)
                for length in [600, 5000] {
                    for size in [6, 11] {
                        for accepted in 0 ... 2 {
                            var host = [Int32](repeating: 7, count: size)
                            host[0] = 11; host[1] = 12
                            host[2] = accepted >= 1 ? 11 : 21
                            host[3] = accepted >= 2 ? 12 : 22
                            host[4] = 13
                            let walk = CBv2MTPDeviceAcceptWalk.Operands(
                                packet: MLXArray(host), count: 2, drafts: 0, targets: 2)
                            func picked(_ loadsFirst: Bool) -> [MLXArray]? {
                                pick(walk: walk, fused: fused, queries: queries, keys: keys, values: values,
                                     firstSpeculated: length - width, length: length, dtype: .bfloat16,
                                     force: true, loadsFirst: loadsFirst)
                                    .map { [$0.first, $0.second, $0.fused, $0.queries, $0.keys, $0.values] }
                            }
                            guard let a = picked(false), let b = picked(true) else { return false }
                            for (x, y) in zip(a, b) {
                                guard x.shape == y.shape, x.dtype == y.dtype else { return false }
                                ok = MLX.logicalAnd(ok, all(bits(x) .== bits(y)))
                            }
                            eval([ok] + a + b)
                            cases += 1
                        }
                    }
                }
                return ok.item(Bool.self)
            }
            loadsFirstVerified = matched
        } catch {
            return report("off (\(error))")
        }
        report(loadsFirstVerified
            ? "on (\(cases) cases bitwise equal to the record launch: masks and walked rows, accept counts 0...2, "
                + "zeros, -0, infinities and NaN included)"
            : "off (differs from the record launch)")
    }
}

enum NemotronH35MTPDeferSecondRow {
    static let enabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_HOLD_SECOND_KV"] != "0"
}

enum NemotronH35MTPDeviceSecondFront {
    static let enabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_DEVICE_SECOND_FRONT"] != "0"
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
