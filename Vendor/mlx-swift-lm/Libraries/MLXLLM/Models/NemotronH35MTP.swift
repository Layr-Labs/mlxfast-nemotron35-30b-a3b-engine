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
        /// Ticket counter of the fused select + final launch
        /// (`NemotronHeadScreen.makeCounter`), set once its ids matched the
        /// two-launch path bitwise.
        var counter: MLXArray? = nil
        /// Weight-only energies and lossless all-pair records. Group selection uses
        /// the current draft activation; target and exact rescore still read the head.
        var activationPlan: (lower: MLXArray, upper: MLXArray, records: MLXArray)? = nil
    }
    private var screen: Screen?
    private var checkSteps = 0, checkMisses = 0, checkRescoreMismatches = 0

    /// Qualified immutable screen backing is outside the registered module
    /// tree. Include it in the existing decode lease, but never the mutable
    /// fused-selection ticket counter or any request tensor.
    var decodeResidencyArrays: [MLXArray] {
        guard let screen else { return [] }
        var arrays = [screen.groups] + [screen.upperGroups, screen.packed, screen.records]
            .compactMap { $0 }
        if let plan = screen.activationPlan {
            arrays += [plan.lower, plan.upper, plan.records]
        }
        return arrays
    }

    /// Greedy draft id [1] int32 for hidden [1, 1, K]: the screen when it
    /// qualified at load and the shapes match, else the stock argmax.
    /// `upper`: also screen the upper vocabulary half (when the plan has it).
    func greedy(_ hidden: MLXArray, upper: Bool = false) -> MLXArray {
        guard let screen, hidden.shape == [1, 1, NemotronHeadScreen.width],
              hidden.dtype == .bfloat16, StreamOrDevice.default == .gpu
        else { return argMax(logits(hidden)[0..., -1, 0...], axis: -1).asType(.int32) }
        if let plan = screen.activationPlan,
           let selected = NemotronHeadScreen.selectActivationGroups(
               hidden, lowerImportance: plan.lower, upperImportance: plan.upper,
               fallbackGroups: screen.groups, fallbackUpperGroups: screen.upperGroups,
               groupCount: screen.groupCount, upperGroupCount: screen.upperGroupCount) {
            let result = NemotronHeadScreen.run(
                hidden, weight: screen.weight, scales: screen.scales, biases: screen.biases,
                groups: selected.groups, groupCount: screen.groupCount, chunk: screen.chunk,
                keep: screen.keep,
                upperGroups: upper && screen.upperGroups != nil ? selected.upperGroups : nil,
                upperGroupCount: upper ? screen.upperGroupCount : 0,
                counter: screen.counter, groupRecords: plan.records)
            if NemotronHeadScreen.check && !(upper && screen.upperGroups != nil) {
                checkAgainstStock(hidden, result, screen)
            }
            return result.id
        }
        let result = NemotronHeadScreen.run(
            hidden, weight: screen.weight, scales: screen.scales, biases: screen.biases,
            groups: screen.groups, groupCount: screen.groupCount, chunk: screen.chunk,
            keep: screen.keep, upperGroups: upper ? screen.upperGroups : nil,
            upperGroupCount: upper ? screen.upperGroupCount : 0, packed: screen.packed,
            records: screen.records,
            counter: screen.counter)
        if NemotronHeadScreen.check && !(upper && screen.upperGroups != nil) {
            checkAgainstStock(hidden, result, screen)
        }
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
        let groupCount = Int(environment["NEMOTRON_MTP_HEAD_SCREEN_GROUPS"] ?? "") ?? 12
        let chunk = Int(environment["NEMOTRON_MTP_HEAD_SCREEN_CHUNK"] ?? "") ?? 1024
        let keep = Int(environment["NEMOTRON_MTP_HEAD_SCREEN_KEEP"] ?? "") ?? 16
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
        var columnEnergy: [Int: MLXArray] = [:]
        func selectGroups(_ rowRange: Range<Int>, count: Int) -> [Int] {
            var columns = MLXArray.zeros([k], dtype: .float32)
            for start in stride(from: rowRange.lowerBound, to: rowRange.upperBound, by: 8192) {
                let r = start..<(start + 8192)
                let d = dequantized(head.weight[r], scales: head.scales[r], biases: biases[r],
                                    groupSize: 64, bits: 4).asType(.float32)
                columns = columns + (d * d).sum(axis: 0)
                eval(columns)
            }
            columnEnergy[rowRange.lowerBound] = columns
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
        let upperCount = Int(environment["NEMOTRON_MTP_HEAD_SCREEN_UPPER_GROUPS"] ?? "") ?? 8
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
        if NemotronHeadScreen.packedRecords {
            adopt("full records") { candidate in
                let records = NemotronHeadScreen.packRecords(
                    weight: head.weight, scales: head.scales, biases: biases, groups: selected,
                    upperGroups: upper)
                candidate.records = records
                return records
            }
        }
        if plan.records == nil, NemotronHeadScreen.packedScalesBiases, (upper?.count ?? 0) <= groupCount {
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
                                    packed: plan.packed, records: plan.records, counter: counter)
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
        // Preserve packed reads for current-activation group selection. Every record
        // holds the original code words and coefficient bits, including groups not
        // used by the static plan. Only adopt after gathered/packed equality and exact
        // candidate-logit qualification on both screened vocabulary sizes.
        var activationNote = "off"
        if NemotronHeadScreen.activationGroupsEnabled, let lower = columnEnergy[0] {
            let upperEnergy = columnEnergy[rows] ?? lower
            do {
                let verified = try withError { () -> MLXArray? in
                    let copy = NemotronHeadScreen.packGroupRecords(
                        weight: head.weight, scales: head.scales, biases: biases)
                    eval(copy)
                    var state: UInt64 = 0xD1B5_4A32_D192_ED03
                    for _ in 0..<4 {
                        let t = (0..<k).map { _ -> Float in
                            state = state &* 6364136223846793005 &+ 1442695040888963407
                            return (Float(state >> 40) / Float(1 << 24) - 0.5) * 3.4641016
                        }
                        let x = (MLXArray(t) * g).asType(.bfloat16).reshaped(1, 1, k)
                        guard let selected = NemotronHeadScreen.selectActivationGroups(
                            x, lowerImportance: lower, upperImportance: upperEnergy,
                            fallbackGroups: plan.groups, fallbackUpperGroups: plan.upperGroups,
                            groupCount: groupCount, upperGroupCount: plan.upperGroupCount)
                        else { return nil }
                        eval(selected.groups, selected.upperGroups)
                        func valid(_ array: MLXArray, count: Int) -> Bool {
                            let ids = array.asArray(UInt32.self)
                            return ids.count == count && Set(ids).count == count
                                && ids == ids.sorted() && ids.allSatisfy({ $0 < 42 })
                                && stride(from: 0, to: count, by: 2).allSatisfy({
                                    ids[$0] % 2 == 0 && ids[$0 + 1] == ids[$0] + 1
                                })
                        }
                        guard valid(selected.groups, count: groupCount),
                              valid(selected.upperGroups,
                                  count: plan.upperGroups == nil ? groupCount : plan.upperGroupCount)
                        else { return nil }
                        let full = head(x)[0, -1].asType(.float32)
                        eval(full)
                        let fullValues = full.asArray(Float.self)
                        for includeUpper in plan.upperGroups == nil ? [false] : [false, true] {
                            func run(_ records: MLXArray?, counter: MLXArray?) -> NemotronHeadScreen.Result {
                                NemotronHeadScreen.run(
                                    x, weight: plan.weight, scales: plan.scales, biases: plan.biases,
                                    groups: selected.groups, groupCount: groupCount, chunk: chunk, keep: keep,
                                    upperGroups: includeUpper ? selected.upperGroups : nil,
                                    upperGroupCount: includeUpper ? plan.upperGroupCount : 0,
                                    counter: counter, groupRecords: records)
                            }
                            let reference = run(nil, counter: nil)
                            let packed = run(copy, counter: plan.counter)
                            let sameScores = (reference.scores.view(dtype: .uint32)
                                .== packed.scores.view(dtype: .uint32)).all()
                            let sameIds = (reference.candidateIds .== packed.candidateIds).all()
                            let sameValues = (reference.candidateValues.view(dtype: .uint32)
                                .== packed.candidateValues.view(dtype: .uint32)).all()
                            let samePick = (reference.id .== packed.id).all()
                            eval(sameScores, sameIds, sameValues, samePick)
                            guard sameScores.item(Bool.self), sameIds.item(Bool.self),
                                  sameValues.item(Bool.self), samePick.item(Bool.self)
                            else { return nil }
                            let ids = packed.candidateIds.asArray(Int32.self)
                            let values = packed.candidateValues.asArray(Float.self)
                            let scoredRows = includeUpper ? NemotronHeadScreen.headRows : rows
                            guard ids.count == values.count, !ids.isEmpty,
                                  Set(ids).count == ids.count,
                                  ids.allSatisfy({ (0..<Int32(scoredRows)).contains($0) }),
                                  zip(ids, values).allSatisfy({ fullValues[Int($0)].bitPattern == $1.bitPattern })
                            else { return nil }
                            var best = 0
                            for i in ids.indices
                            where values[i] > values[best] || (values[i] == values[best] && ids[i] < ids[best]) {
                                best = i
                            }
                            guard packed.id.item(Int32.self) == ids[best] else { return nil }
                        }
                    }
                    return copy
                }
                if let verified {
                    plan.activationPlan = (lower, upperEnergy, verified)
                    activationNote = "on (packed all-pair records, exact candidates qualified)"
                } else {
                    activationNote = "off (qualification failed)"
                }
            } catch {
                activationNote = "off (\(error))"
            }
        }
        screen = plan
        // Requests switch between the full and the half-head screen; compile
        // the half-head variant (on the same packed and fused path) here as
        // well, so neither builds inside a timed window.
        if plan.upperGroups != nil {
            let x = (MLXArray.ones([k]) * g).asType(.bfloat16).reshaped(1, 1, k)
            eval(NemotronHeadScreen.run(
                x, weight: plan.weight, scales: plan.scales, biases: plan.biases,
                groups: plan.groups, groupCount: groupCount, chunk: chunk, keep: keep,
                packed: plan.packed, records: plan.records, counter: plan.counter).id)
        }
        report("on, \(groupCount) of 42 groups \(selected), keep \(keep) per \(chunk) rows,"
            + " upper half \(upper.map { "\($0.count) groups \($0)" } ?? "off"),"
            + " packed screen \(packedNotes.isEmpty ? "off" : packedNotes.joined(separator: ", ")), fused final \(fusedNote),"
            + " qualified (synthetic agreement \(chosen.agree)/4), activation groups \(activationNote)")
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
            decodeResidency = NemotronDecodeResidency(
                head: module.parameters().flattened().map { $0.1 }
                    + draftHead.decodeResidencyArrays,
                layers: NemotronZSkip.decodeCopyLayers(target), owners: NemotronZSkip.decodeCopyOwners(target),
                target: NemotronDecodeResidency.targetGroups(target),
                extras: NemotronZSkip.draftCopyArrays + NemotronSBDict.allArrays)
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
        let out = module.pagedForward(
            hidden: hidden, embedding: target.mtpEmbedding(tokens), cache: s.cache)
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
        s.deviceSecondDraft = nil
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
            s.roots += s.cache.appendTrusted(keys: secondDraft.keys, values: secondDraft.values)
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
        s.paged!.row.rollback(s.cacheOffset - s.trustedOffset)
        // Every row's K/V goes to the head cache; only the row the device
        // picks is attended (`speculateNextFirstDraft`).
        let rows = module.frontRows(
            hidden: targetHidden, embedding: target.mtpEmbedding(targetTokens))
        let written = s.cache.appendTrusted(keys: rows.keys, values: rows.values)
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
        let s = state(requestState)
        guard let spec = s.speculation, acceptedDrafts.shape == [1],
              spec.width == s.stagedInputCount + 1, let paged = s.paged
        else { return }
        let row = acceptedDrafts.asType(.int32)
        // The picked row attends the trusted prefix and post-verify rows 0...row.
        let rowHistory = paged.row.snapshot()
        let rowLength = rowHistory.keys.dim(2)
        let rowFirst = rowLength - spec.width
        let fusedMasks = NemotronMTPDraftMasks.masks(
            row: row, firstSpeculated: rowFirst, length: rowLength, dtype: rowHistory.keys.dtype)
        let rowMask: MLXArray
        if let fusedMasks {
            rowMask = fusedMasks.0
        } else {
            let rowKeep = MLXArray(Int32(0) ..< Int32(rowLength)) .<= (row + Int32(rowFirst))
            rowMask = which(rowKeep, MLXArray(Float(0)), MLXArray(-Float.infinity))
                .asType(rowHistory.keys.dtype).reshaped(1, 1, 1, rowLength)
        }
        let attended = module.maskedRowOutput(
            fused: take(spec.fused, row, axis: 1), queries: take(spec.queries, row, axis: 2),
            keys: rowHistory.keys, values: rowHistory.values, mask: rowMask)
        let out = module.finishPagedAttentionRow(attended)
        let ids = draftHead.greedy(out, upper: s.screensUpper(firstDraft: true))
        let keys = take(spec.keys, row, axis: 2), values = take(spec.values, row, axis: 2)
        s.deviceFirstDraft = (ids: ids, out: out, keys: keys, values: values)
        // The second draft's cache-free part on this first draft, so after the
        // readback only its attention and tail remain to build.
        guard NemotronH35MTPDeviceSecondFront.enabled, draftLimit >= 2 else {
            CBv2MTPSubmitMerge.submit([ids, out, keys, values])
            return
        }
        let inputs = module.pagedDraftInputs(
            hidden: out, embedding: target.mtpEmbedding(ids.reshaped([1, 1])))
        s.deviceSecondInputs = (inputs: inputs, from: out)
        var roots = [ids, out, keys, values] + inputs
        // The rest of the second draft too: its attention reads the head cache,
        // which still holds every post-verify row (they sit last). Attend over
        // the trusted prefix, post-verify rows 0...n for the device-picked n,
        // and the draft's own row; mask out the rows past n.
        if NemotronH35MTPDeviceSecondDraft.enabled, let paged = s.paged {
            // Nothing is appended to the head cache since `rowHistory` was
            // gathered, so its views are this snapshot; gather again only if
            // the row moved.
            let history = paged.row.absoluteOffset == rowHistory.offset
                && NemotronMTPDraftMasks.reuseSnapshot ? rowHistory : paged.row.snapshot()
            let length = history.keys.dim(2)
            let firstSpeculated = length - spec.width
            let own = module.draftKeyValues(inputs: inputs)
            let ownKeys = own.keys.asType(history.keys.dtype)
            let ownValues = own.values.asType(history.values.dtype)
            let mask: MLXArray
            if let fusedMasks, length == rowLength, firstSpeculated == rowFirst,
               history.keys.dtype == rowHistory.keys.dtype {
                mask = fusedMasks.1
            } else {
                let position = MLXArray(Int32(0) ..< Int32(length + 1))
                let keep = (position .<= (row + Int32(firstSpeculated))) .|| (position .== Int32(length))
                mask = which(keep, MLXArray(Float(0)), MLXArray(-Float.infinity))
                    .asType(history.keys.dtype).reshaped(1, 1, 1, length + 1)
            }
            let out2 = module.maskedSecondDraft(
                inputs: inputs,
                keys: concatenated([history.keys, ownKeys], axis: 2),
                values: concatenated([history.values, ownValues], axis: 2), mask: mask)
            let ids2 = draftHead.greedy(out2, upper: s.screensUpper(firstDraft: false))
            s.deviceSecondDraft = (ids: ids2, out: out2, keys: ownKeys, values: ownValues, from: out)
            roots += [ids2, out2, ownKeys, ownValues]
        }
        CBv2MTPSubmitMerge.submit(roots)
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
