// ZSKIP INTERFACE VERSION 2 -- 2026-10-24
// Module: MLXLMCommon. See interface.md for packaging and exact call sites.
import MLX

public enum NemotronZSkipSite: String, CaseIterable {
    case routedWeighted, routedLeader, routedDraft, sharedVerify, sharedDraft
}

public struct NemotronZSkipStock {
    public let weight: MLXArray
    public let scales: MLXArray
    public let biases: MLXArray
    public let k: Int
    public let n: Int
    public let experts: Int
    public init(weight: MLXArray, scales: MLXArray, biases: MLXArray,
                k: Int, n: Int, experts: Int) {
        self.weight = weight; self.scales = scales; self.biases = biases
        self.k = k; self.n = n; self.experts = experts
    }
    public var byteCount: Int { weight.nbytes + scales.nbytes + biases.nbytes }
}

public struct NemotronZSkipDerived {
    public let arrays: [MLXArray]
    public let layout: String
    public init(arrays: [MLXArray], layout: String) {
        self.arrays = arrays; self.layout = layout
    }
    public var byteCount: Int { arrays.reduce(0) { $0 + $1.nbytes } }
}

public protocol NemotronZSkipKernel {
    var version: String { get }
    // Exact allocated bytes, before building. No transient buffers retained.
    func derivedByteCount(_ stock: NemotronZSkipStock) -> Int
    // Must allocate NEW storage; NEVER update or replace stock arrays.
    func build(_ stock: NemotronZSkipStock) -> NemotronZSkipDerived
    // Original shapes and integer/bf16 bits, in original parameter order.
    func inverse(_ derived: NemotronZSkipDerived,
                 stock: NemotronZSkipStock) -> [MLXArray]
    func supports(site: NemotronZSkipSite, rows: Int,
                  stock: NemotronZSkipStock) -> Bool
    // x = [rows*topK,K] routed; [rows,K] shared, bf16.
    // indices = [rows*topK] uint32; scores = [rows*6] fp32, weighted only.
    // routedWeighted -> [1,rows,N] bf16, INCLUDING native weighting/reduction.
    // routedLeader/routedDraft -> [rows*topK,N] bf16 (no weighting).
    // sharedVerify/sharedDraft -> [rows,N] bf16 (no linear bias).
    // `original` is an exact reference closure used by the stub; production
    // kernels ignore it. It ALWAYS reads the unchanged loaded stock parameters.
    func down(_ derived: NemotronZSkipDerived, stock: NemotronZSkipStock,
              x: MLXArray, indices: MLXArray?, scores: MLXArray?,
              site: NemotronZSkipSite, rows: Int, topK: Int,
              original: () -> MLXArray) -> MLXArray
}

import Foundation
import Metal
import Darwin

/// Harness-only provider: full independent copies and qualification, stock math.
public struct NemotronZSkipStub: NemotronZSkipKernel {
    public init() {}
    public var version: String { "stock-stub-v2" }
    public func derivedByteCount(_ s: NemotronZSkipStock) -> Int { s.byteCount }
    public func build(_ s: NemotronZSkipStock) -> NemotronZSkipDerived {
        // MLXArray(data:) allocates a new MLX buffer; asData(.copy) owns new CPU
        // bytes. No contiguous()/view shortcut can alias a loaded parameter.
        let arrays = [s.weight, s.scales, s.biases].map { MLXArray(data: $0.asData()) }
        eval(arrays)
        return NemotronZSkipDerived(arrays: arrays, layout: "independent-plain-copy-v1")
    }
    public func inverse(_ d: NemotronZSkipDerived, stock: NemotronZSkipStock) -> [MLXArray] {
        d.arrays
    }
    public func supports(site: NemotronZSkipSite, rows: Int, stock: NemotronZSkipStock) -> Bool {
        guard stock.experts == 128, stock.k == 1856, stock.n == 2688 else { return false }
        switch site {
        case .routedWeighted, .routedDraft: return (1...4).contains(rows)
        case .routedLeader: return (2...4).contains(rows)
        default: return false
        }
    }
    public func down(_ derived: NemotronZSkipDerived, stock: NemotronZSkipStock,
                     x: MLXArray, indices: MLXArray?, scores: MLXArray?,
                     site: NemotronZSkipSite, rows: Int, topK: Int,
                     original: () -> MLXArray) -> MLXArray { original() }
}

public final class NemotronZSkipBudget {
    public static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_ZSKIP"] != "0"
    public let recommended: Int
    public let resident: Int
    public let margin = 8 * 1024 * 1024 * 1024
    public private(set) var reserved = 0
    public init() {
        recommended = Int(MTLCreateSystemDefaultDevice()?.recommendedMaxWorkingSetSize ?? 0)
        // Includes already evaluated target and embedded head; immutable originals
        // remain live throughout the derived cache's lifetime.
        resident = Memory.activeMemory
    }
    public func reserve(_ bytes: Int) -> Bool {
        guard bytes > 0, resident + reserved + bytes + margin <= recommended else { return false }
        reserved += bytes
        return true
    }
    public func release(_ bytes: Int) { reserved -= bytes }
    public static func log(_ text: String) {
        FileHandle.standardError.write(Data(("zskip " + text + "\n").utf8))
    }
    public static func wiredLimitMB() -> Int {
        var value: Int32 = -1
        var size = MemoryLayout<Int32>.size
        return sysctlbyname("iogpu.wired_limit_mb", &value, &size, nil, 0) == 0 ? Int(value) : -1
    }
}

/// Immutable after load. It is held by the owning projection module, not a
/// global parameter cache. Existing compiler graphs capture its derived arrays.
public final class NemotronZSkipEntry {
    public let derived: NemotronZSkipDerived
    public let stock: NemotronZSkipStock
    private let kernel: any NemotronZSkipKernel
    private let allowed: [NemotronZSkipSite: Set<Int>]
    private init(derived: NemotronZSkipDerived, stock: NemotronZSkipStock,
                 kernel: any NemotronZSkipKernel, allowed: [NemotronZSkipSite: Set<Int>]) {
        self.derived = derived; self.stock = stock; self.kernel = kernel; self.allowed = allowed
    }
    public func down(x: MLXArray, indices: MLXArray? = nil, scores: MLXArray? = nil,
                     site: NemotronZSkipSite, rows: Int, topK: Int = 6,
                     original: () -> MLXArray) -> MLXArray? {
        guard x.dtype == .bfloat16, x.ndim == 2, x.dim(1) == stock.k,
              allowed[site]?.contains(rows) == true,
              (stock.experts == 1 || topK == 6),
              x.dim(0) == rows * (stock.experts == 1 ? 1 : topK) else { return nil }
        return kernel.down(derived, stock: stock, x: x, indices: indices, scores: scores,
                           site: site, rows: rows, topK: topK, original: original)
    }

    public static func make(name: String, stock: NemotronZSkipStock,
                            kernel: any NemotronZSkipKernel, budget: NemotronZSkipBudget,
                            reference: (MLXArray, MLXArray?, MLXArray?, NemotronZSkipSite, Int, Int) -> MLXArray
    ) -> NemotronZSkipEntry? {
        guard NemotronZSkipBudget.enabled else { return nil }
        let env = ProcessInfo.processInfo.environment
        var allowed: [NemotronZSkipSite: Set<Int>] = [:]
        for site in NemotronZSkipSite.allCases {
            guard env["NEMOTRON_ZSKIP_" + site.rawValue.uppercased()] != "0" else { continue }
            for rows in 1...8 where kernel.supports(site: site, rows: rows, stock: stock) {
                allowed[site, default: []].insert(rows)
            }
        }
        guard !allowed.isEmpty else { return nil }
        // Signed affine coefficients are valid. The provider retains the stock
        // scale*zero + zero*bias expression even when weight codes are skipped.
        // Read original values only; never normalize or replace them.
        guard stock.scales.dtype == .bfloat16, stock.biases.dtype == .bfloat16,
              MLX.isFinite(stock.scales).all().item(Bool.self),
              MLX.isFinite(stock.biases).all().item(Bool.self) else {
            NemotronZSkipBudget.log("FAIL layer=\(name) affine_domain; stock")
            return nil
        }
        NemotronZSkipBudget.log("DOMAIN layer=\(name) finite_affine=true signed_scales_preserved=true")
        let bytes = kernel.derivedByteCount(stock)
        guard budget.reserve(bytes) else {
            NemotronZSkipBudget.log("BLOCKED layer=\(name) requested=\(bytes) resident=\(budget.resident) copies=\(budget.reserved) margin=\(budget.margin) recommended=\(budget.recommended); stock")
            return nil
        }
        let derived = kernel.build(stock)
        let inverse = kernel.inverse(derived, stock: stock)
        let originals = [stock.weight, stock.scales, stock.biases]
        var ok = derived.byteCount == bytes && inverse.count == originals.count
        if ok {
            for (a, b) in zip(inverse, originals) {
                guard a.shape == b.shape, a.dtype == b.dtype else { ok = false; break }
                let dtype: DType = a.dtype == .uint32 ? .uint32 : .uint16
                if !(a.view(dtype: dtype) .== b.view(dtype: dtype)).all().item(Bool.self) { ok = false; break }
            }
        }
        guard ok else {
            budget.release(bytes)
            NemotronZSkipBudget.log("FAIL layer=\(name) inverse/bytes; stock")
            return nil
        }
        let entry = NemotronZSkipEntry(derived: derived, stock: stock, kernel: kernel, allowed: allowed)
        var tests = 0
        // Fixed, public, synthetic generator. Never reads a request or prompt.
        // Non-finite and signed-zero outputs compare raw bits, including NaNs.
        for site in NemotronZSkipSite.allCases {
            for rows in (allowed[site] ?? []).sorted() {
                let topK = stock.experts == 1 ? 1 : 6
                for pattern in 0..<12 {
                    var state: UInt64 = 0x6a09e667f3bcc909 ^ UInt64(rows * 101 + pattern)
                    func next() -> UInt32 {
                        state = state &* 6364136223846793005 &+ 1442695040888963407
                        return UInt32(truncatingIfNeeded: state >> 32)
                    }
                    let count = rows * topK * stock.k
                    var values = [Float](repeating: 0, count: count)
                    for i in 0..<count {
                        let r = next()
                        let v = Float((r >> 8) % 1024 + 1) / 1024
                        switch pattern {
                        case 0: values[i] = r % 100 < 80 ? 0 : v * v
                        case 1: values[i] = r % 100 < 90 ? 0 : v * v
                        case 2: values[i] = r % 100 < 97 ? 0 : v * v
                        case 3: values[i] = 0
                        case 4: values[i] = v
                        case 5: values[i] = Float(bitPattern: r % 2 == 0 ? 0x00010000 : 0x80010000)
                        case 6: values[i] = r % 100 < 90 ? 0 : 1e30
                        case 7: values[i] = i / (topK * stock.k) == 0 ? 0 : v * v
                        case 8: values[i] = r % 2 == 0 ? -Float.zero : Float.zero
                        case 9: values[i] = r % 97 == 0 ? Float.infinity : (r % 89 == 0 ? -Float.infinity : v)
                        case 10: values[i] = r % 101 == 0 ? Float.nan : (r % 100 < 90 ? 0 : v)
                        default: values[i] = i % stock.k == 0 ? 1 : 0
                        }
                    }
                    let x = MLXArray(values).asType(.bfloat16).reshaped(rows * topK, stock.k)
                    // Distinct slots within a token, repeated experts across
                    // tokens; offset varies so layers see varied bank entries.
                    let ids = MLXArray((0..<(rows * topK)).map {
                        UInt32((($0 % topK) + pattern * 11) % stock.experts)
                    })
                    let scores = MLXArray((0..<(rows * topK)).map { Float(($0 % topK) + 1) / 21 })
                    let expected = reference(x, ids, scores, site, rows, topK)
                    let result = kernel.down(derived, stock: stock, x: x, indices: ids, scores: scores,
                        site: site, rows: rows, topK: topK, original: { expected })
                    guard result.shape == expected.shape, result.dtype == expected.dtype,
                          (result.view(dtype: .uint16) .== expected.view(dtype: .uint16)).all().item(Bool.self)
                    else {
                        budget.release(bytes)
                        NemotronZSkipBudget.log("FAIL layer=\(name) site=\(site.rawValue) rows=\(rows) pattern=\(pattern); stock")
                        return nil
                    }
                    tests += 1
                }
            }
        }
        NemotronZSkipBudget.log("PASS layer=\(name) kernel=\(kernel.version) layout=\(derived.layout) bytes=\(bytes) tests=\(tests)")
        return entry
    }
}

public struct NemotronZSkipQualifiedKernel: NemotronZSkipKernel {
    private let provider: any NemotronZSkipKernel
    private let selected: [NemotronZSkipSite: Set<Int>]
    private let receiptVersion: String
    public init(provider: any NemotronZSkipKernel,
                selected: [NemotronZSkipSite: Set<Int>], receiptVersion: String) {
        self.provider = provider
        self.selected = selected
        self.receiptVersion = receiptVersion
    }
    public var version: String { provider.version + "-qualified-" + receiptVersion }
    public func derivedByteCount(_ stock: NemotronZSkipStock) -> Int {
        provider.derivedByteCount(stock)
    }
    public func build(_ stock: NemotronZSkipStock) -> NemotronZSkipDerived {
        provider.build(stock)
    }
    public func inverse(_ derived: NemotronZSkipDerived,
                        stock: NemotronZSkipStock) -> [MLXArray] {
        provider.inverse(derived, stock: stock)
    }
    public func supports(site: NemotronZSkipSite, rows: Int,
                         stock: NemotronZSkipStock) -> Bool {
        selected[site]?.contains(rows) == true
            && provider.supports(site: site, rows: rows, stock: stock)
    }
    public func down(_ derived: NemotronZSkipDerived, stock: NemotronZSkipStock,
                     x: MLXArray, indices: MLXArray?, scores: MLXArray?,
                     site: NemotronZSkipSite, rows: Int, topK: Int,
                     original: () -> MLXArray) -> MLXArray {
        // Entry.down already checks the qualified immutable site/row table.
        provider.down(derived, stock: stock, x: x, indices: indices, scores: scores,
                      site: site, rows: rows, topK: topK, original: original)
    }
}
