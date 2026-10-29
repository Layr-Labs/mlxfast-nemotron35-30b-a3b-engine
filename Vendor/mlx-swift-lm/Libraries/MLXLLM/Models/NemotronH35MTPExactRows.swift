import Foundation
import MLX
import MLXLMCommon
import MLXNN

enum NemotronMTPExecution {
    private static func enabledByDefault(_ key: String) -> Bool {
        ProcessInfo.processInfo.environment[key] != "0"
    }
    static let exactMultirowHead = enabledByDefault("DARKBLOOM_NEMOTRON35_MTP_EXACT_MULTIROW_HEAD")
    static let batchedRouter = enabledByDefault("DARKBLOOM_NEMOTRON35_MTP_BATCHED_ROUTER")
    static let compiledMoE = enabledByDefault("DARKBLOOM_NEMOTRON35_MTP_COMPILED_MOE")
    static let batchedM1 = enabledByDefault("DARKBLOOM_NEMOTRON35_MTP_BATCHED_M1")
    static let batchedNorm = enabledByDefault("DARKBLOOM_NEMOTRON35_MTP_BATCHED_NORM")
    static let windowSSM = enabledByDefault("DARKBLOOM_NEMOTRON35_MTP_WINDOW_SSM")
}

/// `prepared`: x's K=2688 activation from the final norm_f join (else
/// prepared here from x).
func nemotronMTPHeadRows(_ x: MLXArray, _ linear: Linear,
                         prepared: MLXArray? = nil) -> MLXArray {
    guard NemotronMTPExecution.exactMultirowHead, x.ndim == 3,
          x.dim(0) == 1, (2...8).contains(x.dim(1)), x.dim(2) == 2688,
          let q = linear as? QuantizedLinear, q.weight.dim(0) == 131072,
          q.groupSize == 64, q.bits == 4, q.mode == .affine,
          q.scales.dtype == x.dtype, q.biases?.dtype == x.dtype
    else { return nemotronMTPLinearRows(x, linear) }
    if let rows = NemotronMultiRowQMV.apply(x.reshaped(x.dim(1), x.dim(2)), q,
                                            prepared: prepared) {
        var result = rows.reshaped(1, x.dim(1), q.weight.dim(0))
        if let bias = q.bias { result = result + bias }
        return result
    }
    let count = x.dim(1), width = x.dim(2), tile = 1024
    let blocks = q.weight.dim(0) / tile
    var result = quantizedMM(x.reshaped(1, count, 1, width),
        q.weight.reshaped(blocks, 1, tile, q.weight.dim(1)),
        scales: q.scales.reshaped(blocks, 1, tile, q.scales.dim(1)),
        biases: q.biases!.reshaped(blocks, 1, tile, q.biases!.dim(1)),
        transpose: true, groupSize: q.groupSize, bits: q.bits, mode: q.mode)
        .transposed(1, 0, 2, 3).reshaped(1, count, q.weight.dim(0))
    if let bias = q.bias { result = result + bias }
    return result
}

/// Norm reductions are row-local. Keep this independently gated while checking
/// bitwise native-dtype equivalence over every captured verification prefix.
func nemotronMTPNormRows(_ x: MLXArray, _ operation: (MLXArray) -> MLXArray) -> MLXArray {
    NemotronMTPExecution.batchedNorm ? operation(x) : nemotronMTPMapRows(x, operation)
}

/// Build adjacent M=1 operations inside one layer-major verification graph.
/// Stock matrix kernels may change accumulation when M changes, so preserve
/// the ordinary decode shape while sharing the surrounding graph/fence.
func nemotronMTPMapRows(_ x: MLXArray, _ operation: (MLXArray) -> MLXArray) -> MLXArray {
    guard x.dim(1) > 1 else { return operation(x) }
    return concatenated((0..<x.dim(1)).map { position in
        operation(x[0..., position..<(position + 1), 0...])
    }, axis: 1)
}

// Multi-row affine 4-bit qmv. Each row repeats the stock M=1 qmv / qmv_fast
// arithmetic (same lane split, block order, qdot and simd_sum), so every row
// is bitwise equal to an ordinary decode step. The weight tile is loaded once
// for all rows instead of once per row.
private let nemotronMultiRowQMVHeader = """
template <typename T, typename U, int values_per_thread>
inline U nmr_load_vector(const device T* x, thread U* x_thread) {
  U sum = 0;
  for (int i = 0; i < values_per_thread; i += 4) {
    sum += U(x[i]) + U(x[i + 1]) + U(x[i + 2]) + U(x[i + 3]);
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 16.0f;
    x_thread[i + 2] = x[i + 2] / 256.0f;
    x_thread[i + 3] = x[i + 3] / 4096.0f;
  }
  return sum;
}

template <typename T, typename U, int values_per_thread>
inline U nmr_load_vector_safe(const device T* x, thread U* x_thread, int N) {
  U sum = 0;
  for (int i = 0; i < N; i += 4) {
    sum += U(x[i]) + U(x[i + 1]) + U(x[i + 2]) + U(x[i + 3]);
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 16.0f;
    x_thread[i + 2] = x[i + 2] / 256.0f;
    x_thread[i + 3] = x[i + 3] / 4096.0f;
  }
  for (int i = N; i < values_per_thread; i++) {
    x_thread[i] = 0;
  }
  return sum;
}

template <typename U, int values_per_thread>
inline U nmr_qdot(
    const device uint8_t* w, const thread U* x_thread, U scale, U bias, U sum) {
  U accum = 0;
  const device uint16_t* ws = (const device uint16_t*)w;
  for (int i = 0; i < (values_per_thread / 4); i++) {
    accum +=
        (x_thread[4 * i] * (ws[i] & 0x000f) +
         x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
         x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
         x_thread[4 * i + 3] * (ws[i] & 0xf000));
  }
  return scale * accum + sum * bias;
}

template <typename U, int values_per_thread>
inline U nmr_qdot_safe(
    const device uint8_t* w, const thread U* x_thread, U scale, U bias, U sum,
    int N) {
  U accum = 0;
  const device uint16_t* ws = (const device uint16_t*)w;
  for (int i = 0; i < (N / 4); i++) {
    accum +=
        (x_thread[4 * i] * (ws[i] & 0x000f) +
         x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
         x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
         x_thread[4 * i + 3] * (ws[i] & 0xf000));
  }
  return scale * accum + sum * bias;
}
"""

// Exact half-pipe nibble decode (the C and D forms of 965e684b change 5,
// extended). HF picks how the four masked nibbles of words[i] reach float:
// 0 int -> float (stock), 1 ushort -> half, 2 n0/n1 from one 32-bit AND/OR
// into half2 magic numbers 1024 + v minus 1024 and n2/n3 through
// ushort -> half, 3 all four through half2 magic numbers (n2, n3 then scaled
// by 2^8, 2^12 in half). Every masked value has <= 4 significant bits and is
// <= 61440, so each form gives the same float operand; the FP32 products,
// their order and the affine close are untouched. Lane i % 2 of a half2 is
// the i-th uint16 of the 32-bit word, as in the D form.
let nemotronHalfDecodeHeader = """
template <int HF>
inline float4 nmr_half_decode4(const device uint16_t* words, int i) {
  const ushort q = words[i];
  if (HF == 0) {
    return float4(float(q & 0x000f), float(q & 0x00f0), float(q & 0x0f00), float(q & 0xf000));
  }
  if (HF == 1) {
    return float4(float(half(ushort(q & 0x000f))), float(half(ushort(q & 0x00f0))),
                  float(half(ushort(q & 0x0f00))), float(half(ushort(q & 0xf000))));
  }
  const uint w2 = ((const device uint*)words)[i / 2];
  const half2 h0 = as_type<half2>((w2 & 0x000f000fu) | 0x64006400u) - half2(1024.0h);
  const half2 h1 = as_type<half2>((w2 & 0x00f000f0u) | 0x64006400u) - half2(1024.0h);
  if (HF == 2) {
    return float4(float(h0[i % 2]), float(h1[i % 2]),
                  float(half(ushort(q & 0x0f00))), float(half(ushort(q & 0xf000))));
  }
  const half2 h2 = (as_type<half2>(((w2 >> 8) & 0x000f000fu) | 0x64006400u)
      - half2(1024.0h)) * half2(256.0h);
  const half2 h3 = (as_type<half2>(((w2 >> 12) & 0x000f000fu) | 0x64006400u)
      - half2(1024.0h)) * half2(4096.0h);
  return float4(float(h0[i % 2]), float(h1[i % 2]), float(h2[i % 2]), float(h3[i % 2]));
}
"""

/// Apply exact text swaps (old, new, occurrences) to a kernel source; each
/// `old` must occur exactly `occurrences` times, so a drifted source fails
/// at first use instead of silently keeping the old decode.
func nemotronSwapped(_ source: String, _ swaps: [(String, String, Int)]) -> String {
    var s = source
    for (old, new, occurrences) in swaps {
        precondition(s.components(separatedBy: old).count - 1 == occurrences,
                     "nemotron half decode source: \(old)")
        s = s.replacingOccurrences(of: old, with: new)
    }
    return s
}

// MARK: Unroll by two (NEMOTRON_U2_*)
//
// Three more decode-window kernels get the plan-free triple FC1's exact
// transform (NEMOTRON_MTP_FC1_UNROLL): `#pragma clang loop unroll_count(2)`
// ahead of their main K loops. The unrolled loop runs the same iteration
// bodies in the same order (iteration i, then i + 1, any odd last iteration
// as before, then the unchanged tail after the loop), so every load, decode,
// product, term order, accumulation, affine close, closing simd_sum and store
// is the record kernel's; only the instruction stream of two consecutive
// iterations is laid out together. Custom kernels compile in MLX's safe math
// mode, so nothing may be reassociated across the two copies.
// - INPROJ: the verify's Mamba in_proj + conv/dt launch (R = 3).
// - HEAD: the verify's 131,072-row target head (R = 3 prepared, ld2/NSG form).
// - DRAFT: the draft head's one-row routed FC1 leader (its pair dictionary).
// Each family's twin is admitted at load only after it matched its record
// kernel bit for bit on the real weights (`NemotronU2.qualify`). Only the
// verify and draft launch sites select it; the stepper's prompt prefill and
// the request prefill never reach them. The same transform measured slower
// on the M4 for the out_proj / o_proj (FAST hoist form, +9%), shared-up,
// fused Q/K/V, FC2 + shared-down co-launch, router and one-row multi-row QMV
// launches, which keep their record kernels.

/// `source` with the pragma ahead of each of exactly `count` occurrences of
/// `loop`, or nil when the record text moved (the record kernel stays).
func nemotronU2Source(_ source: String?, loop: String, count: Int) -> String? {
    guard let source, count > 0, source.components(separatedBy: loop).count == count + 1 else { return nil }
    return source.replacingOccurrences(of: loop, with: "#pragma clang loop unroll_count(2)\n" + loop)
}

enum NemotronU2 {
    /// The prepared QMV main loop (crown, HF, load forms, dictionary).
    static let qmvLoop = "for (; k < k_end; k += block_size) {\n"
    /// The routed FC1 leader's main loops (one-row draft tail).
    static let leaderLoop = "for (; k < K - block_size; k += block_size) {\n"

    enum Family: String, CaseIterable {
        case inProj = "INPROJ", head = "HEAD", draft = "DRAFT"
    }
    /// NEMOTRON_U2_<FAMILY>=0 keeps that family's record kernels.
    static func setting(_ family: Family) -> Bool {
        ProcessInfo.processInfo.environment["NEMOTRON_U2_" + family.rawValue] != "0"
    }

    /// Set once at load by `qualify`, before any verify or draft graph is traced.
    nonisolated(unsafe) static var inProj = false
    nonisolated(unsafe) static var head = false
    nonisolated(unsafe) static var draft = false

    /// The R = 3 prepared launch of an [K, N] projection takes its twin: the
    /// 131,072-row target head once HEAD is admitted.
    static func prepared(rows: Int, n: Int) -> Bool {
        rows == 3 && n == 131072 && head
    }

    static func report(_ family: Family, _ message: String) {
        FileHandle.standardError.write(Data("nemotron u2 \(family.rawValue.lowercased()): \(message)\n".utf8))
    }

    static func equal(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
        guard a.shape == b.shape, a.dtype == b.dtype else { return MLXArray(false) }
        switch a.dtype {
        case .float32, .uint32, .int32: return (a.view(dtype: .uint32) .== b.view(dtype: .uint32)).all()
        default: return (a.view(dtype: .uint16) .== b.view(dtype: .uint16)).all()
        }
    }

    /// Admit one family: `check` runs the family's launches with the record
    /// kernels and with the twins (the same template arguments the window
    /// uses, so it also JITs every twin pipeline outside any timed window)
    /// and returns (all equal, cases), or nil when nothing could run. Any
    /// miss, a missing twin or an error keeps the record kernels.
    static func admit(_ family: Family, _ check: () throws -> (MLXArray, Int)?) -> Bool {
        guard setting(family) else {
            report(family, "off (setting)")
            return false
        }
        var result: (Bool, Int)? = nil
        do {
            result = try withError { () throws -> (Bool, Int)? in
                guard let r = try check(), r.1 > 0 else { return nil }
                return (r.0.item(Bool.self), r.1)
            }
        } catch {
            report(family, "off (\(error))")
            return false
        }
        guard let result else {
            report(family, "off (no twin or no launch ran)")
            return false
        }
        guard result.0 else {
            report(family, "off (\(result.1) cases, a twin differs from its record kernel)")
            return false
        }
        report(family, "on (\(result.1) cases bitwise equal to the record kernels)")
        return true
    }
}

/// The extra template argument of a decode-form twin, or none (crown kernel).
func nemotronHalfTemplate(_ on: Bool, _ form: Int) -> [(String, any KernelTemplateArg)] {
    on ? [("HF", form)] : []
}

/// Decode swaps for the multi-row qmv body and the screen's select rescore
/// (both decode `const U a = words[i] & 0x000f;` ... twice: blocks + tail).
let nemotronMultiRowHalfSwaps: [(String, String, Int)] = [
    ("const U a = words[i] & 0x000f;",
     "const float4 hq = nmr_half_decode4<HF>(words, i);\n        const U a = hq.x;", 2),
    ("const U b1 = words[i] & 0x00f0;", "const U b1 = hq.y;", 2),
    ("const U c = words[i] & 0x0f00;", "const U c = hq.z;", 2),
    ("const U d = words[i] & 0xf000;", "const U d = hq.w;", 2),
]

/// Decode swaps for the prepared-activation QMV body (crown: C form when
/// !FAST and R = 3, else int -> float; here the template HF decides).
let nemotronPreparedHalfSwaps: [(String, String, Int)] = [
    ("constexpr bool half_decode = !FAST && R == 3;",
     "// Decode form: template HF (nmr_half_decode4).", 1),
    ("const U a = half_decode ? U(half(ushort(words[i] & 0x000f))) : U(words[i] & 0x000f);",
     "const float4 hq = nmr_half_decode4<HF>(words, i);\n        const U a = hq.x;", 2),
    ("const U b1 = half_decode ? U(half(ushort(words[i] & 0x00f0))) : U(words[i] & 0x00f0);",
     "const U b1 = hq.y;", 2),
    ("const U c = half_decode ? U(half(ushort(words[i] & 0x0f00))) : U(words[i] & 0x0f00);",
     "const U c = hq.z;", 2),
    ("const U d = half_decode ? U(half(ushort(words[i] & 0xf000))) : U(words[i] & 0xf000);",
     "const U d = hq.w;", 2),
]

// m3 load forms of the prepared-activation QMV main loop (the !FAST tail keeps
// its own crown loads). Only where and how wide the packed words, scales and
// biases are read changes: every decode line, FP32 expression, its order and
// the affine close are the same text. Words are read as one 32-bit word per
// pack (FAST: one 64-bit pair) and split into the same two little-endian
// uint16 halves the 16-bit loads returned; offsets are pack (4 B / 8 B)
// aligned (row K / 2 B with K % 64 == 0, lane packs_per_thread * 4 B).
let nemotronPreparedLoadHeader = """
template <int PPT>
inline void nmr_load_words(const device uint8_t* wl, thread ushort* wv) {
  if (PPT == 2) {
    const uint2 v = *((const device uint2*)wl);
    wv[0] = ushort(v.x);
    wv[1] = ushort(v.x >> 16);
    wv[2] = ushort(v.y);
    wv[3] = ushort(v.y >> 16);
  } else {
    const uint v = *((const device uint*)wl);
    wv[0] = ushort(v);
    wv[1] = ushort(v >> 16);
  }
}
"""

/// nmr_half_decode4 over thread-held words (load forms with an HF decode):
/// the same body; the 32-bit pair is rebuilt from words i & ~1 and i | 1,
/// which is the device version's ((const device uint*)words)[i / 2].
let nemotronHalfDecodeThreadHeader = """
template <int HF>
inline float4 nmr_half_decode4(const thread ushort* words, int i) {
  const ushort q = words[i];
  if (HF == 0) {
    return float4(float(q & 0x000f), float(q & 0x00f0), float(q & 0x0f00), float(q & 0xf000));
  }
  if (HF == 1) {
    return float4(float(half(ushort(q & 0x000f))), float(half(ushort(q & 0x00f0))),
                  float(half(ushort(q & 0x0f00))), float(half(ushort(q & 0xf000))));
  }
  const uint w2 = uint(words[i & ~1]) | (uint(words[i | 1]) << 16);
  const half2 h0 = as_type<half2>((w2 & 0x000f000fu) | 0x64006400u) - half2(1024.0h);
  const half2 h1 = as_type<half2>((w2 & 0x00f000f0u) | 0x64006400u) - half2(1024.0h);
  if (HF == 2) {
    return float4(float(h0[i % 2]), float(h1[i % 2]),
                  float(half(ushort(q & 0x0f00))), float(half(ushort(q & 0xf000))));
  }
  const half2 h2 = (as_type<half2>(((w2 >> 8) & 0x000f000fu) | 0x64006400u)
      - half2(1024.0h)) * half2(256.0h);
  const half2 h3 = (as_type<half2>(((w2 >> 12) & 0x000f000fu) | 0x64006400u)
      - half2(1024.0h)) * half2(4096.0h);
  return float4(float(h0[i % 2]), float(h1[i % 2]), float(h2[i % 2]), float(h3[i % 2]));
}
"""

/// V1 W32: the main loop's per-row words come from one 32-bit (FAST 64-bit) load.
let nemotronPreparedW32Swaps: [(String, String, Int)] = [
    ("accum[r] = 0;\n      const device uint16_t* words = (const device uint16_t*)wl;\n",
     "accum[r] = 0;\n      thread ushort wv1[values_per_thread / 4];\n      nmr_load_words<packs_per_thread>(wl, wv1);\n      const thread ushort* words = wv1;\n", 1),
]

/// V2 HOIST: W32 loads, scales and biases of all RPS rows first, then the
/// prepared activations, then the unchanged per-row math.
let nemotronPreparedHoistSwaps: [(String, String, Int)] = [
    ("    for (int r = 0; r < R; r++) {\n      const int base = (r * tiles + k / block_size) * (values_per_thread + 1) * 32 + simd_lid;\n      sum[r] = prepared[base + values_per_thread * 32];\n      for (int v = 0; v < values_per_thread; v++) x_thread[r][v] = prepared[base + v * 32];\n    }\n    for (int row = 0; row < results_per_simdgroup; row++) {\n      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);\n      const device T* sl = sc + row * in_vec_size_g;\n      const device T* bl = bs + row * in_vec_size_g;\n      U s = sl[0];\n      U b = bl[0];\n",
     "    thread ushort wv[results_per_simdgroup][values_per_thread / 4];\n    thread U sv[results_per_simdgroup];\n    thread U bv[results_per_simdgroup];\n    for (int row = 0; row < results_per_simdgroup; row++) {\n      nmr_load_words<packs_per_thread>(ws + row * in_vec_size_w, wv[row]);\n      sv[row] = sc[row * in_vec_size_g];\n      bv[row] = bs[row * in_vec_size_g];\n    }\n    for (int r = 0; r < R; r++) {\n      const int base = (r * tiles + k / block_size) * (values_per_thread + 1) * 32 + simd_lid;\n      sum[r] = prepared[base + values_per_thread * 32];\n      for (int v = 0; v < values_per_thread; v++) x_thread[r][v] = prepared[base + v * 32];\n    }\n    for (int row = 0; row < results_per_simdgroup; row++) {\n      U s = sv[row];\n      U b = bv[row];\n", 1),
    ("accum[r] = 0;\n      const device uint16_t* words = (const device uint16_t*)wl;\n",
     "accum[r] = 0;\n      const thread ushort* words = wv[row];\n", 1),
]

/// V3 PF: HOIST plus a second register set holding the next main-loop
/// block's words, scales and biases, loaded before this block's math only
/// while k + block_size < k_end (never the tail block or past a row).
let nemotronPreparedPrefetchSwaps: [(String, String, Int)] = [
    ("  const int k_end = FAST ? in_vec_size : in_vec_size - block_size;\n  for (; k < k_end; k += block_size) {\n",
     "  const int k_end = FAST ? in_vec_size : in_vec_size - block_size;\n  thread ushort wv[results_per_simdgroup][values_per_thread / 4];\n  thread U sv[results_per_simdgroup];\n  thread U bv[results_per_simdgroup];\n  if (k < k_end) {\n    for (int row = 0; row < results_per_simdgroup; row++) {\n      nmr_load_words<packs_per_thread>(ws + row * in_vec_size_w, wv[row]);\n      sv[row] = sc[row * in_vec_size_g];\n      bv[row] = bs[row * in_vec_size_g];\n    }\n  }\n  for (; k < k_end; k += block_size) {\n    thread ushort wn[results_per_simdgroup][values_per_thread / 4];\n    thread U sn[results_per_simdgroup];\n    thread U bn[results_per_simdgroup];\n    const bool more = k + block_size < k_end;\n    if (more) {\n      for (int row = 0; row < results_per_simdgroup; row++) {\n        nmr_load_words<packs_per_thread>(ws + block_size * bytes_per_pack / pack_factor + row * in_vec_size_w, wn[row]);\n        sn[row] = sc[block_size / group_size + row * in_vec_size_g];\n        bn[row] = bs[block_size / group_size + row * in_vec_size_g];\n      }\n    }\n", 1),
    ("    for (int r = 0; r < R; r++) {\n      const int base = (r * tiles + k / block_size) * (values_per_thread + 1) * 32 + simd_lid;\n      sum[r] = prepared[base + values_per_thread * 32];\n      for (int v = 0; v < values_per_thread; v++) x_thread[r][v] = prepared[base + v * 32];\n    }\n    for (int row = 0; row < results_per_simdgroup; row++) {\n      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);\n      const device T* sl = sc + row * in_vec_size_g;\n      const device T* bl = bs + row * in_vec_size_g;\n      U s = sl[0];\n      U b = bl[0];\n",
     "    for (int r = 0; r < R; r++) {\n      const int base = (r * tiles + k / block_size) * (values_per_thread + 1) * 32 + simd_lid;\n      sum[r] = prepared[base + values_per_thread * 32];\n      for (int v = 0; v < values_per_thread; v++) x_thread[r][v] = prepared[base + v * 32];\n    }\n    for (int row = 0; row < results_per_simdgroup; row++) {\n      U s = sv[row];\n      U b = bv[row];\n", 1),
    ("accum[r] = 0;\n      const device uint16_t* words = (const device uint16_t*)wl;\n",
     "accum[r] = 0;\n      const thread ushort* words = wv[row];\n", 1),
    ("    ws += block_size * bytes_per_pack / pack_factor;\n    sc += block_size / group_size;\n    bs += block_size / group_size;\n\n  }",
     "    if (more) {\n      for (int row = 0; row < results_per_simdgroup; row++) {\n        for (int i = 0; i < values_per_thread / 4; i++) wv[row][i] = wn[row][i];\n        sv[row] = sn[row];\n        bv[row] = bn[row];\n      }\n    }\n    ws += block_size * bytes_per_pack / pack_factor;\n    sc += block_size / group_size;\n    bs += block_size / group_size;\n\n  }", 1),
]

/// V4 GEO: SIMD groups per threadgroup as template NSG (row -> SIMD-group map only).
let nemotronPreparedNSGSwaps: [(String, String, Int)] = [
    ("constexpr int num_simdgroups = 2;", "constexpr int num_simdgroups = NSG;", 1),
]

/// Prepared-QMV body for a decode form (crown / m1 HF), a load form and the
/// SIMD-group template, through the counted swaps above.
func nemotronPreparedVariantSource(_ base: String, load: Int, nsg: Bool) -> String {
    let loads: [Int: [(String, String, Int)]] = [
        1: nemotronPreparedW32Swaps, 2: nemotronPreparedHoistSwaps, 3: nemotronPreparedPrefetchSwaps]
    var s = base
    if load != 0 { s = nemotronSwapped(s, loads[load]!) }
    if nsg { s = nemotronSwapped(s, nemotronPreparedNSGSwaps) }
    return s
}

private let nemotronMultiRowQMVCrownSource = """
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = RPS;
  constexpr int packs_per_thread = FAST ? 2 : 1;
  constexpr int pack_factor = 8;
  constexpr int bytes_per_pack = 4;
  constexpr int group_size = 64;
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * 32;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  typedef float U;

  const int in_vec_size = K;
  const int out_vec_size = N;
  const uint3 tid = threadgroup_position_in_grid;
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const uint simd_lid = thread_index_in_simdgroup;

  const device uint8_t* ws = (const device uint8_t*)w;
  thread U x_thread[R][values_per_thread];
  thread U result[R][results_per_simdgroup];
  for (int r = 0; r < R; r++) {
    for (int row = 0; row < results_per_simdgroup; row++) {
      result[r][row] = 0;
    }
  }

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  if (out_row >= out_vec_size) {
    return;
  }

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  const device T* sc = scales + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  const device T* bs = biases + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  const device T* xp = x + simd_lid * values_per_thread;
  device T* yp = y + out_row;

  U sum[R];
  int k = 0;
  const int k_end = FAST ? in_vec_size : in_vec_size - block_size;
  for (; k < k_end; k += block_size) {
    for (int r = 0; r < R; r++) {
      sum[r] = nmr_load_vector<T, U, values_per_thread>(xp + r * in_vec_size, x_thread[r]);
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T* sl = sc + row * in_vec_size_g;
      const device T* bl = bs + row * in_vec_size_g;
      U s = sl[0];
      U b = bl[0];
      // Decode packed integer coefficients once for all verification rows.
      // Each row keeps its original i-order and affine accumulation.
      thread U accum[R];
      for (int r = 0; r < R; r++) accum[r] = 0;
      const device uint16_t* words = (const device uint16_t*)wl;
      for (int i = 0; i < values_per_thread / 4; i++) {
        const U a = words[i] & 0x000f;
        const U b1 = words[i] & 0x00f0;
        const U c = words[i] & 0x0f00;
        const U d = words[i] & 0xf000;
        for (int r = 0; r < R; r++) {
          accum[r] += x_thread[r][4*i] * a + x_thread[r][4*i+1] * b1
              + x_thread[r][4*i+2] * c + x_thread[r][4*i+3] * d;
        }
      }
      for (int r = 0; r < R; r++) result[r][row] += s * accum[r] + b * sum[r];
    }
    ws += block_size * bytes_per_pack / pack_factor;
    sc += block_size / group_size;
    bs += block_size / group_size;
    xp += block_size;
  }
  if (!FAST) {
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
        0,
        values_per_thread);
    if (remaining > 0) {
      for (int r = 0; r < R; r++) {
        sum[r] = nmr_load_vector_safe<T, U, values_per_thread>(
            xp + r * in_vec_size, x_thread[r], remaining);
      }
      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = sc + row * in_vec_size_g;
        const device T* bl = bs + row * in_vec_size_g;
        U s = sl[0];
        U b = bl[0];
        // Decode packed integer coefficients once for all verification rows.
        // Each row keeps its original i-order and affine accumulation.
        thread U accum[R];
        for (int r = 0; r < R; r++) accum[r] = 0;
        const device uint16_t* words = (const device uint16_t*)wl;
        for (int i = 0; i < remaining / 4; i++) {
          const U a = words[i] & 0x000f;
          const U b1 = words[i] & 0x00f0;
          const U c = words[i] & 0x0f00;
          const U d = words[i] & 0xf000;
          for (int r = 0; r < R; r++) {
            accum[r] += x_thread[r][4*i] * a + x_thread[r][4*i+1] * b1
                + x_thread[r][4*i+2] * c + x_thread[r][4*i+3] * d;
          }
        }
        for (int r = 0; r < R; r++) result[r][row] += s * accum[r] + b * sum[r];
      }
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    for (int r = 0; r < R; r++) {
      U v = simd_sum(result[r][row]);
      if (simd_lid == 0) {
        T h = static_cast<T>(v);
        if (RELU2) {
          T zero = static_cast<T>(0);
          T m = metal::isnan(h) ? h : (h > zero ? h : zero);
          h = m * m;
        }
        yp[r * out_vec_size + row] = h;
      }
    }
  }
"""

let nemotronMultiRowQMVTail8 =
    ProcessInfo.processInfo.environment["NEMOTRON_MTP_MULTIROW_TAIL8"] != "0"
private let nemotronMultiRowQMVSource = nemotronMultiRowQMVTail8
    ? nemotronSwapped(nemotronMultiRowQMVCrownSource, [
        ("nmr_load_vector_safe<T, U, values_per_thread>(\n            xp + r * in_vec_size, x_thread[r], remaining);",
         "nmr_load_vector<T, U, values_per_thread>(\n            xp + r * in_vec_size, x_thread[r]);", 1),
        ("for (int i = 0; i < remaining / 4; i++) {",
         "for (int i = 0; i < values_per_thread / 4; i++) {", 1),
    ])
    : nemotronMultiRowQMVCrownSource


private let nemotronHeadPrepareSource = """
  const int chunk = thread_position_in_grid.x;
  const int row = thread_position_in_grid.y;
  if (chunk >= K / PV) return;
  float values[PV];
  const float total = nmr_load_vector<T, float, PV>(x + row * K + chunk * PV, values);
  const int tiles = (K / PV + 31) / 32;
  const int base = (row * tiles + chunk / 32) * (PV + 1) * 32 + chunk % 32;
  for (int i = 0; i < PV; i++) prepared[base + i * 32] = values[i];
  prepared[base + PV * 32] = total;
"""

private let nemotronHeadPreparedQMVCrownSource = """
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = RPS;
  constexpr int packs_per_thread = FAST ? 2 : 1;
  constexpr int pack_factor = 8;
  constexpr int bytes_per_pack = 4;
  constexpr int group_size = 64;
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * 32;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  typedef float U;
  // R = 3 without FAST (K % 512 != 0): each masked nibble reaches its float
  // through an exact ushort -> half convert (<= 4 significant bits, <= 61440)
  // widened as an operand. Same float value, same FP32 arithmetic and order.
  // M4: 1.9-2.2% faster at K 2688/3712; neutral or slower with FAST, kept stock.
  constexpr bool half_decode = !FAST && R == 3;

  const int in_vec_size = K;
  const int out_vec_size = N;
  const uint3 tid = threadgroup_position_in_grid;
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const uint simd_lid = thread_index_in_simdgroup;

  const device uint8_t* ws = (const device uint8_t*)w;
  thread U x_thread[R][values_per_thread];
  thread U result[R][results_per_simdgroup];
  for (int r = 0; r < R; r++) {
    for (int row = 0; row < results_per_simdgroup; row++) {
      result[r][row] = 0;
    }
  }

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  if (out_row >= out_vec_size) {
    return;
  }

  ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  const device T* sc = scales + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  const device T* bs = biases + out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  const int tiles = (in_vec_size / values_per_thread + 31) / 32;
  device T* yp = y + out_row;

  U sum[R];
  int k = 0;
  const int k_end = FAST ? in_vec_size : in_vec_size - block_size;
  for (; k < k_end; k += block_size) {
    for (int r = 0; r < R; r++) {
      const int base = (r * tiles + k / block_size) * (values_per_thread + 1) * 32 + simd_lid;
      sum[r] = prepared[base + values_per_thread * 32];
      for (int v = 0; v < values_per_thread; v++) x_thread[r][v] = prepared[base + v * 32];
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
      const device T* sl = sc + row * in_vec_size_g;
      const device T* bl = bs + row * in_vec_size_g;
      U s = sl[0];
      U b = bl[0];
      // Decode packed integer coefficients once for all verification rows.
      // Each row keeps its original i-order and affine accumulation.
      thread U accum[R];
      for (int r = 0; r < R; r++) accum[r] = 0;
      const device uint16_t* words = (const device uint16_t*)wl;
      for (int i = 0; i < values_per_thread / 4; i++) {
        const U a = half_decode ? U(half(ushort(words[i] & 0x000f))) : U(words[i] & 0x000f);
        const U b1 = half_decode ? U(half(ushort(words[i] & 0x00f0))) : U(words[i] & 0x00f0);
        const U c = half_decode ? U(half(ushort(words[i] & 0x0f00))) : U(words[i] & 0x0f00);
        const U d = half_decode ? U(half(ushort(words[i] & 0xf000))) : U(words[i] & 0xf000);
        for (int r = 0; r < R; r++) {
          accum[r] += x_thread[r][4*i] * a + x_thread[r][4*i+1] * b1
              + x_thread[r][4*i+2] * c + x_thread[r][4*i+3] * d;
        }
      }
      for (int r = 0; r < R; r++) result[r][row] += s * accum[r] + b * sum[r];
    }
    ws += block_size * bytes_per_pack / pack_factor;
    sc += block_size / group_size;
    bs += block_size / group_size;

  }
  if (!FAST) {
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
        0,
        values_per_thread);
    if (remaining > 0) {
      for (int r = 0; r < R; r++) {
        const int base = (r * tiles + k / block_size) * (values_per_thread + 1) * 32 + simd_lid;
        sum[r] = prepared[base + values_per_thread * 32];
        for (int v = 0; v < values_per_thread; v++) x_thread[r][v] = v < remaining ? prepared[base + v * 32] : 0;
      }
      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = sc + row * in_vec_size_g;
        const device T* bl = bs + row * in_vec_size_g;
        U s = sl[0];
        U b = bl[0];
        // Decode packed integer coefficients once for all verification rows.
        // Each row keeps its original i-order and affine accumulation.
        thread U accum[R];
        for (int r = 0; r < R; r++) accum[r] = 0;
        const device uint16_t* words = (const device uint16_t*)wl;
        for (int i = 0; i < remaining / 4; i++) {
          const U a = half_decode ? U(half(ushort(words[i] & 0x000f))) : U(words[i] & 0x000f);
          const U b1 = half_decode ? U(half(ushort(words[i] & 0x00f0))) : U(words[i] & 0x00f0);
          const U c = half_decode ? U(half(ushort(words[i] & 0x0f00))) : U(words[i] & 0x0f00);
          const U d = half_decode ? U(half(ushort(words[i] & 0xf000))) : U(words[i] & 0xf000);
          for (int r = 0; r < R; r++) {
            accum[r] += x_thread[r][4*i] * a + x_thread[r][4*i+1] * b1
                + x_thread[r][4*i+2] * c + x_thread[r][4*i+3] * d;
          }
        }
        for (int r = 0; r < R; r++) result[r][row] += s * accum[r] + b * sum[r];
      }
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    for (int r = 0; r < R; r++) {
      U v = simd_sum(result[r][row]);
      if (simd_lid == 0) {
        T h = static_cast<T>(v);
        if (RELU2) {
          T zero = static_cast<T>(0);
          T m = metal::isnan(h) ? h : (h > zero ? h : zero);
          h = m * m;
        }
        yp[r * out_vec_size + row] = h;
      }
    }
  }
"""

// NEMOTRON_MTP_QMV_TAIL8: the prepared QMV's K tail (FAST false) in its
// constant 8-value form. Every caller has K % 64 == 0 and k advances by a
// multiple of 256, so a lane's `remaining` is 0 or 8: the record tail's
// `v < remaining ? x : 0` loads all 8 values and its `i < remaining / 4` loop
// runs i = 0, 1. Both become their constant forms under the same
// `if (remaining > 0)`, so the tail is the same block of arithmetic. The
// runtime-length form indexed x_thread dynamically, which kept the whole
// array out of registers for the entire kernel. Applied to the body every
// prepared kernel is built from (crown and HF twins, load forms, the Mamba
// in_proj conv/dt kernel, the fused Q/K/V launches); decode lines untouched.
let nemotronHeadPreparedQMVTail8 =
    ProcessInfo.processInfo.environment["NEMOTRON_MTP_QMV_TAIL8"] != "0"
let nemotronHeadPreparedQMVTail8Swaps: [(String, String, Int)] = [
    ("x_thread[r][v] = v < remaining ? prepared[base + v * 32] : 0;",
     "x_thread[r][v] = prepared[base + v * 32];", 1),
    ("for (int i = 0; i < remaining / 4; i++) {",
     "for (int i = 0; i < values_per_thread / 4; i++) {", 1),
]
private let nemotronHeadPreparedQMVSource = nemotronHeadPreparedQMVTail8
    ? nemotronSwapped(nemotronHeadPreparedQMVCrownSource, nemotronHeadPreparedQMVTail8Swaps)
    : nemotronHeadPreparedQMVCrownSource

/// Mamba verify in_proj (R = 3 prepared QMV) whose epilogue also produces the
/// window's conv + SiLU, padded history and timestep transform. After the
/// stock store, lane 0 of each simdgroup reads back its own just-stored BF16
/// outputs and, for the xBC channels, runs the `nemotron_packed_conv_silu`
/// expressions per position (padded rows too); for the dt channels it runs the
/// softplus and clip expressions of `nemotron_packed_conv_silu_dt`. Every
/// channel is produced once, by the simdgroup that computed it; the separate
/// conv/timestep launch is not needed. The projection itself is unchanged.
private let nemotronMambaInProjConvDtEpilogueText = """

  // Lane-parallel: lane l < RPS * R owns (row l / R, position l % R). The
  // BF16 outputs it needs are read back from this simdgroup's own stores.
  simdgroup_barrier(mem_flags::mem_device);
  {
    const int l = int(simd_lid);
    if (l < results_per_simdgroup * R) {
      const int row = l / R;
      const int t = l % R;
      const int o = out_row + row;
      if (o >= CS && o < CS + CC) {
        const int64_t channel = int64_t(o - CS);
        NEP_LOADS
        if (t < 3) {
          padded[int64_t(t) * CC + channel] =
              oldState[int64_t(t) * oldState_strides[1] + channel * oldState_strides[2]];
        }
        padded[(int64_t(t) + 3) * CC + channel] = yp[t * out_vec_size + row];
        float acc = 0;
        for (int i = 0; i < 4; ++i) {
          int64_t crow = int64_t(t) + int64_t(i);
          T value = crow < 3
              ? oldState[crow * oldState_strides[1] + channel * oldState_strides[2]]
              : yp[(crow - 3) * out_vec_size + row];
          acc += static_cast<float>(value)
              * cw[channel * cw_strides[0] + int64_t(i) * cw_strides[1]];
        }
        T convolved = static_cast<T>(acc);
        T xc = convolved + cb[channel * cb_strides[0]];
        auto yc = 1 / (1 + metal::exp(metal::abs(xc)));
        T sigmoid = xc < 0 ? yc : 1 - yc;
        conv[int64_t(t) * CC + channel] = xc * sigmoid;
      } else if (o >= DS && o < DS + NH) {
        const int head = o - DS;
        float v = static_cast<float>(yp[t * out_vec_size + row])
            + static_cast<float>(dtBias[int64_t(head) * dtBias_strides[0]]);
        float sp;
        if (metal::isnan(v)) {
          sp = metal::numeric_limits<float>::quiet_NaN();
        } else {
          constexpr float inf = metal::numeric_limits<float>::infinity();
          float maxval = metal::max(v, 0.0f);
          float minval = metal::min(v, 0.0f);
          sp = (minval == -inf || maxval == inf)
              ? maxval : (maxval + log1p(metal::exp(minval - maxval)));
        }
        float lo = dtLo;
        float hi = dtHi;
        float m = metal::isnan(sp) ? sp : (sp > lo ? sp : lo);
        dt[t * NH + head] = metal::isnan(m) ? m : (m < hi ? m : hi);
      }
    }
  }
"""

/// NEMOTRON_MTP_INPROJ_EPILOGUE_LOADS_FIRST=0 keeps the record epilogue's order, where a conv lane loads the old
/// state, the conv weights and bias, and its window outputs after its first `padded` store. Otherwise the lane
/// loads all of them into registers first (`nep_*`: the three old-state rows, the window's R outputs of its row,
/// the four weights and the bias, the same addresses) and the statements below read the registers; every
/// product, sum, rounding and store is the record's.
let nemotronInProjEpilogueLoadsFirst =
    ProcessInfo.processInfo.environment["NEMOTRON_MTP_INPROJ_EPILOGUE_LOADS_FIRST"] != "0"
private let nemotronMambaInProjConvDtEpilogue: String = {
    let s = nemotronMambaInProjConvDtEpilogueText
    guard nemotronInProjEpilogueLoadsFirst else { return s.replacingOccurrences(of: "        NEP_LOADS\n", with: "") }
    let loads = """
            const T nep_o0 = oldState[0 * oldState_strides[1] + channel * oldState_strides[2]];
            const T nep_o1 = oldState[1 * oldState_strides[1] + channel * oldState_strides[2]];
            const T nep_o2 = oldState[2 * oldState_strides[1] + channel * oldState_strides[2]];
            T nep_y[R];
            _Pragma("clang loop unroll(full)")
            for (int k = 0; k < R; ++k) {
              nep_y[k] = yp[k * out_vec_size + row];
            }
            const auto nep_w0 = cw[channel * cw_strides[0] + int64_t(0) * cw_strides[1]];
            const auto nep_w1 = cw[channel * cw_strides[0] + int64_t(1) * cw_strides[1]];
            const auto nep_w2 = cw[channel * cw_strides[0] + int64_t(2) * cw_strides[1]];
            const auto nep_w3 = cw[channel * cw_strides[0] + int64_t(3) * cw_strides[1]];
            const auto nep_b = cb[channel * cb_strides[0]];

    """
    let swaps: [(String, String)] = [
        ("        NEP_LOADS\n", loads),
        ("""
                  padded[int64_t(t) * CC + channel] =
                      oldState[int64_t(t) * oldState_strides[1] + channel * oldState_strides[2]];
        """, """
                  padded[int64_t(t) * CC + channel] = t == 0 ? nep_o0 : (t == 1 ? nep_o1 : nep_o2);
        """),
        ("padded[(int64_t(t) + 3) * CC + channel] = yp[t * out_vec_size + row];",
         "T nep_yt = nep_y[0];\n        _Pragma(\"clang loop unroll(full)\")\n"
            + "        for (int k = 1; k < R; ++k) { nep_yt = t == k ? nep_y[k] : nep_yt; }\n"
            + "        padded[(int64_t(t) + 3) * CC + channel] = nep_yt;"),
        ("""
                for (int i = 0; i < 4; ++i) {
                  int64_t crow = int64_t(t) + int64_t(i);
                  T value = crow < 3
                      ? oldState[crow * oldState_strides[1] + channel * oldState_strides[2]]
                      : yp[(crow - 3) * out_vec_size + row];
                  acc += static_cast<float>(value)
                      * cw[channel * cw_strides[0] + int64_t(i) * cw_strides[1]];
                }
        """, """
                _Pragma("clang loop unroll(full)")
                for (int i = 0; i < 4; ++i) {
                  int64_t crow = int64_t(t) + int64_t(i);
                  T value = nep_y[0];
                  _Pragma("clang loop unroll(full)")
                  for (int k = 1; k < R; ++k) { value = crow - 3 == int64_t(k) ? nep_y[k] : value; }
                  value = crow == 0 ? nep_o0 : (crow == 1 ? nep_o1 : (crow == 2 ? nep_o2 : value));
                  acc += static_cast<float>(value)
                      * (i == 0 ? nep_w0 : (i == 1 ? nep_w1 : (i == 2 ? nep_w2 : nep_w3)));
                }
        """),
        ("T xc = convolved + cb[channel * cb_strides[0]];", "T xc = convolved + nep_b;"),
    ]
    var out = s
    for (from, to) in swaps {
        guard out.components(separatedBy: from).count == 2 else {
            return s.replacingOccurrences(of: "        NEP_LOADS\n", with: "")
        }
        out = out.replacingOccurrences(of: from, with: to)
    }
    return out
}()

/// The loads-first epilogue in 32-bit offsets (`NemotronMambaInProjConvDt`, NEMOTRON_MTP_INPROJ_EPILOGUE32): each
/// element offset the record forms in 64 bits, with the input's runtime strides, is formed in 32 bits with the strides
/// of a row-contiguous input as constants (old state [1, 3, CC]: t * CC + channel; conv weight [CC, 4, 1]:
/// 4 * channel + i; conv bias and dt bias: their index), and the window position arithmetic is in 32 bits. The
/// launches built from it read row-contiguous inputs only (`ensureRowContiguous`), so every element is read and
/// stored at the record's address; each load, value, expression and store is otherwise the record's text. nil when
/// the record text moved (the loads-first form off).
private let nemotronMambaInProjConvDtEpilogue32: String? = {
    var s = nemotronMambaInProjConvDtEpilogue
    let swaps: [(String, String)] = [
        ("""
                const int64_t channel = int64_t(o - CS);
                const T nep_o0 = oldState[0 * oldState_strides[1] + channel * oldState_strides[2]];
                const T nep_o1 = oldState[1 * oldState_strides[1] + channel * oldState_strides[2]];
                const T nep_o2 = oldState[2 * oldState_strides[1] + channel * oldState_strides[2]];

        """, """
                const int channel = o - CS;
                const T nep_o0 = oldState[channel];
                const T nep_o1 = oldState[CC + channel];
                const T nep_o2 = oldState[2 * CC + channel];

        """),
        ("""
                const auto nep_w0 = cw[channel * cw_strides[0] + int64_t(0) * cw_strides[1]];
                const auto nep_w1 = cw[channel * cw_strides[0] + int64_t(1) * cw_strides[1]];
                const auto nep_w2 = cw[channel * cw_strides[0] + int64_t(2) * cw_strides[1]];
                const auto nep_w3 = cw[channel * cw_strides[0] + int64_t(3) * cw_strides[1]];
                const auto nep_b = cb[channel * cb_strides[0]];

        """, """
                const auto nep_w0 = cw[4 * channel];
                const auto nep_w1 = cw[4 * channel + 1];
                const auto nep_w2 = cw[4 * channel + 2];
                const auto nep_w3 = cw[4 * channel + 3];
                const auto nep_b = cb[channel];

        """),
        ("padded[int64_t(t) * CC + channel] = t == 0 ? nep_o0 : (t == 1 ? nep_o1 : nep_o2);",
         "padded[t * CC + channel] = t == 0 ? nep_o0 : (t == 1 ? nep_o1 : nep_o2);"),
        ("padded[(int64_t(t) + 3) * CC + channel] = nep_yt;", "padded[(t + 3) * CC + channel] = nep_yt;"),
        ("int64_t crow = int64_t(t) + int64_t(i);", "int crow = t + i;"),
        ("value = crow - 3 == int64_t(k) ? nep_y[k] : value;", "value = crow - 3 == k ? nep_y[k] : value;"),
        ("conv[int64_t(t) * CC + channel] = xc * sigmoid;", "conv[t * CC + channel] = xc * sigmoid;"),
        ("+ static_cast<float>(dtBias[int64_t(head) * dtBias_strides[0]]);", "+ static_cast<float>(dtBias[head]);"),
    ]
    for (old, new) in swaps {
        guard s.components(separatedBy: old).count == 2 else { return nil }
        s = s.replacingOccurrences(of: old, with: new)
    }
    guard !s.contains("int64_t"), !s.contains("_strides") else { return nil }
    return s
}()

/// NEMOTRON_MTP_INPROJ_EPILOGUE_REGS: the in_proj launch whose epilogue takes
/// its row's BF16 outputs from registers instead of reading them back from its
/// own stores. `simd_sum` leaves each output's sum in every lane, so in the
/// store loop each lane also keeps the stored value of the outputs of the row
/// it serves in the epilogue (`nmc_y[r]`, statically indexed: the loops over
/// the SIMD group's rows and the window positions are unrolled, and the lane's
/// row is a compare). The epilogue picks position t's value with a select
/// chain over the window's positions (no dynamic register indexing), so it
/// reads no device output back and the device fence ahead of it goes. Every
/// stored and computed value is the record kernel's.
private let nemotronInProjStoreLoop = """
  for (int row = 0; row < results_per_simdgroup; row++) {
    for (int r = 0; r < R; r++) {
      U v = simd_sum(result[r][row]);
      if (simd_lid == 0) {
        T h = static_cast<T>(v);
        if (RELU2) {
          T zero = static_cast<T>(0);
          T m = metal::isnan(h) ? h : (h > zero ? h : zero);
          h = m * m;
        }
        yp[r * out_vec_size + row] = h;
      }
    }
  }
"""
private let nemotronInProjStoreLoopRegs = """
  const int nmc_row = int(simd_lid) / R;
  T nmc_y[R];
  #pragma clang loop unroll(full)
  for (int r = 0; r < R; r++) {
    nmc_y[r] = static_cast<T>(0);
  }
  #pragma clang loop unroll(full)
  for (int row = 0; row < results_per_simdgroup; row++) {
    #pragma clang loop unroll(full)
    for (int r = 0; r < R; r++) {
      U v = simd_sum(result[r][row]);
      if (simd_lid == 0) {
        T h = static_cast<T>(v);
        if (RELU2) {
          T zero = static_cast<T>(0);
          T m = metal::isnan(h) ? h : (h > zero ? h : zero);
          h = m * m;
        }
        yp[r * out_vec_size + row] = h;
      }
      if (nmc_row == row) {
        T hc = static_cast<T>(v);
        if (RELU2) {
          T zero = static_cast<T>(0);
          T m = metal::isnan(hc) ? hc : (hc > zero ? hc : zero);
          hc = m * m;
        }
        nmc_y[r] = hc;
      }
    }
  }
"""
private let nemotronMambaInProjConvDtRegsEpilogue = """

  // Lane-parallel: lane l < RPS * R owns (row l / R, position l % R). Its
  // row's BF16 outputs are in nmc_y, kept by the store loop above.
  {
    const int l = int(simd_lid);
    if (l < results_per_simdgroup * R) {
      const int row = l / R;
      const int t = l % R;
      const int o = out_row + row;
      T yt = nmc_y[0];
      #pragma clang loop unroll(full)
      for (int s = 1; s < R; ++s) {
        yt = t == s ? nmc_y[s] : yt;
      }
      if (o >= CS && o < CS + CC) {
        const int64_t channel = int64_t(o - CS);
        if (t < 3) {
          padded[int64_t(t) * CC + channel] =
              oldState[int64_t(t) * oldState_strides[1] + channel * oldState_strides[2]];
        }
        padded[(int64_t(t) + 3) * CC + channel] = yt;
        float acc = 0;
        for (int i = 0; i < 4; ++i) {
          int64_t crow = int64_t(t) + int64_t(i);
          T value = nmc_y[0];
          if (crow < 3) {
            value = oldState[crow * oldState_strides[1] + channel * oldState_strides[2]];
          } else {
            #pragma clang loop unroll(full)
            for (int s = 1; s < R; ++s) {
              value = crow - 3 == int64_t(s) ? nmc_y[s] : value;
            }
          }
          acc += static_cast<float>(value)
              * cw[channel * cw_strides[0] + int64_t(i) * cw_strides[1]];
        }
        T convolved = static_cast<T>(acc);
        T xc = convolved + cb[channel * cb_strides[0]];
        auto yc = 1 / (1 + metal::exp(metal::abs(xc)));
        T sigmoid = xc < 0 ? yc : 1 - yc;
        conv[int64_t(t) * CC + channel] = xc * sigmoid;
      } else if (o >= DS && o < DS + NH) {
        const int head = o - DS;
        float v = static_cast<float>(yt)
            + static_cast<float>(dtBias[int64_t(head) * dtBias_strides[0]]);
        float sp;
        if (metal::isnan(v)) {
          sp = metal::numeric_limits<float>::quiet_NaN();
        } else {
          constexpr float inf = metal::numeric_limits<float>::infinity();
          float maxval = metal::max(v, 0.0f);
          float minval = metal::min(v, 0.0f);
          sp = (minval == -inf || maxval == inf)
              ? maxval : (maxval + log1p(metal::exp(minval - maxval)));
        }
        float lo = dtLo;
        float hi = dtHi;
        float m = metal::isnan(sp) ? sp : (sp > lo ? sp : lo);
        dt[t * NH + head] = metal::isnan(m) ? m : (m < hi ? m : hi);
      }
    }
  }
"""
/// The record in_proj source (prepared body + epilogue) with the register store
/// loop and epilogue, or nil when the body's store loop text is not found.
private func nemotronInProjRegsSource() -> String? {
    guard nemotronHeadPreparedQMVSource.components(separatedBy: nemotronInProjStoreLoop).count == 2
    else { return nil }
    return nemotronHeadPreparedQMVSource.replacingOccurrences(
        of: nemotronInProjStoreLoop, with: nemotronInProjStoreLoopRegs) + nemotronMambaInProjConvDtRegsEpilogue
}

enum NemotronMambaInProjConvDt {
    /// NEMOTRON_MTP_INPROJ_CONV=0 keeps the separate conv/timestep launch.
    static let enabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_INPROJ_CONV"] != "0"
    /// NEMOTRON_MTP_INPROJ_CONV_CHECK=1 also runs the separate launch on the
    /// same projection and poisons the outputs with NaN on any bit difference.
    static let check =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_INPROJ_CONV_CHECK"] == "1"

    private static let kernel = MLXFast.metalKernel(
        name: "nemotron_mamba_in_proj_conv_dt",
        inputNames: ["w", "scales", "biases", "prepared", "K", "N",
                     "oldState", "cw", "cb", "dtBias", "dtLo", "dtHi"],
        outputNames: ["y", "conv", "padded", "dt"],
        source: nemotronHeadPreparedQMVSource + nemotronMambaInProjConvDtEpilogue,
        header: nemotronMultiRowQMVHeader, ensureRowContiguous: false)
    /// The same launch reading (scale, bias) through `NemotronSBDict`.
    private static let dictKernel = nemotronSBDictSource(
        nemotronHeadPreparedQMVSource + nemotronMambaInProjConvDtEpilogue).map {
        MLXFast.metalKernel(
            name: "nemotron_mamba_in_proj_conv_dt_sbd",
            inputNames: ["w", "scales", "biases", "prepared", "K", "N",
                         "oldState", "cw", "cb", "dtBias", "dtLo", "dtHi"],
            outputNames: ["y", "conv", "padded", "dt"],
            source: $0, header: nemotronMultiRowQMVHeader, ensureRowContiguous: false)
    }
    static var dictAvailable: Bool { dictKernel != nil }
    /// The two launches with the projection's main K loop unrolled by two
    /// (`nemotronU2Source`, NEMOTRON_U2_INPROJ); the epilogue is unchanged.
    private static let kernelU2 = nemotronU2Source(
        nemotronHeadPreparedQMVSource + nemotronMambaInProjConvDtEpilogue,
        loop: NemotronU2.qmvLoop, count: 1).map {
        MLXFast.metalKernel(
            name: "nemotron_mamba_in_proj_conv_dt_u2",
            inputNames: ["w", "scales", "biases", "prepared", "K", "N",
                         "oldState", "cw", "cb", "dtBias", "dtLo", "dtHi"],
            outputNames: ["y", "conv", "padded", "dt"],
            source: $0, header: nemotronMultiRowQMVHeader, ensureRowContiguous: false)
    }
    private static let dictKernelU2 = nemotronU2Source(
        nemotronSBDictSource(nemotronHeadPreparedQMVSource + nemotronMambaInProjConvDtEpilogue),
        loop: NemotronU2.qmvLoop, count: 1).map {
        MLXFast.metalKernel(
            name: "nemotron_mamba_in_proj_conv_dt_sbd_u2",
            inputNames: ["w", "scales", "biases", "prepared", "K", "N",
                         "oldState", "cw", "cb", "dtBias", "dtLo", "dtHi"],
            outputNames: ["y", "conv", "padded", "dt"],
            source: $0, header: nemotronMultiRowQMVHeader, ensureRowContiguous: false)
    }
    static var u2Available: Bool { kernelU2 != nil && (dictKernel == nil || dictKernelU2 != nil) }

    /// NEMOTRON_MTP_INPROJ_EPILOGUE32=0 keeps the record epilogue. Otherwise each of the four launches above runs as a
    /// twin built from the same text with `nemotronMambaInProjConvDtEpilogue32` (32-bit offsets, the row-contiguous
    /// strides of the old state, conv weight, conv bias and dt bias as constants), launched with `ensureRowContiguous`
    /// (an input of any other layout is first copied to a row-contiguous array). The register-epilogue launches keep
    /// their text, and a requested register epilogue (NEMOTRON_MTP_INPROJ_EPILOGUE_REGS=1) keeps this off. Used only
    /// after `qualifyEpilogue32` found every twin bitwise equal to its record launch.
    static let epilogue32Requested =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_INPROJ_EPILOGUE32"] != "0"
    nonisolated(unsafe) private(set) static var epilogue32Verified = false
    private static func e32Kernel(_ name: String, _ source: String?) -> MLXFast.MLXFastKernel? {
        source.map {
            MLXFast.metalKernel(
                name: name,
                inputNames: ["w", "scales", "biases", "prepared", "K", "N",
                             "oldState", "cw", "cb", "dtBias", "dtLo", "dtHi"],
                outputNames: ["y", "conv", "padded", "dt"],
                source: $0, header: nemotronMultiRowQMVHeader, ensureRowContiguous: true)
        }
    }
    private static let source32 = nemotronMambaInProjConvDtEpilogue32.map { nemotronHeadPreparedQMVSource + $0 }
    private static let kernel32 = e32Kernel("nemotron_mamba_in_proj_conv_dt_e32", source32)
    private static let dictKernel32 = e32Kernel(
        "nemotron_mamba_in_proj_conv_dt_sbd_e32", source32.flatMap { nemotronSBDictSource($0) })
    private static let kernelU2_32 = e32Kernel(
        "nemotron_mamba_in_proj_conv_dt_u2_e32",
        nemotronU2Source(source32, loop: NemotronU2.qmvLoop, count: 1))
    private static let dictKernelU2_32 = e32Kernel(
        "nemotron_mamba_in_proj_conv_dt_sbd_u2_e32",
        nemotronU2Source(source32.flatMap { nemotronSBDictSource($0) }, loop: NemotronU2.qmvLoop, count: 1))
    private static var epilogue32Available: Bool {
        kernel32 != nil && dictKernel32 != nil && kernelU2_32 != nil && dictKernelU2_32 != nil
    }

    /// NEMOTRON_MTP_INPROJ_EPILOGUE_REGS=0 keeps the epilogue that reads the
    /// stored outputs back behind a device fence (see `nemotronInProjRegsSource`).
    static let regsRequested =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_INPROJ_EPILOGUE_REGS"] == "1"
    nonisolated(unsafe) private(set) static var regsVerified = false
    private static func regsKernel(_ name: String, _ source: String?) -> MLXFast.MLXFastKernel? {
        source.map {
            MLXFast.metalKernel(
                name: name,
                inputNames: ["w", "scales", "biases", "prepared", "K", "N",
                             "oldState", "cw", "cb", "dtBias", "dtLo", "dtHi"],
                outputNames: ["y", "conv", "padded", "dt"],
                source: $0, header: nemotronMultiRowQMVHeader, ensureRowContiguous: false)
        }
    }
    private static let kernelRegs = regsKernel("nemotron_mamba_in_proj_conv_dt_rg", nemotronInProjRegsSource())
    private static let dictKernelRegs = regsKernel(
        "nemotron_mamba_in_proj_conv_dt_sbd_rg", nemotronSBDictSource(nemotronInProjRegsSource() ?? ""))
    private static let kernelU2Regs = regsKernel(
        "nemotron_mamba_in_proj_conv_dt_u2_rg",
        nemotronU2Source(nemotronInProjRegsSource(), loop: NemotronU2.qmvLoop, count: 1))
    private static let dictKernelU2Regs = regsKernel(
        "nemotron_mamba_in_proj_conv_dt_sbd_u2_rg",
        nemotronU2Source(nemotronSBDictSource(nemotronInProjRegsSource() ?? ""), loop: NemotronU2.qmvLoop, count: 1))
    /// The register-epilogue twin of each record launch, all four built.
    private static var regsAvailable: Bool {
        kernelRegs != nil && dictKernelRegs != nil && kernelU2Regs != nil && dictKernelU2Regs != nil
    }
    /// `NemotronQMVTune`: the record launch's source before the dictionary and unroll transforms (prepared body +
    /// epilogue, its 32-bit epilogue form, or its register-epilogue form).
    static func tuneBase(regs: Bool, e32: Bool) -> String? {
        regs ? nemotronInProjRegsSource()
            : e32 ? source32 : nemotronHeadPreparedQMVSource + nemotronMambaInProjConvDtEpilogue
    }

    /// x: [3, 2688] BF16 rows (or their prepared activation), the Mamba
    /// in_proj [z | xBC | dt]. Returns the projection [3, N] plus the window's
    /// convolution [1, 3, convDim], padded history [1, 6, convDim] and FP32
    /// timestep [1, 3, heads], or nil for any other shape.
    static func apply(
        _ x: MLXArray, _ q: QuantizedLinear, prepared: MLXArray?,
        oldState: MLXArray, convWeight: MLXArray, convBias: MLXArray,
        dtBias: MLXArray, timeStepLimit: (Float, Float),
        convStart: Int, convDim: Int, heads: Int, sbDict: Bool = true, u2: Bool? = nil, regs: Bool? = nil,
        epilogue32: Bool? = nil, qmvTune: NemotronQMVTune.Config? = nil
    ) -> (projected: MLXArray, convolution: MLXArray, padded: MLXArray, dt: MLXArray)? {
        let rows = x.dim(0), k = x.dim(1), n = q.weight.dim(0)
        guard enabled, NemotronMultiRowQMV.enabled, NemotronMultiRowQMV.narrowEnabled,
              x.ndim == 2, rows == 3, x.dtype == .bfloat16, k == 2688, k % 512 != 0,
              q.mode == .affine, q.bits == 4, q.groupSize == 64, q.bias == nil,
              q.weight.dtype == .uint32, q.weight.ndim == 2, q.weight.dim(1) * 8 == k,
              q.scales.dtype == .bfloat16, let biases = q.biases, biases.dtype == .bfloat16,
              n == convStart + convDim + heads, n % 4 == 0,
              Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              oldState.shape == [1, 3, convDim], oldState.dtype == .bfloat16,
              convWeight.shape == [convDim, 4, 1], convWeight.dtype == .bfloat16,
              convBias.shape == [convDim], convBias.dtype == .bfloat16,
              dtBias.shape == [heads], [DType.bfloat16, .float32].contains(dtBias.dtype)
        else { return nil }
        let pv = 8
        let packed = NemotronHeadPreparedActivation.compatible(prepared, rows: rows, k: k)
            ?? NemotronHeadPreparedActivation.prepareContiguousRows(x, pv: pv)
        let args = NemotronMultiRowQMV.scalarPair(k, n)
        let dict = sbDict && NemotronSBDict.enabled && dictKernel != nil ? NemotronSBDict.dict(q) : nil
        let unrolled = (u2 ?? NemotronU2.inProj) ? (dict != nil ? dictKernelU2 : kernelU2) : nil
        var chosen = unrolled ?? (dict != nil ? dictKernel! : kernel)
        let registerEpilogue = regs ?? regsVerified
        if !registerEpilogue, epilogue32 ?? epilogue32Verified {
            // The 32-bit epilogue twin of the same launch (same unroll, same dictionary form).
            let twin = unrolled != nil ? (dict != nil ? dictKernelU2_32 : kernelU2_32)
                : (dict != nil ? dictKernel32 : kernel32)
            if let twin { chosen = twin }
        }
        if registerEpilogue {
            // The register-epilogue twin of the same launch (same unroll, same dictionary form).
            let twin = unrolled != nil ? (dict != nil ? dictKernelU2Regs : kernelU2Regs)
                : (dict != nil ? dictKernelRegs : kernelRegs)
            if let twin { chosen = twin }
        }
        // NemotronQMVTune: the adopted geometry of the dictionary launch (`qmvTune`: a load-time check's).
        let tuned = dict != nil
            ? NemotronQMVTune.inProjLaunch(qmvTune, production: u2 == nil && regs == nil && epilogue32 == nil) : nil
        if let tuned { chosen = tuned.0 }
        let launchRPS = tuned?.2 ?? 2, launchNSG = tuned?.3 ?? 2
        let outputs = chosen(
            [q.weight, dict?.index ?? q.scales, dict?.table ?? biases, packed, args.0, args.1,
             oldState, convWeight, convBias, dtBias,
             MLXArray(timeStepLimit.0), MLXArray(timeStepLimit.1)],
            template: [("T", DType.bfloat16), ("R", rows), ("FAST", false), ("RELU2", false),
                       ("RPS", launchRPS), ("CS", convStart), ("CC", convDim),
                       ("DS", convStart + convDim), ("NH", heads)] + (tuned?.1 ?? []),
            grid: (32, n / launchRPS, 1), threadGroup: (32, launchNSG, 1),
            outputShapes: [[rows, n], [1, rows, convDim], [1, rows + 3, convDim], [1, rows, heads]],
            outputDTypes: [.bfloat16, .bfloat16, .bfloat16, .float32])
        // Self-test: the projection against the multi-row route on the same
        // rows (itself checked against the non-prepared multi-row kernel).
        if NemotronMoELeader.selfTest,
           let reference = NemotronMultiRowQMV.apply(x, q, prepared: packed) {
            return (nemotronBitwiseOrNaN(outputs[0], reference), outputs[1], outputs[2], outputs[3])
        }
        return (outputs[0], outputs[1], outputs[2], outputs[3])
    }

    /// NemotronU2 (INPROJ): the unrolled launch against the record launch on
    /// every Mamba layer's real in_proj, conv and timestep parameters (the
    /// window's dictionary or plain form), two seeded row sets each, all four
    /// outputs bit for bit.
    static func u2Check(_ model: Module) -> (MLXArray, Int)? {
        guard u2Available else { return nil }
        var same = MLXArray(true)
        var cases = 0
        let keys = MLXRandom.split(key: MLXRandom.key(0x52a1), into: 2)
        for module in model.modules() {
            guard let mixer = module as? NemotronHMamba2Mixer, let q = mixer.inProj as? QuantizedLinear,
                  let convBias = mixer.conv1d.bias else { continue }
            for key in keys {
                let x = (MLXRandom.normal([3, 2688], key: key) * 0.5).asType(.bfloat16)
                let state = (MLXRandom.normal([1, 3, mixer.convDim], key: key) * 0.5).asType(.bfloat16)
                func run(_ on: Bool) -> [MLXArray]? {
                    apply(x, q, prepared: nil, oldState: state, convWeight: mixer.conv1d.weight,
                          convBias: convBias, dtBias: mixer.dtBias, timeStepLimit: mixer.timeStepLimit,
                          convStart: mixer.intermediateSize, convDim: mixer.convDim, heads: mixer.numHeads,
                          u2: on).map { [$0.projected, $0.convolution, $0.padded, $0.dt] }
                }
                guard let a = run(false), let b = run(true) else { continue }
                for (u, v) in zip(a, b) { same = logicalAnd(same, NemotronU2.equal(u, v)) }
                eval(same)
                cases += 1
            }
        }
        return cases > 0 ? (same, cases) : nil
    }

    /// Load time, after `NemotronU2.qualify` and before `qualifyRegs` and `NemotronQMVTune`: the 32-bit epilogue twin
    /// against the record launch (the unroll and dictionary form the verify takes) on every Mamba layer's real in_proj,
    /// conv and timestep parameters, from the rows and from their prepared activation, with three old-state layouts
    /// each (a dense [1, 3, CC] state, rows 3 ... 5 of a [1, 6, CC] history as the window passes it, and a transposed
    /// [1, CC, 3] array that reaches the twin through MLX's row-contiguous copy), all four outputs bit for bit.
    /// Engages only if every case matches.
    static func qualifyEpilogue32(_ model: Module) {
        epilogue32Verified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron mtp in_proj epilogue32: \(message)\n".utf8))
        }
        guard epilogue32Requested else { return report("off (setting)") }
        guard !regsRequested else { return report("off (register epilogue requested)") }
        guard enabled, epilogue32Available else { return report("off (source not found)") }
        var same = MLXArray(true)
        var cases = 0
        do {
            try withError {
                for module in model.modules() {
                    guard let mixer = module as? NemotronHMamba2Mixer, let q = mixer.inProj as? QuantizedLinear,
                          let convBias = mixer.conv1d.bias else { continue }
                    let cc = mixer.convDim
                    for seed in 0 ..< 2 {
                        let keys = MLXRandom.split(key: MLXRandom.key(UInt64(0x52e1 + 2 * cases + seed)), into: 4)
                        let x = (MLXRandom.normal([3, 2688], key: keys[0]) * Float(seed == 0 ? 0.5 : 4))
                            .asType(.bfloat16)
                        let prepared = seed == 1 ? NemotronHeadPreparedActivation.pack(x) : nil
                        let states = [
                            (MLXRandom.normal([1, 3, cc], key: keys[1]) * 0.5).asType(.bfloat16),
                            (MLXRandom.normal([1, 6, cc], key: keys[2]) * 2).asType(.bfloat16)[0..., 3 ..< 6, 0...],
                            (MLXRandom.normal([1, cc, 3], key: keys[3]) * 0.5).asType(.bfloat16).transposed(0, 2, 1),
                        ]
                        for state in states {
                            func run(_ on: Bool) -> [MLXArray]? {
                                apply(x, q, prepared: prepared, oldState: state, convWeight: mixer.conv1d.weight,
                                      convBias: convBias, dtBias: mixer.dtBias, timeStepLimit: mixer.timeStepLimit,
                                      convStart: mixer.intermediateSize, convDim: cc, heads: mixer.numHeads,
                                      epilogue32: on)
                                    .map { [$0.projected, $0.convolution, $0.padded, $0.dt] }
                            }
                            guard let a = run(false), let b = run(true) else { same = MLXArray(false); return }
                            for (u, v) in zip(a, b) {
                                same = logicalAnd(same, u.dtype == .float32
                                    ? (u.view(dtype: .uint32) .== v.view(dtype: .uint32)).all()
                                    : (u.view(dtype: .uint16) .== v.view(dtype: .uint16)).all())
                            }
                        }
                    }
                    cases += 1
                    eval(same)
                }
            }
        } catch {
            return report("off (\(error))")
        }
        epilogue32Verified = cases > 0 && same.item(Bool.self)
        report(epilogue32Verified
            ? "on (\(cases) layers x 2 row sets x 3 state layouts: projection, conv, padded history and timestep"
                + " bitwise equal)"
            : "off (differs from the record epilogue)")
    }

    /// Load time, after `NemotronU2.qualify`: the register-epilogue launch
    /// against the launch it replaces (the same unroll and dictionary form as
    /// the verify takes) on every Mamba layer's real in_proj, conv and timestep
    /// parameters, from the rows and from their prepared activation, two seeded
    /// row sets and conv states each, all four outputs bit for bit. Engages only
    /// if every case matches.
    static func qualifyRegs(_ model: Module) {
        regsVerified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron mtp in_proj epilogue regs: \(message)\n".utf8))
        }
        guard regsRequested else { return report("off (setting)") }
        guard enabled, regsAvailable else { return report("off (source not found)") }
        var same = MLXArray(true)
        var cases = 0
        do {
            try withError {
                for module in model.modules() {
                    guard let mixer = module as? NemotronHMamba2Mixer, let q = mixer.inProj as? QuantizedLinear,
                          let convBias = mixer.conv1d.bias else { continue }
                    for seed in 0 ..< 2 {
                        let keys = MLXRandom.split(key: MLXRandom.key(UInt64(0x52c1 + 2 * cases + seed)), into: 2)
                        let x = (MLXRandom.normal([3, 2688], key: keys[0]) * Float(seed == 0 ? 0.5 : 4))
                            .asType(.bfloat16)
                        let state = (MLXRandom.normal([1, 3, mixer.convDim], key: keys[1]) * 0.5)
                            .asType(.bfloat16)
                        let prepared = seed == 1 ? NemotronHeadPreparedActivation.pack(x) : nil
                        func run(_ on: Bool) -> [MLXArray]? {
                            apply(x, q, prepared: prepared, oldState: state, convWeight: mixer.conv1d.weight,
                                  convBias: convBias, dtBias: mixer.dtBias, timeStepLimit: mixer.timeStepLimit,
                                  convStart: mixer.intermediateSize, convDim: mixer.convDim,
                                  heads: mixer.numHeads, regs: on)
                                .map { [$0.projected, $0.convolution, $0.padded, $0.dt] }
                        }
                        guard let a = run(false), let b = run(true) else { same = MLXArray(false); return }
                        for (u, v) in zip(a, b) {
                            same = logicalAnd(same, u.dtype == .float32
                                ? (u.view(dtype: .uint32) .== v.view(dtype: .uint32)).all()
                                : (u.view(dtype: .uint16) .== v.view(dtype: .uint16)).all())
                        }
                    }
                    cases += 1
                    eval(same)
                }
            }
        } catch {
            return report("off (\(error))")
        }
        regsVerified = cases > 0 && same.item(Bool.self)
        report(regsVerified
            ? "on (\(cases) layers x 2 row sets: projection, conv, padded history and timestep bitwise equal)"
            : "off (differs from the record epilogue)")
    }
}

enum NemotronHeadPreparedActivation {
    static let prepare = MLXFast.metalKernel(
        name: "nemotron_head_prepare_activation", inputNames: ["x"],
        outputNames: ["prepared"], source: nemotronHeadPrepareSource,
        header: nemotronMultiRowQMVHeader)
    static let projectRecord = MLXFast.metalKernel(
        name: "nemotron_head_prepared_qmv", inputNames: ["w", "scales", "biases", "prepared", "K", "N"],
        outputNames: ["y"], source: nemotronHeadPreparedQMVSource,
        header: nemotronMultiRowQMVHeader)
    /// The crown launch with the vector store (`nemotronQMVVectorStoreSource`), used once
    /// `NemotronTargetQKV.qualifyVectorStore` matched it to the lane-0 launch.
    static let projectVS: MLXFast.MLXFastKernel? = nemotronQMVVectorStoreSource(nemotronHeadPreparedQMVSource).map {
        MLXFast.metalKernel(
            name: "nemotron_head_prepared_qmv_vs", inputNames: ["w", "scales", "biases", "prepared", "K", "N"],
            outputNames: ["y"], source: $0, header: nemotronMultiRowQMVHeader)
    }
    nonisolated(unsafe) static var projectVSVerified = false
    static var project: MLXFast.MLXFastKernel {
        projectVSVerified ? (projectVS ?? projectRecord) : projectRecord
    }
    private static func rowSource() -> String {
        let address = "x + row * K + chunk * PV"
        precondition(nemotronHeadPrepareSource.components(separatedBy: address).count == 2)
        return "const device T* xr[3] = {x0, x1, x2};\n" +
            nemotronHeadPrepareSource.replacingOccurrences(
                of: address, with: "xr[row] + chunk * PV")
    }
    private static let prepareRows = MLXFast.metalKernel(
        name: "nemotron_prepare_activation_query_rows3",
        inputNames: ["x0", "x1", "x2"], outputNames: ["prepared"],
        source: rowSource(), header: nemotronMultiRowQMVHeader)

    /// Kill switches (code constants) for the R = 3 decode form, picked per
    /// instantiation by the synthetic-weight microbenchmark (bitwise equal
    /// outputs required). Values equal to the crown's (1 without FAST,
    /// 0 with FAST) keep the crown `project` kernel object and source.
    static let halfDecodeForm = 1
    static let fastHalfDecodeForm = 2
    static let preparedHalfSource = nemotronSwapped(
        nemotronHeadPreparedQMVSource, nemotronPreparedHalfSwaps)
    static let preparedHalfHeader = nemotronMultiRowQMVHeader + nemotronHalfDecodeHeader
    /// `NemotronQMVTune`: the crown prepared body (the HF body is `preparedHalfSource`).
    static var crownSource: String { nemotronHeadPreparedQMVSource }
    static let projectHalf = MLXFast.metalKernel(
        name: "nemotron_head_prepared_qmv_hf", inputNames: ["w", "scales", "biases", "prepared", "K", "N"],
        outputNames: ["y"], source: preparedHalfSource, header: preparedHalfHeader)
    /// The non-crown decode form for (R, FAST), or nil (crown kernel).
    static func decodeForm(rows: Int, fast: Bool) -> Int? {
        guard rows == 3 else { return nil }
        let form = fast ? fastHalfDecodeForm : halfDecodeForm
        return form == (fast ? 0 : 1) ? nil : form
    }
    private static func projectKernel(rows: Int, fast: Bool)
        -> (MLXFast.MLXFastKernel, [(String, any KernelTemplateArg)]) {
        guard let form = decodeForm(rows: rows, fast: fast) else { return (project, []) }
        return (projectHalf, [("HF", form)])
    }

    /// m3 kill switches (code constants) for the R = 3 prepared QMV: main-loop
    /// load form (0 crown, 1 W32, 2 W32 + hoist, 3 W32 + hoist + next-block
    /// prefetch) and the row -> SIMD-group geometry per [K, N]. Crown values
    /// (0, empty tables) keep the crown / m1 kernel objects, template lists,
    /// grid and threadgroup.
    static let loadForm = 0
    static let fastLoadForm = 0
    static let qkvLoadForm = 0
    static let loadFormOverrides: [[Int]: Int] = [[2688, 131072]: 2, [4096, 2688]: 2]
    static let preparedGeometry: [[Int]: (rps: Int, nsg: Int)] = [[2688, 131072]: (rps: 2, nsg: 8)]
    static func loadForm(k: Int, n: Int) -> Int {
        loadFormOverrides[[k, n]] ?? (k % 512 == 0 ? fastLoadForm : loadForm)
    }

    nonisolated(unsafe) private static var variants: [String: MLXFast.MLXFastKernel] = [:]
    private static let variantsLock = NSLock()
    /// Prepared QMV twin for (decode form twin, load form, NSG template).
    static func variantKernel(half: Bool, load: Int, nsg: Bool) -> MLXFast.MLXFastKernel {
        let name = "nemotron_head_prepared_qmv" + (half ? "_hf" : "") + (load != 0 ? "_ld\(load)" : "")
            + (nsg ? "_nsg" : "")
        variantsLock.lock(); defer { variantsLock.unlock() }
        if let kernel = variants[name] { return kernel }
        let kernel = MLXFast.metalKernel(
            name: name, inputNames: ["w", "scales", "biases", "prepared", "K", "N"],
            outputNames: ["y"],
            source: nemotronPreparedVariantSource(
                half ? preparedHalfSource : nemotronHeadPreparedQMVSource, load: load, nsg: nsg),
            header: variantHeader(half: half, load: load))
        variants[name] = kernel
        return kernel
    }
    nonisolated(unsafe) private static var dictVariants: [String: MLXFast.MLXFastKernel?] = [:]
    /// The prepared QMV twin reading (scale, bias) through `NemotronSBDict`
    /// (`nemotronSBDictSource` of the same body), or nil.
    static func dictKernel(half: Bool, load: Int, nsg: Bool) -> MLXFast.MLXFastKernel? {
        let name = "nemotron_head_prepared_qmv" + (half ? "_hf" : "") + (load != 0 ? "_ld\(load)" : "")
            + (nsg ? "_nsg" : "") + "_sbd"
        variantsLock.lock(); defer { variantsLock.unlock() }
        if let kernel = dictVariants[name] { return kernel }
        let kernel = nemotronSBDictSource(nemotronPreparedVariantSource(
            half ? preparedHalfSource : nemotronHeadPreparedQMVSource, load: load, nsg: nsg)).map {
            MLXFast.metalKernel(
                name: name, inputNames: ["w", "scales", "biases", "prepared", "K", "N"],
                outputNames: ["y"], source: $0, header: variantHeader(half: half, load: load))
        }
        dictVariants[name] = kernel
        return kernel
    }
    static func variantHeader(half: Bool, load: Int) -> String {
        nemotronMultiRowQMVHeader + (half ? nemotronHalfDecodeHeader : "")
            + (load != 0 ? nemotronPreparedLoadHeader + (half ? nemotronHalfDecodeThreadHeader : "") : "")
    }

    nonisolated(unsafe) private static var u2Variants: [String: MLXFast.MLXFastKernel?] = [:]
    /// The unroll-by-two twin (NEMOTRON_U2_HEAD) of the R = 3 prepared kernel
    /// for (decode form twin, load form, NSG template, dictionary): the same
    /// body with the pragma ahead of its main K loop, or nil. With load 0 and
    /// no NSG the body is the crown (`project`) or HF (`projectHalf`) source.
    static func u2Kernel(half: Bool, load: Int, nsg: Bool, dict: Bool) -> MLXFast.MLXFastKernel? {
        let name = "nemotron_head_prepared_qmv" + (half ? "_hf" : "") + (load != 0 ? "_ld\(load)" : "")
            + (nsg ? "_nsg" : "") + (dict ? "_sbd" : "") + "_u2"
        variantsLock.lock(); defer { variantsLock.unlock() }
        if let kernel = u2Variants[name] { return kernel }
        let body = nemotronPreparedVariantSource(
            half ? preparedHalfSource : nemotronHeadPreparedQMVSource, load: load, nsg: nsg)
        let kernel = nemotronU2Source(dict ? nemotronSBDictSource(body) : body,
                                      loop: NemotronU2.qmvLoop, count: 1).map {
            MLXFast.metalKernel(
                name: name, inputNames: ["w", "scales", "biases", "prepared", "K", "N"],
                outputNames: ["y"], source: $0, header: variantHeader(half: half, load: load))
        }
        u2Variants[name] = kernel
        return kernel
    }

    /// Kernel, extra template arguments, RPS and SIMD groups for one prepared
    /// launch of `rows` x [K, N] at the caller's RPS. `u2` (R = 3): the
    /// selected kernel's unroll-by-two twin, when built.
    private static func projectLaunch(rows: Int, k: Int, n: Int, rps: Int, u2: Bool = false)
        -> (MLXFast.MLXFastKernel, [(String, any KernelTemplateArg)], Int, Int) {
        let (kernel, hf) = projectKernel(rows: rows, fast: k % 512 == 0)
        guard rows == 3 else { return (kernel, hf, rps, 2) }
        let load = loadForm(k: k, n: n)
        let geometry = preparedGeometry[[k, n]] ?? (rps: rps, nsg: 2)
        let nsg = geometry.nsg != 2
        let extra: [(String, any KernelTemplateArg)] = nsg ? [("NSG", geometry.nsg)] : []
        if u2, let twin = u2Kernel(half: !hf.isEmpty, load: load, nsg: nsg, dict: false) {
            return (twin, hf + extra, geometry.rps, geometry.nsg)
        }
        guard load != 0 || geometry.nsg != 2 else { return (kernel, hf, geometry.rps, 2) }
        return (variantKernel(half: !hf.isEmpty, load: load, nsg: nsg), hf + extra, geometry.rps, geometry.nsg)
    }

    /// projectLaunch's R = 3 launch with its kernel's dictionary twin, or nil.
    private static func dictLaunch(k: Int, n: Int, rps: Int, u2: Bool = false)
        -> (MLXFast.MLXFastKernel, [(String, any KernelTemplateArg)], Int, Int)? {
        let (_, hf) = projectKernel(rows: 3, fast: k % 512 == 0)
        let load = loadForm(k: k, n: n)
        let geometry = preparedGeometry[[k, n]] ?? (rps: rps, nsg: 2)
        let nsg = geometry.nsg != 2
        let extra: [(String, any KernelTemplateArg)] = nsg ? [("NSG", geometry.nsg)] : []
        if u2, let twin = u2Kernel(half: !hf.isEmpty, load: load, nsg: nsg, dict: true) {
            return (twin, hf + extra, geometry.rps, geometry.nsg)
        }
        guard let kernel = dictKernel(half: !hf.isEmpty, load: load, nsg: nsg) else { return nil }
        return (kernel, hf + extra, geometry.rps, geometry.nsg)
    }

    /// The unroll-by-two twin of [K, N]'s R = 3 launch (dictionary or plain form) exists.
    static func u2Available(k: Int, n: Int, dict: Bool) -> Bool {
        let (_, hf) = projectKernel(rows: 3, fast: k % 512 == 0)
        let nsg = (preparedGeometry[[k, n]]?.nsg ?? 2) != 2
        return u2Kernel(half: !hf.isEmpty, load: loadForm(k: k, n: n), nsg: nsg, dict: dict) != nil
    }

    /// NemotronU2 (HEAD): the verify head's R = 3 prepared launch (the
    /// dictionary or plain form the window takes) on the real 131,072-row
    /// head, unrolled against record over three seeded row sets, bit for bit.
    static func u2CheckHead(_ model: Module) -> (MLXArray, Int)? {
        guard let q = model.modules().lazy.compactMap({ $0 as? QuantizedLinear })
                  .first(where: { $0.weight.shape == [131072, 336] }),
              q.mode == .affine, q.bits == 4, q.groupSize == 64, q.bias == nil, q.weight.dtype == .uint32,
              q.scales.dtype == .bfloat16, let biases = q.biases, biases.dtype == .bfloat16
        else { return nil }
        let k = 2688, n = 131072
        let dict = NemotronSBDict.enabled && NemotronSBDict.dict(q) != nil && dictLaunchAvailable(k: k, n: n)
        guard u2Available(k: k, n: n, dict: dict) else { return nil }
        var same = MLXArray(true)
        var cases = 0
        for key in MLXRandom.split(key: MLXRandom.key(0x52a3), into: 3) {
            let x = (MLXRandom.normal([3, k], key: key) * 0.5).asType(.bfloat16)
            let rps = NemotronMultiRowQMV.headRPS(n: n)
            let a = apply(x, q, relu2: false, rps: rps, u2: false)
            let b = apply(x, q, relu2: false, rps: rps, u2: true)
            same = logicalAnd(same, NemotronU2.equal(a, b))
            eval(same)
            cases += 1
        }
        return (same, cases)
    }
    static func dictLaunchAvailable(k: Int, n: Int) -> Bool {
        dictLaunch(k: k, n: n, rps: 2) != nil
    }

    /// The real [K, N] of every R = 3 prepared launch: JIT the non-crown
    /// selections at load (crown ones are compiled by the loops above).
    private static let verifyShapes: [[Int]] = [
        [2688, 10304], [2688, 3712], [3712, 2688], [4096, 2688], [5376, 2688], [2688, 4096],
        [2688, 131072],
    ]
    private static func warmVariants(_ outputs: inout [MLXArray]) {
        for shape in verifyShapes {
            let k = shape[0], n = shape[1]
            let base = n == 131072 ? NemotronMultiRowQMV.headRPS(n: n) : 2
            let (kernel, extra, rps, nsg) = projectLaunch(rows: 3, k: k, n: n, rps: base)
            guard loadForm(k: k, n: n) != 0 || nsg != 2 || rps != base else { continue }
            let pv = k % 512 == 0 ? 16 : 8
            let x = MLXArray.zeros([3, k], dtype: .bfloat16)
            let w = MLXArray.zeros([16, k / 8], dtype: .uint32)
            let s = MLXArray.zeros([16, k / 64], dtype: .bfloat16)
            let packed = prepare([x], template: [("T", x.dtype), ("K", k), ("PV", pv)],
                grid: (k / pv, 3, 1), threadGroup: (32, 1, 1),
                outputShapes: [[3, (k / pv + 31) / 32, (pv + 1) * 32]], outputDTypes: [.float32])
            let args = NemotronMultiRowQMV.scalarPair(k, 16)
            for relu2 in [false, true] {
                outputs.append(kernel([w, s, s, packed[0], args.0, args.1],
                    template: [("T", x.dtype), ("R", 3), ("FAST", k % 512 == 0),
                               ("RELU2", relu2), ("RPS", rps)] + extra,
                    grid: (32, 16 / rps, 1), threadGroup: (32, nsg, 1),
                    outputShapes: [[3, 16]], outputDTypes: [x.dtype])[0])
            }
        }
    }

    /// Feed the winner's prepared projection directly from the serialized
    /// SDPA rows. Same preparation arithmetic; only the source address moves.
    static func applyRows(_ rows: [MLXArray], _ q: QuantizedLinear) -> MLXArray {
        precondition(rows.count == 3)
        return projectPrepared(packRows(rows, k: q.weight.dim(1) * 8), q)
    }

    /// The prepared activation `applyRows` builds from three row pointers.
    static func packRows(_ rows: [MLXArray], k: Int) -> MLXArray {
        let pv = k % 512 == 0 ? 16 : 8
        return prepareRows(rows,
            template: [("T", DType.bfloat16), ("K", k), ("PV", pv)],
            grid: (k / pv, 3, 1), threadGroup: (32, 1, 1),
            outputShapes: [[3, (k / pv + 31) / 32, (pv + 1) * 32]], outputDTypes: [.float32])[0]
    }

    /// `applyRows`'s projection of an already prepared R = 3 activation.
    static func projectPrepared(_ packed: MLXArray, _ q: QuantizedLinear) -> MLXArray {
        let k = q.weight.dim(1) * 8, n = q.weight.dim(0)
        let packed = [packed]
        let args = NemotronMultiRowQMV.scalarPair(k, n)
        if NemotronSBDict.enabled, let d = NemotronSBDict.dict(q) {
            // A tuned launch binds its unread `dep` input to the prepared activation.
            let tuned = NemotronQMVTune.preparedLaunch(k: k, n: n, relu2: false, u2: false)
            if case let (kernel, hf, rps, nsg)? = tuned ?? dictLaunch(k: k, n: n, rps: 2) {
                return kernel([q.weight, d.index, d.table, packed[0], args.0, args.1] + (tuned != nil ? packed : []),
                    template: [("T", DType.bfloat16), ("R", 3), ("FAST", k % 512 == 0),
                               ("RELU2", false), ("RPS", rps)] + hf,
                    grid: (32, n / rps, 1), threadGroup: (32, nsg, 1),
                    outputShapes: [[3, n]], outputDTypes: [.bfloat16])[0]
            }
        }
        let (kernel, hf, rps, nsg) = projectLaunch(rows: 3, k: k, n: n, rps: 2)
        return kernel([q.weight, q.scales, q.biases!, packed[0], args.0, args.1],
            template: [("T", DType.bfloat16), ("R", 3), ("FAST", k % 512 == 0),
                       ("RELU2", false), ("RPS", rps)] + hf,
            grid: (32, n / rps, 1), threadGroup: (32, nsg, 1),
            outputShapes: [[3, n]], outputDTypes: [.bfloat16])[0]
    }
    // JIT each actual prepare K specialization and the two exact QMV
    // epilogues before timed verification. N is a scalar, so eight rows of
    // dummy weights cover the same project pipelines as the full matrices.
    static func warm() {
        var outputs: [MLXArray] = []
        for k in [2688, 3712, 4096, 5376] {
            let pv = k % 512 == 0 ? 16 : 8
            let x = MLXArray.zeros([3, k], dtype: .bfloat16)
            let w = MLXArray.zeros([8, k / 8], dtype: .uint32)
            let s = MLXArray.zeros([8, k / 64], dtype: .bfloat16)
            let packed = prepare([x], template: [("T", x.dtype), ("K", k), ("PV", pv)],
                grid: (k / pv, 3, 1), threadGroup: (32, 1, 1),
                outputShapes: [[3, (k / pv + 31) / 32, (pv + 1) * 32]], outputDTypes: [.float32])
            let args = NemotronMultiRowQMV.scalarPair(k, 8)
            let (kernel, hf) = projectKernel(rows: 3, fast: k % 512 == 0)
            for relu2 in [false, true] {
                outputs.append(kernel([w, s, s, packed[0], args.0, args.1],
                    template: [("T", x.dtype), ("R", 3), ("FAST", k % 512 == 0),
                               ("RELU2", relu2), ("RPS", 2)] + hf,
                    grid: (32, 4, 1), threadGroup: (32, 2, 1),
                    outputShapes: [[3, 8]], outputDTypes: [x.dtype])[0])
            }
            // The target head's pipeline (K 2688, no ReLU2) at its own RPS.
            if k == 2688 && NemotronMultiRowQMV.headRPS(n: 131072) != 2 {
                let rps = NemotronMultiRowQMV.headRPS(n: 131072)
                outputs.append(kernel([w, s, s, packed[0], args.0, args.1],
                    template: [("T", x.dtype), ("R", 3), ("FAST", false),
                               ("RELU2", false), ("RPS", rps)] + hf,
                    grid: (32, 8 / rps, 1), threadGroup: (32, 2, 1),
                    outputShapes: [[3, 8]], outputDTypes: [x.dtype])[0])
            }
        }
        warmVariants(&outputs)
        eval(outputs)
    }

    static func prepareContiguousRows(_ x: MLXArray, pv: Int) -> MLXArray {
        let rows = x.dim(0), k = x.dim(1)
        return prepare([x], template: [("T", x.dtype), ("K", k), ("PV", pv)],
            grid: (k / pv, rows, 1), threadGroup: (32, 1, 1),
            outputShapes: [[rows, (k / pv + 31) / 32, (pv + 1) * 32]],
            outputDTypes: [.float32])[0]
    }

    /// x: [rows, K] BF16. The lane-coalesced FP32 buffer the prepared QMV reads.
    static func pack(_ x: MLXArray) -> MLXArray {
        let rows = x.dim(0), k = x.dim(1)
        let pv = k % 512 == 0 ? 16 : 8
        return prepare([x], template: [("T", x.dtype), ("K", k), ("PV", pv)],
            grid: (k / pv, rows, 1), threadGroup: (32, 1, 1),
            outputShapes: [[rows, (k / pv + 31) / 32, (pv + 1) * 32]], outputDTypes: [.float32])[0]
    }

    static func compatible(_ prepared: MLXArray?, rows: Int, k: Int) -> MLXArray? {
        let pv = k % 512 == 0 ? 16 : 8
        guard let prepared, prepared.dtype == .float32,
              prepared.shape == [rows, (k / pv + 31) / 32, (pv + 1) * 32]
        else { return nil }
        return prepared
    }

    /// `u2`: the R = 3 launch's unroll-by-two twin (nil: the admitted family, `NemotronU2.prepared`).
    /// `qmvDep`: a `NemotronQMVTune` launch's unread FP32 `dep` binding (nil: the prepared activation).
    static func apply(_ x: MLXArray, _ q: QuantizedLinear, relu2: Bool, rps: Int,
                      prepared: MLXArray? = nil, sbDict: Bool = true, u2: Bool? = nil,
                      qmvTune: NemotronQMVTune.Config? = nil, qmvDep: MLXArray? = nil) -> MLXArray {
        let rows = x.dim(0), k = x.dim(1), n = q.weight.dim(0)
        let pv = k % 512 == 0 ? 16 : 8
        let packed = compatible(prepared, rows: rows, k: k)
            ?? prepare([x], template: [("T", x.dtype), ("K", k), ("PV", pv)],
            grid: (k / pv, rows, 1), threadGroup: (32, 1, 1),
            outputShapes: [[rows, (k / pv + 31) / 32, (pv + 1) * 32]], outputDTypes: [.float32])[0]
        let args = NemotronMultiRowQMV.scalarPair(k, n)
        let unroll = rows == 3 && (u2 ?? NemotronU2.prepared(rows: rows, n: n))
        if sbDict, rows == 3, NemotronSBDict.enabled, let d = NemotronSBDict.dict(q) {
            let tuned = NemotronQMVTune.preparedLaunch(k: k, n: n, relu2: relu2, u2: unroll, config: qmvTune)
            if case let (kernel, hf, launchRPS, nsg)? = tuned ?? dictLaunch(k: k, n: n, rps: rps, u2: unroll) {
                return kernel([q.weight, d.index, d.table, packed, args.0, args.1]
                              + (tuned != nil ? [qmvDep ?? packed] : []),
                    template: [("T", x.dtype), ("R", rows), ("FAST", k % 512 == 0), ("RELU2", relu2),
                               ("RPS", launchRPS)] + hf,
                    grid: (32, n / launchRPS, 1), threadGroup: (32, nsg, 1),
                    outputShapes: [[rows, n]], outputDTypes: [x.dtype])[0]
            }
        }
        // `NemotronDraftTune` (ehp): the geometry adopted at load for the head's R = 3 eh_proj (unread `dep`: the
        // prepared activation).
        if rows == 3, !relu2, k == 5376, n == 2688, u2 == nil, qmvTune == nil, let biases = q.biases,
           let g = NemotronDraftTune.adopted[.ehp],
           let y = draftTuneLaunch(g, packed: packed, w: q.weight, scales: q.scales, biases: biases, dep: packed) {
            return y
        }
        let (kernel, hf, launchRPS, nsg) = projectLaunch(rows: rows, k: k, n: n, rps: rps, u2: unroll)
        return kernel([q.weight, q.scales, q.biases!, packed, args.0, args.1],
            template: [("T", x.dtype), ("R", rows), ("FAST", k % 512 == 0), ("RELU2", relu2), ("RPS", launchRPS)] + hf,
            grid: (32, n / launchRPS, 1), threadGroup: (32, nsg, 1),
            outputShapes: [[rows, n]], outputDTypes: [x.dtype])[0]
    }
}

private let nemotronSharedUpPackHeader = nemotronMultiRowQMVHeader + """

template <typename T, typename U, int values_per_thread>
inline U nmr_load_vector_tg(const threadgroup T* x, thread U* x_thread) {
  U sum = 0;
  for (int i = 0; i < values_per_thread; i += 4) {
    sum += U(x[i]) + U(x[i + 1]) + U(x[i + 2]) + U(x[i + 3]);
    x_thread[i] = x[i];
    x_thread[i + 1] = x[i + 1] / 16.0f;
    x_thread[i + 2] = x[i + 2] / 256.0f;
    x_thread[i + 3] = x[i + 3] / 4096.0f;
  }
  return sum;
}
"""

private func nemotronSharedUpPackSource() -> String {
    let groups = "constexpr int num_simdgroups = 2;"
    let store = "yp[r * out_vec_size + row] = h;"
    precondition(nemotronHeadPreparedQMVSource.components(separatedBy: groups).count == 2)
    precondition(nemotronHeadPreparedQMVSource.components(separatedBy: store).count == 2)
    let body = nemotronHeadPreparedQMVSource
        .replacingOccurrences(of: groups, with: "constexpr int num_simdgroups = 4;")
        .replacingOccurrences(
            of: store,
            with: store + "\n        chunk_h[r][simd_gid * results_per_simdgroup + row] = h;")
    return "  threadgroup T chunk_h[R][8];\n" + body + """

  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0 && simd_lid < uint(R)) {
    const int r = int(simd_lid);
    float values[8];
    const float total = nmr_load_vector_tg<T, float, 8>(&chunk_h[r][0], values);
    const int chunk = int(tid.y);
    const int chunk_tiles = (out_vec_size / 8 + 31) / 32;
    const int base = (r * chunk_tiles + chunk / 32) * (8 + 1) * 32 + chunk % 32;
    for (int i = 0; i < 8; i++) packed[base + i * 32] = values[i];
    packed[base + 8 * 32] = total;
  }
"""
}

enum NemotronSharedUpPacking {
    static let enabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_SHARED_UP_PACK"] != "0"
    nonisolated(unsafe) private static var verified = false
    private static let unpackedEnabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_SHARED_UP_UNPACKED"] != "0"
    private static let unpackedLock = NSLock()
    nonisolated(unsafe) private static var unpackedOwners: Set<ObjectIdentifier> = []
    private static let kernel = MLXFast.metalKernel(
        name: "nemotron_shared_up_relu2_pack_r3",
        inputNames: ["w", "scales", "biases", "prepared", "K", "N"],
        outputNames: ["y", "packed"], source: nemotronSharedUpPackSource(),
        header: nemotronSharedUpPackHeader)
    /// The same launch reading (scale, bias) through `NemotronSBDict`.
    private static let dictKernel = nemotronSBDictSource(nemotronSharedUpPackSource()).map {
        MLXFast.metalKernel(
            name: "nemotron_shared_up_relu2_pack_r3_sbd",
            inputNames: ["w", "scales", "biases", "prepared", "K", "N"],
            outputNames: ["y", "packed"], source: $0, header: nemotronSharedUpPackHeader)
    }

    /// Load check of the dictionary twin on `q`: both outputs of both
    /// kernels on the same rows, bit for bit (nil when either cannot run).
    /// The packed output is compared on the chunks the kernel writes and the
    /// prepared QMV reads: [3, tiles, 9, 32] holds chunk t * 32 + lane at
    /// [r, t, i, lane], and only chunks < N / 8 exist. The last tile's lanes
    /// past N / 8 are never stored (their bytes are whatever the allocator
    /// handed out) and never loaded (the reader's `remaining` is 0 there).
    static func dictMatches(_ x: MLXArray, _ q: QuantizedLinear) -> MLXArray? {
        guard dictKernel != nil, NemotronSBDict.dict(q) != nil,
              let a = dispatch(x, q, prepared: nil, sbDict: false),
              let b = dispatch(x, q, prepared: nil) else { return nil }
        let chunks = q.weight.dim(0) / 8, tiles = (chunks + 31) / 32
        func written(_ p: MLXArray) -> MLXArray {
            p.view(dtype: .uint32).reshaped(3, tiles, 9, 32).transposed(0, 1, 3, 2)
                .reshaped(3, tiles * 32, 9)[0..., ..<chunks, 0...]
        }
        return logicalAnd((a.0.view(dtype: .uint16) .== b.0.view(dtype: .uint16)).all(),
                          (written(a.1) .== written(b.1)).all())
    }

    private static func dispatch(_ x: MLXArray, _ q: QuantizedLinear,
                                 prepared: MLXArray?, sbDict: Bool = true) -> (MLXArray, MLXArray)? {
        guard enabled, NemotronMultiRowQMV.enabled, NemotronMultiRowQMV.narrowEnabled,
              Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              x.ndim == 2, x.dim(0) == 3, x.dtype == .bfloat16,
              q.mode == .affine, q.bits == 4, q.groupSize == 64, q.bias == nil,
              let biases = q.biases, q.weight.ndim == 2, q.weight.dtype == .uint32,
              q.scales.dtype == .bfloat16, biases.dtype == .bfloat16
        else { return nil }
        let k = x.dim(1), n = q.weight.dim(0)
        guard k % 64 == 0, k > 128, q.weight.dim(1) * 8 == k,
              n >= 2688, n % 8 == 0, n % 512 != 0, NemotronMultiRowQMV.headRPS(n: n) == 2
        else { return nil }
        let pv = k % 512 == 0 ? 16 : 8
        let input = NemotronHeadPreparedActivation.compatible(prepared, rows: 3, k: k)
            ?? NemotronHeadPreparedActivation.prepareContiguousRows(x, pv: pv)
        let args = NemotronMultiRowQMV.scalarPair(k, n)
        let dict = sbDict && NemotronSBDict.enabled && dictKernel != nil ? NemotronSBDict.dict(q) : nil
        let outputs = (dict != nil ? dictKernel! : kernel)(
            [q.weight, dict?.index ?? q.scales, dict?.table ?? biases, input, args.0, args.1],
            template: [("T", DType.bfloat16), ("R", 3), ("FAST", k % 512 == 0),
                       ("RELU2", true), ("RPS", 2)],
            grid: (32, n / 2, 1), threadGroup: (32, 4, 1),
            outputShapes: [[3, n], [3, (n / 8 + 31) / 32, 9 * 32]],
            outputDTypes: [.bfloat16, .float32])
        return (outputs[0], outputs[1])
    }

    /// True when `unpackedActivation` would return the prepared QMV's rows for these operands.
    static func unpackedApplies(_ x: MLXArray, up: Linear, down: Linear) -> Bool {
        guard verified, !NemotronMoELeader.selfTest, x.ndim == 3, x.dim(0) == 1, x.dim(1) == 3,
              let q = up as? QuantizedLinear, unpackedEnabled, NemotronSharedDownZSkip.covers(down)
        else { return false }
        return unpackedLock.withLock { unpackedOwners.contains(ObjectIdentifier(q)) }
    }

    /// The shared relu2 rows [3, N] that `apply` hands the zero-skip shared down on its unpacked route, or nil
    /// when `apply` would take another route (`NemotronFC2SharedColaunch` then keeps the separate launches).
    static func unpackedActivation(_ x: MLXArray, up: Linear, down: Linear,
                                   prepared: MLXArray?) -> MLXArray? {
        guard verified, !NemotronMoELeader.selfTest, x.ndim == 3, x.dim(0) == 1, x.dim(1) == 3,
              let q = up as? QuantizedLinear, unpackedEnabled, NemotronSharedDownZSkip.covers(down),
              unpackedLock.withLock({ unpackedOwners.contains(ObjectIdentifier(q)) })
        else { return nil }
        return NemotronHeadPreparedActivation.apply(x.reshaped(3, x.dim(2)), q, relu2: true, rps: 2,
                                                    prepared: prepared)
    }

    static func apply(_ x: MLXArray, up: Linear, down: Linear,
                      prepared: MLXArray?) -> MLXArray? {
        guard verified, x.ndim == 3, x.dim(0) == 1, x.dim(1) == 3,
              let q = up as? QuantizedLinear
        else { return nil }
        let rows = x.reshaped(3, x.dim(2))
        let fused: (MLXArray, MLXArray?)
        if unpackedEnabled, NemotronSharedDownZSkip.covers(down),
           unpackedLock.withLock({ unpackedOwners.contains(ObjectIdentifier(q)) }) {
            fused = (NemotronHeadPreparedActivation.apply(rows, q, relu2: true, rps: 2, prepared: prepared), nil)
        } else {
            guard let packed = dispatch(rows, q, prepared: prepared) else { return nil }
            fused = (packed.0, packed.1)
        }
        var activated = fused.0
        if NemotronMoELeader.selfTest {
            activated = nemotronBitwiseOrNaN(activated, NemotronHeadPreparedActivation.apply(
                x.reshaped(3, x.dim(2)), q, relu2: true, rps: 2, prepared: prepared))
        }
        if let y = NemotronSharedDownZSkip.run(activated, owner: down) {
            guard NemotronMoELeader.selfTest else { return y }
            return nemotronBitwiseOrNaN(y, nemotronMTPLinearRows(
                activated.reshaped(1, 3, q.weight.dim(0)), down, prepared: fused.1))
        }
        return nemotronMTPLinearRows(
            activated.reshaped(1, 3, q.weight.dim(0)), down, prepared: fused.1)
    }

    static func prepareUnpacked(_ model: Module) {
        unpackedLock.withLock { unpackedOwners = [] }
        guard unpackedEnabled, verified else { return }
        var built: Set<ObjectIdentifier> = []
        var accepted = false
        do {
            accepted = try withError { () -> Bool in
                for module in model.modules() {
                    guard let shared = (module as? NemotronHMoE)?.sharedExperts,
                          NemotronSharedDownZSkip.covers(shared.downProj),
                          let q = shared.upProj as? QuantizedLinear,
                          !built.contains(ObjectIdentifier(q)) else { continue }
                    let k = q.weight.dim(1) * 8
                    let dense = MLXRandom.normal([3, k], dtype: .bfloat16,
                        key: MLXRandom.key(UInt64(0x51a0 + built.count)))
                    let zero = MLXArray.zeros([3, k], dtype: .bfloat16)
                    for x in [dense, zero] {
                        guard let old = dispatch(x, q, prepared: nil) else { return false }
                        let new = NemotronHeadPreparedActivation.apply(x, q, relu2: true, rps: 2, prepared: nil)
                        guard (old.0.view(dtype: .uint16) .== new.view(dtype: .uint16)).all().item(Bool.self)
                        else { return false }
                    }
                    built.insert(ObjectIdentifier(q))
                }
                return !built.isEmpty
            }
        } catch {
            accepted = false
        }
        unpackedLock.withLock { unpackedOwners = accepted ? built : [] }
        FileHandle.standardError.write(Data((accepted
            ? "nemotron shared up unpacked on: \(built.count) owners\n"
            : "nemotron shared up unpacked off\n").utf8))
    }

    static func verify(_ model: Module) {
        verified = false
        guard enabled,
              let mlp = model.modules().lazy.compactMap({ ($0 as? NemotronHMoE)?.sharedExperts }).first,
              let up = mlp.upProj as? QuantizedLinear
        else { return }
        var matches = true
        for seed in 0 ..< 3 {
            let x = MLXRandom.normal(
                [3, up.weight.dim(1) * 8], dtype: .bfloat16,
                key: MLXRandom.key(UInt64(0x5a17 + seed)))
            guard let fused = dispatch(x, up, prepared: nil) else { return }
            let activated = NemotronHeadPreparedActivation.apply(
                x, up, relu2: true, rps: 2, prepared: nil)
            let lhs = nemotronMTPLinearRows(
                fused.0.reshaped(1, 3, up.weight.dim(0)), mlp.downProj, prepared: fused.1)
            let rhs = nemotronMTPLinearRows(
                activated.reshaped(1, 3, up.weight.dim(0)), mlp.downProj)
            let sameUp = (fused.0.view(dtype: .uint16) .== activated.view(dtype: .uint16)).all()
            let sameDown = (lhs.view(dtype: .uint16) .== rhs.view(dtype: .uint16)).all()
            matches = matches && sameUp.item(Bool.self) && sameDown.item(Bool.self)
        }
        verified = matches
    }
}

private final class NemotronMultiRowQMVKernel: Sendable {
    static let shared = NemotronMultiRowQMVKernel()
    let kernel: MLXFast.MLXFastKernel
    let attentionOutputAdding: MLXFast.MLXFastKernel
    private init() {
        kernel = MLXFast.metalKernel(
            name: "nemotron_multirow_qmv_affine4_gs64",
            inputNames: ["w", "scales", "biases", "x", "K", "N"],
            outputNames: ["y"],
            source: nemotronMultiRowQMVSource,
            header: nemotronMultiRowQMVHeader)
        // Same reduction and BF16 narrowing; the residual add moves into the
        // epilogue as one BF16 + BF16, like the separate add it replaces.
        let addingSource = nemotronMultiRowQMVSource.replacingOccurrences(
            of: "yp[r * out_vec_size + row] = h;",
            with: "yp[r * out_vec_size + row] = residual[r * out_vec_size + out_row + row] + h;")
        attentionOutputAdding = MLXFast.metalKernel(
            name: "nemotron_mtp_attention_output_add_affine4_gs64",
            inputNames: ["residual", "w", "scales", "biases", "x", "K", "N"],
            outputNames: ["y"], source: addingSource,
            header: nemotronMultiRowQMVHeader, ensureRowContiguous: true)
        // Decode-form twins (template HF): the same bodies through
        // nemotronMultiRowHalfSwaps; used only where a switch below is on.
        let halfSource = nemotronSwapped(nemotronMultiRowQMVSource, nemotronMultiRowHalfSwaps)
        let halfHeader = nemotronMultiRowQMVHeader + nemotronHalfDecodeHeader
        kernelHalf = MLXFast.metalKernel(
            name: "nemotron_multirow_qmv_affine4_gs64_hf",
            inputNames: ["w", "scales", "biases", "x", "K", "N"],
            outputNames: ["y"], source: halfSource, header: halfHeader)
        attentionOutputAddingHalf = MLXFast.metalKernel(
            name: "nemotron_mtp_attention_output_add_affine4_gs64_hf",
            inputNames: ["residual", "w", "scales", "biases", "x", "K", "N"],
            outputNames: ["y"], source: nemotronSwapped(addingSource, nemotronMultiRowHalfSwaps),
            header: halfHeader, ensureRowContiguous: true)
        kernelDict = nemotronSBDictSource(nemotronMultiRowQMVSource).map {
            MLXFast.metalKernel(
                name: "nemotron_multirow_qmv_affine4_gs64_sbd",
                inputNames: ["w", "scales", "biases", "x", "K", "N"],
                outputNames: ["y"], source: $0, header: nemotronMultiRowQMVHeader)
        }
        kernelHalfDict = nemotronSBDictSource(halfSource).map {
            MLXFast.metalKernel(
                name: "nemotron_multirow_qmv_affine4_gs64_hf_sbd",
                inputNames: ["w", "scales", "biases", "x", "K", "N"],
                outputNames: ["y"], source: $0, header: halfHeader)
        }
        // `NemotronDraftColaunch`: the R = 1 decode-form twins with one more input, `dep`, bound first and never
        // read (same source text); it only orders the draft shared down after the routed FC1 in the eval tape.
        kernelHalfDep = MLXFast.metalKernel(
            name: "nemotron_multirow_qmv_affine4_gs64_hf_dep",
            inputNames: ["dep", "w", "scales", "biases", "x", "K", "N"],
            outputNames: ["y"], source: halfSource, header: halfHeader)
        kernelHalfDictDep = nemotronSBDictSource(halfSource).map {
            MLXFast.metalKernel(
                name: "nemotron_multirow_qmv_affine4_gs64_hf_sbd_dep",
                inputNames: ["dep", "w", "scales", "biases", "x", "K", "N"],
                outputNames: ["y"], source: $0, header: halfHeader)
        }
    }
    let kernelHalfDep: MLXFast.MLXFastKernel
    let kernelHalfDictDep: MLXFast.MLXFastKernel?
    let kernelHalf: MLXFast.MLXFastKernel
    let attentionOutputAddingHalf: MLXFast.MLXFastKernel
    /// `kernel` and `kernelHalf` reading (scale, bias) through a draft head
    /// tensor's `NemotronSBDict` pair dictionary (R = 1 only), or nil.
    let kernelDict: MLXFast.MLXFastKernel?
    let kernelHalfDict: MLXFast.MLXFastKernel?
    /// Kill switches (code constants; 0 = crown int -> float decode and the
    /// crown kernel objects): post-verify MTP wo + residual at R = 3 (FAST),
    /// and the R = 1 multi-row qmv (draft shared expert up).
    static let attentionHalfDecodeForm = 2
    static let r1HalfDecodeForm = 2
}

enum NemotronMultiRowQMV {
    /// Input-independent int32 scalar arguments, built once per (K, N)
    /// instead of twice per call. Same values as the implicit Int conversion.
    nonisolated(unsafe) private static var scalars: [Int: (MLXArray, MLXArray)] = [:]
    private static let scalarsLock = NSLock()
    static func scalarPair(_ k: Int, _ n: Int) -> (MLXArray, MLXArray) {
        scalarsLock.lock(); defer { scalarsLock.unlock() }
        let key = k &* 1_000_003 &+ n
        if let pair = scalars[key] { return pair }
        let pair = (MLXArray(Int32(k)), MLXArray(Int32(n)))
        scalars[key] = pair
        return pair
    }
    static let enabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_MULTIROW_QMV"] != "0"
    /// NEMOTRON_MTP_QMV_NARROW=0 keeps four outputs per SIMD group everywhere.
    static let narrowEnabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_QMV_NARROW"] != "0"
    /// NEMOTRON_MTP_R1_TWIN=0 keeps the stock call for the draft step's
    /// one-row eh_proj, attention wo and shared-expert down (`nemotronR1Linear`).
    static let r1Route = ProcessInfo.processInfo.environment["NEMOTRON_MTP_R1_TWIN"] != "0"
    /// The dictionary twin of the R = 1 kernel `dispatch` selects, if built.
    static var draftDictAvailable: Bool {
        let kernels = NemotronMultiRowQMVKernel.shared
        return (NemotronMultiRowQMVKernel.r1HalfDecodeForm != 0
            ? kernels.kernelHalfDict : kernels.kernelDict) != nil
    }
    /// NEMOTRON_MTP_HEAD_RPS4=0 keeps two outputs per SIMD group on the
    /// 131,072-row target head. Four halves the threadgroups that each
    /// re-read the prepared rows; every output keeps its lane split, block
    /// order and closing simd_sum, so only the row-to-SIMD-group map moves.
    static let headRPS4 =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_HEAD_RPS4"] != "0"
    /// m3 kill switch (code constant): the target head's outputs per SIMD
    /// group when NEMOTRON_MTP_HEAD_RPS4 is on (crown 4). 2 is the crown's own
    /// RPS 2 pipeline (no new source); only the row-to-SIMD-group map moves.
    static let headRPSValue = 4
    static func headRPS(n: Int) -> Int { headRPS4 && n == 131072 ? headRPSValue : 2 }

    /// DARKBLOOM_BONSAI_R8B=0 keeps the MTP wo projection and residual add
    /// as two launches. Otherwise the fused route engages only after the
    /// load-time bit check below matches the two-launch route.
    static let attentionOutputAdd =
        ProcessInfo.processInfo.environment["DARKBLOOM_BONSAI_R8B"] != "0"
    nonisolated(unsafe) private static var attentionOutputAddVerified = false

    /// Post-verify MTP wo (K 4096 -> N 2688) plus residual, R = 2...4, in one
    /// launch. Nil unless the load-time check passed; callers then keep
    /// `residual + nemotronMTPLinearRows(x, linear)`.
    static func applyAttentionOutputAdding(
        _ x: MLXArray, _ linear: Linear, residual: MLXArray
    ) -> MLXArray? {
        guard attentionOutputAddVerified else { return nil }
        return attentionOutputAdding(x, linear, residual: residual)
    }

    private static func attentionOutputAdding(
        _ x: MLXArray, _ linear: Linear, residual: MLXArray
    ) -> MLXArray? {
        guard enabled, attentionOutputAdd, NemotronMTPExecution.batchedM1,
              Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              x.ndim == 3, x.dim(0) == 1, (2...4).contains(x.dim(1)),
              x.dim(2) == 4096, x.dtype == .bfloat16,
              residual.shape == [1, x.dim(1), 2688], residual.dtype == x.dtype,
              type(of: linear) == QuantizedLinear.self,
              let q = linear as? QuantizedLinear,
              q.mode == .affine, q.bits == 4, q.groupSize == 64, q.bias == nil,
              q.weight.dtype == .uint32, q.weight.shape == [2688, 512],
              q.scales.dtype == .bfloat16, q.scales.shape == [2688, 64],
              let offsets = q.biases, offsets.dtype == .bfloat16,
              offsets.shape == q.scales.shape else { return nil }
        let rows = x.dim(1)
        // Same outputs per SIMD group as the unfused `apply`, so the same
        // lane split and simd_sum produce each output.
        let rps = resultsPerSimdgroup(rows: rows, k: 4096, n: 2688)
        let (k, n) = scalarPair(4096, 2688)
        let form = NemotronMultiRowQMVKernel.attentionHalfDecodeForm
        let half = rows == 3 && form != 0
        let y = (half ? NemotronMultiRowQMVKernel.shared.attentionOutputAddingHalf
                      : NemotronMultiRowQMVKernel.shared.attentionOutputAdding)(
            [residual.reshaped(rows, 2688), q.weight, q.scales, offsets,
             x.reshaped(rows, 4096), k, n],
            template: [("T", x.dtype), ("R", rows), ("FAST", true),
                       ("RELU2", false), ("RPS", rps)] + nemotronHalfTemplate(half, form),
            grid: (32, 2688 / rps, 1), threadGroup: (32, 2, 1),
            outputShapes: [[1, rows, 2688]], outputDTypes: [x.dtype])[0]
        guard NemotronMoELeader.selfTest else { return y }
        return nemotronBitwiseOrNaN(y, residual + nemotronMTPLinearRows(x, linear))
    }

    /// Load-time bit check with the loaded wo on seeded random rows, R = 2...4:
    /// the fused route engages only if every output equals the two-launch
    /// route bit for bit. Also JITs the fused pipelines outside any request.
    static func verifyAttentionOutputAdding(_ wo: Linear) {
        attentionOutputAddVerified = false
        var matches = true
        for rows in 2 ... 4 {
            let x = MLXRandom.normal(
                [1, rows, 4096], dtype: .bfloat16, key: MLXRandom.key(UInt64(0x8b40 + rows)))
            let residual = MLXRandom.normal(
                [1, rows, 2688], dtype: .bfloat16, scale: 4,
                key: MLXRandom.key(UInt64(0x8b48 + rows)))
            guard let fused = attentionOutputAdding(x, wo, residual: residual) else { return }
            let expected = residual + nemotronMTPLinearRows(x, wo)
            let same = fused.view(dtype: .uint16) .== expected.view(dtype: .uint16)
            matches = matches && same.all().item(Bool.self)
        }
        attentionOutputAddVerified = matches
    }

    /// Outputs per SIMD group for one verify width and (K, N). Two outputs
    /// halve each threadgroup's rows and double the threadgroups; each output
    /// keeps its lane split over K, block order, qdot, affine step and closing
    /// simd_sum, so only the output-to-SIMD-group mapping changes. A fixed
    /// table from an M1 Max per-shape microbenchmark (dependent chains, 24
    /// weight copies per shape): two outputs won on every listed shape at
    /// R = 3 (the depth-2 verify and post-verify width) and lost on most at
    /// R = 1 and 2, so only R = 3 and 4 narrow. Never a run-time timing pick.
    static func resultsPerSimdgroup(rows: Int, k: Int, n: Int) -> Int {
        guard narrowEnabled else { return 4 }
        switch rows {
        case 3: return narrowShapes.contains([k, n]) ? 2 : 4
        // R = 4 out_proj / o_proj / wo (4096 -> 2688) measured flat.
        case 4: return narrowShapes.contains([k, n]) && [k, n] != [4096, 2688] ? 2 : 4
        default: return 4
        }
    }
    /// [K, N]: Mamba out_proj, attention o_proj and MTP wo; shared expert up
    /// and down; MTP eh_proj, wq, wk/wv; Mamba in_proj; draft and target head.
    private static let narrowShapes: Set<[Int]> = [
        [4096, 2688], [2688, 3712], [3712, 2688], [5376, 2688], [2688, 4096],
        [2688, 256], [2688, 10304], [2688, 65536], [2688, 131072],
    ]

    private static func dispatch(_ w: MLXArray, _ scales: MLXArray, _ biases: MLXArray,
                                 _ x: MLXArray, relu2: Bool, rps: Int,
                                 dict: NemotronSBDict.Dict? = nil, dep: MLXArray? = nil) -> MLXArray {
        let rows = x.dim(0), k = x.dim(1), n = w.dim(0)
        let (kArray, nArray) = scalarPair(k, n)
        let form = NemotronMultiRowQMVKernel.r1HalfDecodeForm
        let half = rows == 1 && form != 0
        let kernels = NemotronMultiRowQMVKernel.shared
        // `NemotronDraftTune`: the geometry adopted at load for this draft head R = 1 dictionary launch (its unread
        // `dep`: the launch's own `dep`, else its input row).
        if rows == 1, half, let dict, let family = NemotronDraftTune.qmvFamily(k: k, n: n, relu2: relu2),
           let g = NemotronDraftTune.adopted[family],
           let y = NemotronDraftTune.qmvLaunch(g, w: w, index: dict.index, table: dict.table, x: x, relu2: relu2,
                                               dep: dep ?? x) {
            return y
        }
        // `dep` (one row, decode form only; `NemotronDraftColaunch`): the selected kernel's `dep` twin.
        if let dep, rows == 1, half {
            if let dict {
                if let twin = kernels.kernelHalfDictDep {
                    return twin(
                        [dep, w, dict.index, dict.table, x, kArray, nArray],
                        template: [("T", x.dtype), ("R", rows), ("FAST", k % 512 == 0),
                                   ("RELU2", relu2), ("RPS", rps)] + nemotronHalfTemplate(half, form),
                        grid: (32, n / rps, 1),
                        threadGroup: (32, 2, 1),
                        outputShapes: [[rows, n]],
                        outputDTypes: [x.dtype])[0]
                }
            } else {
                return kernels.kernelHalfDep(
                    [dep, w, scales, biases, x, kArray, nArray],
                    template: [("T", x.dtype), ("R", rows), ("FAST", k % 512 == 0),
                               ("RELU2", relu2), ("RPS", rps)] + nemotronHalfTemplate(half, form),
                    grid: (32, n / rps, 1),
                    threadGroup: (32, 2, 1),
                    outputShapes: [[rows, n]],
                    outputDTypes: [x.dtype])[0]
            }
        }
        // `dict` (one row only): the selected kernel's dictionary twin.
        if let dict, rows == 1, let twin = half ? kernels.kernelHalfDict : kernels.kernelDict {
            return twin(
                [w, dict.index, dict.table, x, kArray, nArray],
                template: [("T", x.dtype), ("R", rows), ("FAST", k % 512 == 0),
                           ("RELU2", relu2), ("RPS", rps)] + nemotronHalfTemplate(half, form),
                grid: (32, n / rps, 1),
                threadGroup: (32, 2, 1),
                outputShapes: [[rows, n]],
                outputDTypes: [x.dtype])[0]
        }
        return (half ? kernels.kernelHalf : kernels.kernel)(
            [w, scales, biases, x, kArray, nArray],
            template: [("T", x.dtype), ("R", rows), ("FAST", k % 512 == 0),
                       ("RELU2", relu2), ("RPS", rps)] + nemotronHalfTemplate(half, form),
            grid: (32, n / rps, 1),
            threadGroup: (32, 2, 1),
            outputShapes: [[rows, n]],
            outputDTypes: [x.dtype])[0]
    }

    /// x: [rows, K] contiguous. Returns [rows, N] or nil when the shape is not
    /// one the stock M=1 dispatch would send to qmv / qmv_fast.
    /// Width one is used by the exact BF16 ReLU2 epilogue, not a GEMM.
    /// `sbDict`: one row of a draft head tensor reads its admitted pair
    /// dictionary (`NemotronSBDict.draftDict`); false keeps the record kernel.
    static func apply(_ x: MLXArray, _ q: QuantizedLinear, relu2: Bool = false,
                      prepared: MLXArray? = nil, sbDict: Bool = true, dep: MLXArray? = nil) -> MLXArray? {
        guard enabled, x.ndim == 2, (1...8).contains(x.dim(0)),
              q.mode == .affine, q.bits == 4, q.groupSize == 64,
              let biases = q.biases, x.dtype == .bfloat16,
              q.scales.dtype == x.dtype, biases.dtype == x.dtype,
              q.weight.ndim == 2, q.weight.dtype == .uint32
        else { return nil }
        let k = x.dim(1), n = q.weight.dim(0)
        // Stock dispatch_qmv sends K == 64/128 to qmv_quad; skip those.
        guard k % 64 == 0, k > 128, n % 8 == 0, q.weight.dim(1) * 8 == k else { return nil }
        if narrowEnabled && x.dim(0) == 3 && n >= 2688 {
            let y = NemotronHeadPreparedActivation.apply(x, q, relu2: relu2,
                rps: headRPS(n: n), prepared: prepared)
            guard NemotronMoELeader.selfTest else { return y }
            return nemotronBitwiseOrNaN(y, dispatch(q.weight, q.scales, biases, x, relu2: relu2, rps: 2))
        }
        let rps = resultsPerSimdgroup(rows: x.dim(0), k: k, n: n)
        let dict = sbDict && x.dim(0) == 1 ? NemotronSBDict.draftDict(q) : nil
        let y = dispatch(q.weight, q.scales, biases, x, relu2: relu2, rps: rps, dict: dict,
                         dep: x.dim(0) == 1 ? dep : nil)
        guard rps != 4, NemotronMoELeader.selfTest else { return y }
        return nemotronBitwiseOrNaN(
            y, dispatch(q.weight, q.scales, biases, x, relu2: relu2, rps: 4))
    }

    /// JIT every narrow pipeline (R = 3, 4; FAST; RELU2) at load, so no
    /// timed window compiles one. Tiny dummy shapes; same template keys.
    static func warmNarrow() {
        guard enabled, narrowEnabled else { return }
        NemotronHeadPreparedActivation.warm()
        var outputs: [MLXArray] = []
        for k in [512, 576] {
            let w = MLXArray.zeros([8, k / 8], dtype: .uint32)
            let s = MLXArray.zeros([8, k / 64], dtype: .bfloat16)
            for rows in [3, 4] {
                let x = MLXArray.zeros([rows, k], dtype: .bfloat16)
                for relu2 in [false, true] {
                    outputs.append(dispatch(w, s, s, x, relu2: relu2, rps: 2))
                }
            }
            // The one-row route (`nemotronR1Linear`, shared-expert up) at RPS 4.
            let x1 = MLXArray.zeros([1, k], dtype: .bfloat16)
            for relu2 in [false, true] {
                outputs.append(dispatch(w, s, s, x1, relu2: relu2, rps: 4))
            }
        }
        outputs += NemotronTargetQKV.warmNarrow()
        eval(outputs)
    }
}

/// One-row twin route for M = 1 quantized projections (the draft step): the
/// multi-row qmv at R = 1 repeats the stock qmv / qmv_fast per-row body at the
/// stock four rows per SIMD group, so each output is bitwise the stock one.
/// It takes the twin's R = 1 decode form (`r1HalfDecodeForm`).
func nemotronR1Linear(_ x: MLXArray, _ linear: Linear) -> MLXArray {
    guard NemotronMultiRowQMV.r1Route, x.size == x.dim(-1), x.dtype == .bfloat16,
          type(of: linear) == QuantizedLinear.self, let q = linear as? QuantizedLinear,
          q.bias == nil, let y = NemotronMultiRowQMV.apply(x.reshaped(1, x.dim(-1)), q)
    else { return linear(x) }
    var shape = x.shape
    shape[shape.count - 1] = q.weight.dim(0)
    let out = y.reshaped(shape)
    guard NemotronMoELeader.selfTest else { return out }
    return nemotronBitwiseOrNaN(out, linear(x))
}

extension NemotronMultiRowQMV {
    /// One row of `q` through the R = 1 twin `dispatch` selects, reading
    /// (scale, bias) through `dict` (the tensor's `NemotronSBDict`), at the
    /// stock four outputs per SIMD group. nil when the shape is not the stock
    /// qmv / qmv_fast one.
    static func dictRow(_ x: MLXArray, _ q: QuantizedLinear, dict: NemotronSBDict.Dict) -> MLXArray? {
        guard enabled, draftDictAvailable, x.ndim == 2, x.dim(0) == 1, x.dtype == .bfloat16,
              q.mode == .affine, q.bits == 4, q.groupSize == 64, q.bias == nil,
              let biases = q.biases, q.scales.dtype == x.dtype, biases.dtype == x.dtype,
              q.weight.ndim == 2, q.weight.dtype == .uint32,
              dict.index.shape == q.scales.shape, dict.index.dtype == .uint16,
              dict.table.dtype == .float32
        else { return nil }
        let k = x.dim(1), n = q.weight.dim(0)
        guard k % 64 == 0, k > 128, n % 8 == 0, q.weight.dim(1) * 8 == k else { return nil }
        return dispatch(q.weight, q.scales, biases, x, relu2: false, rps: 4, dict: dict)
    }
}

/// Seed target head only: move each lane's input widening, three nibble
/// rescalings per four values and ordered partial sum out of the 16,384
/// output threadgroups. One 336-thread preparation produces the same lane
/// operands once; the R = 1 HF2 dictionary QMV consumes them in the original
/// K order at the original four outputs / SIMD group and two SIMD groups / TG.
/// No activation cache: the packed array is a local lazy graph dependency.
enum NemotronSeedHeadPrepared {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_SEED_HEAD_PREPARED"] != "0"
    private static let kernel = nemotronSBDictSource(
        NemotronHeadPreparedActivation.preparedHalfSource).map {
        MLXFast.metalKernel(
            name: "nemotron_seed_head_prepared_r1_hf2_sbd",
            inputNames: ["w", "scales", "biases", "prepared", "K", "N"],
            outputNames: ["y"], source: $0,
            header: NemotronHeadPreparedActivation.preparedHalfHeader)
    }

    static func launch(_ x: MLXArray, _ q: QuantizedLinear,
                       dict: NemotronSBDict.Dict) -> MLXArray? {
        guard enabled, Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              x.shape == [1, 1, 2688], x.dtype == .bfloat16,
              type(of: q) == QuantizedLinear.self, q.bias == nil,
              q.mode == .affine, q.bits == 4, q.groupSize == 64,
              q.weight.dtype == .uint32, q.weight.shape == [131072, 336],
              q.scales.dtype == .bfloat16, q.scales.shape == [131072, 42],
              let biases = q.biases, biases.dtype == .bfloat16, biases.shape == q.scales.shape,
              dict.index.dtype == .uint16, dict.index.shape == q.scales.shape,
              dict.table.dtype == .float32, let kernel
        else { return nil }
        let packed = NemotronHeadPreparedActivation.prepareContiguousRows(x.reshaped(1, 2688), pv: 8)
        let args = NemotronMultiRowQMV.scalarPair(2688, 131072)
        return kernel([q.weight, dict.index, dict.table, packed, args.0, args.1],
            template: [("T", DType.bfloat16), ("R", 1), ("FAST", false),
                       ("RELU2", false), ("RPS", 4), ("HF", 2)],
            grid: (32, 131072 / 4, 1), threadGroup: (32, 2, 1),
            outputShapes: [[1, 1, 131072]], outputDTypes: [.bfloat16])[0]
    }
}

/// Seed only (request prefill, `NemotronHPrefillX5.active`): the target head's
/// one-row logits (`logits(last)` of the sampling prefill chunk) through the
/// R = 1 multi-row qmv twin reading the head's `NemotronSBDict` pair
/// dictionary (uint16 index + FP32 pair table, 11 MB) instead of the stock qmv
/// reading its BF16 scales and biases (22 MB). With prepared reuse enabled,
/// one small prepare precedes the projection and amortizes input arithmetic;
/// without it the imported route remains one projection. Both twins keep the
/// stock per-row body, lane split, block order and simd_sum, and the
/// table holds the exact FP32 widenings of the BF16 pairs. `qualify` compares
/// it with `head(x)` bit for bit on the loaded head; only then does the seed
/// use it. NEMOTRON_SEED_HEAD_DICT=0 keeps the stock call.
enum NemotronSeedHeadDict {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_SEED_HEAD_DICT"] != "0"
    nonisolated(unsafe) static var verified = false

    /// `head(x)` for x: [1, 1, K] inside the request prefill; nil otherwise.
    static func apply(_ x: MLXArray, _ head: Linear?) -> MLXArray? {
        guard verified, NemotronHPrefillX5.active, let head, x.ndim == 3, x.dim(0) == 1, x.dim(1) == 1
        else { return nil }
        guard let y = launch(x, head) else { return nil }
        guard NemotronMoELeader.selfTest else { return y }
        return nemotronBitwiseOrNaN(y, head(x))
    }

    private static func launch(_ x: MLXArray, _ head: Linear,
                               reusePrepared: Bool = true) -> MLXArray? {
        guard type(of: head) == QuantizedLinear.self, let q = head as? QuantizedLinear,
              let d = NemotronSBDict.dict(q) else { return nil }
        if reusePrepared && NemotronSeedHeadPrepared.enabled {
            return NemotronSeedHeadPrepared.launch(x, q, dict: d)
        }
        guard let y = NemotronMultiRowQMV.dictRow(x.reshaped(1, x.dim(-1)), q, dict: d)
        else { return nil }
        return y.reshaped(1, 1, q.weight.dim(0))
    }

    /// Load time, after `NemotronSBDict.prepare`: the head must have a pair
    /// dictionary in its scales' shape whose table decodes to the scales' and
    /// biases' raw BF16 words, and the twin must equal `head(x)` bit for bit
    /// on seeded rows of several scales.
    static func qualify(_ target: NemotronHModel) {
        verified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron seed head dict: \(message)\n".utf8))
        }
        guard enabled, NemotronSBDict.enabled else { return report("off (setting)") }
        var checked = 0
        defer {
            if NemotronSeedHeadPrepared.enabled {
                precondition(verified && checked == 6,
                    "seed prepared R1 head qualification refused; no unchanged-fallback score")
            }
        }
        guard let head = target.lmHead, type(of: head) == QuantizedLinear.self,
              let q = head as? QuantizedLinear, let biases = q.biases
        else { return report("off (no quantized head)") }
        guard let d = NemotronSBDict.dict(q), d.index.shape == q.scales.shape,
              d.index.dtype == .uint16, d.table.dtype == .float32
        else { return report("off (no pair dictionary for the head)") }
        do {
            let matches = try withError { () -> Bool in
                // Raw words: every group's table entry is the head's (scale, bias) BF16 pair.
                let pairs = take(d.table, d.index.reshaped(-1).asType(.int32), axis: 0)
                let words = stacked([q.scales.reshaped(-1).asType(.float32),
                                     biases.reshaped(-1).asType(.float32)], axis: 1)
                guard (pairs.view(dtype: .uint32) .== words.view(dtype: .uint32)).all().item(Bool.self)
                else { return false }
                let width = q.weight.dim(1) * 8
                for (index, scale) in [Float(1), 0.25, 4, 16, 1, 0.05].enumerated() {
                    let x = MLXRandom.normal([1, 1, width], scale: scale,
                                             key: MLXRandom.key(UInt64(0x5eed_d1c7 + index)))
                        .asType(.bfloat16)
                    guard let twin = launch(x, head) else { return false }
                    let stock = head(x)
                    guard let raw = launch(x, head, reusePrepared: false),
                          twin.shape == stock.shape, raw.shape == stock.shape,
                          (raw.view(dtype: .uint16) .== stock.view(dtype: .uint16)).all().item(Bool.self),
                          (twin.view(dtype: .uint16) .== stock.view(dtype: .uint16)).all().item(Bool.self)
                    else { return false }
                    checked += 1
                }
                return true
            }
            verified = matches
        } catch {
            report("off (\(error))")
        }
        report(verified ? "on (dictionary raw words, raw R1 and prepared R1 equal to stock, \(checked) cases)"
                        : "off (differs from the stock head after \(checked) cases)")
    }
}

/// Self-test select: `y` where it is bitwise equal to `expected`, else NaN
/// everywhere, so the token gate fails loudly on any difference.
func nemotronBitwiseOrNaN(_ y: MLXArray, _ expected: MLXArray) -> MLXArray {
    let ok = (y.view(dtype: .uint16) .== expected.view(dtype: .uint16)).all()
    return which(ok, y, MLXArray(Float.nan).asType(y.dtype))
}

// Row-pointer twin of the multi-row qmv: row r reads its activations from its
// own buffer x<r> instead of from x + r * K. The body is the same source with
// only the three activation address expressions replaced, so every load,
// coefficient decode, product, K-order and simd_sum is unchanged; only where
// the row's K values live moves. It lets the MTP verify's o_proj read the R
// per-query SDPA outputs in place, without the axis-2 concatenate (R copies)
// and the token-major transpose copy that build the [R, K] input today.
enum NemotronMultiRowQMVRowInputs {
    private static func source(rows: Int) -> String {
        var s = nemotronMultiRowQMVSource
        func swap(_ old: String, _ new: String, count: Int) {
            precondition(s.components(separatedBy: old).count - 1 == count,
                         "multirow qmv row-pointer source: \(old)")
            s = s.replacingOccurrences(of: old, with: new)
        }
        let names = (0..<rows).map { "x\($0)" }.joined(separator: ", ")
        swap("const device T* xp = x + simd_lid * values_per_thread;",
             "const device T* xr[\(rows)] = {\(names)};\n  int xo = simd_lid * values_per_thread;",
             count: 1)
        swap("xp + r * in_vec_size", "xr[r] + xo", count: 2)
        swap("xp += block_size;", "xo += block_size;", count: 1)
        precondition(s.range(of: "\\bxp\\b", options: .regularExpression) == nil,
                     "multirow qmv row-pointer source: stray xp")
        return s
    }

    nonisolated(unsafe) private static var kernels: [Int: MLXFast.MLXFastKernel] = [:]
    private static let lock = NSLock()
    private static func kernel(rows: Int) -> MLXFast.MLXFastKernel {
        lock.lock(); defer { lock.unlock() }
        if let kernel = kernels[rows] { return kernel }
        let kernel = MLXFast.metalKernel(
            name: "nemotron_multirow_qmv_affine4_gs64_rowptr\(rows)",
            inputNames: ["w", "scales", "biases"] + (0..<rows).map { "x\($0)" } + ["K", "N"],
            outputNames: ["y"],
            source: source(rows: rows),
            header: nemotronMultiRowQMVHeader)
        kernels[rows] = kernel
        return kernel
    }

    /// rows: R row-contiguous activations of K elements each (any shape).
    /// Returns [R, N], bitwise equal to NemotronMultiRowQMV.apply on the
    /// [R, K] stack of the same rows, or nil wherever that call would not
    /// take the multi-row kernel.
    /// `prepared`: the rows' prepared activation when their producer wrote
    /// it (the verify SDPA launch), used where the rows would be prepared.
    static func apply(_ rows: [MLXArray], _ q: QuantizedLinear, prepared: MLXArray? = nil) -> MLXArray? {
        guard NemotronMultiRowQMV.enabled, NemotronMTPExecution.batchedM1,
              (2...8).contains(rows.count),
              q.mode == .affine, q.bits == 4, q.groupSize == 64,
              let biases = q.biases, q.weight.ndim == 2, q.weight.dtype == .uint32
        else { return nil }
        let k = q.weight.dim(1) * 8, n = q.weight.dim(0)
        guard k % 64 == 0, k > 128, n % 8 == 0,
              q.scales.dtype == .bfloat16, biases.dtype == .bfloat16,
              rows.allSatisfy({ $0.dtype == .bfloat16 && $0.size == k })
        else { return nil }
        if NemotronMultiRowQMV.narrowEnabled, rows.count == 3, n >= 2688 {
            if let prepared, k == 4096, prepared.shape == [3, 8, 17 * 32], prepared.dtype == .float32 {
                return NemotronHeadPreparedActivation.projectPrepared(prepared, q)
            }
            return NemotronHeadPreparedActivation.applyRows(rows, q)
        }
        let (kArray, nArray) = NemotronMultiRowQMV.scalarPair(k, n)
        // The same outputs-per-SIMD-group the stacked call picks for (R, K, N).
        let rps = NemotronMultiRowQMV.resultsPerSimdgroup(rows: rows.count, k: k, n: n)
        return kernel(rows: rows.count)(
            [q.weight, q.scales, biases] + rows + [kArray, nArray],
            template: [("T", DType.bfloat16), ("R", rows.count), ("FAST", k % 512 == 0),
                       ("RELU2", false), ("RPS", rps)],
            grid: (32, n / rps, 1),
            threadGroup: (32, 2, 1),
            outputShapes: [[rows.count, n]],
            outputDTypes: [DType.bfloat16])[0]
    }
}

/// MTP verify attention output projection over the serialized per-query
/// attention outputs. NEMOTRON_MTP_ATTN_DIRECT=0 restores the concatenate +
/// transpose copy + multi-row qmv.
enum NemotronMTPAttentionDirect {
    static let enabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_ATTN_DIRECT"] != "0"

    /// rows: L outputs [1, heads, 1, headDim] in query order. Returns
    /// [1, L, N], bitwise equal to the original
    /// nemotronMTPLinearRows(concat(rows, axis 2) -> [1, L, heads * headDim], wo).
    static func project(_ rows: [MLXArray], _ wo: Linear) -> MLXArray {
        let length = rows.count
        func original() -> MLXArray {
            let output = length == 1 ? rows[0] : concatenated(rows, axis: 2)
            let merged = output.transposed(0, 2, 1, 3).reshaped(1, length, -1)
            return nemotronMTPLinearRows(merged, wo)
        }
        guard let q = wo as? QuantizedLinear,
              rows.allSatisfy({ $0.ndim == 4 && $0.dim(0) == 1 && $0.dim(2) == 1 }),
              let direct = NemotronMultiRowQMVRowInputs.apply(
                  rows, q, prepared: NemotronMTPVerifySDPA.takePrepared(rows))
        else { return original() }
        var y = direct.reshaped(1, length, q.weight.dim(0))
        if let bias = q.bias { y = y + bias }
        if NemotronMoELeader.selfTest { y = nemotronBitwiseOrNaN(y, original()) }
        return y
    }

    /// Compile the row-pointer pipelines for every verify width (R = 2...4)
    /// on each distinct attention o_proj shape, at load, with the RPS each
    /// width uses.
    static func warm(_ model: Module) {
        guard enabled else { return }
        var outputs: [MLXArray] = []
        var seen = Set<[Int]>()
        for module in model.modules() {
            guard let attention = module as? NemotronHAttention,
                  let q = attention.wo as? QuantizedLinear else { continue }
            let k = q.weight.dim(1) * 8, n = q.weight.dim(0)
            for rows in 2...4 {
                guard seen.insert([rows, k, n]).inserted else { continue }
                let xs = (0..<rows).map { _ in
                    MLXArray.zeros([1, attention.numHeads, 1, k / attention.numHeads],
                                   dtype: .bfloat16)
                }
                if let y = NemotronMultiRowQMVRowInputs.apply(xs, q) { outputs.append(y) }
            }
        }
        if !outputs.isEmpty { eval(outputs) }
    }
}

// MTP verify SDPA over the R serialized query rows in one launch. The verify
// attends query row r alone over keys [0, history + r + 1) with one one-pass
// sdpa_vector launch of 32 threadgroups per row; this is one grid of 32 x R
// threadgroups with z = row. A prologue sets row r's query and output pointers
// and key length; below it the text is sdpa_vector's (kernels/sdpa_vector.h)
// with its function constants folded to the stock call's values (no mask, no
// sinks, not causal), so each (head, row) threadgroup runs the stock program
// on the same inputs: 1024 threads, lane-to-element map, key order per
// simdgroup, simd_sum, fast::exp and the cross-simdgroup reduction. K/V and
// the queries are read in place through their strides (cache views, no copy).
// MLX JIT-compiles custom kernels with MathMode::Safe; the metallib is built
// with -fno-fast-math. Used only after the load-time bit check in `qualify`.
private let nemotronMTPVerifySDPASource = """
  // Verify row r = tid.z of the tpg.z serialized query rows: query row r,
  // keys [0, N) with N = keys_shape[2] - (tpg.z - 1 - r), output row r.
  // Below this prologue the text is sdpa_vector's, with its function
  // constants folded to the stock verify call's values (no mask, no sinks,
  // not causal) and its scalar arguments taken from the inputs.
  constexpr bool query_transposed = false;
  constexpr bool do_causal = false;
  constexpr bool bool_mask = false;
  constexpr bool float_mask = false;
  constexpr bool has_sinks = false;
  const device bool* bmask = nullptr;
  const device T* fmask = nullptr;
  constexpr int mask_kv_seq_stride = 0;
  constexpr int mask_q_seq_stride = 0;
  constexpr int mask_head_stride = 0;
  const device T* sinks = nullptr;
  constexpr int num_q_heads = 1;
  const uint3 tid = threadgroup_position_in_grid;
  const uint3 tpg = threadgroups_per_grid;
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const uint simd_lid = thread_index_in_simdgroup;
  const int gqa_factor = int(tpg.x) / keys_shape[1];
  const int N = keys_shape[2] - int(tpg.z - tid.z) + 1;
  const size_t k_head_stride = keys_strides[1];
  const size_t k_seq_stride = keys_strides[2];
  const size_t v_head_stride = values_strides[1];
  const size_t v_seq_stride = values_strides[2];
  queries += tid.z * queries_strides[2];
  out += tid.z * tpg.x * V;
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  int inner_k_stride = BN * int(k_seq_stride);
  int inner_v_stride = BN * int(v_seq_stride);

  typedef float U;

  thread U q[qk_per_thread];
  thread U k[qk_per_thread];
  thread U o[v_per_thread];

  threadgroup U outputs[BN * BD];
  threadgroup U max_scores[BN];
  threadgroup U sum_exp_scores[BN];

  // Adjust positions
  const int q_batch_head_idx = tid.x;
  const int q_seq_idx = tid.y;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;
  const int o_offset = q_batch_head_idx * tpg.y + q_seq_idx;
  const int q_offset =
      query_transposed ? tpg.x * q_seq_idx + q_batch_head_idx : o_offset;
  queries += q_offset * D + simd_lid * qk_per_thread;
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
      simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
      simd_lid * v_per_thread;
  if (bool_mask) {
    bmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }
  if (float_mask) {
    fmask += q_batch_head_idx * mask_head_stride +
        simd_gid * mask_kv_seq_stride + q_seq_idx * mask_q_seq_stride;
  }

  out += o_offset * V + simd_gid * v_per_thread;

  // Read the query and 0 the output accumulator
  for (int i = 0; i < qk_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = Limits<U>::finite_min;
  U sum_exp_score = 0;
  if (has_sinks && simd_gid == 0) {
    max_score = static_cast<U>(sinks[q_batch_head_idx % num_q_heads]);
    sum_exp_score = 1;
  }

  // For each key
  for (int i = simd_gid; i < N; i += BN) {
    bool use_key = true;
    if (do_causal) {
      use_key = i <= (N - int(tpg.y) + int(q_seq_idx));
    } else if (bool_mask) {
      use_key = bmask[0];
    } else if (float_mask) {
      use_key = (fmask[0] >= Limits<T>::finite_min);
    }
    if (use_key) {
      // Read the key
      for (int j = 0; j < qk_per_thread; j++) {
        k[j] = keys[j];
      }

      // Compute the i-th score
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);
      if (float_mask) {
        score += static_cast<U>(fmask[0]);
      }

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * values[j];
      }
    }

    // Move the pointers to the next kv
    keys += inner_k_stride;
    values += inner_v_stride;
    if (bool_mask) {
      bmask += BN * mask_kv_seq_stride;
    }
    if (float_mask) {
      fmask += BN * mask_kv_seq_stride;
    }
  }

  // Each thread has a partial part of the output so we need to combine them.

  // First let's communicate the max and sum_exp
  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

  // Now we need to aggregate all the outputs
  for (int i = 0; i < v_per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < v_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
"""

// The lab's unroll4 key loop (patches/sdpa-unroll4.patch) for the unmasked
// call: K and V of keys i, i+32, i+64, i+96 load first, then the four scores
// and online-softmax updates run in key order with the stock statements; the
// remainder runs the stock loop. Only load issue moves.
private let nemotronMTPVerifySDPAUnroll4Loop = """
  int i = simd_gid;
  if (!do_causal && !bool_mask && !float_mask) {
    // Four keys per step: issue their K and V loads first, then apply the
    // per-key updates in the original key order.
    for (; i + 3 * BN < N; i += 4 * BN) {
      U kk[4][qk_per_thread];
      U vv[4][v_per_thread];
      for (int u = 0; u < 4; u++) {
        for (int j = 0; j < qk_per_thread; j++) {
          kk[u][j] = keys[u * inner_k_stride + j];
        }
        for (int j = 0; j < v_per_thread; j++) {
          vv[u][j] = values[u * inner_v_stride + j];
        }
      }
      U key_scores[4];
      for (int u = 0; u < 4; u++) {
        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) {
          score += q[j] * kk[u][j];
        }
        key_scores[u] = simd_sum(score);
      }
      for (int u = 0; u < 4; u++) {
        U new_max = max(max_score, key_scores[u]);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(key_scores[u] - new_max);

        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        for (int j = 0; j < v_per_thread; j++) {
          o[j] = o[j] * factor + exp_score * vv[u][j];
        }
      }
      keys += 4 * inner_k_stride;
      values += 4 * inner_v_stride;
    }
  }
  for (; i < N; i += BN) {
"""

// The R = 3 prepared verify launch with one threadgroup per query head
// instead of one per head and row: grid 32 x 1 x 1 of 1024 threads. SIMD
// group g holds the three rows' query slices and online-softmax states and
// runs, for each row, the statements the row launch's SIMD group g runs on
// the same keys (g, g + 32, ... below that row's key count) in the same
// order; each key's K and V are read once for the three rows. The
// aggregation forms every output with the row launch's simd_max / exp /
// simd_sum operations on the same operands in the same lanes, through a
// padded threadgroup transpose in two halves, and the prepared epilogue
// writes the same values. Used only after `NemotronMTPVerifySDPA.qualifyFast`
// found it bitwise equal to the row launch.
private let nemotronMTPVerifySDPAFastSource = """
  // One threadgroup per query head (tid.x) for the R = 3 serialized rows.
  // SIMD group g keeps, for every row r, the online-softmax state the
  // reference row launch keeps in its SIMD group g: keys g, g + 32, ...
  // below N_r = keys_shape[2] - (R - 1 - r), in key order, with the same
  // statements. Each key's K and V are read once for the three rows.
  constexpr int R = 3;
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int qk_per_thread = D / BD;
  constexpr int v_per_thread = V / BD;
  typedef float U;
  const uint3 tid = threadgroup_position_in_grid;
  const uint3 tpg = threadgroups_per_grid;
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const uint simd_lid = thread_index_in_simdgroup;
  const int gqa_factor = int(tpg.x) / keys_shape[1];
  const int M = keys_shape[2];
  const size_t k_head_stride = keys_strides[1];
  const size_t k_seq_stride = keys_strides[2];
  const size_t v_head_stride = values_strides[1];
  const size_t v_seq_stride = values_strides[2];
  int inner_k_stride = BN * int(k_seq_stride);
  int inner_v_stride = BN * int(v_seq_stride);
  const int q_batch_head_idx = tid.x;
  const int kv_head_idx = q_batch_head_idx / gqa_factor;

  thread U q[R][qk_per_thread];
  thread U o[R][v_per_thread];
  U max_score[R];
  U sum_exp_score[R];
  for (int r = 0; r < R; r++) {
    const device T* qr = queries + r * queries_strides[2] + q_batch_head_idx * D + simd_lid * qk_per_thread;
    for (int i = 0; i < qk_per_thread; i++) {
      q[r][i] = static_cast<U>(scale) * qr[i];
    }
    for (int i = 0; i < v_per_thread; i++) {
      o[r][i] = 0;
    }
    max_score[r] = Limits<U>::finite_min;
    sum_exp_score[r] = 0;
  }
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride + simd_lid * qk_per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride + simd_lid * v_per_thread;

  // Keys i and i + 32 valid for every row: both load first, then each
  // row's two scores and two updates in key order.
  int i = simd_gid;
  for (; i + BN < M - (R - 1); i += 2 * BN) {
    U kk[2][qk_per_thread];
    U vv[2][v_per_thread];
    for (int u = 0; u < 2; u++) {
      for (int j = 0; j < qk_per_thread; j++) {
        kk[u][j] = keys[u * inner_k_stride + j];
      }
      for (int j = 0; j < v_per_thread; j++) {
        vv[u][j] = values[u * inner_v_stride + j];
      }
    }
    for (int r = 0; r < R; r++) {
      U key_scores[2];
      for (int u = 0; u < 2; u++) {
        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) {
          score += q[r][j] * kk[u][j];
        }
        key_scores[u] = simd_sum(score);
      }
      for (int u = 0; u < 2; u++) {
        U new_max = max(max_score[r], key_scores[u]);
        U factor = fast::exp(max_score[r] - new_max);
        U exp_score = fast::exp(key_scores[u] - new_max);

        max_score[r] = new_max;
        sum_exp_score[r] = sum_exp_score[r] * factor + exp_score;

        for (int j = 0; j < v_per_thread; j++) {
          o[r][j] = o[r][j] * factor + exp_score * vv[u][j];
        }
      }
    }
    keys += 2 * inner_k_stride;
    values += 2 * inner_v_stride;
  }
  // The remaining keys one at a time, each row only below its N_r.
  for (; i < M; i += BN) {
    U k[qk_per_thread];
    U v[v_per_thread];
    for (int j = 0; j < qk_per_thread; j++) {
      k[j] = keys[j];
    }
    for (int j = 0; j < v_per_thread; j++) {
      v[j] = values[j];
    }
    for (int r = 0; r < R; r++) {
      if (i < M - (R - 1 - r)) {
        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) {
          score += q[r][j] * k[j];
        }
        score = simd_sum(score);

        U new_max = max(max_score[r], score);
        U factor = fast::exp(max_score[r] - new_max);
        U exp_score = fast::exp(score - new_max);

        max_score[r] = new_max;
        sum_exp_score[r] = sum_exp_score[r] * factor + exp_score;

        for (int j = 0; j < v_per_thread; j++) {
          o[r][j] = o[r][j] * factor + exp_score * v[j];
        }
      }
    }
    keys += inner_k_stride;
    values += inner_v_stride;
  }

  // The rows' aggregation in two halves: outputs 0 and 1 of the three
  // rows, then outputs 2 and 3 (slot (r, i) of 32 rows of 33 floats).
  // Every output is the same simd_sum of the same products in the same
  // lanes; the exp-sum division then runs once per lane (lane l < 12: row
  // l / 4, output l % 4) and lanes 0 ... 11 store.
  constexpr int PITCH = BD + 1;
  threadgroup U outputs[R * 2 * BN * PITCH];
  threadgroup U max_scores[R * BN];
  threadgroup U sum_exp_scores[R * BN];
  if (simd_lid == 0) {
    for (int r = 0; r < R; r++) {
      max_scores[r * BN + simd_gid] = max_score[r];
      sum_exp_scores[r * BN + simd_gid] = sum_exp_score[r];
    }
  }
  for (int r = 0; r < R; r++) {
    for (int i = 0; i < 2; i++) {
      outputs[((r * 2 + i) * BN + simd_lid) * PITCH + simd_gid] = o[r][i];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  U factor[R];
  U sum[R];
  U res[R][v_per_thread];
  for (int r = 0; r < R; r++) {
    U ms = max_scores[r * BN + simd_lid];
    U new_max = simd_max(ms);
    factor[r] = fast::exp(ms - new_max);
    sum[r] = simd_sum(sum_exp_scores[r * BN + simd_lid] * factor[r]);
  }
  for (int r = 0; r < R; r++) {
    for (int i = 0; i < 2; i++) {
      res[r][i] = simd_sum(outputs[((r * 2 + i) * BN + simd_gid) * PITCH + simd_lid] * factor[r]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int r = 0; r < R; r++) {
    for (int i = 0; i < 2; i++) {
      outputs[((r * 2 + i) * BN + simd_lid) * PITCH + simd_gid] = o[r][i + 2];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int r = 0; r < R; r++) {
    for (int i = 0; i < 2; i++) {
      res[r][i + 2] = simd_sum(outputs[((r * 2 + i) * BN + simd_gid) * PITCH + simd_lid] * factor[r]);
    }
  }
  const int lr = min(int(simd_lid) / 4, R - 1);
  const int li = int(simd_lid) % 4;
  U x = res[0][0];
  U s = sum[0];
  for (int r = 0; r < R; r++) {
    for (int i = 0; i < v_per_thread; i++) {
      x = (lr == r && li == i) ? res[r][i] : x;
    }
    s = lr == r ? sum[r] : s;
  }
  x = s == 0 ? x : (x / s);
  if (int(simd_lid) < 4 * R) {
    out[(lr * int(tpg.x) + q_batch_head_idx) * V + simd_gid * v_per_thread + li] = static_cast<T>(x);
  }

  // The o_proj's prepared activation (PV 16) of the three rows: lane l
  // < 12 writes its scaled output, lane 4r the SIMD group's part of row r.
  threadgroup U prepared_parts[R * BN];
  const int prepared_tiles = (int(tpg.x) * V / 16 + 31) / 32;
  {
    const U xb = U(static_cast<T>(x));
    const U x1 = simd_shuffle_down(xb, 1);
    const U x2 = simd_shuffle_down(xb, 2);
    const U x3 = simd_shuffle_down(xb, 3);
    if (int(simd_lid) < 4 * R) {
      const int chunk = int(tid.x) * (V / 16) + int(simd_gid) / 4;
      const int base = (lr * prepared_tiles + chunk / 32) * 17 * 32 + chunk % 32;
      const int p = (int(simd_gid) % 4) * 4;
      prepared[base + (p + li) * 32] =
          li == 0 ? xb : (li == 1 ? xb / 16.0f : (li == 2 ? xb / 256.0f : xb / 4096.0f));
      if (li == 0) {
        prepared_parts[lr * BN + simd_gid] = xb + x1 + x2 + x3;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0 && int(simd_lid) < R * (V / 16)) {
    const int r = int(simd_lid) / (V / 16);
    const int c = int(simd_lid) % (V / 16);
    const int chunk = int(tid.x) * (V / 16) + c;
    const int base = (r * prepared_tiles + chunk / 32) * 17 * 32 + chunk % 32;
    U total = 0;
    total += prepared_parts[r * BN + 4 * c];
    total += prepared_parts[r * BN + 4 * c + 1];
    total += prepared_parts[r * BN + 4 * c + 2];
    total += prepared_parts[r * BN + 4 * c + 3];
    prepared[base + 16 * 32] = total;
  }
"""

enum NemotronMTPVerifySDPA {
    /// NEMOTRON_MTP_VERIFY_SDPA_MERGED=0 keeps the per-row stock launches;
    /// NEMOTRON_MTP_VERIFY_SDPA_UNROLL4=0 keeps the stock key loop in the
    /// merged launch.
    static let enabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_VERIFY_SDPA_MERGED"] != "0"
    static let unroll4 =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_VERIFY_SDPA_UNROLL4"] != "0"
    nonisolated(unsafe) private static var verified = false

    private static func source() -> String {
        guard unroll4 else { return nemotronMTPVerifySDPASource }
        let loop = "  for (int i = simd_gid; i < N; i += BN) {"
        precondition(nemotronMTPVerifySDPASource.components(separatedBy: loop).count == 2,
                     "verify sdpa source: key loop")
        return nemotronMTPVerifySDPASource.replacingOccurrences(
            of: loop, with: nemotronMTPVerifySDPAUnroll4Loop)
    }
    private static let kernel = MLXFast.metalKernel(
        name: unroll4 ? "nemotron_mtp_verify_sdpa_rows_unroll4" : "nemotron_mtp_verify_sdpa_rows",
        inputNames: ["queries", "keys", "values", "scale"], outputNames: ["out"],
        source: source(), ensureRowContiguous: false)

    /// NEMOTRON_MTP_VERIFY_SDPA_ONE_BARRIER=0 keeps sdpa_vector's aggregation (one barrier for the maxima and
    /// exp-sums, then two per output value). Otherwise every SIMD group stores its maximum, its exp-sum and
    /// its four partial outputs (output i in threadgroup slice i) before one barrier, and after it forms the
    /// same factor, exp-sum and four output sums from the same operands in the same lane order (the slices
    /// replace the reuse of one slice, so the later barriers are not needed). Used only after
    /// `qualifyOneBarrier` found both launches bitwise equal to the aggregation they replace.
    static let oneBarrier =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_VERIFY_SDPA_ONE_BARRIER"] != "0"
    nonisolated(unsafe) private static var oneBarrierVerified = false
    private static func oneBarrierSource(_ s: String) -> String? {
        let scratch = "  threadgroup U outputs[BN * BD];\n"
        let block = """
          // First let's communicate the max and sum_exp
          if (simd_lid == 0) {
            max_scores[simd_gid] = max_score;
            sum_exp_scores[simd_gid] = sum_exp_score;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          max_score = max_scores[simd_lid];
          U new_max = simd_max(max_score);
          U factor = fast::exp(max_score - new_max);
          sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

          // Now we need to aggregate all the outputs
          for (int i = 0; i < v_per_thread; i++) {
            outputs[simd_lid * BD + simd_gid] = o[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
            o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
            threadgroup_barrier(mem_flags::mem_threadgroup);
          }

        """
        guard oneBarrier, s.components(separatedBy: scratch).count == 2,
              s.components(separatedBy: block).count == 2 else { return nil }
        return s.replacingOccurrences(of: scratch, with: "  threadgroup U outputs[v_per_thread * BN * BD];\n")
            .replacingOccurrences(of: block, with: """
              // One barrier: the maxima, the exp-sums and the four partial
              // outputs (output i in slice i) are stored first.
              if (simd_lid == 0) {
                max_scores[simd_gid] = max_score;
                sum_exp_scores[simd_gid] = sum_exp_score;
              }
              for (int i = 0; i < v_per_thread; i++) {
                outputs[i * (BN * BD) + simd_lid * BD + simd_gid] = o[i];
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              max_score = max_scores[simd_lid];
              U new_max = simd_max(max_score);
              U factor = fast::exp(max_score - new_max);
              sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);
              for (int i = 0; i < v_per_thread; i++) {
                o[i] = simd_sum(outputs[i * (BN * BD) + simd_gid * BD + simd_lid] * factor);
                o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
              }

            """)
    }
    private static let oneBarrierKernel = oneBarrierSource(source()).map {
        MLXFast.metalKernel(
            name: (unroll4 ? "nemotron_mtp_verify_sdpa_rows_unroll4" : "nemotron_mtp_verify_sdpa_rows") + "_1b",
            inputNames: ["queries", "keys", "values", "scale"], outputNames: ["out"],
            source: $0, ensureRowContiguous: false)
    }

    /// The launch above whose epilogue also writes the o_proj's prepared
    /// activation of the R rows (`NemotronHeadPreparedActivation.packRows`,
    /// K = 4096, PV = 16): lane 0 of each SIMD group takes its four BF16
    /// outputs just stored, one prepare group, and writes the group's scaled
    /// values; then one lane per 16-value chunk adds the chunk's four group
    /// sums in the prepare kernel's order. The separate prepare launch is not
    /// needed. Used only after `qualify` found both outputs bitwise equal.
    /// NEMOTRON_MTP_VERIFY_SDPA_PREPARED=0 keeps the prepare launch.
    static let preparedEnabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_VERIFY_SDPA_PREPARED"] != "0"
    nonisolated(unsafe) private static var preparedVerified = false
    private static let preparedEpilogue = """

  // The o_proj's prepared activation (PV 16) of this head's outputs.
  threadgroup U prepared_parts[BN];
  const int prepared_tiles = (int(tpg.x) * V / 16 + 31) / 32;
  if (simd_lid == 0) {
    const U x0 = U(static_cast<T>(o[0]));
    const U x1 = U(static_cast<T>(o[1]));
    const U x2 = U(static_cast<T>(o[2]));
    const U x3 = U(static_cast<T>(o[3]));
    prepared_parts[simd_gid] = x0 + x1 + x2 + x3;
    const int chunk = int(tid.x) * (V / 16) + int(simd_gid) / 4;
    const int base = (int(tid.z) * prepared_tiles + chunk / 32) * 17 * 32 + chunk % 32;
    const int p = (int(simd_gid) % 4) * 4;
    prepared[base + p * 32] = x0;
    prepared[base + (p + 1) * 32] = x1 / 16.0f;
    prepared[base + (p + 2) * 32] = x2 / 256.0f;
    prepared[base + (p + 3) * 32] = x3 / 4096.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0 && simd_lid < V / 16) {
    const int chunk = int(tid.x) * (V / 16) + int(simd_lid);
    const int base = (int(tid.z) * prepared_tiles + chunk / 32) * 17 * 32 + chunk % 32;
    U total = 0;
    total += prepared_parts[4 * simd_lid];
    total += prepared_parts[4 * simd_lid + 1];
    total += prepared_parts[4 * simd_lid + 2];
    total += prepared_parts[4 * simd_lid + 3];
    prepared[base + 16 * 32] = total;
  }
"""
    private static let preparedKernel = MLXFast.metalKernel(
        name: (unroll4 ? "nemotron_mtp_verify_sdpa_rows_unroll4" : "nemotron_mtp_verify_sdpa_rows") + "_prepared",
        inputNames: ["queries", "keys", "values", "scale"], outputNames: ["out", "prepared"],
        source: source() + preparedEpilogue, ensureRowContiguous: false)
    private static let oneBarrierPreparedKernel = oneBarrierSource(source()).map {
        MLXFast.metalKernel(
            name: (unroll4 ? "nemotron_mtp_verify_sdpa_rows_unroll4" : "nemotron_mtp_verify_sdpa_rows")
                + "_prepared_1b",
            inputNames: ["queries", "keys", "values", "scale"], outputNames: ["out", "prepared"],
            source: $0 + preparedEpilogue, ensureRowContiguous: false)
    }

    /// NEMOTRON_MTP_VERIFY_SDPA_FAST=0 keeps the per-row R = 3 prepared launch (32 x 3 threadgroups). Otherwise
    /// that launch runs as `fastKernel`, one threadgroup per query head for the three rows (see
    /// `nemotronMTPVerifySDPAFastSource`). Used only after `qualifyFast` found its output rows and prepared
    /// activation bitwise equal to the launch it replaces.
    static let fastEnabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_VERIFY_SDPA_FAST"] != "0"
    nonisolated(unsafe) private static var fastVerified = false
    private static let fastKernel = MLXFast.metalKernel(
        name: "nemotron_mtp_verify_sdpa_heads_prepared",
        inputNames: ["queries", "keys", "values", "scale"], outputNames: ["out", "prepared"],
        source: nemotronMTPVerifySDPAFastSource, ensureRowContiguous: false)

    /// `mergedPrepared` as one launch of 32 threadgroups (one per query head): the same R = 3 shapes, the
    /// same outputs [3, 32, 128] and [3, 8, 17 * 32].
    private static func fastPrepared(
        _ queries: MLXArray, _ keys: MLXArray, _ values: MLXArray, _ scale: Float
    ) -> (MLXArray, MLXArray)? {
        guard enabled, Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              queries.ndim == 4, queries.dim(0) == 1, queries.dim(1) == 32, queries.dim(2) == 3,
              queries.dim(3) == 128,
              keys.ndim == 4, keys.dim(0) == 1, keys.dim(1) == 2, keys.dim(3) == 128,
              values.shape == keys.shape, (3 ..< 1024).contains(keys.dim(2)),
              queries.dtype == .bfloat16, keys.dtype == .bfloat16, values.dtype == .bfloat16
        else { return nil }
        let outputs = fastKernel(
            [queries, keys, values, scaleArray(scale)],
            template: [("T", DType.bfloat16), ("D", 128), ("V", 128)],
            grid: (32 * 1024, 1, 1), threadGroup: (1024, 1, 1),
            outputShapes: [[3, 32, 128], [3, 8, 17 * 32]], outputDTypes: [DType.bfloat16, .float32])
        return (outputs[0], outputs[1])
    }

    /// The last prepared activation `rows` built, keyed by its first row.
    nonisolated(unsafe) private static var pendingPrepared: (key: ObjectIdentifier, prepared: MLXArray)?

    /// The prepared activation `rows` built with these rows, once.
    static func takePrepared(_ rows: [MLXArray]) -> MLXArray? {
        guard let pending = pendingPrepared, rows.count == 3, pending.key == ObjectIdentifier(rows[0])
        else { return nil }
        pendingPrepared = nil
        return pending.prepared
    }

    nonisolated(unsafe) private static var scales: [UInt32: MLXArray] = [:]
    private static let lock = NSLock()
    private static func scaleArray(_ scale: Float) -> MLXArray {
        lock.lock(); defer { lock.unlock() }
        if let array = scales[scale.bitPattern] { return array }
        let array = MLXArray(scale)
        scales[scale.bitPattern] = array
        return array
    }

    /// queries [1, 32, R, 128], R = 2...4; keys/values the updated cache
    /// views [1, 2, history + R, 128]. Returns [R, 32, 128], or nil where the
    /// stock per-query calls would not all be one-pass sdpa_vector: MLX
    /// switches to the two-pass kernel from 1024 keys ('d'/'s' GPUs) or 4096
    /// (GQA elsewhere), so every row stays below 1024 keys.
    private static func merged(
        _ queries: MLXArray, _ keys: MLXArray, _ values: MLXArray, _ scale: Float,
        oneBarrier: Bool? = nil
    ) -> MLXArray? {
        let rows = queries.ndim == 4 ? queries.dim(2) : 0
        guard enabled, Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              queries.ndim == 4, queries.dim(0) == 1, queries.dim(1) == 32,
              (2...4).contains(rows), queries.dim(3) == 128,
              keys.ndim == 4, keys.dim(0) == 1, keys.dim(1) == 2, keys.dim(3) == 128,
              values.shape == keys.shape, (rows ..< 1024).contains(keys.dim(2)),
              queries.dtype == .bfloat16, keys.dtype == .bfloat16, values.dtype == .bfloat16
        else { return nil }
        let chosen = ((oneBarrier ?? oneBarrierVerified) ? oneBarrierKernel : nil) ?? kernel
        return chosen(
            [queries, keys, values, scaleArray(scale)],
            template: [("T", DType.bfloat16), ("D", 128), ("V", 128)],
            grid: (32 * 1024, 1, rows), threadGroup: (1024, 1, 1),
            outputShapes: [[rows, 32, 128]], outputDTypes: [DType.bfloat16])[0]
    }

    /// `merged` plus the R = 3 prepared activation from the same launch.
    private static func mergedPrepared(
        _ queries: MLXArray, _ keys: MLXArray, _ values: MLXArray, _ scale: Float,
        oneBarrier: Bool? = nil
    ) -> (MLXArray, MLXArray)? {
        guard queries.ndim == 4, queries.dim(2) == 3,
              merged(queries, keys, values, scale) != nil else { return nil }
        let chosen = ((oneBarrier ?? oneBarrierVerified) ? oneBarrierPreparedKernel : nil) ?? preparedKernel
        let outputs = chosen(
            [queries, keys, values, scaleArray(scale)],
            template: [("T", DType.bfloat16), ("D", 128), ("V", 128)],
            grid: (32 * 1024, 1, 3), threadGroup: (1024, 1, 1),
            outputShapes: [[3, 32, 128], [3, 8, 17 * 32]], outputDTypes: [DType.bfloat16, .float32])
        return (outputs[0], outputs[1])
    }

    /// The per-query calls `CBv2AttentionV1.queryBlockOutputs` makes for the
    /// serialized verify of a full-attention layer (block size 1, no sinks):
    /// row r attends keys [0, history + r + 1) with no mask.
    static func stock(
        _ queries: MLXArray, _ keys: MLXArray, _ values: MLXArray, _ scale: Float
    ) -> [MLXArray] {
        let rows = queries.dim(2), history = keys.dim(2) - rows
        return (0..<rows).map { r in
            MLXFast.scaledDotProductAttention(
                queries: queries[0..., 0..., r ..< (r + 1), 0...],
                keys: keys[0..., 0..., 0 ..< (history + r + 1), 0...],
                values: values[0..., 0..., 0 ..< (history + r + 1), 0...],
                scale: scale, mask: .none)
        }
    }

    /// Row r as [1, 32, 1, 128]: a slice view (an Int subscript would gather).
    private static func split(_ out: MLXArray) -> [MLXArray] {
        (0..<out.dim(0)).map { out[$0 ..< ($0 + 1)].reshaped(1, 32, 1, 128) }
    }

    /// `attendRows` for the serialized verify: the R outputs [1, 32, 1, 128]
    /// as views of the one launch's output, bitwise the stock calls'; nil
    /// until `qualify` passed or outside the shapes `merged` serves.
    static func rows(
        _ queries: MLXArray, _ keys: MLXArray, _ values: MLXArray, _ scale: Float
    ) -> [MLXArray]? {
        guard verified else { return nil }
        if preparedVerified,
           let (out, prepared) = (fastVerified && !perRowAdopted ? fastPrepared(queries, keys, values, scale) : nil)
               ?? mergedPrepared(queries, keys, values, scale) {
            let outputs = split(out)
            pendingPrepared = (ObjectIdentifier(outputs[0]), prepared)
            guard NemotronMoELeader.selfTest else { return outputs }
            return zip(outputs, stock(queries, keys, values, scale)).map { nemotronBitwiseOrNaN($0, $1) }
        }
        guard let out = merged(queries, keys, values, scale) else { return nil }
        let outputs = split(out)
        guard NemotronMoELeader.selfTest else { return outputs }
        return zip(outputs, stock(queries, keys, values, scale)).map { nemotronBitwiseOrNaN($0, $1) }
    }

    /// Load-time bit check on seeded random inputs against the stock
    /// per-query launches, at the target's attention scale: R = 2...4, key
    /// lengths across the 32-key and unroll boundaries up to 1023, cache
    /// capacity past the keys (head stride != N * 128), query/key scales
    /// 1 to 16. Engages only if every output is bitwise equal; an MLX error
    /// (a kernel that fails to build, or a threadgroup limit below 1024)
    /// keeps the stock launches. Also JITs the pipeline outside any request.
    static func qualify(_ target: Module) {
        verified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron mtp verify sdpa merged: \(message)\n".utf8))
        }
        guard enabled, NemotronMTPAttentionDirect.enabled else { return report("off (setting)") }
        var scaleBits = Set<UInt32>()
        for module in target.modules() {
            guard let attention = module as? NemotronHAttention, attention.numHeads == 32,
                  attention.numKeyValueHeads == 2, attention.headDim == 128 else { continue }
            scaleBits.insert(attention.scale.bitPattern)
        }
        guard !scaleBits.isEmpty else { return report("off (no 32 x 2 x 128 attention layer)") }
        let cases: [(rows: Int, histories: [Int])] = [
            (3, [0, 1, 29, 30, 31, 92, 93, 94, 125, 126, 127, 254, 488, 552, 616, 1020]),
            (2, [94, 95, 616, 1021]), (4, [92, 93, 616, 1019]),
        ]
        let inputScales: [(Float, Float)] = [(1, 1), (4, 4), (16, 8)]
        var checked = 0
        do {
            let matches = try withError { () -> Bool in
                for bits in scaleBits.sorted() {
                    let scale = Float(bitPattern: bits)
                    for (rows, histories) in cases {
                        for (i, history) in histories.enumerated() {
                            let seed = UInt64(0x5d9a_0000 + rows * 256 + i)
                            let (qScale, kvScale) = inputScales[i % inputScales.count]
                            let capacity = history + rows + 37
                            let q = MLXRandom.normal(
                                [1, rows, 4096], dtype: .bfloat16, scale: qScale,
                                key: MLXRandom.key(seed))
                            let k = MLXRandom.normal(
                                [1, 2, capacity, 128], dtype: .bfloat16, scale: kvScale,
                                key: MLXRandom.key(seed &+ 0x100))
                            let v = MLXRandom.normal(
                                [1, 2, capacity, 128], dtype: .bfloat16, scale: kvScale,
                                key: MLXRandom.key(seed &+ 0x200))
                            // The views the verify passes: the transposed Q
                            // projection and the cache's [..<offset] slices.
                            let queries = q.reshaped(1, rows, 32, 128).transposed(0, 2, 1, 3)
                            let keys = k[.ellipsis, ..<(history + rows), 0...]
                            let values = v[.ellipsis, ..<(history + rows), 0...]
                            guard let out = merged(queries, keys, values, scale) else { return false }
                            var same = MLXArray(true)
                            for (row, expected) in zip(split(out), stock(queries, keys, values, scale)) {
                                same = same .&& (row.view(dtype: .uint16) .== expected.view(dtype: .uint16)).all()
                            }
                            if !same.item(Bool.self) { return false }
                            checked += 1
                        }
                    }
                }
                return true
            }
            verified = matches
        } catch {
            return report("off (\(error))")
        }
        report(verified
            ? "on (\(checked) cases bitwise equal, key loop \(unroll4 ? "unroll4" : "stock"))"
            : "off (outputs differ from the stock launches after \(checked) cases)")
        if verified { qualifyPrepared(scaleBits.sorted()) }
        if verified { qualifyOneBarrier(scaleBits.sorted()) }
        if verified { qualifyFast(scaleBits.sorted()) }
    }

    /// NEMOTRON_SDPA_TUNE=0 keeps the launch selected above (one threadgroup per query head once `qualifyFast`
    /// passed). Otherwise `tune` times that launch and the per-row prepared launch it was checked against
    /// (`mergedPrepared`: 32 x 3 threadgroups, the aggregation selected above) and adopts the per-row launch only
    /// when `NemotronTuneAdoption.decide` admits it (at least 3% faster by median and faster in at least 75% of the
    /// cycles). NEMOTRON_SDPA_FORCE=1 adopts the per-row launch, 0 keeps the per-head launch.
    static let tuneEnabled = ProcessInfo.processInfo.environment["NEMOTRON_SDPA_TUNE"] != "0"
    static let tuneForced = Int(ProcessInfo.processInfo.environment["NEMOTRON_SDPA_FORCE"] ?? "")
    nonisolated(unsafe) private static var perRowAdopted = false

    /// Load time, after `qualify`, before any verify graph is traced: with seeded caches per full-attention layer at
    /// that layer's scale and key counts of a decode window (491, 555, 619), the per-head and per-row launches run
    /// round robin (`NemotronTuneAdoption.order`, median) as one chain per sample in which each launch takes the
    /// previous launch's output rows as its queries (so the launches run one after another, as the verify's attention
    /// layers do), after one chain of each was compared bit for bit (output rows and prepared activation of every
    /// launch). The per-row launch is adopted only when `NemotronTuneAdoption.decide` admits it.
    static func tune(_ target: Module) {
        perRowAdopted = false
        let t0 = DispatchTime.now().uptimeNanoseconds
        func elapsed() -> Double { Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9 }
        func report(_ message: String) {
            FileHandle.standardError.write(Data(
                "nemotron sdpa tune: \(message) (\(String(format: "%.2f", elapsed())) s)\n".utf8))
        }
        guard tuneEnabled else { return report("off (setting)") }
        guard verified, preparedVerified, fastVerified else { return report("off (per-head launch not in use)") }
        var layerScales: [Float] = []
        for module in target.modules() {
            guard let attention = module as? NemotronHAttention, attention.numHeads == 32,
                  attention.numKeyValueHeads == 2, attention.headDim == 128 else { continue }
            layerScales.append(attention.scale)
        }
        guard !layerScales.isEmpty else { return report("off (no 32 x 2 x 128 attention layer)") }
        let keyCounts = [491, 555, 619]
        let capacity = keyCounts.max()! + 37
        var caches: [(MLXArray, MLXArray)] = []
        var start: MLXArray = MLXArray(0)
        do {
            try withError {
                for l in layerScales.indices {
                    let k = MLXRandom.normal([1, 2, capacity, 128], dtype: .bfloat16, scale: 1,
                                             key: MLXRandom.key(UInt64(0x5d70 + 2 * l)))
                    let v = MLXRandom.normal([1, 2, capacity, 128], dtype: .bfloat16, scale: 1,
                                             key: MLXRandom.key(UInt64(0x5d71 + 2 * l)))
                    caches.append((k, v))
                }
                start = MLXRandom.normal([1, 3, 4096], dtype: .bfloat16, scale: 1, key: MLXRandom.key(0x5d6f))
                eval(caches.flatMap { [$0.0, $0.1] } + [start])
            }
        } catch {
            return report("off (inputs: \(error))")
        }
        let chain = layerScales.count * keyCounts.count
        // One chain: (output rows, prepared activation) of every launch; nil when a launch is unavailable.
        func run(_ perRow: Bool) -> [MLXArray]? {
            var queries = start.reshaped(1, 3, 32, 128).transposed(0, 2, 1, 3)
            var outs: [MLXArray] = []
            for (l, scale) in layerScales.enumerated() {
                for keys in keyCounts {
                    let k = caches[l].0[.ellipsis, ..<keys, 0...], v = caches[l].1[.ellipsis, ..<keys, 0...]
                    guard let (out, prepared) = perRow ? mergedPrepared(queries, k, v, scale)
                        : fastPrepared(queries, k, v, scale) else { return nil }
                    outs += [out, prepared]
                    queries = out.reshaped(1, 3, 32, 128).transposed(0, 2, 1, 3)
                }
            }
            return outs
        }
        var same = false
        var medians: [Double] = []
        var verdict: NemotronTuneAdoption.Verdict? = nil
        var reps = 0
        var failure: String?
        do {
            try withError {
                guard let a = run(false), let b = run(true) else { failure = "launch unavailable"; return }
                eval(a + b)
                var ok = MLXArray(true)
                for (i, (x, y)) in zip(a, b).enumerated() {
                    let dtype: DType = i % 2 == 0 ? .uint16 : .uint32
                    ok = ok .&& (x.view(dtype: dtype) .== y.view(dtype: dtype)).all()
                }
                same = ok.item(Bool.self)
                guard same else { failure = "per-row launch differs"; return }
                // A round: both launches' chains built first, then evaluated back to back in `order`.
                func round(_ order: [Int]) -> [Double?] {
                    let chains = order.map { run($0 == 1) }
                    return chains.map { outs -> Double? in
                        guard let outs else { return nil }
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
                reps = NemotronTuneAdoption.cycles(target: 100, wall: wall, cap: 61)
                var samples: [[Double]] = [[], []]
                for rep in 0 ..< reps {
                    let order = NemotronTuneAdoption.order(2, cycle: rep)
                    for (i, ms) in zip(order, round(order)) {
                        if let ms { samples[i].append(ms) }
                    }
                }
                guard samples.allSatisfy({ $0.count == reps }) else { failure = "timing incomplete"; return }
                verdict = NemotronTuneAdoption.decide(samples)
                medians = verdict?.medians ?? []
            }
        } catch {
            failure = "error: \(error)"
        }
        var why = "kept"
        if failure == nil, medians.count == 2 {
            if let f = tuneForced {
                perRowAdopted = f == 1
                why = f == 1 ? "forced" : f == 0 ? "forced V0" : "forced \(f) not qualified, kept"
            } else if verdict?.adopted == 1 {
                perRowAdopted = true
                why = "adopted"
            }
        }
        func ms(_ i: Int) -> String { i < medians.count ? String(format: "%.4f", medians[i]) : "n/a" }
        let rule = verdict?.rule({ _ in "rows" }) ?? "no candidate timed"
        report("\(why) \(perRowAdopted ? "rows" : "V0") \(ms(perRowAdopted ? 1 : 0)) ms vs V0 (heads) \(ms(0)) ms"
            + " per launch (\(rule); median of \(reps), \(chain) chained launches/sample on \(layerScales.count)"
            + " layers;"
            + " \(same ? 1 : 0)/1 bitwise over \(chain) launches and the 58 cases of the per-head check"
            + (medians.count == 2 ? "; [rows \(ms(1)) \(verdict?.won(1) ?? "n/a")]" : "")
            + (failure.map { "; dropped: rows \($0)" } ?? "") + ")")
    }

    /// Load time: `fastPrepared` against `mergedPrepared` (the launch it replaces, with the aggregation selected
    /// above), output rows as uint16 and prepared activation as uint32, on seeded finite inputs at the target's
    /// attention scales. Random rows at key counts 3 ... 1023 across the 32-, 64- and 128-key loop boundaries,
    /// with the cache view ending at the buffer end or 1, 37 or 256 rows short of it, query/key scales 1/2 to 16;
    /// and at three key counts: zero and -0 queries (every score ties at 0), one key row repeated (ties between
    /// keys), zero keys, -0 values, and scales where all but the largest exponentials underflow. Engages only if
    /// every case matches; an MLX error keeps the per-row launch.
    private static func qualifyFast(_ scales: [UInt32]) {
        fastVerified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron mtp verify sdpa fast: \(message)\n".utf8))
        }
        guard fastEnabled else { return report("off (setting)") }
        guard preparedVerified else { return report("off (prepared launch off)") }
        let histories = [0, 1, 2, 29, 30, 31, 32, 33, 61, 62, 63, 64, 93, 94, 95, 96, 125, 126, 127, 128, 129,
                         254, 255, 381, 382, 383, 488, 509, 510, 511, 552, 616, 637, 638, 639, 765, 893,
                         1018, 1019, 1020]
        let inputScales: [(Float, Float)] = [(1, 1), (4, 4), (16, 8), (0.5, 2)]
        let slack = [37, 0, 256, 1]
        var checked = 0
        func matches(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, _ keys: Int, _ scale: Float) -> Bool {
            let queries = q.reshaped(1, 3, 32, 128).transposed(0, 2, 1, 3)
            let k = k[.ellipsis, ..<keys, 0...], v = v[.ellipsis, ..<keys, 0...]
            guard let (a, pa) = mergedPrepared(queries, k, v, scale),
                  let (b, pb) = fastPrepared(queries, k, v, scale) else { return false }
            let same = (a.view(dtype: .uint16) .== b.view(dtype: .uint16)).all()
                .&& (pa.view(dtype: .uint32) .== pb.view(dtype: .uint32)).all()
            return same.item(Bool.self)
        }
        do {
            let allMatch = try withError { () -> Bool in
                for bits in scales {
                    let scale = Float(bitPattern: bits)
                    for (i, history) in histories.enumerated() {
                        let seed = UInt64(0x5da1_0000 + i)
                        let (qScale, kvScale) = inputScales[i % inputScales.count]
                        let capacity = history + 3 + slack[i % slack.count]
                        let q = MLXRandom.normal(
                            [1, 3, 4096], dtype: .bfloat16, scale: qScale, key: MLXRandom.key(seed))
                        let k = MLXRandom.normal(
                            [1, 2, capacity, 128], dtype: .bfloat16, scale: kvScale,
                            key: MLXRandom.key(seed &+ 0x100))
                        let v = MLXRandom.normal(
                            [1, 2, capacity, 128], dtype: .bfloat16, scale: kvScale,
                            key: MLXRandom.key(seed &+ 0x200))
                        guard matches(q, k, v, history + 3, scale) else { return false }
                        checked += 1
                    }
                    for (i, history) in [93, 552, 1020].enumerated() {
                        let seed = UInt64(0x5da2_0000 + i)
                        let capacity = history + 3 + 37
                        let normal = { (shape: [Int], std: Float, salt: UInt64) in
                            MLXRandom.normal(shape, dtype: .bfloat16, scale: std, key: MLXRandom.key(seed &+ salt))
                        }
                        let q = normal([1, 3, 4096], 1, 0), k = normal([1, 2, capacity, 128], 1, 0x100)
                        let v = normal([1, 2, capacity, 128], 1, 0x200)
                        let zeroQ = MLXArray.zeros([1, 3, 4096], dtype: .bfloat16)
                        let zeroKV = MLXArray.zeros([1, 2, capacity, 128], dtype: .bfloat16)
                        let cases: [(MLXArray, MLXArray, MLXArray)] = [
                            (zeroQ, k, v), (-zeroQ, k, v),
                            (q, tiled(k[0..., 0..., 0 ..< 1, 0...], repetitions: [1, 1, capacity, 1]), v),
                            (q, zeroKV, v), (q, k, -zeroKV),
                            (normal([1, 3, 4096], 64, 0x300), normal([1, 2, capacity, 128], 32, 0x400), v),
                        ]
                        for (q, k, v) in cases {
                            guard matches(q, k, v, history + 3, scale) else { return false }
                            checked += 1
                        }
                    }
                }
                return true
            }
            fastVerified = allMatch
        } catch {
            return report("off (\(error))")
        }
        report(fastVerified
            ? "on (\(checked) cases bitwise equal to the per-row launch: output rows and prepared activation, "
                + "key counts 3 ... 1023, ties, zeros, -0, underflow)"
            : "off (differs from the per-row launch after \(checked) cases)")
    }

    /// Load time: the one-barrier aggregation launches against the launches they replace, bit for bit, on
    /// seeded inputs at the target's attention scales: the merged launch at R = 2...4 and the prepared launch
    /// at R = 3 (output rows and prepared activation), key lengths across the 32-key and unroll boundaries up
    /// to 1023, query/key scales 1 to 16. Engages only if every case matches.
    private static func qualifyOneBarrier(_ scales: [UInt32]) {
        oneBarrierVerified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron mtp verify sdpa one barrier: \(message)\n".utf8))
        }
        guard oneBarrier else { return report("off (setting)") }
        guard oneBarrierKernel != nil, oneBarrierPreparedKernel != nil else {
            return report("off (aggregation text not found)")
        }
        let cases: [(rows: Int, histories: [Int])] = [
            (3, [0, 1, 29, 30, 31, 92, 93, 94, 125, 126, 127, 254, 488, 552, 616, 1020]),
            (2, [94, 95, 616, 1021]), (4, [92, 93, 616, 1019]),
        ]
        let inputScales: [(Float, Float)] = [(1, 1), (4, 4), (16, 8)]
        var checked = 0
        do {
            let matches = try withError { () -> Bool in
                for bits in scales {
                    let scale = Float(bitPattern: bits)
                    for (rows, histories) in cases {
                        for (i, history) in histories.enumerated() {
                            let seed = UInt64(0x5d9e_0000 + rows * 256 + i)
                            let (qScale, kvScale) = inputScales[i % inputScales.count]
                            let capacity = history + rows + 37
                            let q = MLXRandom.normal(
                                [1, rows, 4096], dtype: .bfloat16, scale: qScale, key: MLXRandom.key(seed))
                            let k = MLXRandom.normal(
                                [1, 2, capacity, 128], dtype: .bfloat16, scale: kvScale,
                                key: MLXRandom.key(seed &+ 0x100))
                            let v = MLXRandom.normal(
                                [1, 2, capacity, 128], dtype: .bfloat16, scale: kvScale,
                                key: MLXRandom.key(seed &+ 0x200))
                            let queries = q.reshaped(1, rows, 32, 128).transposed(0, 2, 1, 3)
                            let keys = k[.ellipsis, ..<(history + rows), 0...]
                            let values = v[.ellipsis, ..<(history + rows), 0...]
                            guard let a = merged(queries, keys, values, scale, oneBarrier: false),
                                  let b = merged(queries, keys, values, scale, oneBarrier: true)
                            else { return false }
                            var same = (a.view(dtype: .uint16) .== b.view(dtype: .uint16)).all()
                            if rows == 3 {
                                guard let pa = mergedPrepared(queries, keys, values, scale, oneBarrier: false),
                                      let pb = mergedPrepared(queries, keys, values, scale, oneBarrier: true)
                                else { return false }
                                same = same .&& (pa.0.view(dtype: .uint16) .== pb.0.view(dtype: .uint16)).all()
                                    .&& (pa.1.view(dtype: .uint32) .== pb.1.view(dtype: .uint32)).all()
                            }
                            if !same.item(Bool.self) { return false }
                            checked += 1
                        }
                    }
                }
                return true
            }
            oneBarrierVerified = matches
        } catch {
            return report("off (\(error))")
        }
        report(oneBarrierVerified
            ? "on (\(checked) cases bitwise equal to the stock aggregation, prepared output included at R = 3)"
            : "off (differs from the stock aggregation after \(checked) cases)")
    }

    /// Load-time bit check of `mergedPrepared` (R = 3) against `merged` and
    /// the prepare launch on the same rows, on seeded random inputs.
    private static func qualifyPrepared(_ scales: [UInt32]) {
        preparedVerified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron mtp verify sdpa prepared: \(message)\n".utf8))
        }
        guard preparedEnabled else { return report("off (setting)") }
        var checked = 0
        do {
            let matches = try withError { () -> Bool in
                for bits in scales {
                    let scale = Float(bitPattern: bits)
                    for (i, history) in [0, 31, 94, 127, 488, 616, 1020].enumerated() {
                        let seed = UInt64(0x5d9b_0000 + i)
                        let qScale: Float = [1, 4, 16][i % 3]
                        let kvScale: Float = [1, 4, 8][i % 3]
                        let capacity = history + 3 + 37
                        let q = MLXRandom.normal(
                            [1, 3, 4096], dtype: .bfloat16, scale: qScale, key: MLXRandom.key(seed))
                        let k = MLXRandom.normal(
                            [1, 2, capacity, 128], dtype: .bfloat16, scale: kvScale,
                            key: MLXRandom.key(seed &+ 0x100))
                        let v = MLXRandom.normal(
                            [1, 2, capacity, 128], dtype: .bfloat16, scale: kvScale,
                            key: MLXRandom.key(seed &+ 0x200))
                        let queries = q.reshaped(1, 3, 32, 128).transposed(0, 2, 1, 3)
                        let keys = k[.ellipsis, ..<(history + 3), 0...]
                        let values = v[.ellipsis, ..<(history + 3), 0...]
                        guard let out = merged(queries, keys, values, scale),
                              let (fusedOut, prepared) = mergedPrepared(queries, keys, values, scale)
                        else { return false }
                        let expected = NemotronHeadPreparedActivation.packRows(split(out), k: 4096)
                        let same = (fusedOut.view(dtype: .uint16) .== out.view(dtype: .uint16)).all()
                            .&& (prepared.view(dtype: .uint32) .== expected.view(dtype: .uint32)).all()
                        if !same.item(Bool.self) { return false }
                        checked += 1
                    }
                }
                return true
            }
            preparedVerified = matches
        } catch {
            return report("off (\(error))")
        }
        report(preparedVerified ? "on (\(checked) cases bitwise equal)"
                                : "off (differs from the prepare launch after \(checked) cases)")
    }
}

// Multi-row BF16 router projection. Stock `matmul(x_row, weight.T)` at M=1,
// N=128, K=2688 takes gemv_axbpy's non-transposed-matrix branch (K >= 16 * N),
// i.e. gemv_bfloat16_bm1_bn8_sm1_sn32_tm4_tn4: 8 simdgroups split K in
// 1024-wide blocks, each lane owns 4 output rows x 4 columns, then a
// simd_shuffle_down tree and a simdgroup-ordered threadgroup sum. This body is
// GEMVKernel::run for those parameters with the vector side repeated per row,
// so every row is bitwise equal to its own M=1 gemv. Each weight tile is
// loaded once for all rows instead of once per row.
private let nemotronMTPRouterRowsSource = """
  constexpr int BN = 8;
  constexpr int SN = 32;
  constexpr int TM = 4;
  constexpr int TN = 4;
  constexpr int blockM = TM;
  constexpr int blockN = BN * SN * TN;
  typedef float AccT;
  threadgroup AccT tgp_memory[R * BN * (blockM + TM)];

  const uint3 tid = threadgroup_position_in_grid;
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const uint simd_lid = thread_index_in_simdgroup;
  const int in_vec_size = K;
  const int out_vec_size = N;

  thread AccT result[R][TM];
  thread T inter[TN];
  thread AccT v_coeff[R][TN];
  for (int r = 0; r < R; r++) {
    for (int tm = 0; tm < TM; tm++) {
      result[r][tm] = 0;
    }
  }

  const int sgN = int(simd_gid);
  const int simdN = SN * int(simd_gid);
  int bn = (simdN + int(simd_lid)) * TN;
  // The shape guard sends N % 4 == 0 only, so no tail tile is ever shifted.
  const int out_row = int(tid.x) * blockM;
  const device T* mat = w + size_t(out_row) * in_vec_size;

  constexpr const uniform<int> loop_stride = make_uniform(blockN);
  const uniform<int> in_size = make_uniform(in_vec_size);
  const uniform<int> n_iter = in_size / loop_stride;
  const uniform<int> last_iter = loop_stride * n_iter;
  const uniform<int> leftover = in_size - last_iter;

  for (int i = 0; i < n_iter; ++i) {
    for (int r = 0; r < R; r++) {
      for (int tn = 0; tn < TN; tn++) {
        v_coeff[r][tn] = static_cast<AccT>(x[r * in_vec_size + bn + tn]);
      }
    }
    int mat_offset = 0;
    for (int tm = 0; tm < TM; tm++) {
      for (int tn = 0; tn < TN; tn++) {
        inter[tn] = mat[mat_offset + bn + tn];
      }
      // One weight load, the same per-row tn-order accumulation as stock.
      for (int r = 0; r < R; r++) {
        for (int tn = 0; tn < TN; tn++) {
          result[r][tm] += inter[tn] * v_coeff[r][tn];
        }
      }
      mat_offset += in_vec_size;
    }
    bn += blockN;
  }

  if (leftover > 0) {
    // GEMVKernel::load_safe: whole in-range quads, zero-filled otherwise.
    const bool whole = bn + TN <= in_size;
    for (int r = 0; r < R; r++) {
      for (int tn = 0; tn < TN; tn++) {
        v_coeff[r][tn] = (whole || bn + tn < in_size)
            ? static_cast<AccT>(x[r * in_vec_size + bn + tn]) : AccT(0);
      }
    }
    for (int tm = 0; tm < TM; tm++) {
      for (int tn = 0; tn < TN; tn++) {
        inter[tn] = (whole || bn + tn < in_size)
            ? mat[tm * in_vec_size + bn + tn] : T(0);
      }
      for (int r = 0; r < R; r++) {
        for (int tn = 0; tn < TN; tn++) {
          result[r][tm] += inter[tn] * v_coeff[r][tn];
        }
      }
    }
  }

  for (int r = 0; r < R; r++) {
    for (int tm = 0; tm < TM; tm++) {
      for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
        result[r][tm] += simd_shuffle_down(result[r][tm], sn);
      }
    }
  }

  if (simd_lid == 0) {
    for (int r = 0; r < R; r++) {
      for (int tm = 0; tm < TM; tm++) {
        tgp_memory[(r * BN + sgN) * (blockM + TM) + tm] = result[r][tm];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgN == 0 && simd_lid == 0) {
    for (int r = 0; r < R; r++) {
      // Stock order: result += simdgroup 1, then 2, ... then 7.
      for (int sgn = 1; sgn < BN; sgn++) {
        for (int tm = 0; tm < TM; tm++) {
          result[r][tm] += tgp_memory[(r * BN + sgn) * (blockM + TM) + tm];
        }
      }
      for (int tm = 0; tm < TM; tm++) {
        y[r * out_vec_size + out_row + tm] = static_cast<T>(result[r][tm]);
      }
    }
  }
"""

enum NemotronMTPRouterRows {
    static let enabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_ROUTER_ROWS"] != "0"
    private static let kernel = MLXFast.metalKernel(
        name: "nemotron_mtp_router_rows_gemv_bf16",
        inputNames: ["w", "x", "K", "N"],
        outputNames: ["y"],
        source: nemotronMTPRouterRowsSource)

    /// x: [1, R, 2688] bf16, weight: [128, 2688] bf16. Returns the gates
    /// [1, R, 128], or nil for any shape the stock gemv parameters above do
    /// not describe (the caller then keeps one M=1 matmul per row).
    static func gates(_ x: MLXArray, _ weight: MLXArray) -> MLXArray? {
        guard enabled, Device.defaultDevice().deviceType == .gpu,
              StreamOrDevice.default == .gpu,
              x.ndim == 3, x.dim(0) == 1, (1...8).contains(x.dim(1)),
              x.dim(2) == 2688, x.dtype == .bfloat16,
              weight.dtype == .bfloat16, weight.shape == [128, 2688],
              rowMajor.contains(ObjectIdentifier(weight))
        else { return nil }
        let rows = x.dim(1)
        let (kArray, nArray) = NemotronMultiRowQMV.scalarPair(2688, 128)
        var y = kernel([weight, x, kArray, nArray],
            template: [("T", x.dtype), ("R", rows)],
            grid: (32 * 32, 8, 1),
            threadGroup: (32, 8, 1),
            outputShapes: [[1, rows, 128]],
            outputDTypes: [x.dtype])[0]
        if NemotronMoELeader.selfTest {
            let expected = nemotronMTPMapRows(x) { MLX.matmul($0, weight.transposed()) }
            let ok = (y.view(dtype: .uint16) .== expected.view(dtype: .uint16)).all()
            y = which(ok, y, MLXArray(Float.nan).asType(y.dtype))
        }
        return y
    }

    // Router weights whose loaded layout is row-major [128, 2688]: there the
    // reference matmul(row, weight.T) takes the non-transposed gemv this kernel
    // repeats. Another layout makes MLX pick gemv_t, whose order differs, so
    // only weights verified here at load take the fast path. Filled by warm()
    // (the target's routers, then the draft head's), before any compile
    // trace, from the frozen loaded weights.
    nonisolated(unsafe) private static var rowMajor = Set<ObjectIdentifier>()
    static func isRowMajor(_ weight: MLXArray) -> Bool { rowMajor.contains(ObjectIdentifier(weight)) }

    /// Record row-major router weights, then compile every pipeline the
    /// verify window can request (R = 1...4) at load.
    static func warm(_ model: Module) {
        guard enabled else { return }
        let routers = model.modules().compactMap { $0 as? NemotronHMoEGate }
        for gate in routers {
            eval(gate.weight)
            if gate.weight.strides == [2688, 1] { rowMajor.insert(ObjectIdentifier(gate.weight)) }
        }
        guard let gate = routers.first(where: { rowMajor.contains(ObjectIdentifier($0.weight)) })
        else { return }
        let outputs = (1...4).compactMap { rows in
            gates(MLXArray.zeros([1, rows, 2688], dtype: .bfloat16), gate.weight)
        }
        if !outputs.isEmpty { eval(outputs) }
    }
}

// The router gemv above with the selection kernel's body as an epilogue of the
// grid's last threadgroup. Every threadgroup computes and stores its gates
// exactly as the gemv does (the text above is unchanged); then it orders that
// store before a device atomic. The threadgroup that takes the last ticket
// orders every other store before its reads and runs, per verify row in its
// own SIMD group, `nemotron_h_router_select`'s statements on those BF16
// gates: the same sigmoid, bias add, six descending picks with the lowest id
// on ties, sequential sum, divide and scale. It resets the ticket counter.
private let nemotronMTPRouterRowsSelectEpilogue = """

  threadgroup uint nmr_ticket[1];
  threadgroup float nmr_original[R * 128];
  threadgroup float nmr_normalized[R * 6];
  if (sgN == 0 && simd_lid == 0) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    nmr_ticket[0] = atomic_fetch_add_explicit((device atomic_uint*)counter, 1u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (nmr_ticket[0] != uint(threadgroups_per_grid.x - 1)) { return; }
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  const uint lane = simd_lid;
  const int row = sgN;
  float corrected[4];
  bool selected[4] = {false, false, false, false};
  float chosen[6];
  if (row < R) {
    for (uint i = 0; i < 4; ++i) {
      const uint id = lane + 32 * i;
      nmr_original[row * 128 + id] = nemotron_h_router_sigmoid(float(y[row * out_vec_size + id]));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (row < R) {
    for (uint i = 0; i < 4; ++i) {
      const uint id = lane + 32 * i;
      corrected[i] = nmr_original[row * 128 + id] + bias[id];
    }
    for (uint slot = 0; slot < 6; ++slot) {
      uint local_id = 0xffffffff;
      float local_score = -INFINITY;
      for (uint i = 0; i < 4; ++i) {
        if (!selected[i] && (local_id == 0xffffffff
            || (!metal::isnan(corrected[i])
                && (metal::isnan(local_score) || corrected[i] > local_score)))) {
          local_id = lane + 32 * i;
          local_score = corrected[i];
        }
      }
      const bool ordered = local_id != 0xffffffff && !metal::isnan(local_score);
      const bool any_ordered = simd_any(ordered);
      const float maximum = simd_max(ordered ? local_score : -INFINITY);
      const uint winner = simd_min(
          ((!any_ordered && local_id != 0xffffffff)
              || (ordered && local_score == maximum)) ? local_id : 0xffffffff);
      if (lane == 0) {
        indices[row * 6 + slot] = winner;
        chosen[slot] = nmr_original[row * 128 + winner];
      }
      for (uint i = 0; i < 4; ++i) {
        if (lane + 32 * i == winner) { selected[i] = true; }
      }
    }
    if (lane == 0) {
      float total = 0.0f;
      for (uint slot = 0; slot < 6; ++slot) { total = chosen[slot] + total; }
      const float denominator = total + 1e-20f;
      for (uint slot = 0; slot < 6; ++slot) {
        nmr_normalized[row * 6 + slot] = chosen[slot] / denominator;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (row < R && lane < 6) { weights[row * 6 + lane] = nmr_normalized[row * 6 + lane] * 2.5f; }
  if (sgN == 0 && simd_lid == 0) {
    atomic_store_explicit((device atomic_uint*)counter, 0u, memory_order_relaxed);
  }
"""

/// One launch for a verify window's router: `NemotronMTPRouterRows`'s gemv and
/// `NemotronHRouterSelection`'s top-six selection (see the epilogue above).
/// Used only after `qualify` found indices, weights and gates bitwise equal to
/// the two launches. NEMOTRON_MTP_ROUTER_SELECT_FUSED=0 keeps the two launches.
/// The draft head's one-row router (`exactOneTokenTail`) takes the same launch
/// on its own counter once `qualifyDraft` matched it on the head's gate.
enum NemotronMTPRouterRowsSelect {
    static let enabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_ROUTER_SELECT_FUSED"] != "0"
    nonisolated(unsafe) static var verified = false
    nonisolated(unsafe) static var draftVerified = false
    nonisolated(unsafe) fileprivate static var counter: MLXArray? = nil
    nonisolated(unsafe) private static var draftCounter: MLXArray? = nil
    /// Router rows per threadgroup: one (128 threadgroups) instead of the
    /// gemv's four (32). A row's per-thread terms, shuffle tree and simdgroup
    /// order do not depend on which rows share a threadgroup, so the gates,
    /// indices and weights are the same; `qualify` checks them bit for bit.
    /// NEMOTRON_MTP_ROUTER_ROWS_PER_TG=4 keeps four rows per threadgroup.
    static let rowsPerThreadgroup =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_ROUTER_ROWS_PER_TG"] == "4" ? 4 : 1
    private static let kernel = MLXFast.metalKernel(
        name: "nemotron_mtp_router_rows_select",
        inputNames: ["w", "x", "K", "N", "bias", "counter"],
        outputNames: ["y", "indices", "weights"],
        source: nemotronSwapped(nemotronMTPRouterRowsSource, [
            ("  constexpr int TM = 4;\n", "  constexpr int TM = \(rowsPerThreadgroup);\n", 1),
        ]) + nemotronMTPRouterRowsSelectEpilogue,
        header: NemotronHRouterSelection.sigmoidHeader,
        ensureRowContiguous: true, mutableInputs: ["counter"])

    /// x: [1, R, 2688] bf16 (R = 1...4), weight: a row-major [128, 2688] bf16
    /// router, bias: [128] fp32. Returns (indices [1, R, 6] uint32,
    /// weights [1, R, 6] fp32), or nil outside the qualified shapes.
    static func run(_ x: MLXArray, _ weight: MLXArray, bias: MLXArray, force: Bool = false,
                    draft: Bool = false)
        -> (MLXArray, MLXArray)?
    {
        launch(x, weight, bias: bias, force: force, draft: draft).map { ($0[1], $0[2]) }
    }

    /// `run`'s launch: [gates, indices, weights].
    private static func launch(_ x: MLXArray, _ weight: MLXArray, bias: MLXArray, force: Bool,
                               draft: Bool) -> [MLXArray]?
    {
        guard enabled, force || (draft ? draftVerified : verified),
              let counter = draft ? Self.draftCounter : Self.counter,
              Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              x.ndim == 3, x.dim(0) == 1, (1...4).contains(x.dim(1)),
              x.dim(2) == 2688, x.dtype == .bfloat16,
              weight.dtype == .bfloat16, weight.shape == [128, 2688],
              NemotronMTPRouterRows.isRowMajor(weight),
              bias.shape == [128], bias.dtype == .float32
        else { return nil }
        let rows = x.dim(1)
        let (kArray, nArray) = NemotronMultiRowQMV.scalarPair(2688, 128)
        return kernel([weight, x, kArray, nArray, bias, counter],
            template: [("T", x.dtype), ("R", rows)],
            grid: (32 * (128 / rowsPerThreadgroup), 8, 1), threadGroup: (32, 8, 1),
            outputShapes: [[1, rows, 128], [1, rows, 6], [1, rows, 6]],
            outputDTypes: [x.dtype, .uint32, .float32])
    }

    /// Load time: the fused launch against the gemv + selection pair, bit for
    /// bit, on seeded rows (R = 1...4, twice each so the counter reset is
    /// exercised) with a real router weight and bias. Engages only if all match.
    static func qualify(_ model: Module) {
        verified = false
        guard enabled else { return }
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron mtp router select fused: \(message)\n".utf8))
        }
        let routers = model.modules().compactMap { $0 as? NemotronHMoEGate }
        guard let gate = routers.first(where: { NemotronMTPRouterRows.isRowMajor($0.weight) }),
              gate.eSCB.shape == [128], gate.eSCB.dtype == .float32
        else { return report("off (no row-major router)") }
        let zero = MLXArray.zeros([1], dtype: .uint32)
        eval(zero)
        counter = zero
        let draftZero = MLXArray.zeros([1], dtype: .uint32)
        eval(draftZero)
        draftCounter = draftZero
        var ok = MLXArray(true)
        do {
            let matched = try withError { () -> Bool in
                for rows in 1...4 {
                    for seed in 0 ..< 2 {
                        let key = MLXRandom.key(UInt64(0x5e1ec7 + rows * 16 + seed))
                        let x = MLXRandom.normal([1, rows, 2688], key: key).asType(.bfloat16)
                        guard let fused = run(x, gate.weight, bias: gate.eSCB, force: true),
                              let gates = NemotronMTPRouterRows.gates(x, gate.weight),
                              let reference = NemotronHRouterSelection.apply(
                                  gates: gates, eSCB: gate.eSCB, topK: 6, nGroup: 1, topkGroup: 1,
                                  routedScalingFactor: 2.5, normTopkProb: true)
                        else { return false }
                        ok = MLX.logicalAnd(ok, MLX.logicalAnd(
                            (fused.0 .== reference.0).all(),
                            (fused.1.view(dtype: .uint32) .== reference.1.view(dtype: .uint32)).all()))
                        eval(ok, fused.0, fused.1)
                    }
                }
                return ok.item(Bool.self) && (counter!.item(UInt32.self) == 0)
            }
            verified = matched
        } catch {
            verified = false
        }
        report(verified ? "on (bitwise equal at R = 1...4)" : "off (differs or failed)")
    }

    /// Load time, after `qualify` and before any draft graph is traced: the
    /// head's router (`exactOneTokenTail`, one row per draft step). Records the
    /// head's row-major gate (`NemotronMTPRouterRows.warm`), then compares the
    /// R = 1 launch on the draft counter with the two launches the draft takes
    /// without it, the head gate's stock gemv `matmul(row, weight.T)` and
    /// `NemotronHRouterSelection`: gates, indices and weights bit for bit on
    /// seeded rows (each launched twice, so the counter reset is exercised).
    /// Engages only if all match. NEMOTRON_MTP_DRAFT_ROUTER_FUSED=0 (or
    /// NEMOTRON_MTP_DRAFT_FUSED=0) keeps the two launches.
    static func qualifyDraft(_ head: Module) {
        draftVerified = false
        guard enabled, NemotronMTPDraftTail.oneLaunchRouter else { return }
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron mtp draft router select fused: \(message)\n".utf8))
        }
        NemotronMTPRouterRows.warm(head)
        let routers = head.modules().compactMap { $0 as? NemotronHMoEGate }
        guard routers.count == 1, let gate = routers.first,
              NemotronMTPRouterRows.isRowMajor(gate.weight),
              gate.eSCB.shape == [128], gate.eSCB.dtype == .float32,
              gate.topK == 6, gate.nGroup == 1, gate.topkGroup == 1, gate.normTopkProb,
              gate.routedScalingFactor == 2.5
        else { return report("off (no row-major head router)") }
        if draftCounter == nil {
            let zero = MLXArray.zeros([1], dtype: .uint32)
            eval(zero)
            draftCounter = zero
        }
        var ok = MLXArray(true)
        do {
            let matched = try withError { () -> Bool in
                for seed in 0 ..< 8 {
                    let key = MLXRandom.key(UInt64(0xd7a5e1 + seed))
                    let x = (MLXRandom.normal([1, 1, 2688], key: key) * Float(1 + seed % 4))
                        .asType(.bfloat16)
                    let gates = MLX.matmul(x, gate.weight.transposed())
                    let reference = gate(x)
                    for _ in 0 ..< 2 {
                        guard let fused = launch(x, gate.weight, bias: gate.eSCB, force: true, draft: true)
                        else { return false }
                        ok = MLX.logicalAnd(ok, MLX.logicalAnd(
                            (fused[0].view(dtype: .uint16) .== gates.view(dtype: .uint16)).all(),
                            MLX.logicalAnd(
                                (fused[1] .== reference.0).all(),
                                (fused[2].view(dtype: .uint32) .== reference.1.view(dtype: .uint32)).all())))
                        eval(ok, fused[0], fused[1], fused[2])
                    }
                }
                return ok.item(Bool.self) && (draftCounter!.item(UInt32.self) == 0)
            }
            draftVerified = matched
        } catch {
            draftVerified = false
        }
        report(draftVerified ? "on (bitwise equal to the head's gemv + selection at R = 1)"
            : "off (differs or failed)")
    }
}

// MARK: - Router and shared up in one launch
//
// In the captured verify, a MoE layer's router (`NemotronMTPRouterRowsSelect`: 128 gemv threadgroups, the last
// ticket runs the top-six selection) and its shared-expert up projection with ReLU² (the prepared QMV the
// zero-skip shared down reads) both read only the layer's input, and only later launches read their results.
// One launch runs both: threadgroups y < 128 run the router body, the rest the shared up body at eight SIMD
// groups per threadgroup instead of two (each SIMD group keeps its own rows, loads, order, simd_sum and store).
// Each body is its kernel's source text with its threadgroup position (and the router's grid size) shadowed and
// the router's threadgroup arrays declared once at kernel scope. `qualify` compares every layer with the two
// separate launches bit for bit at load. The separate launches are the default; NEMOTRON_MTP_ROUTER_UP_COLAUNCH=1
// takes the co-launch.
enum NemotronRouterSharedUpColaunch {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_MTP_ROUTER_UP_COLAUNCH"] == "1"
        && !NemotronMoELeader.selfTest
    private static let routerThreadgroups = 128
    private static let upSimdgroups = 8
    private static let lock = NSLock()
    /// Shared up projections whose layer matched the separate launches at load.
    nonisolated(unsafe) private static var owners: Set<ObjectIdentifier> = []

    private static func source(up: String) -> String {
        let router = nemotronSwapped(
            nemotronSwapped(nemotronMTPRouterRowsSource, [
                ("  constexpr int TM = 4;\n", "  constexpr int TM = 1;\n", 1),
                ("  threadgroup AccT tgp_memory[R * BN * (blockM + TM)];\n",
                 "  threadgroup AccT* tgp_memory = nrc_tgp;\n", 1),
            ]) + nemotronMTPRouterRowsSelectEpilogue, [
                ("  threadgroup uint nmr_ticket[1];\n", "  threadgroup uint* nmr_ticket = nrc_ticket;\n", 1),
                ("  threadgroup float nmr_original[R * 128];\n",
                 "  threadgroup float* nmr_original = nrc_original;\n", 1),
                ("  threadgroup float nmr_normalized[R * 6];\n",
                 "  threadgroup float* nmr_normalized = nrc_normalized;\n", 1),
            ])
        return """
          threadgroup float nrc_tgp[R * 8 * 2];
          threadgroup uint nrc_ticket[1];
          threadgroup float nrc_original[R * 128];
          threadgroup float nrc_normalized[R * 6];
          const uint nrc_tg = threadgroup_position_in_grid.y;
          if (nrc_tg >= uint(RT)) {
            const auto w = uw;
            const auto scales = uscales;
            const auto biases = ubiases;
            const int K = UK;
            const int N = UN;
            device T* y = uy;
            const uint3 threadgroup_position_in_grid = uint3(0, nrc_tg - uint(RT), 0);
          \(up)
            return;
          }
          {
            const uint3 threadgroup_position_in_grid = uint3(nrc_tg, 0, 0);
            const uint3 threadgroups_per_grid = uint3(RT, 1, 1);
          \(router)
          }
          """
    }

    private static func make(_ name: String, up: String?) -> MLXFast.MLXFastKernel? {
        guard let up else { return nil }
        return MLXFast.metalKernel(
            name: name,
            inputNames: ["w", "x", "K", "N", "bias", "counter", "uw", "uscales", "ubiases", "prepared", "UK", "UN"],
            outputNames: ["y", "indices", "weights", "uy"],
            source: source(up: up),
            header: NemotronHRouterSelection.sigmoidHeader + nemotronMultiRowQMVHeader,
            ensureRowContiguous: true, mutableInputs: ["counter"])
    }
    private static let upBody = nemotronPreparedVariantSource(nemotronHeadPreparedQMVSource, load: 0, nsg: true)
    private static let plainKernel = make("nemotron_router_select_shared_up_r3", up: upBody)
    private static let dictKernel = make("nemotron_router_select_shared_up_r3_sbd", up: nemotronSBDictSource(upBody))

    /// The router's (indices, weights) and the shared up's ReLU² rows [3, N] of one captured-verify MoE layer,
    /// or nil when the layer did not qualify (the caller keeps the separate launches).
    static func run(_ x: MLXArray, gate: NemotronHMoEGate, up: Linear, prepared: MLXArray)
        -> (indices: MLXArray, weights: MLXArray, up: MLXArray)?
    {
        guard enabled, let q = up as? QuantizedLinear,
              lock.withLock({ owners.contains(ObjectIdentifier(q)) }) else { return nil }
        return dispatch(x, gate: gate, q: q, prepared: prepared)
    }

    private static func dispatch(_ x: MLXArray, gate: NemotronHMoEGate, q: QuantizedLinear, prepared: MLXArray)
        -> (indices: MLXArray, weights: MLXArray, up: MLXArray)?
    {
        let k = 2688, n = q.weight.dim(0)
        guard NemotronMTPRouterRowsSelect.enabled, NemotronMTPRouterRowsSelect.verified,
              NemotronMTPRouterRowsSelect.rowsPerThreadgroup == 1,
              let counter = NemotronMTPRouterRowsSelect.counter,
              gate.topK == 6, gate.nGroup == 1, gate.topkGroup == 1, gate.normTopkProb,
              gate.routedScalingFactor == 2.5,
              Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              x.shape == [1, 3, k], x.dtype == .bfloat16,
              gate.weight.dtype == .bfloat16, gate.weight.shape == [128, k],
              NemotronMTPRouterRows.isRowMajor(gate.weight),
              gate.eSCB.shape == [128], gate.eSCB.dtype == .float32,
              q.mode == .affine, q.bits == 4, q.groupSize == 64, q.bias == nil,
              let biases = q.biases, biases.dtype == .bfloat16, q.scales.dtype == .bfloat16,
              q.weight.dtype == .uint32, q.weight.shape == [n, k / 8],
              n % (2 * upSimdgroups) == 0,
              let packed = NemotronHeadPreparedActivation.compatible(prepared, rows: 3, k: k)
        else { return nil }
        // The separate shared up takes the dictionary twin exactly when this holds.
        let dict = NemotronSBDict.enabled && NemotronHeadPreparedActivation.dictLaunchAvailable(k: k, n: n)
            ? NemotronSBDict.dict(q) : nil
        guard let kernel = dict != nil ? dictKernel : plainKernel else { return nil }
        let (kArray, nArray) = NemotronMultiRowQMV.scalarPair(k, 128)
        let (ukArray, unArray) = NemotronMultiRowQMV.scalarPair(k, n)
        let upGroups = n / (2 * upSimdgroups)
        let out = kernel(
            [gate.weight, x, kArray, nArray, gate.eSCB, counter,
             q.weight, dict?.index ?? q.scales, dict?.table ?? biases, packed, ukArray, unArray],
            template: [("T", DType.bfloat16), ("R", 3), ("FAST", false), ("RELU2", true), ("RPS", 2),
                       ("NSG", upSimdgroups), ("RT", routerThreadgroups)],
            grid: (32, 8 * (routerThreadgroups + upGroups), 1), threadGroup: (32, 8, 1),
            outputShapes: [[1, 3, 128], [1, 3, 6], [1, 3, 6], [3, n]],
            outputDTypes: [.bfloat16, .uint32, .float32, .bfloat16])
        return (out[1], out[2], out[3])
    }

    /// Load time, after `NemotronMTPRouterRowsSelect.qualify` and `NemotronSBDict.prepare`: every MoE layer's
    /// launch against the router launch and the shared up launch it replaces, bit for bit on the indices, the
    /// weights and the ReLU² rows, over seeded rows (twice each, so the ticket reset is exercised). Any miss keeps
    /// every layer on the separate launches.
    static func qualify(_ model: Module) {
        lock.withLock { owners = [] }
        guard enabled, NemotronMTPRouterRowsSelect.verified else { return }
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron router+shared up colaunch: \(message)\n".utf8))
        }
        var built: Set<ObjectIdentifier> = []
        var accepted = false
        do {
            accepted = try withError { () -> Bool in
                for module in model.modules() {
                    guard let moe = module as? NemotronHMoE, let shared = moe.sharedExperts,
                          let q = shared.upProj as? QuantizedLinear else { continue }
                    for seed in 0 ..< 2 {
                        let key = MLXRandom.key(UInt64(0x7a0c + built.count * 4 + seed))
                        let x = MLXRandom.normal([1, 3, 2688], key: key).asType(.bfloat16)
                        let packed = NemotronHeadPreparedActivation.pack(x.reshaped(3, 2688))
                        guard let fused = dispatch(x, gate: moe.gate, q: q, prepared: packed),
                              let router = NemotronMTPRouterRowsSelect.run(x, moe.gate.weight, bias: moe.gate.eSCB)
                        else { return false }
                        let up = NemotronHeadPreparedActivation.apply(
                            x.reshaped(3, 2688), q, relu2: true, rps: 2, prepared: packed)
                        let ok = MLX.logicalAnd(MLX.logicalAnd(
                            (fused.indices .== router.0).all(),
                            (fused.weights.view(dtype: .uint32) .== router.1.view(dtype: .uint32)).all()),
                            (fused.up.view(dtype: .uint16) .== up.view(dtype: .uint16)).all())
                        eval(ok)
                        guard ok.item(Bool.self),
                              NemotronMTPRouterRowsSelect.counter?.item(UInt32.self) == 0 else { return false }
                    }
                    built.insert(ObjectIdentifier(q))
                }
                return !built.isEmpty
            }
        } catch {
            accepted = false
        }
        lock.withLock { owners = accepted ? built : [] }
        report(accepted ? "on (\(built.count) MoE layers bitwise equal)" : "off (differs or failed)")
    }
}

/// Put verification tokens on the native *batch* axis, keeping matrix M=1.
/// A singleton leading weight axis prevents MLX's flatten-to-M optimization
/// and therefore avoids qmv_wide's different accumulation. The underlying
/// matrix/scales/offset storage is shared; no weight bank is duplicated.
func nemotronMTPLinearRows(_ x: MLXArray, _ linear: Linear,
                           prepared: MLXArray? = nil) -> MLXArray {
    guard NemotronMTPExecution.batchedM1, x.ndim == 3, x.dim(0) == 1, x.dim(1) > 1,
        let q = linear as? QuantizedLinear, q.mode == .affine,
        q.scales.dtype == x.dtype, let biases = q.biases, biases.dtype == x.dtype
    else { return nemotronMTPMapRows(x) { linear($0) } }
    if let rows = NemotronMultiRowQMV.apply(x.reshaped(x.dim(1), x.dim(2)), q,
        prepared: prepared) {
        var result = rows.reshaped(1, x.dim(1), q.weight.dim(0))
        if let bias = q.bias { result = result + bias }
        return result
    }
    let count = x.dim(1), width = x.dim(2)
    var result = quantizedMM(x.reshaped(count, 1, width), q.weight.expandedDimensions(axis: 0),
        scales: q.scales.expandedDimensions(axis: 0), biases: biases.expandedDimensions(axis: 0),
        transpose: true, groupSize: q.groupSize, bits: q.bits, mode: q.mode)
        .reshaped(1, count, q.weight.dim(0))
    if let bias = q.bias { result = result + bias }
    return result
}

// The epilogue must first round to BF16, exactly as the separate projection.
func nemotronMTPLinearRelu2Rows(_ x: MLXArray, _ linear: Linear,
                              prepared: MLXArray? = nil) -> MLXArray {
    if NemotronMTPExecution.batchedM1, x.ndim == 3, x.dim(0) == 1,
       x.dim(1) >= 1, let q = linear as? QuantizedLinear, q.bias == nil,
       let rows = NemotronMultiRowQMV.apply(x.reshaped(x.dim(1), x.dim(2)), q,
           relu2: true, prepared: prepared) {
        return rows.reshaped(1, x.dim(1), q.weight.dim(0))
    }
    let projected = nemotronMTPLinearRows(x, linear)
    let positive = maximum(projected, MLXArray(0))
    return positive * positive
}

// Target Q/K/V share one dispatch; every tile keeps the existing row body.
// A tile is one threadgroup of 2 * RPS output rows: Q tiles, then K, then V.
enum NemotronTargetQKV {
    /// NEMOTRON_MTP_QKV_ACTPREP=0 keeps the fused launch on BF16 activations.
    static let preparedEnabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_QKV_ACTPREP"] != "0"
    private static let tilePrefix = """
      constexpr uint q_tiles = 4096 / (2 * RPS);
      constexpr uint kv_tiles = 256 / (2 * RPS);
      const uint tile = threadgroup_position_in_grid.y;
      const uint local_tile = tile < q_tiles ? tile
          : (tile < q_tiles + kv_tiles ? tile - q_tiles : tile - q_tiles - kv_tiles);
      {
        const uint3 threadgroup_position_in_grid = uint3(0, local_tile, 0);
        const device uint32_t* w = tile < q_tiles ? qw : (tile < q_tiles + kv_tiles ? kw : vw);
        const device T* scales = tile < q_tiles ? qs : (tile < q_tiles + kv_tiles ? ks : vs);
        const device T* biases = tile < q_tiles ? qb : (tile < q_tiles + kv_tiles ? kb : vb);
        device T* y = tile < q_tiles ? qy : (tile < q_tiles + kv_tiles ? ky : vy);
        const int N = tile < q_tiles ? 4096 : 256;
    """
    private static let kernel = MLXFast.metalKernel(
        name: "nemotron_target_qkv_affine4_gs64",
        inputNames: ["qw", "qs", "qb", "kw", "ks", "kb", "vw", "vs", "vb", "x", "K"],
        outputNames: ["qy", "ky", "vy"], source: tilePrefix + nemotronMultiRowQMVSource + "\n}",
        header: nemotronMultiRowQMVHeader)
    // newjordan's f0513638 prepared-activation QMV body behind the same tile
    // prefix: the three verify rows are packed once per launch instead of
    // once per output tile. Each Q/K/V row keeps the body's lane split, block
    // order, qdot, affine update, simd_sum and BF16 store.
    private static let preparedKernelRecord = MLXFast.metalKernel(
        name: "nemotron_target_qkv_prepared_affine4_gs64",
        inputNames: ["qw", "qs", "qb", "kw", "ks", "kb", "vw", "vs", "vb", "prepared", "K"],
        outputNames: ["qy", "ky", "vy"], source: tilePrefix + nemotronHeadPreparedQMVSource + "\n}",
        header: nemotronMultiRowQMVHeader)
    /// The prepared launch (the MTP head's post-verify rows) with the vector store, used once
    /// `qualifyVectorStore` matched it to the lane-0 launch.
    private static let preparedKernelVS: MLXFast.MLXFastKernel? =
        nemotronQMVVectorStoreSource(tilePrefix + nemotronHeadPreparedQMVSource + "\n}").map {
            MLXFast.metalKernel(
                name: "nemotron_target_qkv_prepared_affine4_gs64_vs",
                inputNames: ["qw", "qs", "qb", "kw", "ks", "kb", "vw", "vs", "vb", "prepared", "K"],
                outputNames: ["qy", "ky", "vy"], source: $0, header: nemotronMultiRowQMVHeader)
        }
    private static var preparedKernel: MLXFast.MLXFastKernel {
        vstoreVerified ? (preparedKernelVS ?? preparedKernelRecord) : preparedKernelRecord
    }
    // CBV2_KV_IN_PLACE_APPEND: the prepared launch above, whose K and V tiles
    // also store each BF16 result into the layer's contiguous cache
    // [1, 2, cap, 128] at token kvpos[0] + r (kvpos = [offset, capacity]).
    // The stored value is the same `h` the body writes to ky/vy; every other
    // instruction is the prepared body's.
    private static let storeLine = "yp[r * out_vec_size + row] = h;"
    private static let storeAndCache = """
        yp[r * out_vec_size + row] = h;
                if (tile >= q_tiles) {
                  const uint n_out = uint(out_row) + uint(row);
                  device T* cache = tile < q_tiles + kv_tiles ? kcache : vcache;
                  cache[((size_t)(n_out / 128) * (size_t)nkv_cap
                         + (size_t)(nkv_off + r)) * 128 + (n_out % 128)] = h;
                }
        """
    // The cache coordinates the store loops above read, loaded once at the start of the launch: the launch never
    // writes `kvpos` or `kvdst`, so each holds the value the record text loads in the store loop, where every
    // load followed the thread's previous output store.
    private static let kvPreload = """
      const int nkv_off = kvpos[0];
      const int nkv_cap = kvpos[1];

    """
    private static let pagedPreload = """
      const int nkv_plane = kvdst[0];
      int nkv_page[R];
      int nkv_slot[R];
      _Pragma("clang loop unroll(full)")
      for (int r = 0; r < R; r++) {
        nkv_page[r] = kvdst[1 + 2 * r];
        nkv_slot[r] = kvdst[2 + 2 * r];
      }

    """
    private static let preparedWritingKVSource: String = {
        precondition(nemotronHeadPreparedQMVSource.components(separatedBy: storeLine).count == 2,
                     "NemotronTargetQKV: prepared body store not unique")
        return kvPreload + tilePrefix
            + nemotronHeadPreparedQMVSource.replacingOccurrences(of: storeLine, with: storeAndCache) + "\n}"
    }()
    private static func makeWritingKV(_ name: String, _ source: String) -> MLXFast.MLXFastKernel {
        MLXFast.metalKernel(
            name: name,
            inputNames: ["qw", "qs", "qb", "kw", "ks", "kb", "vw", "vs", "vb", "prepared", "K",
                         "kcache", "vcache", "kvpos"],
            outputNames: ["qy", "ky", "vy"], source: source,
            header: nemotronMultiRowQMVHeader, mutableInputs: ["kcache", "vcache"])
    }
    private static let preparedWritingKVKernelRecord =
        makeWritingKV("nemotron_target_qkv_prepared_kv_affine4_gs64", preparedWritingKVSource)
    /// The cache-writing launch with the vector store (`nemotronQMVVectorStoreSource`), used once
    /// `qualifyVectorStore` matched it to the lane-0 launch.
    private static let preparedWritingKVKernelVS: MLXFast.MLXFastKernel? =
        nemotronQMVVectorStoreSource(preparedWritingKVSource).map {
            makeWritingKV("nemotron_target_qkv_prepared_kv_affine4_gs64_vs", $0)
        }
    private static var preparedWritingKVKernel: MLXFast.MLXFastKernel {
        vstoreVerified ? (preparedWritingKVKernelVS ?? preparedWritingKVKernelRecord) : preparedWritingKVKernelRecord
    }
    // NEMOTRON_MTP_FRONT_KV_DIRECT: the same prepared launch for the MTP head's
    // post-verify rows, whose K and V tiles also store each BF16 result into the
    // head cache's paged segment `storage` [2, pages, 2, 16, 128]: row r's token
    // goes to slot kvdst[2 + 2r] of segment-local page kvdst[1 + 2r], at the
    // segment write's element address (the value plane kvdst[0] elements after
    // the key plane). The stored value is the same `h` the body writes to ky/vy.
    // Thread 0 stores fence[0] = previous[0] + 1, the segment write's fence,
    // before the body runs. Every other instruction is the prepared body's.
    private static let storeAndPage = """
        yp[r * out_vec_size + row] = h;
                if (tile >= q_tiles) {
                  const uint n_out = uint(out_row) + uint(row);
                  const int page = nkv_page[r];
                  if (page > 0) {
                    const size_t target = (((size_t)page * 2 + n_out / 128) * 16
                        + (size_t)nkv_slot[r]) * 128 + (n_out % 128);
                    storage[(tile < q_tiles + kv_tiles ? (size_t)0 : (size_t)uint(nkv_plane)) + target] = h;
                  }
                }
        """
    private static let pagedFence = """
      if (thread_position_in_grid.x == 0 && thread_position_in_grid.y == 0) {
        fence[0] = previous[0] + 1;
      }

    """
    private static let preparedWritingPagedKernel: MLXFast.MLXFastKernel = {
        precondition(nemotronHeadPreparedQMVSource.components(separatedBy: storeLine).count == 2,
                     "NemotronTargetQKV: prepared body store not unique")
        return MLXFast.metalKernel(
            name: "nemotron_target_qkv_prepared_paged_kv_affine4_gs64",
            inputNames: ["qw", "qs", "qb", "kw", "ks", "kb", "vw", "vs", "vb", "prepared", "K",
                         "storage", "kvdst", "previous"],
            outputNames: ["qy", "ky", "vy", "fence"],
            source: pagedPreload + pagedFence + tilePrefix
                + nemotronHeadPreparedQMVSource.replacingOccurrences(of: storeLine, with: storeAndPage)
                + "\n}",
            header: nemotronMultiRowQMVHeader, mutableInputs: ["storage"])
    }()

    /// Kill switch (code constant; 0 = crown kernel): decode form of the
    /// R = 1 fused Q/K/V launch (MTP draft attention). The R = 3 prepared
    /// launch follows NemotronHeadPreparedActivation.halfDecodeForm.
    static let r1HalfDecodeForm = 2
    private static let kernelHalf = MLXFast.metalKernel(
        name: "nemotron_target_qkv_affine4_gs64_hf",
        inputNames: ["qw", "qs", "qb", "kw", "ks", "kb", "vw", "vs", "vb", "x", "K"],
        outputNames: ["qy", "ky", "vy"],
        source: tilePrefix + nemotronSwapped(nemotronMultiRowQMVSource, nemotronMultiRowHalfSwaps) + "\n}",
        header: nemotronMultiRowQMVHeader + nemotronHalfDecodeHeader)
    // The R = 1 launches reading (scale, bias) through the draft head's Q, K
    // and V pair dictionaries (`NemotronSBDict.draftDict`): qs / ks / vs carry
    // the uint16 pair indices and qb / kb / vb the FP32 pair tables; the body
    // is the same row body through `nemotronSBDictSource`.
    private static let dictTilePrefix = nemotronSwapped(tilePrefix, [
        ("const device T* scales = ", "const device uint16_t* scales = ", 1),
        ("const device T* biases = ", "const device float* biases = ", 1),
    ])
    private static let kernelDict = nemotronSBDictSource(nemotronMultiRowQMVSource).map {
        MLXFast.metalKernel(
            name: "nemotron_target_qkv_affine4_gs64_sbd",
            inputNames: ["qw", "qs", "qb", "kw", "ks", "kb", "vw", "vs", "vb", "x", "K"],
            outputNames: ["qy", "ky", "vy"], source: dictTilePrefix + $0 + "\n}",
            header: nemotronMultiRowQMVHeader)
    }
    private static let kernelHalfDict = nemotronSBDictSource(
        nemotronSwapped(nemotronMultiRowQMVSource, nemotronMultiRowHalfSwaps)).map {
        MLXFast.metalKernel(
            name: "nemotron_target_qkv_affine4_gs64_hf_sbd",
            inputNames: ["qw", "qs", "qb", "kw", "ks", "kb", "vw", "vs", "vb", "x", "K"],
            outputNames: ["qy", "ky", "vy"], source: dictTilePrefix + $0 + "\n}",
            header: nemotronMultiRowQMVHeader + nemotronHalfDecodeHeader)
    }
    /// The dictionary twin of the R = 1 launch `dispatch` selects, if built.
    static var draftDictAvailable: Bool {
        (r1HalfDecodeForm != 0 ? kernelHalfDict : kernelDict) != nil
    }
    private static let preparedKernelHalf = MLXFast.metalKernel(
        name: "nemotron_target_qkv_prepared_affine4_gs64_hf",
        inputNames: ["qw", "qs", "qb", "kw", "ks", "kb", "vw", "vs", "vb", "prepared", "K"],
        outputNames: ["qy", "ky", "vy"],
        source: tilePrefix + NemotronHeadPreparedActivation.preparedHalfSource + "\n}",
        header: NemotronHeadPreparedActivation.preparedHalfHeader)

    /// m3 load-form twins of the prepared launch (kill switch
    /// NemotronHeadPreparedActivation.qkvLoadForm; geometry stays crown).
    nonisolated(unsafe) private static var loadKernels: [String: MLXFast.MLXFastKernel] = [:]
    private static let loadKernelsLock = NSLock()
    private static func preparedLoadKernel(half: Bool, load: Int) -> MLXFast.MLXFastKernel {
        let name = "nemotron_target_qkv_prepared_affine4_gs64" + (half ? "_hf" : "") + "_ld\(load)"
        loadKernelsLock.lock(); defer { loadKernelsLock.unlock() }
        if let kernel = loadKernels[name] { return kernel }
        let body = nemotronPreparedVariantSource(
            half ? NemotronHeadPreparedActivation.preparedHalfSource : nemotronHeadPreparedQMVSource,
            load: load, nsg: false)
        let kernel = MLXFast.metalKernel(
            name: name,
            inputNames: ["qw", "qs", "qb", "kw", "ks", "kb", "vw", "vs", "vb", "prepared", "K"],
            outputNames: ["qy", "ky", "vy"], source: tilePrefix + body + "\n}",
            header: NemotronHeadPreparedActivation.variantHeader(half: half, load: load))
        loadKernels[name] = kernel
        return kernel
    }

    /// Outputs per SIMD group for the fused Q/K/V launch: narrow at R = 3
    /// and 4 (M1 Max: -9 % and -12 %; R = 1 +7 %). Same kill switch as
    /// NemotronMultiRowQMV.
    static func resultsPerSimdgroup(rows: Int) -> Int {
        NemotronMultiRowQMV.narrowEnabled && (rows == 3 || rows == 4) ? 2 : 4
    }

    private static func buffers(_ linear: Linear, rows: Int) -> [MLXArray]? {
        guard type(of: linear) == QuantizedLinear.self, let q = linear as? QuantizedLinear,
              q.mode == .affine, q.bits == 4, q.groupSize == 64, q.bias == nil,
              q.weight.dtype == .uint32, q.weight.shape == [rows, 336],
              q.scales.dtype == .bfloat16, q.scales.shape == [rows, 42],
              let offsets = q.biases, offsets.dtype == .bfloat16, offsets.shape == [rows, 42]
        else { return nil }
        return [q.weight, q.scales, offsets]
    }

    private static func dispatch(_ weights: [MLXArray], _ x: MLXArray, rps: Int,
                                 prepared: Bool = false,
                                 packed: MLXArray? = nil,
                                 dicts: [NemotronSBDict.Dict]? = nil) -> [MLXArray] {
        let rows = x.dim(1)
        // `packed`: the same K=2688 (PV 8) layout `pack` writes, prepared by
        // the preceding join from these exact BF16 rows.
        let activations = prepared
            ? (NemotronHeadPreparedActivation.compatible(packed, rows: rows, k: 2688)
                ?? NemotronHeadPreparedActivation.pack(x.reshaped(rows, 2688)))
            : x
        let preparedForm = NemotronHeadPreparedActivation.decodeForm(rows: rows, fast: false)
        let r1Half = !prepared && rows == 1 && r1HalfDecodeForm != 0
        let qkvLoad = prepared && rows == 3 ? NemotronHeadPreparedActivation.qkvLoadForm : 0
        // `dicts` (Q, K, V; one row only): the selected R = 1 kernel's dictionary twin.
        let twin = dicts == nil || prepared || rows != 1 ? nil : (r1Half ? kernelHalfDict : kernelDict)
        // `NemotronDraftTune`: the geometry adopted at load for the draft head's R = 1 dictionary launch.
        if twin != nil, r1Half, let dicts, dicts.count == 3, let g = NemotronDraftTune.adopted[.qkv],
           let outputs = tuneLaunch(g, weights: [weights[0], weights[3], weights[6]], indices: dicts.map(\.index),
                                    tables: dicts.map(\.table), x: x, dep: x) {
            return outputs
        }
        let chosen = twin ?? (prepared
            ? (qkvLoad != 0 ? preparedLoadKernel(half: preparedForm != nil, load: qkvLoad)
               : (preparedForm == nil ? preparedKernel : preparedKernelHalf))
            : (r1Half ? kernelHalf : kernel))
        let extra = prepared
            ? nemotronHalfTemplate(preparedForm != nil, preparedForm ?? 0)
            : nemotronHalfTemplate(r1Half, r1HalfDecodeForm)
        let bound = twin == nil ? weights
            : (0 ..< 3).flatMap { [weights[3 * $0], dicts![$0].index, dicts![$0].table] }
        return chosen(
            bound + [activations, NemotronMultiRowQMV.scalarPair(2688, 4096).0],
            template: [("T", x.dtype), ("R", rows), ("FAST", false), ("RELU2", false),
                       ("RPS", rps)] + extra,
            grid: (32, 4608 / rps, 1), threadGroup: (32, 2, 1),
            outputShapes: [[1, rows, 4096], [1, rows, 256], [1, rows, 256]],
            outputDTypes: [x.dtype, x.dtype, x.dtype])
    }

    /// `sbDict`: one row of the draft head reads its admitted Q, K and V pair
    /// dictionaries (`NemotronSBDict.draftDict`, all three or none); false
    /// keeps the record kernel.
    static func apply(
        _ x: MLXArray, query: Linear, key: Linear, value: Linear, captureMTP: Bool,
        prepared packed: MLXArray? = nil, sbDict: Bool = true
    ) -> (MLXArray, MLXArray, MLXArray)? {
        guard Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              x.ndim == 3, x.dim(0) == 1, x.dim(2) == 2688, x.dtype == .bfloat16,
              x.dim(1) == 1 || (captureMTP && [3, 4].contains(x.dim(1))
                && NemotronMTPExecution.batchedM1 && NemotronMultiRowQMV.enabled),
              let q = buffers(query, rows: 4096), let k = buffers(key, rows: 256),
              let v = buffers(value, rows: 256)
        else { return nil }
        let rps = resultsPerSimdgroup(rows: x.dim(1))
        let prepared = preparedEnabled && rps == 2 && x.dim(1) == 3
        let dicts = sbDict && x.dim(1) == 1
            ? [query, key, value].compactMap { NemotronSBDict.draftDict($0) } : []
        var outputs = dispatch(q + k + v, x, rps: rps, prepared: prepared, packed: packed,
                               dicts: dicts.count == 3 ? dicts : nil)
        if rps != 4, NemotronMoELeader.selfTest {
            let expected = dispatch(q + k + v, x, rps: 4)
            outputs = zip(outputs, expected).map { nemotronBitwiseOrNaN($0, $1) }
        }
        return (outputs[0], outputs[1], outputs[2])
    }

    /// `apply` for the captured [1, 3] verify window on the prepared launch,
    /// also storing the K/V rows straight into the layer's contiguous cache
    /// `[1, 2, capacity, 128]` at tokens `offset ..< offset + 3` (mutable
    /// inputs; the caller orders later reads after this launch). Returns the
    /// usual (q, k, v), or nil (nothing launched) when the shapes, dtypes or
    /// switches do not fit.
    static func applyWritingKV(
        _ x: MLXArray, query: Linear, key: Linear, value: Linear,
        prepared packed: MLXArray?, kCache: MLXArray, vCache: MLXArray,
        offset: Int, capacity: Int
    ) -> (MLXArray, MLXArray, MLXArray)? {
        guard CBv2CriticalPathCuts.kvInPlaceAppend,
              Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              x.ndim == 3, x.dim(0) == 1, x.dim(1) == 3, x.dim(2) == 2688, x.dtype == .bfloat16,
              NemotronMTPExecution.batchedM1, NemotronMultiRowQMV.enabled,
              preparedEnabled, resultsPerSimdgroup(rows: 3) == 2,
              kCache.dtype == .bfloat16, vCache.dtype == .bfloat16,
              kCache.shape == [1, 2, capacity, 128], vCache.shape == kCache.shape,
              offset >= 0, offset + 3 <= capacity,
              let q = buffers(query, rows: 4096), let k = buffers(key, rows: 256),
              let v = buffers(value, rows: 256)
        else { return nil }
        let activations = NemotronHeadPreparedActivation.compatible(packed, rows: 3, k: 2688)
            ?? NemotronHeadPreparedActivation.pack(x.reshaped(3, 2688))
        var outputs = preparedWritingKVKernel(
            q + k + v + [activations, NemotronMultiRowQMV.scalarPair(2688, 4096).0,
                         kCache, vCache, MLXArray([Int32(offset), Int32(capacity)])],
            template: [("T", x.dtype), ("R", 3), ("FAST", false), ("RELU2", false), ("RPS", 2)],
            grid: (32, 4608 / 2, 1), threadGroup: (32, 2, 1),
            outputShapes: [[1, 3, 4096], [1, 3, 256], [1, 3, 256]],
            outputDTypes: [x.dtype, x.dtype, x.dtype])
        if NemotronMoELeader.selfTest {
            let expected = dispatch(q + k + v, x, rps: 4)
            outputs = zip(outputs, expected).map { nemotronBitwiseOrNaN($0, $1) }
        }
        return (outputs[0], outputs[1], outputs[2])
    }

    /// `apply` for the MTP head's three post-verify rows on the prepared launch `dispatch` selects for them,
    /// also storing their K/V rows into a paged segment: row r at slot `slots[r]` of segment-local page
    /// `pages[r]` of `storage` ([2, pages, 2, 16, 128] BF16, the value plane `valueOffset` elements after the
    /// key plane), the segment write's addresses. `storage` is a mutable input; `fence` (the group's write fence,
    /// [1] int32) is bound, and the returned fence is fence + 1 from the same launch, so later reads and writes
    /// order after the stores when it replaces the group's fence. Returns (q, k, v, fence), or nil (nothing
    /// launched) when the shapes, dtypes, switches or the selected body do not fit.
    static func applyWritingPaged(
        _ x: MLXArray, query: Linear, key: Linear, value: Linear, prepared packed: MLXArray?,
        storage: MLXArray, valueOffset: Int, pages: [Int32], slots: [Int32], fence: MLXArray
    ) -> (MLXArray, MLXArray, MLXArray, MLXArray)? {
        guard Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              x.ndim == 3, x.dim(0) == 1, x.dim(1) == 3, x.dim(2) == 2688, x.dtype == .bfloat16,
              NemotronMTPExecution.batchedM1, NemotronMultiRowQMV.enabled,
              preparedEnabled, resultsPerSimdgroup(rows: 3) == 2,
              NemotronHeadPreparedActivation.qkvLoadForm == 0,
              NemotronHeadPreparedActivation.decodeForm(rows: 3, fast: false) == nil,
              storage.dtype == .bfloat16, storage.ndim == 5, storage.dim(0) == 2,
              storage.shape == [2, storage.dim(1), 2, 16, 128],
              valueOffset == storage.dim(1) * 2 * 16 * 128, valueOffset < Int(Int32.max),
              pages.count == 3, slots.count == 3,
              pages.allSatisfy({ $0 > 0 && Int($0) < storage.dim(1) }),
              slots.allSatisfy({ (0 ..< 16).contains($0) }),
              fence.shape == [1], fence.dtype == .int32,
              let q = buffers(query, rows: 4096), let k = buffers(key, rows: 256),
              let v = buffers(value, rows: 256)
        else { return nil }
        let activations = NemotronHeadPreparedActivation.compatible(packed, rows: 3, k: 2688)
            ?? NemotronHeadPreparedActivation.pack(x.reshaped(3, 2688))
        // [value plane offset, (page, slot) x 3, pad]: eight entries keep the device address space.
        var destination: [Int32] = [Int32(valueOffset)]
        for r in 0 ..< 3 {
            destination += [pages[r], slots[r]]
        }
        destination.append(0)
        var outputs = preparedWritingPagedKernel(
            q + k + v + [activations, NemotronMultiRowQMV.scalarPair(2688, 4096).0,
                         storage, MLXArray(destination), fence],
            template: [("T", x.dtype), ("R", 3), ("FAST", false), ("RELU2", false), ("RPS", 2)],
            grid: (32, 4608 / 2, 1), threadGroup: (32, 2, 1),
            outputShapes: [[1, 3, 4096], [1, 3, 256], [1, 3, 256], [1]],
            outputDTypes: [x.dtype, x.dtype, x.dtype, .int32])
        if NemotronMoELeader.selfTest {
            let expected = dispatch(q + k + v, x, rps: 4)
            for i in 0 ..< 3 {
                outputs[i] = nemotronBitwiseOrNaN(outputs[i], expected[i])
            }
        }
        return (outputs[0], outputs[1], outputs[2], outputs[3])
    }

    /// Load-time JIT of the cache-writing launch (scratch caches only).
    static func warmWritingKV() -> [MLXArray] {
        guard CBv2CriticalPathCuts.kvInPlaceAppend, preparedEnabled,
              resultsPerSimdgroup(rows: 3) == 2 else { return [] }
        let weights = [4096, 256, 256].flatMap { n in
            [MLXArray.zeros([n, 336], dtype: .uint32),
             MLXArray.zeros([n, 42], dtype: .bfloat16),
             MLXArray.zeros([n, 42], dtype: .bfloat16)]
        }
        let x = MLXArray.zeros([1, 3, 2688], dtype: .bfloat16)
        let cache = { MLXArray.zeros([1, 2, 16, 128], dtype: .bfloat16) }
        let kc = cache(), vc = cache()
        eval(kc, vc)
        return preparedWritingKVKernel(
            weights + [NemotronHeadPreparedActivation.pack(x.reshaped(3, 2688)),
                       NemotronMultiRowQMV.scalarPair(2688, 4096).0, kc, vc,
                       MLXArray([Int32(0), Int32(16)])],
            template: [("T", DType.bfloat16), ("R", 3), ("FAST", false), ("RELU2", false),
                       ("RPS", 2)],
            grid: (32, 4608 / 2, 1), threadGroup: (32, 2, 1),
            outputShapes: [[1, 3, 4096], [1, 3, 256], [1, 3, 256]],
            outputDTypes: [.bfloat16, .bfloat16, .bfloat16])
    }

    /// Narrow pipelines for the widths that use them (R = 3, 4).
    static func warmNarrow() -> [MLXArray] {
        let weights = [4096, 256, 256].flatMap { n in
            [MLXArray.zeros([n, 336], dtype: .uint32),
             MLXArray.zeros([n, 42], dtype: .bfloat16),
             MLXArray.zeros([n, 42], dtype: .bfloat16)]
        }
        var outputs = [3, 4].flatMap { rows in
            dispatch(weights, MLXArray.zeros([1, rows, 2688], dtype: .bfloat16), rps: 2)
        }
        if preparedEnabled {
            outputs += dispatch(weights, MLXArray.zeros([1, 3, 2688], dtype: .bfloat16),
                                rps: 2, prepared: true)
        }
        outputs += warmWritingKV()
        return outputs
    }

    /// NEMOTRON_MORE_VSTORE: the prepared and cache-writing launches take their vector-store twins.
    nonisolated(unsafe) static var vstoreVerified = false

    /// Load time, before any verify or draft graph is traced: the vector-store twins of the prepared launch (the
    /// MTP head's post-verify rows) and of the cache-writing launch (the verify's attention layers) against their
    /// lane-0 launches, bit for bit (q, k, v, and every element of both caches), on the real Q/K/V weights of the
    /// first and last target attention layers and the MTP head's, with seeded rows: Gaussian rows at two scales,
    /// an all-zero row, -0 entries, a one-hot row, tiny (1e-30) and large (±3e4) values; the cache rows land at
    /// two offsets of a cache holding Gaussian values. The crown prepared QMV's twin
    /// (`NemotronHeadPreparedActivation.projectVS`) the same way on those layers' Q and K weights and on the MTP
    /// head's eh_proj (K 5376), with and without ReLU². The lane-0 launches stay on any miss. One stderr line.
    static func qualifyVectorStore(target: Module, head: Module) {
        vstoreVerified = false
        NemotronHeadPreparedActivation.projectVSVerified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron more vstore: \(message)\n".utf8))
        }
        guard nemotronMoreVectorStore else { return report("off (setting)") }
        guard Device.defaultDevice().deviceType == .gpu, preparedEnabled, resultsPerSimdgroup(rows: 3) == 2
        else { return report("off (prepared Q/K/V launch not used)") }
        guard let preparedVS = preparedKernelVS, let writingVS = preparedWritingKVKernelVS
        else { return report("off (a prepared Q/K/V body no longer matches)") }
        let targets = target.modules().compactMap { $0 as? NemotronHAttention }
        let heads = head.modules().compactMap { $0 as? NemotronHAttention }
        let layers = [targets.first, targets.last, heads.first].compactMap { $0 }
        let weightSets = layers.compactMap { a -> [MLXArray]? in
            guard let q = buffers(a.wq, rows: 4096), let k = buffers(a.wk, rows: 256),
                  let v = buffers(a.wv, rows: 256) else { return nil }
            return q + k + v
        }
        guard weightSets.count == 3 else { return report("off (no prepared Q/K/V weights)") }
        // The MTP head's eh_proj (K 5376, N 2688) when it is plain affine 4-bit: the crown launch it takes is the one
        // `NemotronDraftTune` (ehp) rebuilds, so the crown check also runs on its real weights.
        let ehProj: [MLXArray]? = {
            guard let linear = head.modules().lazy.compactMap({ $0 as? NemotronH35MTPAttentionBlock }).first?
                      .projection,
                  type(of: linear) == QuantizedLinear.self, let q = linear as? QuantizedLinear,
                  q.mode == .affine, q.bits == 4, q.groupSize == 64, q.bias == nil,
                  q.weight.dtype == .uint32, q.weight.shape == [2688, 672],
                  q.scales.dtype == .bfloat16, q.scales.shape == [2688, 84],
                  let biases = q.biases, biases.dtype == .bfloat16, biases.shape == [2688, 84]
            else { return nil }
            return [q.weight, q.scales, biases]
        }()
        var ehCases = 0
        let keys = MLXRandom.split(key: MLXRandom.key(0x7652_5153), into: 6)
        let bf16 = { (v: Float) in MLXArray(v).asType(.bfloat16) }
        let row = MLXArray((0 ..< 3).map(Int32.init)).reshaped(1, 3, 1)
        let column = MLXArray((0 ..< 2688).map(Int32.init)).reshaped(1, 1, 2688)
        let gaussian = MLXRandom.normal([1, 3, 2688], key: keys[0]).asType(.bfloat16)
        let small = (MLXRandom.normal([1, 3, 2688], key: keys[1]) * 0.01).asType(.bfloat16)
        let u = MLXRandom.uniform(Float(0) ..< 1, [1, 3, 2688], key: keys[2])
        // Row 0 sparse with -0 for some zeros, row 1 all zero, row 2 one-hot.
        let sparse = which(row .== MLXArray(Int32(0)),
                           which(u .< MLXArray(Float(0.3)), small, which(u .< MLXArray(Float(0.6)), bf16(-0.0), bf16(0))),
                           which(row .== MLXArray(Int32(1)), bf16(0),
                                 which(column .== MLXArray(Int32(1337)), bf16(1.5), bf16(0))))
        // Row 0 tiny, row 1 large, row 2 Gaussian with -0 and zero entries.
        let edges = which(row .== MLXArray(Int32(0)), (gaussian.asType(.float32) * 1e-30).asType(.bfloat16),
                          which(row .== MLXArray(Int32(1)), (small.asType(.float32) * 3e6).asType(.bfloat16),
                                which(u .< MLXArray(Float(0.1)), bf16(-0.0),
                                      which(u .< MLXArray(Float(0.2)), bf16(0), gaussian))))
        let rows = [gaussian, small, sparse, edges]
        let template: [(String, any KernelTemplateArg)] =
            [("T", DType.bfloat16), ("R", 3), ("FAST", false), ("RELU2", false), ("RPS", 2)]
        let outputShapes = [[1, 3, 4096], [1, 3, 256], [1, 3, 256]]
        let outputDTypes: [DType] = [.bfloat16, .bfloat16, .bfloat16]
        let k = NemotronMultiRowQMV.scalarPair(2688, 4096).0
        func same(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
            (a.view(dtype: .uint16) .== b.view(dtype: .uint16)).all()
        }
        var cases = 0
        var verdict = ""
        do {
            let matched = try withError { () -> Bool in
                var ok = MLXArray(true)
                var okProject = MLXArray(true)
                var okWriting = MLXArray(true)
                for (l, weights) in weightSets.enumerated() {
                    for (i, x) in rows.enumerated() {
                        let packed = NemotronHeadPreparedActivation.pack(x.reshaped(3, 2688))
                        let a = preparedKernelRecord(weights + [packed, k], template: template,
                            grid: (32, 4608 / 2, 1), threadGroup: (32, 2, 1),
                            outputShapes: outputShapes, outputDTypes: outputDTypes)
                        let b = preparedVS(weights + [packed, k], template: template,
                            grid: (32, 4608 / 2, 1), threadGroup: (32, 2, 1),
                            outputShapes: outputShapes, outputDTypes: outputDTypes)
                        for j in 0 ..< 3 { ok = ok .&& same(a[j], b[j]) }
                        cases += 1
                        // The crown prepared QMV (`NemotronHeadPreparedActivation.project`) on this layer's
                        // Q (N = 4096) and K (N = 256) weights, with and without ReLU².
                        if let projectVS = NemotronHeadPreparedActivation.projectVS {
                            for (index, n) in [(0, 4096), (3, 256)] {
                                let args = NemotronMultiRowQMV.scalarPair(2688, n)
                                for relu2 in [false, true] {
                                    let inputs = Array(weights[index ..< index + 3]) + [packed, args.0, args.1]
                                    let projectTemplate: [(String, any KernelTemplateArg)] = [
                                        ("T", DType.bfloat16), ("R", 3), ("FAST", false), ("RELU2", relu2), ("RPS", 2)]
                                    let e = NemotronHeadPreparedActivation.projectRecord(inputs, template: projectTemplate,
                                        grid: (32, n / 2, 1), threadGroup: (32, 2, 1),
                                        outputShapes: [[3, n]], outputDTypes: [.bfloat16])[0]
                                    let f = projectVS(inputs, template: projectTemplate,
                                        grid: (32, n / 2, 1), threadGroup: (32, 2, 1),
                                        outputShapes: [[3, n]], outputDTypes: [.bfloat16])[0]
                                    okProject = okProject .&& same(e, f)
                                    cases += 1
                                }
                            }
                        }
                        guard l < 2 else { continue }
                        // The cache-writing launch on the target layers, rows at two offsets.
                        let offset = i % 2 == 0 ? 0 : 13
                        // Two caches with Gaussian contents, and a copy of each for the twin.
                        let initial = (0 ..< 2).map {
                            MLXRandom.normal([1, 2, 16, 128], key: MLXRandom.key(UInt64(0x7653_0000 + 2 * i + $0)))
                                .asType(.bfloat16)
                        }
                        let caches = initial + initial.map { $0 * bf16(1) }
                        eval(caches)
                        let position = MLXArray([Int32(offset), Int32(16)])
                        let c = preparedWritingKVKernelRecord(
                            weights + [packed, k, caches[0], caches[1], position], template: template,
                            grid: (32, 4608 / 2, 1), threadGroup: (32, 2, 1),
                            outputShapes: outputShapes, outputDTypes: outputDTypes)
                        let d = writingVS(
                            weights + [packed, k, caches[2], caches[3], position], template: template,
                            grid: (32, 4608 / 2, 1), threadGroup: (32, 2, 1),
                            outputShapes: outputShapes, outputDTypes: outputDTypes)
                        eval(c + d)
                        for j in 0 ..< 3 { okWriting = okWriting .&& same(c[j], d[j]) .&& same(a[j], c[j]) }
                        let stored = depends(inputs: caches, dependencies: c + d)
                        okWriting = okWriting .&& same(stored[0], stored[2]) .&& same(stored[1], stored[3])
                        cases += 1
                    }
                }
                // The crown prepared QMV on the MTP head's eh_proj: each row set beside the next one (K 5376),
                // with and without ReLU².
                if let projectVS = NemotronHeadPreparedActivation.projectVS, let ehProj {
                    let args = NemotronMultiRowQMV.scalarPair(5376, 2688)
                    for (i, x) in rows.enumerated() {
                        let packed = NemotronHeadPreparedActivation.pack(
                            concatenated([x, rows[(i + 1) % rows.count]], axis: 2).reshaped(3, 5376))
                        for relu2 in [false, true] {
                            let inputs = ehProj + [packed, args.0, args.1]
                            let projectTemplate: [(String, any KernelTemplateArg)] = [
                                ("T", DType.bfloat16), ("R", 3), ("FAST", false), ("RELU2", relu2), ("RPS", 2)]
                            let e = NemotronHeadPreparedActivation.projectRecord(inputs, template: projectTemplate,
                                grid: (32, 2688 / 2, 1), threadGroup: (32, 2, 1),
                                outputShapes: [[3, 2688]], outputDTypes: [.bfloat16])[0]
                            let f = projectVS(inputs, template: projectTemplate,
                                grid: (32, 2688 / 2, 1), threadGroup: (32, 2, 1),
                                outputShapes: [[3, 2688]], outputDTypes: [.bfloat16])[0]
                            okProject = okProject .&& same(e, f)
                            cases += 1
                            ehCases += 1
                        }
                    }
                }
                eval(ok, okProject, okWriting)
                verdict = "prepared Q/K/V \(ok.item(Bool.self)), crown prepared QMV \(okProject.item(Bool.self)), "
                    + "cache-writing Q/K/V \(okWriting.item(Bool.self))"
                return (ok .&& okProject .&& okWriting).item(Bool.self)
            }
            vstoreVerified = matched
            NemotronHeadPreparedActivation.projectVSVerified = matched && NemotronHeadPreparedActivation.projectVS != nil
        } catch {
            return report("off (\(error))")
        }
        report(vstoreVerified
            ? "on (\(cases) cases bitwise equal to the lane-0 store: prepared Q/K/V on the first and last target "
                + "attention layers and the MTP head, cache-writing Q/K/V with both caches on the target layers, "
                + "crown prepared QMV \(NemotronHeadPreparedActivation.projectVSVerified ? "on" : "off") "
                + "(Q and K weights" + (ehCases > 0 ? ", the MTP head's eh_proj" : "")
                + ", ReLU² off and on); zero, -0, one-hot, tiny and large rows included)"
            : "off (differs from the lane-0 store: \(verdict))")
    }
}

// Draft-only greedy head: partial-K screen, per-chunk keep, exact rescore.
// The draft head's argmax over the half-vocabulary prefix of the shared
// affine 4-bit g64 lm_head reads every row's full K. This pipeline reads, for
// every row, only S of the 42 K-groups (chosen once at load from the weights),
// keeps the top C rows of each CH-row chunk by that partial score, rescores
// the kept rows with the stock qmv's own per-row arithmetic (so each kept
// value is bitwise the stock BF16 logit), and returns the argmax with the
// stock lowest-id tie-break. It changes draft proposals only when the stock
// argmax is not kept; the target still verifies every token.
enum NemotronHeadScreen {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_MTP_HEAD_SCREEN"] != "0"
    static let check = ProcessInfo.processInfo.environment["NEMOTRON_MTP_HEAD_SCREEN_CHECK"] == "1"
    /// NEMOTRON_MTP_HEAD_SCREEN_PACKED=0 gathers scales/biases from the head rows.
    static var packedScalesBiases: Bool {
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_HEAD_SCREEN_PACKED"] != "0"
            && packedPartialSource.components(separatedBy: "sb[size_t(row) * (2 * S)").count == 3
            && !packedPartialSource.contains("scales[") && !packedPartialSource.contains("biases[")
    }
    /// NEMOTRON_MTP_HEAD_SCREEN_PACK_FULL=0 keeps the weight words in place
    /// (packed scales/biases only); NEMOTRON_MTP_HEAD_SCREEN_PACKED=0 turns
    /// off both.
    static var packedRecords: Bool {
        packedScalesBiases
            && ProcessInfo.processInfo.environment["NEMOTRON_MTP_HEAD_SCREEN_PACK_FULL"] != "0"
            && recordPartialSource.contains("(rp + (lane >> 1) * 8 + part * 4)")
            && !recordPartialSource.contains("(w + ") && !recordPartialSource.contains("sb[size_t(row)")
    }
    static let width = 2688
    static let wordsPerRow = 336
    static let groupsPerRow = 42
    static let headRows = 131072
    static let rows = 65536

    // One simdgroup scores RPS rows. Lane l owns half-groups l, l + 32, ...
    // (32 values = one uint4 of packed words) of the selected groups. Rows
    // from LO up (the upper vocabulary half, when scored) read their own,
    // smaller group set S2 / groups2; LO is a multiple of RPS, so the choice
    // is uniform per simdgroup.
    private static let partialSource = """
      // S, S2 <= 16: lane l owns half-group l (32 values, 16 bytes).
      constexpr int KW = 336;
      constexpr int KG = 42;
      constexpr int VPL = 32;
      const int lane = int(thread_index_in_simdgroup);
      const int sg = int(threadgroup_position_in_grid.y) * NSG + int(simdgroup_index_in_threadgroup);
      const int row0 = sg * RPS;
      const bool upper = row0 >= LO;
      const int H = upper ? 2 * S2 : 2 * S;
      const bool on = lane < H;
      const int g = on ? int(upper ? groups2[lane >> 1] : groups[lane >> 1]) : 0;
      const int part = lane & 1;
      const int woff = g * 8 + part * 4;
      const device T* xp = x + g * 64 + part * 32;
      float xv[VPL];
      float xs = 0.0f;
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < VPL; i += 4) {
        const float x0 = on ? float(xp[i]) : 0.0f;
        const float x1 = on ? float(xp[i + 1]) : 0.0f;
        const float x2 = on ? float(xp[i + 2]) : 0.0f;
        const float x3 = on ? float(xp[i + 3]) : 0.0f;
        xs += x0 + x1 + x2 + x3;
        xv[i] = x0;
        xv[i + 1] = x1 / 16.0f;
        xv[i + 2] = x2 / 256.0f;
        xv[i + 3] = x3 / 4096.0f;
      }
      _Pragma("clang loop unroll(full)")
      for (int r = 0; r < RPS; r++) {
        const int row = row0 + r;
        float acc = 0.0f;
        if (on) {
          const device uint16_t* ws = (const device uint16_t*)(w + size_t(row) * KW + woff);
          const float sc = float(scales[size_t(row) * KG + g]);
          const float bi = float(biases[size_t(row) * KG + g]);
          float a = 0.0f;
          _Pragma("clang loop unroll(full)")
          for (int i = 0; i < VPL / 4; i++) {
            const uint16_t q = ws[i];
            a += xv[4 * i] * float(q & 0x000f) + xv[4 * i + 1] * float(q & 0x00f0)
                + xv[4 * i + 2] * float(q & 0x0f00) + xv[4 * i + 3] * float(q & 0xf000);
          }
          acc = sc * a + bi * xs;
        }
        acc = simd_sum(acc);
        if (lane == 0) {
          scores[row] = acc;
        }
      }
    """

    // One threadgroup per CH-row chunk. Simdgroup 0 keeps the top C partial
    // scores (ties: lowest id); then NSG
    // simdgroups rescore 4 kept rows each with the stock non-fast qmv per-row
    // body (K = 2688: ten 256-blocks plus a 128-value tail), as
    // nemotronMultiRowQMVSource does for R = 1, and round to T.
    private static let selectCrownSource = """
      constexpr int P1 = CH / 32;
      constexpr int RPSG = 4;
      constexpr int in_vec_size = 2688;
      constexpr int in_vec_size_w = 1344;
      constexpr int in_vec_size_g = 42;
      constexpr int values_per_thread = 8;
      constexpr int block_size = 256;
      constexpr int scale_step_per_thread = 8;
      typedef float U;
      const int lane = int(thread_index_in_simdgroup);
      const int sgid = int(simdgroup_index_in_threadgroup);
      const int chunk = int(threadgroup_position_in_grid.y);
      const int base = chunk * CH;
      threadgroup int cand[C];
      threadgroup float cval[C];
      // Simdgroup 0 keeps the chunk's top C partial scores (larger score,
      // then lower id); lane l holds the P1 contiguous rows l * P1 + i.
      if (sgid == 0) {
        float v[P1];
        const device float* sp = scores + base + lane * P1;
        for (int i = 0; i < P1; i++) v[i] = sp[i];
        ulong taken = 0;
        float bv = -INFINITY;
        int bi = -1;
        for (int i = 0; i < P1; i++) {
          if (bi < 0 || v[i] > bv) { bv = v[i]; bi = i; }
        }
        for (int c = 0; c < C; c++) {
          float mv = bv;
          int mi = bi < 0 ? 0x7fffffff : lane * P1 + bi;
          for (ushort off = 16; off > 0; off >>= 1) {
            const float ov = simd_shuffle_xor(mv, off);
            const int oi = simd_shuffle_xor(mi, off);
            if (oi != 0x7fffffff && (mi == 0x7fffffff || ov > mv || (ov == mv && oi < mi))) {
              mv = ov; mi = oi;
            }
          }
          if (lane == 0) cand[c] = base + mi;
          if (mi / P1 == lane) {
            taken |= (1ul << ulong(mi - lane * P1));
            bv = -INFINITY;
            bi = -1;
            for (int i = 0; i < P1; i++) {
              if (((taken >> ulong(i)) & 1ul) == 0 && (bi < 0 || v[i] > bv)) { bv = v[i]; bi = i; }
            }
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      const device uint8_t* wbytes = (const device uint8_t*)w;
      for (int slot0 = sgid * RPSG; slot0 < C; slot0 += NSG * RPSG) {
        const device uint8_t* wl[RPSG];
        const device T* sl[RPSG];
        const device T* bl[RPSG];
        for (int row = 0; row < RPSG; row++) {
          const size_t id = size_t(cand[slot0 + row]);
          wl[row] = wbytes + id * in_vec_size_w + lane * 4;
          sl[row] = scales + id * in_vec_size_g + lane / scale_step_per_thread;
          bl[row] = biases + id * in_vec_size_g + lane / scale_step_per_thread;
        }
        const device T* xp = x + lane * values_per_thread;
        thread U x_thread[values_per_thread];
        thread U result[RPSG];
        for (int row = 0; row < RPSG; row++) result[row] = 0;
        U sum;
        int k = 0;
        constexpr int k_end = in_vec_size - block_size;
        for (; k < k_end; k += block_size) {
          sum = nmr_load_vector<T, U, values_per_thread>(xp, x_thread);
          for (int row = 0; row < RPSG; row++) {
            U s = sl[row][0];
            U b = bl[row][0];
            U accum = 0;
            const device uint16_t* words = (const device uint16_t*)wl[row];
            for (int i = 0; i < values_per_thread / 4; i++) {
              const U a = words[i] & 0x000f;
              const U b1 = words[i] & 0x00f0;
              const U c = words[i] & 0x0f00;
              const U d = words[i] & 0xf000;
              accum += x_thread[4*i] * a + x_thread[4*i+1] * b1
                  + x_thread[4*i+2] * c + x_thread[4*i+3] * d;
            }
            result[row] += s * accum + b * sum;
            wl[row] += block_size / 2;
            sl[row] += block_size / 64;
            bl[row] += block_size / 64;
          }
          xp += block_size;
        }
        const int remaining = clamp(
            static_cast<int>(in_vec_size - k - lane * values_per_thread),
            0,
            values_per_thread);
        if (remaining > 0) {
          sum = nmr_load_vector_safe<T, U, values_per_thread>(xp, x_thread, remaining);
          for (int row = 0; row < RPSG; row++) {
            U s = sl[row][0];
            U b = bl[row][0];
            U accum = 0;
            const device uint16_t* words = (const device uint16_t*)wl[row];
            for (int i = 0; i < remaining / 4; i++) {
              const U a = words[i] & 0x000f;
              const U b1 = words[i] & 0x00f0;
              const U c = words[i] & 0x0f00;
              const U d = words[i] & 0xf000;
              accum += x_thread[4*i] * a + x_thread[4*i+1] * b1
                  + x_thread[4*i+2] * c + x_thread[4*i+3] * d;
            }
            result[row] += s * accum + b * sum;
          }
        }
        for (int row = 0; row < RPSG; row++) {
          U v = simd_sum(result[row]);
          if (lane == 0) {
            T h = static_cast<T>(v);
            cval[slot0 + row] = float(h);
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (sgid == 0) {
        float mv = -INFINITY;
        int mi = 0x7fffffff;
        for (int s2 = lane; s2 < C; s2 += 32) {
          const float v = cval[s2];
          const int id = cand[s2];
          cand_ids[chunk * C + s2] = id;
          cand_vals[chunk * C + s2] = v;
          if (mi == 0x7fffffff || v > mv || (v == mv && id < mi)) { mv = v; mi = id; }
        }
        for (ushort off = 16; off > 0; off >>= 1) {
          const float ov = simd_shuffle_xor(mv, off);
          const int oi = simd_shuffle_xor(mi, off);
          if (oi != 0x7fffffff && (mi == 0x7fffffff || ov > mv || (ov == mv && oi < mi))) {
            mv = ov; mi = oi;
          }
        }
        if (lane == 0) {
          best_val[chunk] = mv;
          best_id[chunk] = mi;
        }
      }
    """
    private static let selectSource = nemotronMultiRowQMVTail8
        ? nemotronSwapped(selectCrownSource, [
            ("sum = nmr_load_vector_safe<T, U, values_per_thread>(xp, x_thread, remaining);",
             "sum = nmr_load_vector<T, U, values_per_thread>(xp, x_thread);", 1),
            ("for (int i = 0; i < remaining / 4; i++) {",
             "for (int i = 0; i < values_per_thread / 4; i++) {", 1),
        ])
        : selectCrownSource

    private static let finalSource = """
      const int lane = int(thread_index_in_simdgroup);
      float mv = -INFINITY;
      int mi = 0x7fffffff;
      for (int c = lane; c < NCH; c += 32) {
        const float v = best_val[c];
        const int id = best_id[c];
        if (mi == 0x7fffffff || v > mv || (v == mv && id < mi)) { mv = v; mi = id; }
      }
      for (ushort off = 16; off > 0; off >>= 1) {
        const float ov = simd_shuffle_xor(mv, off);
        const int oi = simd_shuffle_xor(mi, off);
        if (oi != 0x7fffffff && (mi == 0x7fffffff || ov > mv || (ov == mv && oi < mi))) {
          mv = ov; mi = oi;
        }
      }
      if (lane == 0) {
        out[0] = mi;
      }
    """

    private static let partialKernel = MLXFast.metalKernel(
        name: "nemotron_head_screen_partial_affine4_gs64",
        inputNames: ["w", "scales", "biases", "x", "groups", "groups2"],
        outputNames: ["scores"], source: partialSource)
    /// The same body reading each row's selected scales and biases from one
    /// packed bf16 record ([selected scales, selected biases], 2 * S per row;
    /// upper rows use their first 2 * S2) instead of gathering them from the
    /// 42-group rows: the same values in the same order, so scores are bitwise
    /// equal (checked at load).
    private static let packedPartialSource = partialSource
        .replacingOccurrences(
            of: "const float sc = float(scales[size_t(row) * KG + g]);",
            with: "const float sc = float(sb[size_t(row) * (2 * S) + (lane >> 1)]);")
        .replacingOccurrences(
            of: "const float bi = float(biases[size_t(row) * KG + g]);",
            with: "const float bi = float(sb[size_t(row) * (2 * S) + (upper ? S2 : S) + (lane >> 1)]);")
    private static let packedPartialKernel = MLXFast.metalKernel(
        name: "nemotron_head_screen_partial_packed_affine4_gs64",
        inputNames: ["w", "sb", "x", "groups", "groups2"],
        outputNames: ["scores"], source: packedPartialSource)
    /// The same body reading each row's whole screen record from one flat
    /// array: the selected groups' 8 words each, then that row's packed
    /// scales/biases (9 * S words per row below LO, 9 * S2 from LO up). With
    /// RPS 8 and 16 / 8 groups a simdgroup's rows are one contiguous run of
    /// whole 128-byte lines (4,608 / 2,304 bytes) instead of 8 row-strided
    /// gathers; the same words and values in the same order, so scores are
    /// bitwise equal (checked at load).
    private static let recordPartialSource = packedPartialSource
        .replacingOccurrences(
            of: "const device uint16_t* ws = (const device uint16_t*)(w + size_t(row) * KW + woff);",
            with: """
                const device uint32_t* rp = rec + (upper
                          ? size_t(LO) * (9 * S) + size_t(row - LO) * (9 * S2) : size_t(row) * (9 * S));
                      const device uint16_t* ws = (const device uint16_t*)(rp + (lane >> 1) * 8 + part * 4);
                      const device T* sb = (const device T*)(rp + 8 * (upper ? S2 : S));
                """)
        .replacingOccurrences(of: "sb[size_t(row) * (2 * S) + ", with: "sb[")
    private static let recordPartialKernel = MLXFast.metalKernel(
        name: "nemotron_head_screen_partial_record_affine4_gs64",
        inputNames: ["rec", "x", "groups", "groups2"],
        outputNames: ["scores"], source: recordPartialSource)

    /// [rows, 2 * groupCount] bf16: each row's selected scales then biases
    /// (rows from `rows` up: the upper group set, zero-padded). Copies of the
    /// head's own bytes, built once at load for the draft screen only.
    static func packScalesBiases(
        scales: MLXArray, biases: MLXArray, groups: [Int], groupCount: Int,
        upperGroups: [Int]?
    ) -> MLXArray {
        func pack(_ range: Range<Int>, _ selected: [Int]) -> MLXArray {
            let index = MLXArray(selected.map { Int32($0) })
            var parts = [take(scales[range], index, axis: 1), take(biases[range], index, axis: 1)]
            if selected.count < groupCount {
                parts.append(MLXArray.zeros(
                    [range.count, 2 * (groupCount - selected.count)], dtype: scales.dtype))
            }
            return concatenated(parts, axis: 1)
        }
        let lower = pack(0..<rows, groups)
        guard let upperGroups else { return lower }
        return concatenated([lower, pack(rows..<headRows, upperGroups)], axis: 0)
    }

    /// Flat uint32 screen records: per row the selected groups' words
    /// (`groups` order), then its selected scales and biases as above, 9 words
    /// per selected group (rows from `rows` up: the upper group set). Copies
    /// of the head's own bytes, built once at load for the draft screen only;
    /// the rescore keeps reading the head.
    static func packRecords(
        weight: MLXArray, scales: MLXArray, biases: MLXArray, groups: [Int],
        upperGroups: [Int]?
    ) -> MLXArray {
        func pack(_ range: Range<Int>, _ selected: [Int]) -> MLXArray {
            let words = MLXArray(selected.flatMap { g in (0..<8).map { Int32(8 * g + $0) } })
            let index = MLXArray(selected.map { Int32($0) })
            let sb = concatenated(
                [take(scales[range], index, axis: 1), take(biases[range], index, axis: 1)], axis: 1)
            return concatenated(
                [take(weight[range], words, axis: 1), sb.view(dtype: .uint32)], axis: 1
            ).reshaped(-1)
        }
        let lower = pack(0..<rows, groups)
        guard let upperGroups else { return lower }
        return concatenated([lower, pack(rows..<headRows, upperGroups)], axis: 0)
    }

    /// Draft-only 2-bit screen records: each selected group's 64 head values
    /// (dequantized from the head's own 4-bit codes) re-quantized on load to
    /// 2 bits with their own minimum and step (round to nearest, clamped to
    /// 0...3). Per row: 4 words of codes per selected group (value j of a
    /// word at bits 2j), then one word per group holding the minimum (low
    /// half) and the step (high half) as float16; 5 words per selected group
    /// (rows from `rows` up: the upper group set). Only the screen's partial
    /// pass reads them; the kept rows are rescored on the head itself.
    static func packRecords2(
        weight: MLXArray, scales: MLXArray, biases: MLXArray, groups: [Int],
        upperGroups: [Int]?
    ) -> MLXArray {
        let shifts = MLXArray((0..<16).map { UInt32(2 * $0) })
        func pack(_ range: Range<Int>, _ selected: [Int]) -> MLXArray {
            let columns = MLXArray(selected.flatMap { g in (0..<64).map { Int32(64 * g + $0) } })
            var parts: [MLXArray] = []
            for start in stride(from: range.lowerBound, to: range.upperBound, by: 8192) {
                let r = start..<min(start + 8192, range.upperBound)
                let values = take(
                    dequantized(weight[r], scales: scales[r], biases: biases[r], groupSize: 64, bits: 4)
                        .asType(.float32), columns, axis: 1
                ).reshaped(r.count, selected.count, 64)
                let low = values.min(axis: -1, keepDims: true)
                let step = (values.max(axis: -1, keepDims: true) - low) / 3
                let safe = which(step .== 0, MLXArray(Float(1)), step)
                let codes = clip(round((values - low) / safe), min: 0, max: 3).asType(.uint32)
                let words = (codes.reshaped(r.count, selected.count, 4, 16) << shifts).sum(axis: -1)
                    .reshaped(r.count, selected.count * 4)
                let lowBits = low.squeezed(axis: -1).asType(.float16).view(dtype: .uint16).asType(.uint32)
                let stepBits = step.squeezed(axis: -1).asType(.float16).view(dtype: .uint16).asType(.uint32)
                let part = concatenated([words.asType(.uint32), lowBits | (stepBits << 16)], axis: 1)
                eval(part)
                parts.append(part)
            }
            return concatenated(parts, axis: 0).reshaped(-1)
        }
        let lower = pack(0..<rows, groups)
        guard let upperGroups else { return lower }
        return concatenated([lower, pack(rows..<headRows, upperGroups)], axis: 0)
    }

    // The 2-bit record twin of the partial pass: lane l owns half-group l
    // (32 values = two code words) of the selected groups; scores are
    // step * sum(code * x) + minimum * sum(x) over the selected groups.
    private static let record2PartialSource = """
      constexpr int VPL = 32;
      const int lane = int(thread_index_in_simdgroup);
      const int sg = int(threadgroup_position_in_grid.y) * NSG + int(simdgroup_index_in_threadgroup);
      const int row0 = sg * RPS;
      const bool upper = row0 >= LO;
      const int H = upper ? 2 * S2 : 2 * S;
      const bool on = lane < H;
      const int gi = lane >> 1;
      const int g = on ? int(upper ? groups2[gi] : groups[gi]) : 0;
      const int part = lane & 1;
      const device T* xp = x + g * 64 + part * 32;
      float xv[VPL];
      float xs = 0.0f;
      _Pragma("clang loop unroll(full)")
      for (int i = 0; i < VPL; i += 8) {
        _Pragma("clang loop unroll(full)")
        for (int k = 0; k < 8; k++) {
          const float xk = on ? float(xp[i + k]) : 0.0f;
          xs += xk;
          xv[i + k] = xk / float(1 << (2 * k));
        }
      }
      _Pragma("clang loop unroll(full)")
      for (int r = 0; r < RPS; r++) {
        const int row = row0 + r;
        float acc = 0.0f;
        if (on) {
          const device uint32_t* rp = rec + (upper
              ? size_t(LO) * (5 * S) + size_t(row - LO) * (5 * S2) : size_t(row) * (5 * S));
          const device uint16_t* ws = (const device uint16_t*)(rp + gi * 4 + part * 2);
          const uint ld = rp[4 * (upper ? S2 : S) + gi];
          const float lo = float(as_type<half>(ushort(ld & 0xffffu)));
          const float d = float(as_type<half>(ushort(ld >> 16)));
          float a = 0.0f;
          _Pragma("clang loop unroll(full)")
          for (int i = 0; i < VPL / 8; i++) {
            const uint q = uint(ws[i]);
            a += xv[8 * i] * float(q & 0x0003u) + xv[8 * i + 1] * float(q & 0x000cu)
                + xv[8 * i + 2] * float(q & 0x0030u) + xv[8 * i + 3] * float(q & 0x00c0u)
                + xv[8 * i + 4] * float(q & 0x0300u) + xv[8 * i + 5] * float(q & 0x0c00u)
                + xv[8 * i + 6] * float(q & 0x3000u) + xv[8 * i + 7] * float(q & 0xc000u);
          }
          acc = d * a + lo * xs;
        }
        acc = simd_sum(acc);
        if (lane == 0) {
          scores[row] = acc;
        }
      }
    """
    private static let record2PartialKernel = MLXFast.metalKernel(
        name: "nemotron_head_screen_partial_record2_affine2",
        inputNames: ["rec", "x", "groups", "groups2"],
        outputNames: ["scores"], source: record2PartialSource)

    // The 2-bit partial pass with the upper rows two per pass. An upper row
    // has 2 * S2 <= 16 active lanes, so lanes 16..31 take the next row with
    // lane - 16's half-group: each lane computes the record kernel's partial
    // for its (row, half-group) with the same statements. The closing sums
    // keep the record kernel's per-row input: simd_sum over that row's
    // partials in lanes 0..2 * S2 - 1 and +0.0f in every other lane
    // (simd_shuffle_down moves the second row's partials into lanes 0..15,
    // the lane < 16 selects zero the rest), so every score keeps its bits.
    // The lane's 32 activations load as four 16-byte words when x is 16-byte
    // aligned (each half converted from T as before; else the scalar loads)
    // and its two code words as one 8-byte word (S, S2 even). Lower rows keep
    // the record kernel's one row per pass.
    private static let record2PairedSource: String? = {
        var s = record2PartialSource
        func swap(_ from: String, _ to: String) -> Bool {
            guard s.components(separatedBy: from).count == 2 else { return false }
            s = s.replacingOccurrences(of: from, with: to)
            return true
        }
        let lanes = """
              const int H = upper ? 2 * S2 : 2 * S;
              const bool on = lane < H;
              const int gi = lane >> 1;
              const int g = on ? int(upper ? groups2[gi] : groups[gi]) : 0;
              const int part = lane & 1;

            """
        let pairedLanes = """
              constexpr bool PAIRABLE = 2 * S2 <= 16 && RPS % 2 == 0;
              const bool paired = PAIRABLE && upper;
              const int ln = paired ? (lane & 15) : lane;
              const int H = upper ? 2 * S2 : 2 * S;
              const bool on = ln < H;
              const int gi = ln >> 1;
              const int g = on ? int(upper ? groups2[gi] : groups[gi]) : 0;
              const int part = ln & 1;

            """
        let xLoop = """
              _Pragma("clang loop unroll(full)")
              for (int i = 0; i < VPL; i += 8) {
                _Pragma("clang loop unroll(full)")
                for (int k = 0; k < 8; k++) {
                  const float xk = on ? float(xp[i + k]) : 0.0f;
                  xs += xk;
                  xv[i + k] = xk / float(1 << (2 * k));
                }
              }

            """
        let wideXLoop = """
              if ((reinterpret_cast<ulong>(x) & 15) == 0) {
                const device uint4* xq = (const device uint4*)xp;
                _Pragma("clang loop unroll(full)")
                for (int i = 0; i < VPL; i += 8) {
                  const uint4 xw = xq[i / 8];
                  const ushort xh[8] = {ushort(xw.x & 0xffffu), ushort(xw.x >> 16), ushort(xw.y & 0xffffu),
                                        ushort(xw.y >> 16), ushort(xw.z & 0xffffu), ushort(xw.z >> 16),
                                        ushort(xw.w & 0xffffu), ushort(xw.w >> 16)};
                  _Pragma("clang loop unroll(full)")
                  for (int k = 0; k < 8; k++) {
                    const float xk = on ? float(as_type<T>(xh[k])) : 0.0f;
                    xs += xk;
                    xv[i + k] = xk / float(1 << (2 * k));
                  }
                }
              } else {

            """ + xLoop + "  }\n"
        let codes = "const device uint16_t* ws = (const device uint16_t*)(rp + gi * 4 + part * 2);"
        let wideCodes = "const uint2 cw = *((const device uint2*)(rp + gi * 4 + part * 2));\n"
            + "      const uint ws[4] = {cw.x & 0xffffu, cw.x >> 16, cw.y & 0xffffu, cw.y >> 16};"
        guard swap(lanes, pairedLanes), swap(xLoop, wideXLoop), swap(codes, wideCodes) else { return nil }
        // The row loop runs to the end of the source (no newline after it).
        let loopHead = "  _Pragma(\"clang loop unroll(full)\")\n  for (int r = 0; r < RPS; r++) {\n"
            + "    const int row = row0 + r;\n"
        let close = "    acc = simd_sum(acc);\n    if (lane == 0) {\n      scores[row] = acc;\n    }\n  }"
        guard let start = s.range(of: loopHead), let end = s.range(of: close),
              start.upperBound <= end.lowerBound, s[end.upperBound...].allSatisfy(\.isWhitespace)
        else { return nil }
        let body = String(s[start.upperBound..<end.lowerBound])
        guard !screenVectorStore else {
            // Lane r keeps row row0 + r's sum (the same simd_sum results, selected), and lanes 0 ... RPS - 1 store
            // the pass's RPS scores at once after the loop instead of lane 0 storing each row's in turn.
            s = String(s[..<start.lowerBound]) + "  float nsc_kept = 0.0f;\n  if (!paired) {\n" + loopHead + body
                + "    acc = simd_sum(acc);\n    nsc_kept = lane == r ? acc : nsc_kept;\n  }\n  } else {\n"
                + "  _Pragma(\"clang loop unroll(full)\")\n  for (int r = 0; r < RPS; r += 2) {\n"
                + "    const int row = row0 + r + (lane >> 4);\n" + body
                + "    const float next = simd_shuffle_down(acc, ushort(16));\n"
                + "    const float first = simd_sum(lane < 16 ? acc : 0.0f);\n"
                + "    const float second = simd_sum(lane < 16 ? next : 0.0f);\n"
                + "    nsc_kept = lane == r ? first : nsc_kept;\n    nsc_kept = lane == r + 1 ? second : nsc_kept;\n"
                + "  }\n  }\n  if (lane < RPS) {\n    scores[row0 + lane] = nsc_kept;\n  }\n"
            return s
        }
        let pairedLoop = "  if (!paired) {\n" + loopHead + body + close + "\n  } else {\n"
            + "  _Pragma(\"clang loop unroll(full)\")\n  for (int r = 0; r < RPS; r += 2) {\n"
            + "    const int row = row0 + r + (lane >> 4);\n" + body
            + "    const float next = simd_shuffle_down(acc, ushort(16));\n"
            + "    const float first = simd_sum(lane < 16 ? acc : 0.0f);\n"
            + "    const float second = simd_sum(lane < 16 ? next : 0.0f);\n"
            + "    if (lane == 0) {\n      scores[row0 + r] = first;\n      scores[row0 + r + 1] = second;\n    }\n"
            + "  }\n  }\n"
        s = String(s[..<start.lowerBound]) + pairedLoop
        return s
    }()
    private static let record2PairedKernel = record2PairedSource.map {
        MLXFast.metalKernel(
            name: "nemotron_head_screen_partial_record2_paired_affine2",
            inputNames: ["rec", "x", "groups", "groups2"],
            outputNames: ["scores"], source: $0)
    }
    /// NEMOTRON_MTP_HEAD_SCREEN_PAIRED=0 keeps the record kernel's one row per
    /// pass for the 2-bit records.
    static let pairedEnabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_HEAD_SCREEN_PAIRED"] != "0"
    /// NEMOTRON_MTP_HEAD_SCREEN_VSTORE=0 keeps the paired pass's lane-0 score stores (one per row).
    static let screenVectorStore = ProcessInfo.processInfo.environment["NEMOTRON_MTP_HEAD_SCREEN_VSTORE"] != "0"
    /// The paired pass needs an upper group set of at most 8 groups, even
    /// group counts (8-byte code words) and an even RPS (at most 32 with the
    /// vector store: one lane per row of the pass).
    static func pairable(groupCount: Int, upperGroupCount: Int?, rowsPerSimdgroup: Int) -> Bool {
        record2PairedKernel != nil && groupCount % 2 == 0 && rowsPerSimdgroup % 2 == 0
            && (!screenVectorStore || rowsPerSimdgroup <= 32)
            && upperGroupCount.map { $0 % 2 == 0 && 2 * $0 <= 16 } ?? true
    }

    /// NEMOTRON_MTP_SCREEN_PARTIAL_O32=0 keeps the paired partial's record addresses. Otherwise a screen of the full
    /// head (both halves) runs the paired 2-bit partial as a twin (`record2PairedOffset32Source`) that forms each
    /// row's record offset in 32 bits (`upper ? LO * 5S + (row - LO) * 5S2 : row * 5S` as `uint`) and reads the row's
    /// code words and (minimum, step) word at `rec[offset + ...]`, where the record forms the same offset as a
    /// `size_t` expression and a 64-bit row pointer for every row of the pass. Half-head screens keep the record
    /// kernel. Used only after `qualifyPartialOffset32` found it bitwise equal to the paired partial.
    static let partialOffset32Enabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_SCREEN_PARTIAL_O32"] != "0"
    nonisolated(unsafe) private(set) static var partialOffset32Verified = false
    /// The paired partial's text with each row's record offset in 32 bits (nil when a record statement moved or a
    /// `size_t` or row pointer remains). The records are one row-contiguous uint32 array (`ensureRowContiguous`, as
    /// the record), so `rec[offset + i]` is the word the record reads at `rp[i]`; the full head's records hold
    /// 65,536 x 5S + 65,536 x 5S2 words, far below 2^32. Every load, value, product, sum, shuffle and store is the
    /// record's statement.
    fileprivate static let record2PairedOffset32Source: String? = record2PairedSource.flatMap { source -> String? in
        var s = source
        let swaps: [(String, String, Int)] = [
            ("      const device uint32_t* rp = rec + (upper\n"
                + "          ? size_t(LO) * (5 * S) + size_t(row - LO) * (5 * S2) : size_t(row) * (5 * S));\n",
             "      const uint nso_r = upper ? uint(LO) * (5 * S) + uint(row - LO) * (5 * S2) : uint(row) * (5 * S);\n",
             2),
            ("const uint2 cw = *((const device uint2*)(rp + gi * 4 + part * 2));",
             "const uint2 cw = *((const device uint2*)(rec + (nso_r + gi * 4 + part * 2)));", 2),
            ("const uint ld = rp[4 * (upper ? S2 : S) + gi];",
             "const uint ld = rec[nso_r + 4 * (upper ? S2 : S) + gi];", 2),
        ]
        for (old, new, count) in swaps {
            guard s.components(separatedBy: old).count == count + 1 else { return nil }
            s = s.replacingOccurrences(of: old, with: new)
        }
        guard !s.contains("size_t"), !s.contains(" rp "), !s.contains("rp["), !s.contains("(rp ") else { return nil }
        return s
    }
    private static let record2PairedOffset32Kernel = record2PairedOffset32Source.map {
        MLXFast.metalKernel(
            name: "nemotron_head_screen_partial_record2_paired_o32_affine2",
            inputNames: ["rec", "x", "groups", "groups2"],
            outputNames: ["scores"], source: $0)
    }
    /// Whether a paired partial over `records2` runs as the 32-bit twin: a full-head screen (`fullHead`), the twin's
    /// text found, the records' word offsets below 2^32, and `offset32` (nil: the load check's verdict).
    static func partialOffset32(fullHead: Bool, records2: MLXArray, offset32: Bool?) -> Bool {
        (offset32 ?? (partialOffset32Enabled && partialOffset32Verified)) && fullHead
            && record2PairedOffset32Source != nil && records2.dtype == .uint32 && records2.size <= Int(UInt32.max)
    }
    private static let selectKernel = MLXFast.metalKernel(
        name: "nemotron_head_screen_select_affine4_gs64",
        inputNames: ["scores", "w", "scales", "biases", "x"],
        outputNames: ["cand_ids", "cand_vals", "best_val", "best_id"],
        source: selectSource, header: nemotronMultiRowQMVHeader)
    private static let finalKernel = MLXFast.metalKernel(
        name: "nemotron_head_screen_final",
        inputNames: ["best_val", "best_id"], outputNames: ["out"], source: finalSource)

    /// Kill switches (code constants; 0 = crown int -> float decode and the
    /// crown kernel objects): the draft screen's partial pass (gathered and
    /// packed) and its exact select rescore.
    static let partialHalfDecodeForm = 0
    static let selectHalfDecodeForm = 0
    static let partialHalfSwaps: [(String, String, Int)] = [
        ("const uint16_t q = ws[i];", "const float4 hq = nmr_half_decode4<HF>(ws, i);", 1),
        ("float(q & 0x000f)", "hq.x", 1),
        ("float(q & 0x00f0)", "hq.y", 1),
        ("float(q & 0x0f00)", "hq.z", 1),
        ("float(q & 0xf000)", "hq.w", 1),
    ]
    private static let partialHalfKernel = MLXFast.metalKernel(
        name: "nemotron_head_screen_partial_affine4_gs64_hf",
        inputNames: ["w", "scales", "biases", "x", "groups", "groups2"],
        outputNames: ["scores"], source: nemotronSwapped(partialSource, partialHalfSwaps),
        header: nemotronHalfDecodeHeader)
    /// The packed twin of the crown chain, on the same decode swaps.
    private static let packedPartialHalfKernel = MLXFast.metalKernel(
        name: "nemotron_head_screen_partial_packed_affine4_gs64_hf",
        inputNames: ["w", "sb", "x", "groups", "groups2"],
        outputNames: ["scores"], source: nemotronSwapped(packedPartialSource, partialHalfSwaps),
        header: nemotronHalfDecodeHeader)
    /// The full-record twin, on the same decode swaps.
    private static let recordPartialHalfKernel = MLXFast.metalKernel(
        name: "nemotron_head_screen_partial_record_affine4_gs64_hf",
        inputNames: ["rec", "x", "groups", "groups2"],
        outputNames: ["scores"], source: nemotronSwapped(recordPartialSource, partialHalfSwaps),
        header: nemotronHalfDecodeHeader)
    private static let selectHalfKernel = MLXFast.metalKernel(
        name: "nemotron_head_screen_select_affine4_gs64_hf",
        inputNames: ["scores", "w", "scales", "biases", "x"],
        outputNames: ["cand_ids", "cand_vals", "best_val", "best_id"],
        source: nemotronSwapped(selectSource, nemotronMultiRowHalfSwaps),
        header: nemotronMultiRowQMVHeader + nemotronHalfDecodeHeader)
    /// NEMOTRON_MTP_SCREEN_FUSED_FINAL=0 keeps the separate final kernel.
    static var fusedFinal: Bool {
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_SCREEN_FUSED_FINAL"] != "0"
            && !fusedSelectSource.contains("best_val[") && !fusedSelectSource.contains("best_id[")
            && fusedSelectSource.components(separatedBy: "atomic_load_explicit(").count == 3
    }
    // Select + final in one launch: each chunk's threadgroup publishes its
    // (best_val, best_id) with device atomics (same bits), fences at device
    // scope and takes a ticket from `counter`; the threadgroup that draws
    // NCH - 1 runs finalSource over all NCH pairs (atomic loads: same values,
    // lane order and comparator, so the same id) and resets the counter.
    // One launch and one dependent barrier fewer per screen.
    private static let fusedFinalHead = """

      if (sgid == 0) {
        uint ticket = 0;
        if (lane == 0) {
          atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
          ticket = atomic_fetch_add_explicit((device atomic_uint*)counter, 1u, memory_order_relaxed);
        }
        if (simd_broadcast(ticket, 0) == uint(NCH - 1)) {
          atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
          {

    """
    private static let fusedFinalTail = """

          }
          if (lane == 0) {
            atomic_store_explicit((device atomic_uint*)counter, 0u, memory_order_relaxed);
          }
        }
      }
    """
    private static let fusedSelectSource = selectSource
        .replacingOccurrences(
            of: "best_val[chunk] = mv;",
            with: "atomic_store_explicit((device atomic_uint*)(best_val + chunk), as_type<uint>(mv), memory_order_relaxed);")
        .replacingOccurrences(
            of: "best_id[chunk] = mi;",
            with: "atomic_store_explicit((device atomic_int*)(best_id + chunk), mi, memory_order_relaxed);")
        + fusedFinalHead
        + finalSource
            .replacingOccurrences(
                of: "const float v = best_val[c];",
                with: "const float v = as_type<float>(atomic_load_explicit((device atomic_uint*)(best_val + c), memory_order_relaxed));")
            .replacingOccurrences(
                of: "const int id = best_id[c];",
                with: "const int id = atomic_load_explicit((device atomic_int*)(best_id + c), memory_order_relaxed);")
        + fusedFinalTail
    private static let fusedSelectKernel = MLXFast.metalKernel(
        name: "nemotron_head_screen_select_final_affine4_gs64",
        inputNames: ["scores", "w", "scales", "biases", "x", "counter"],
        outputNames: ["cand_ids", "cand_vals", "best_val", "best_id", "out"],
        source: fusedSelectSource, header: nemotronMultiRowQMVHeader,
        mutableInputs: ["counter"])
    /// The fused select on the select rescore's decode swaps.
    private static let fusedSelectHalfKernel = MLXFast.metalKernel(
        name: "nemotron_head_screen_select_final_affine4_gs64_hf",
        inputNames: ["scores", "w", "scales", "biases", "x", "counter"],
        outputNames: ["cand_ids", "cand_vals", "best_val", "best_id", "out"],
        source: nemotronSwapped(fusedSelectSource, nemotronMultiRowHalfSwaps),
        header: nemotronMultiRowQMVHeader + nemotronHalfDecodeHeader,
        mutableInputs: ["counter"])

    /// NEMOTRON_MTP_SCREEN_SELECT_B1=0 keeps the record fused select.
    static let b1Enabled = ProcessInfo.processInfo.environment["NEMOTRON_MTP_SCREEN_SELECT_B1"] != "0"
    // The fused select with one threadgroup barrier fewer (B1). In the record,
    // SIMD group 0 alone picks the chunk's top C partial scores, stores the ids
    // to threadgroup memory, and every SIMD group waits at a barrier before the
    // rescore reads them. Here every SIMD group that rescores rows (sgid * RPSG
    // < C) runs SIMD group 0's selection itself, with the same statements on the
    // same scores, and keeps the ids it needs in registers: at pick c, lane c
    // keeps lane 0's id (SIMD group 0's closing pass reads them as before from
    // cand[lane]), and the group's own rescore slots r keep it at a constant
    // register index. The rescore and the closing pass then read the same ids,
    // so every rescored value, kept id, best and pick are the record's. One
    // pass of C <= 32 kept rows (C <= NSG * RPSG). Nil when the text moved.
    private static let fusedSelectB1Source: String? = b1Swapped(fusedSelectSource)
    private static func b1Swapped(_ source: String) -> String? {
        var s = source
        let swaps: [(String, String)] = [
            ("  threadgroup int cand[C];\n",
             "  static_assert(C <= 32 && C <= NSG * RPSG, \"one kept id per lane, one rescore pass\");\n"
                + "  int nsb_lane_cand = 0;\n  int nsb_row_cand[RPSG];\n"),
            ("  if (sgid == 0) {\n    float v[P1];\n", "  if (sgid * RPSG < C) {\n    float v[P1];\n"),
            ("      if (lane == 0) cand[c] = base + mi;\n",
             "      const int nsb_pick = base + simd_broadcast(mi, ushort(0));\n"
                + "      if (lane == c) nsb_lane_cand = nsb_pick;\n"
                + "      _Pragma(\"clang loop unroll(full)\")\n"
                + "      for (int r = 0; r < RPSG; r++) {\n"
                + "        if (c == sgid * RPSG + r) nsb_row_cand[r] = nsb_pick;\n      }\n"),
            ("  }\n  threadgroup_barrier(mem_flags::mem_threadgroup);\n\n  const device uint8_t* wbytes",
             "  }\n\n  const device uint8_t* wbytes"),
            ("    for (int row = 0; row < RPSG; row++) {\n      const size_t id = size_t(cand[slot0 + row]);\n",
             "    _Pragma(\"clang loop unroll(full)\")\n"
                + "    for (int row = 0; row < RPSG; row++) {\n      const size_t id = size_t(nsb_row_cand[row]);\n"),
            ("      const int id = cand[s2];\n", "      const int id = nsb_lane_cand;\n"),
        ]
        for (old, new) in swaps {
            guard s.components(separatedBy: old).count == 2 else { return nil }
            s = s.replacingOccurrences(of: old, with: new)
        }
        guard !s.contains(" cand["), s.components(separatedBy: "threadgroup_barrier(").count == 2 else { return nil }
        return s
    }
    private static let fusedSelectB1Kernel = fusedSelectB1Source.map {
        MLXFast.metalKernel(
            name: "nemotron_head_screen_select_final_b1_affine4_gs64",
            inputNames: ["scores", "w", "scales", "biases", "x", "counter"],
            outputNames: ["cand_ids", "cand_vals", "best_val", "best_id", "out"],
            source: $0, header: nemotronMultiRowQMVHeader,
            mutableInputs: ["counter"])
    }
    /// The B1 select serves a plan with a fused counter and at most 32 kept rows per chunk.
    static func b1Available(keep: Int) -> Bool { fusedSelectB1Kernel != nil && keep <= 32 }

    /// NEMOTRON_MTP_SCREEN_NO_TICKET=0 keeps the fused B1 select's ticket. Otherwise the screen runs the B1
    /// select without the fused final (`selectB1Kernel`: the same B1 swaps on the two-launch select, so each
    /// chunk's best value and id are plain stores of the same values; no counter, no device fence, no ticket)
    /// and the pick is `finalKernel` over the chunk bests, the two-launch path's final. Launches that read the
    /// pick on the draft chain form it themselves from the chunk bests with `finalKernel`'s statements
    /// (`pickStatements`); every other reader takes the final's id. Used only after the plan's check found it
    /// bitwise equal to the fused B1 select (`NemotronH35MTPDraftHead`).
    static let noTicketEnabled = ProcessInfo.processInfo.environment["NEMOTRON_MTP_SCREEN_NO_TICKET"] != "0"
    private static let selectB1Source: String? = b1Swapped(selectSource)
    private static let selectB1Kernel = selectB1Source.map {
        MLXFast.metalKernel(
            name: "nemotron_head_screen_select_b1_affine4_gs64",
            inputNames: ["scores", "w", "scales", "biases", "x"],
            outputNames: ["cand_ids", "cand_vals", "best_val", "best_id"],
            source: $0, header: nemotronMultiRowQMVHeader)
    }
    static func noTicketAvailable(keep: Int) -> Bool { selectB1Kernel != nil && keep <= 32 }

    /// NEMOTRON_MTP_SCREEN_SELECT_O32=0 keeps the B1 select's address arithmetic. Otherwise the ticket-free screen
    /// launches a twin of `selectB1Kernel` (`selectB1Offset32Source`) whose rescore reads every kept row's code
    /// words, scales and biases, and the activations, at 32-bit element offsets from the arrays' bases, where the
    /// record steps 64-bit row pointers (`size_t` id times the row size, then a 64-bit pointer add per row and
    /// 256-value block), and whose pick mask is 32 bits wide where the record's is a 64-bit `ulong` (CH / 32 <= 32
    /// rows per lane). Used only after `qualifySelectOffset32` found it bitwise equal to the B1 select.
    static let selectOffset32Enabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_SCREEN_SELECT_O32"] != "0"
    nonisolated(unsafe) private(set) static var selectOffset32Verified = false
    /// The B1 select's text with its 64-bit offsets in 32 bits (nil when a record statement moved or a 64-bit
    /// offset remains). The kernel reads row-contiguous inputs (as the record: `ensureRowContiguous`), so element
    /// e of a kept row `id` sits at `id * row size + e` from the array's base, below 2^32 for the head (131,072 rows
    /// x 672 code halves); every load reads the record's address and every value, product, sum, rounding, pick and
    /// store is the record's statement.
    private static let selectB1Offset32Source: String? = selectB1Source.flatMap { source -> String? in
        var s = source
        let swaps: [(String, String, Int)] = [
            ("    ulong taken = 0;\n", "    uint taken = 0;\n", 1),
            ("        taken |= (1ul << ulong(mi - lane * P1));\n",
             "        taken |= (1u << uint(mi - lane * P1));\n", 1),
            ("          if (((taken >> ulong(i)) & 1ul) == 0 && (bi < 0 || v[i] > bv)) { bv = v[i]; bi = i; }\n",
             "          if (((taken >> uint(i)) & 1u) == 0 && (bi < 0 || v[i] > bv)) { bv = v[i]; bi = i; }\n", 1),
            ("""
              const device uint8_t* wbytes = (const device uint8_t*)w;
              for (int slot0 = sgid * RPSG; slot0 < C; slot0 += NSG * RPSG) {
                const device uint8_t* wl[RPSG];
                const device T* sl[RPSG];
                const device T* bl[RPSG];
                _Pragma("clang loop unroll(full)")
                for (int row = 0; row < RPSG; row++) {
                  const size_t id = size_t(nsb_row_cand[row]);
                  wl[row] = wbytes + id * in_vec_size_w + lane * 4;
                  sl[row] = scales + id * in_vec_size_g + lane / scale_step_per_thread;
                  bl[row] = biases + id * in_vec_size_g + lane / scale_step_per_thread;
                }
                const device T* xp = x + lane * values_per_thread;

            """, """
              const device uint16_t* nso_w16 = (const device uint16_t*)w;
              static_assert(P1 <= 32, "one 32-bit pick mask per lane");
              for (int slot0 = sgid * RPSG; slot0 < C; slot0 += NSG * RPSG) {
                uint nso_w[RPSG];
                uint nso_sb[RPSG];
                _Pragma("clang loop unroll(full)")
                for (int row = 0; row < RPSG; row++) {
                  const uint id = uint(nsb_row_cand[row]);
                  nso_w[row] = id * (in_vec_size_w / 2) + uint(lane) * 2;
                  nso_sb[row] = id * in_vec_size_g + uint(lane / scale_step_per_thread);
                }
                uint nso_x = uint(lane) * values_per_thread;

            """, 1),
            ("sum = nmr_load_vector<T, U, values_per_thread>(xp, x_thread);",
             "sum = nmr_load_vector<T, U, values_per_thread>(x + nso_x, x_thread);", 2),
            ("        U s = sl[row][0];\n        U b = bl[row][0];\n",
             "        U s = scales[nso_sb[row]];\n        U b = biases[nso_sb[row]];\n", 2),
            ("        const device uint16_t* words = (const device uint16_t*)wl[row];\n",
             "        const uint nso_wo = nso_w[row];\n", 2),
            ("words[i]", "nso_w16[nso_wo + i]", 8),
            ("        wl[row] += block_size / 2;\n        sl[row] += block_size / 64;\n        bl[row] += block_size / 64;\n",
             "        nso_w[row] += block_size / 4;\n        nso_sb[row] += block_size / 64;\n", 1),
            ("      xp += block_size;\n", "      nso_x += block_size;\n", 1),
        ]
        for (old, new, count) in swaps {
            guard s.components(separatedBy: old).count == count + 1 else { return nil }
            s = s.replacingOccurrences(of: old, with: new)
        }
        guard !s.contains("size_t"), !s.contains("ulong"), !s.contains("1ul"), !s.contains("wbytes"),
              !s.contains("(xp"), !s.contains(" xp"), !s.contains("wl["), !s.contains("sl["), !s.contains("bl["),
              !s.contains("words")
        else { return nil }
        return s
    }
    private static let selectB1Offset32Kernel = selectB1Offset32Source.map {
        MLXFast.metalKernel(
            name: "nemotron_head_screen_select_b1_o32_affine4_gs64",
            inputNames: ["scores", "w", "scales", "biases", "x"],
            outputNames: ["cand_ids", "cand_vals", "best_val", "best_id"],
            source: $0, header: nemotronMultiRowQMVHeader)
    }

    /// The ticket-free screen's select launch: the B1 select, or its 32-bit twin when `offset32` (nil: the load
    /// check's verdict) and the twin applies (chunks of at most 1,024 rows, every code half and scale index of the
    /// head below 2^32). Outputs: kept ids, kept values, chunk best values, chunk best ids.
    static func selectB1(
        _ scores: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray, hidden: MLXArray,
        chunk: Int, keep: Int, offset32: Bool? = nil
    ) -> [MLXArray]? {
        guard let selectB1Kernel, keep <= 32 else { return nil }
        let twin = (offset32 ?? (selectOffset32Enabled && selectOffset32Verified)) && chunk <= 1024
            && weight.dtype == .uint32 && weight.size * 2 <= Int(UInt32.max) && scales.size <= Int(UInt32.max)
            && biases.size <= Int(UInt32.max)
            ? selectB1Offset32Kernel : nil
        let chunks = scores.dim(0) / chunk
        let selectGroups = 8
        return (twin ?? selectB1Kernel)(
            [scores, weight, scales, biases, hidden],
            template: [("T", hidden.dtype), ("CH", chunk), ("C", keep), ("NSG", selectGroups)],
            grid: (32, selectGroups * chunks, 1), threadGroup: (32, selectGroups, 1),
            outputShapes: [[chunks * keep], [chunks * keep], [chunks], [chunks]],
            outputDTypes: [.int32, .float32, .float32, .int32])
    }

    /// Load time, right after the plan's ticket-free check (`prepareScreen`): the 32-bit select twin against the B1
    /// select, all four outputs bit for bit (kept ids, kept values as bits, chunk best values as bits, chunk best
    /// ids), on the plan's head arrays, chunk and keep, for each row count the plan screens. Partial scores of ten
    /// seeded kinds: Gaussian (three seeds), rounded to integers (ties), NaNs, both infinities, signed zeros among
    /// rounded values, a constant row (every row tied), mostly -inf with a few finite values (chunks with fewer
    /// finite rows than kept rows), and rising scores (each chunk keeps its last rows, the head's last row
    /// included); each with activations at three scales (one with -0 entries, one with an infinity). Then the plan's
    /// screen end to end (`screen`: `run` with the twin off and on, each half the plan screens) on six seeded
    /// activations. Returns the stderr note; engages only if every case matches.
    static func qualifySelectOffset32(
        weight: MLXArray, scales: MLXArray, biases: MLXArray, chunk: Int, keep: Int, rowCounts: [Int],
        gamma: MLXArray, screen: (MLXArray, Bool, Bool) -> Result
    ) -> String {
        selectOffset32Verified = false
        guard selectOffset32Enabled else { return "off (setting)" }
        guard selectB1Offset32Kernel != nil else { return "off (select text moved)" }
        guard chunk <= 1024, keep <= 32 else { return "off (chunk or keep does not qualify)" }
        var cases = 0
        do {
            let equal = try withError { () -> Bool in
                func same(_ a: [MLXArray], _ b: [MLXArray]) -> MLXArray {
                    (a[0] .== b[0]).all()
                        .&& (a[1].view(dtype: .uint32) .== b[1].view(dtype: .uint32)).all()
                        .&& (a[2].view(dtype: .uint32) .== b[2].view(dtype: .uint32)).all()
                        .&& (a[3] .== b[3]).all()
                }
                let k = width
                for rows in rowCounts {
                    var ok = MLXArray(true)
                    for xKind in 0 ..< 3 {
                        let xKey = MLXRandom.split(key: MLXRandom.key(UInt64(0x5e1e_0000 + rows + xKind)), into: 2)
                        var x = MLXRandom.normal([k], key: xKey[0]) * gamma.asType(.float32)
                            * Float([1, 8, 0.25][xKind])
                        if xKind == 1 {
                            x = which(MLXRandom.uniform(Float(0) ..< Float(1), [k], key: xKey[1]) .< MLXArray(Float(0.1)),
                                      MLXArray(Float(-0.0)), x)
                        }
                        if xKind == 2 {
                            x = which(MLXArray.arange(k, dtype: .int32) .== MLXArray(Int32(1000)),
                                      MLXArray(Float.infinity), x)
                        }
                        let hidden = x.asType(.bfloat16).reshaped(1, 1, k)
                        for kind in 0 ..< 10 {
                            let key = MLXRandom.split(
                                key: MLXRandom.key(UInt64(0x5e1f_0000 + rows + 16 * xKind + kind)), into: 2)
                            let g = MLXRandom.normal([rows], key: key[0]) * 3
                            let scores: MLXArray
                            switch kind {
                            case 3: scores = round(g)
                            case 4: scores = which(abs(g) .> MLXArray(Float(4.5)), MLXArray(Float.nan), g)
                            case 5:
                                scores = which(g .> MLXArray(Float(4.5)), MLXArray(Float.infinity),
                                    which(g .< MLXArray(Float(-4.5)), MLXArray(-Float.infinity), g))
                            case 6: scores = which(abs(g) .< MLXArray(Float(1)), MLXArray(Float(-0.0)), round(g))
                            case 7: scores = MLXArray.zeros([rows], dtype: .float32) + MLXArray(Float(0.5))
                            case 8:
                                scores = which(MLXRandom.uniform(Float(0) ..< Float(1), [rows], key: key[1]) .< MLXArray(Float(0.99)),
                                               MLXArray(-Float.infinity), g)
                            case 9: scores = MLXArray.arange(rows, dtype: .float32)
                            default: scores = g
                            }
                            guard let a = selectB1(scores, weight: weight, scales: scales, biases: biases,
                                                   hidden: hidden, chunk: chunk, keep: keep, offset32: false),
                                  let b = selectB1(scores, weight: weight, scales: scales, biases: biases,
                                                   hidden: hidden, chunk: chunk, keep: keep, offset32: true)
                            else { return false }
                            ok = ok .&& same(a, b)
                            cases += 1
                        }
                    }
                    if !ok.item(Bool.self) { return false }
                }
                var ok = MLXArray(true)
                var state: UInt64 = 0x7F4A_7C15_9E37_79B9
                for round in 0 ..< 6 {
                    let t = (0 ..< k).map { _ -> Float in
                        state = state &* 6364136223846793005 &+ 1442695040888963407
                        return (Float(state >> 40) / Float(1 << 24) - 0.5) * 3.4641016
                    }
                    let x = (MLXArray(t) * gamma.asType(.float32) * Float(1 + round % 3)).asType(.bfloat16)
                        .reshaped(1, 1, k)
                    for upper in rowCounts.count > 1 ? [false, true] : [false] {
                        let a = screen(x, upper, false), b = screen(x, upper, true)
                        guard let av = a.bestValues, let ai = a.bestIds, let bv = b.bestValues, let bi = b.bestIds
                        else { return false }
                        ok = ok .&& (a.id .== b.id).all()
                            .&& same([a.candidateIds, a.candidateValues, av, ai], [b.candidateIds, b.candidateValues, bv, bi])
                        cases += 1
                    }
                }
                return ok.item(Bool.self)
            }
            selectOffset32Verified = equal
        } catch {
            return "off (\(error))"
        }
        return selectOffset32Verified
            ? "on (\(cases) cases: kept ids, kept values and chunk bests bitwise equal to the B1 select)"
            : "off (differs from the B1 select after \(cases) cases)"
    }

    /// Load time, after the plan's paired check (`prepareScreen`): the paired partial's 32-bit twin against the
    /// paired partial, every score bit for bit, on the plan's real 2-bit records and group sets over the full head
    /// (the only screen the twin serves), through `record2Scores` with and without the tune's unread `dep` input at
    /// the production geometry (8 rows per SIMD group, 2 SIMD groups) and at 4 x 2 and 16 x 1, on eight seeded
    /// activations (four scales, one with -0 entries, one all zero, one with an infinity, one with a NaN); then the
    /// plan's full-head screen end to end (`screen`: `run` with the twin off and on; scores, pick, kept ids and kept
    /// values) on four seeded activations. Returns the stderr note; engages only if every case matches.
    static func qualifyPartialOffset32(
        records2: MLXArray, groups: MLXArray, groupCount: Int, upperGroups: MLXArray, upperGroupCount: Int,
        gamma: MLXArray, screen: (MLXArray, Bool) -> Result
    ) -> String {
        partialOffset32Verified = false
        guard partialOffset32Enabled else { return "off (setting)" }
        guard record2PairedOffset32Kernel != nil else { return "off (partial text moved)" }
        guard partialOffset32(fullHead: true, records2: records2, offset32: true) else {
            return "off (records do not qualify)"
        }
        var cases = 0
        do {
            let equal = try withError { () -> Bool in
                let k = width
                var ok = MLXArray(true)
                for kind in 0 ..< 8 {
                    let key = MLXRandom.split(key: MLXRandom.key(UInt64(0x5e20_0000 + kind)), into: 2)
                    var x = MLXRandom.normal([k], key: key[0]) * gamma.asType(.float32)
                        * Float([0.25, 1, 4, 16, 1, 1, 1, 1][kind])
                    switch kind {
                    case 4:
                        x = which(MLXRandom.uniform(Float(0) ..< Float(1), [k], key: key[1]) .< MLXArray(Float(0.1)),
                                  MLXArray(Float(-0.0)), x)
                    case 5: x = MLXArray.zeros([k], dtype: .float32)
                    case 6:
                        x = which(MLXArray.arange(k, dtype: .int32) .== MLXArray(Int32(70)), MLXArray(Float.infinity), x)
                    case 7: x = which(MLXArray.arange(k, dtype: .int32) .== MLXArray(Int32(130)), MLXArray(Float.nan), x)
                    default: break
                    }
                    let hidden = x.asType(.bfloat16).reshaped(1, 1, k)
                    for (rps, nsg, dep) in [(8, 2, false), (8, 2, true), (4, 2, false), (16, 1, false)] {
                        func scores(_ on: Bool) -> MLXArray? {
                            record2Scores(hidden, records2: records2, groups: groups, groupCount: groupCount,
                                          upperGroups: upperGroups, upperGroupCount: upperGroupCount, rps: rps,
                                          nsg: nsg, dep: dep ? hidden : nil, offset32: on)
                        }
                        guard let a = scores(false), let b = scores(true) else { return false }
                        ok = ok .&& (a.view(dtype: .uint32) .== b.view(dtype: .uint32)).all()
                        cases += 1
                    }
                }
                if !ok.item(Bool.self) { return false }
                var state: UInt64 = 0x2545_F491_4F6C_DD1D
                for round in 0 ..< 4 {
                    let t = (0 ..< k).map { _ -> Float in
                        state = state &* 6364136223846793005 &+ 1442695040888963407
                        return (Float(state >> 40) / Float(1 << 24) - 0.5) * 3.4641016
                    }
                    let x = (MLXArray(t) * gamma.asType(.float32) * Float(1 + round % 3)).asType(.bfloat16)
                        .reshaped(1, 1, k)
                    let a = screen(x, false), b = screen(x, true)
                    ok = ok .&& (a.scores.view(dtype: .uint32) .== b.scores.view(dtype: .uint32)).all()
                        .&& (a.id .== b.id).all() .&& (a.candidateIds .== b.candidateIds).all()
                        .&& (a.candidateValues.view(dtype: .uint32) .== b.candidateValues.view(dtype: .uint32)).all()
                    cases += 1
                }
                return ok.item(Bool.self)
            }
            partialOffset32Verified = equal
        } catch {
            return "off (\(error))"
        }
        return partialOffset32Verified
            ? "on (\(cases) full-head cases: scores bitwise equal to the paired partial; half-head screens keep it)"
            : "off (differs from the paired partial after \(cases) cases)"
    }

    /// `finalKernel` on chunk bests ([NCH] float32 values, [NCH] int32 ids): the pick, [1] int32.
    static func pick(values: MLXArray, ids: MLXArray) -> MLXArray {
        finalKernel(
            [values, ids], template: [("NCH", values.dim(0))],
            grid: (32, 1, 1), threadGroup: (32, 1, 1),
            outputShapes: [[1]], outputDTypes: [.int32])[0]
    }

    /// `finalSource`'s statements as a block inside another launch's SIMD group: the same lane loop over the
    /// `count` chunk bests in `values` / `ids`, the same comparator and xor butterfly; every lane takes lane 0's
    /// id (the id the final kernel stores) into the int `result`, declared before the block. All 32 lanes of the
    /// SIMD group must run it.
    static func pickStatements(values: String, ids: String, count: String, result: String) -> String {
        var body = finalSource
        let swaps: [(String, String)] = [
            ("best_val[c]", values + "[c]"), ("best_id[c]", ids + "[c]"), ("c < NCH", "c < " + count),
            ("  if (lane == 0) {\n    out[0] = mi;\n  }", "  \(result) = simd_broadcast(mi, ushort(0));"),
        ]
        for (old, new) in swaps {
            precondition(body.components(separatedBy: old).count == 2, "screen pick statements: \(old)")
            body = body.replacingOccurrences(of: old, with: new)
        }
        return "int \(result);\n{\n" + body + "\n}\n"
    }

    /// NEMOTRON_MTP_SCREEN_PICK_HOIST=0 keeps the pick statements' loop form. Otherwise each lane first loads all
    /// its chunk bests (slots lane, lane + 32, ... below `count`; at most `(maxCount + 31) / 32` of them, a
    /// compile-time bound) and then runs the loop's comparisons on them in the same slot order, so the loads issue
    /// together instead of one iteration at a time. The comparator, the butterfly and lane 0's result are
    /// unchanged, so the pick is the same id, NaNs and ties included.
    static let pickHoistEnabled = ProcessInfo.processInfo.environment["NEMOTRON_MTP_SCREEN_PICK_HOIST"] != "0"

    /// `pickStatements` with the loads hoisted ahead of the comparisons; `maxCount` is a compile-time bound on
    /// `count` (a template constant or a constexpr expression).
    static func pickStatementsHoisted(
        values: String, ids: String, count: String, maxCount: String, result: String
    ) -> String {
        let loop = "  for (int c = lane; c < NCH; c += 32) {\n    const float v = best_val[c];\n"
            + "    const int id = best_id[c];\n"
        let hoisted = """
              constexpr int nsp_pk = ((\(maxCount)) + 31) / 32;
              float nsp_v[nsp_pk];
              int nsp_i[nsp_pk];
              _Pragma("clang loop unroll(full)")
              for (int k = 0; k < nsp_pk; k++) {
                const int c = lane + 32 * k;
                nsp_v[k] = c < \(count) ? \(values)[c] : 0.0f;
                nsp_i[k] = c < \(count) ? \(ids)[c] : 0;
              }
              _Pragma("clang loop unroll(full)")
              for (int k = 0; k < nsp_pk; k++) {
                if (lane + 32 * k >= \(count)) continue;
                const float v = nsp_v[k];
                const int id = nsp_i[k];

            """
        var body = finalSource
        let swaps: [(String, String)] = [
            (loop, hoisted),
            ("  if (lane == 0) {\n    out[0] = mi;\n  }", "  \(result) = simd_broadcast(mi, ushort(0));"),
        ]
        for (old, new) in swaps {
            precondition(body.components(separatedBy: old).count == 2, "screen pick statements (hoisted): \(old)")
            body = body.replacingOccurrences(of: old, with: new)
        }
        precondition(!body.contains("best_val") && !body.contains("best_id") && !body.contains("c < NCH;"))
        return "int \(result);\n{\n" + body + "\n}\n"
    }

    /// The fused select's ticket counter: one zeroed uint32, allocated once
    /// per screen plan, which every fused dispatch returns to 0. MLX treats
    /// a mutable input as that dispatch's output, so it puts a barrier
    /// between any two fused dispatches and their tickets never interleave.
    static func makeCounter() -> MLXArray {
        let counter = MLXArray.zeros([1], dtype: .uint32)
        eval(counter)
        return counter
    }

    struct Result {
        let id: MLXArray
        let scores: MLXArray
        let candidateIds: MLXArray
        let candidateValues: MLXArray
        /// The chunk bests the pick reduces ([NCH] float32 values, [NCH] int32 ids).
        var bestValues: MLXArray? = nil
        var bestIds: MLXArray? = nil
    }

    /// hidden [1, 1, 2688] bf16; the head arrays are the full shared head.
    /// `upperGroups` (count `upperGroupCount`) also screens rows 65536..<131072
    /// with that smaller group set; candidate ids stay global head rows, and
    /// every chunk keeps the same per-chunk selection and exact rescore.
    /// `records` (`packRecords`) wins over `packed` (`packScalesBiases`); each
    /// must be built from the same groups and upper groups.
    /// `counter` (`makeCounter`): one fused select + final launch.
    static func run(
        _ hidden: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray,
        groups: MLXArray, groupCount: Int, chunk: Int, keep: Int,
        upperGroups: MLXArray? = nil, upperGroupCount: Int = 0,
        packed: MLXArray? = nil, records: MLXArray? = nil, records2: MLXArray? = nil,
        counter: MLXArray? = nil, paired: Bool = false,
        rowsPerSimdgroup: Int = 8, simdgroups: Int = 2, b1: Bool = false, noTicket: Bool = false,
        tunedPartial: Bool = false, selectOffset32: Bool? = nil, partialOffset32: Bool? = nil
    ) -> Result {
        let rows = upperGroups == nil ? Self.rows : headRows
        let template: [(String, any KernelTemplateArg)] = [
            ("T", hidden.dtype), ("S", groupCount), ("RPS", rowsPerSimdgroup),
            ("NSG", simdgroups), ("LO", Self.rows),
            ("S2", upperGroups == nil ? groupCount : upperGroupCount)]
        let partialHalf = partialHalfDecodeForm != 0
        let partialTemplate = template + nemotronHalfTemplate(partialHalf, partialHalfDecodeForm)
        let pairedPass = paired
            && pairable(groupCount: groupCount, upperGroupCount: upperGroups == nil ? nil : upperGroupCount,
                        rowsPerSimdgroup: rowsPerSimdgroup)
        // `partialOffset32`: the paired pass's 32-bit twin on a full-head screen (`qualifyPartialOffset32`).
        let pairedTwin = pairedPass && records2.map {
            Self.partialOffset32(fullHead: upperGroups != nil, records2: $0, offset32: partialOffset32)
        } ?? false
        let record2Kernel = pairedPass
            ? (pairedTwin ? record2PairedOffset32Kernel : nil) ?? record2PairedKernel ?? record2PartialKernel
            : record2PartialKernel
        // `tunedPartial` (`NemotronDraftTune`): the paired pass at the given geometry through the tune's kernel.
        let tunedScores: MLXArray? = tunedPartial && paired ? records2.flatMap {
            record2Scores(hidden, records2: $0, groups: groups, groupCount: groupCount, upperGroups: upperGroups,
                          upperGroupCount: upperGroupCount, rps: rowsPerSimdgroup, nsg: simdgroups, dep: hidden,
                          offset32: partialOffset32)
        } : nil
        let scores = tunedScores ?? (records2.map {
            record2Kernel(
                [$0, hidden, groups, upperGroups ?? groups], template: template,
                grid: (32, rows / rowsPerSimdgroup, 1), threadGroup: (32, simdgroups, 1),
                outputShapes: [[rows]], outputDTypes: [.float32])
        } ?? records.map {
            (partialHalf ? recordPartialHalfKernel : recordPartialKernel)(
                [$0, hidden, groups, upperGroups ?? groups], template: partialTemplate,
                grid: (32, rows / rowsPerSimdgroup, 1), threadGroup: (32, simdgroups, 1),
                outputShapes: [[rows]], outputDTypes: [.float32])
        } ?? packed.map {
            (partialHalf ? packedPartialHalfKernel : packedPartialKernel)(
                [weight, $0, hidden, groups, upperGroups ?? groups], template: partialTemplate,
                grid: (32, rows / rowsPerSimdgroup, 1), threadGroup: (32, simdgroups, 1),
                outputShapes: [[rows]], outputDTypes: [.float32])
        } ?? (partialHalf ? partialHalfKernel : partialKernel)(
            [weight, scales, biases, hidden, groups, upperGroups ?? groups], template: partialTemplate,
            grid: (32, rows / rowsPerSimdgroup, 1), threadGroup: (32, simdgroups, 1),
            outputShapes: [[rows]], outputDTypes: [.float32]))[0]
        let chunks = rows / chunk
        let selectGroups = 8
        let selectHalf = selectHalfDecodeForm != 0
        if noTicket, b1, !selectHalf, keep <= 32,
           let picked = selectB1(scores, weight: weight, scales: scales, biases: biases, hidden: hidden,
                                 chunk: chunk, keep: keep, offset32: selectOffset32) {
            return Result(
                id: pick(values: picked[2], ids: picked[3]), scores: scores, candidateIds: picked[0],
                candidateValues: picked[1], bestValues: picked[2], bestIds: picked[3])
        }
        if let counter {
            let fused = (b1 && !selectHalf && b1Available(keep: keep) ? fusedSelectB1Kernel : nil)
                ?? (selectHalf ? fusedSelectHalfKernel : fusedSelectKernel)
            let picked = fused(
                [scores, weight, scales, biases, hidden, counter],
                template: [("T", hidden.dtype), ("CH", chunk), ("C", keep), ("NSG", selectGroups),
                           ("NCH", chunks)] + nemotronHalfTemplate(selectHalf, selectHalfDecodeForm),
                grid: (32, selectGroups * chunks, 1), threadGroup: (32, selectGroups, 1),
                outputShapes: [[chunks * keep], [chunks * keep], [chunks], [chunks], [1]],
                outputDTypes: [.int32, .float32, .float32, .int32, .int32])
            return Result(id: picked[4], scores: scores, candidateIds: picked[0], candidateValues: picked[1],
                          bestValues: picked[2], bestIds: picked[3])
        }
        let picked = (selectHalf ? selectHalfKernel : selectKernel)(
            [scores, weight, scales, biases, hidden],
            template: [("T", hidden.dtype), ("CH", chunk), ("C", keep), ("NSG", selectGroups)]
                + nemotronHalfTemplate(selectHalf, selectHalfDecodeForm),
            grid: (32, selectGroups * chunks, 1), threadGroup: (32, selectGroups, 1),
            outputShapes: [[chunks * keep], [chunks * keep], [chunks], [chunks]],
            outputDTypes: [.int32, .float32, .float32, .int32])
        let id = finalKernel(
            [picked[2], picked[3]], template: [("NCH", chunks)],
            grid: (32, 1, 1), threadGroup: (32, 1, 1),
            outputShapes: [[1]], outputDTypes: [.int32])[0]
        return Result(id: id, scores: scores, candidateIds: picked[0], candidateValues: picked[1])
    }
}

/// The ticket-free draft screen's picks (`NemotronHeadScreen.noTicketEnabled`): for each pick array the draft head
/// hands out, the chunk bests it is the final reduction of, so a launch that reads the pick can form it itself
/// (`NemotronHeadScreen.pickStatements`) instead of waiting on the final. Matched by the pick array's identity;
/// the last eight picks are kept.
enum NemotronScreenPicks {
    struct Bests {
        let values: MLXArray
        let ids: MLXArray
        var count: Int { values.dim(0) }
    }
    private static let lock = NSLock()
    nonisolated(unsafe) private static var recent: [(pick: MLXArray, bests: Bests)] = []
    /// The chunk counts the ticket-free screen picks reduce (half head, then full head when the plan screens the
    /// upper half); set when the plan adopts it, empty otherwise.
    nonisolated(unsafe) static var chunkCounts: [Int] = []

    static func register(_ pick: MLXArray, _ bests: Bests) {
        lock.withLock {
            recent.append((pick, bests))
            if recent.count > 8 { recent.removeFirst(recent.count - 8) }
        }
    }

    /// The chunk bests `pick` reduces, when it is a registered pick.
    static func bests(of pick: MLXArray) -> Bests? {
        lock.withLock { recent.last(where: { $0.pick === pick })?.bests }
    }

    /// Seeded chunk bests for the consumers' load checks: `count` values (kind 0 Gaussian, 1 rounded (exact
    /// ties), 2 all equal, 3 NaN in every fifth slot from slot 0, 4 infinities, 5 every eighth value tied to the
    /// maximum) and distinct ids, chunk c's id inside chunk c's slice of the vocabulary.
    static func seededBests(count: Int, vocabulary: Int, kind: Int, seed: UInt64) -> Bests {
        let keys = MLXRandom.split(key: MLXRandom.key(seed), into: 2)
        let slot = MLXArray(Int32(0) ..< Int32(count))
        var values = MLXRandom.normal([count], key: keys[0]) * Float(3)
        switch kind {
        case 1: values = round(values)
        case 2: values = MLXArray.zeros([count]) + MLXArray(Float(1.5))
        case 3: values = which((slot % 5) .== MLXArray(Int32(0)), MLXArray(Float.nan), values)
        case 4:
            values = which(values .> MLXArray(Float(2.5)), MLXArray(Float.infinity),
                which(values .< MLXArray(Float(-2.5)), MLXArray(-Float.infinity), values))
        case 5: values = which((slot % 8) .== MLXArray(Int32(3)), values.max(), values)
        default: break
        }
        let span = max(1, vocabulary / count)
        let ids = slot * Int32(span) + MLXRandom.randInt(Int32(0) ..< Int32(span), [count], key: keys[1])
        return Bests(values: values.asType(.float32), ids: ids.asType(.int32))
    }
}

// MARK: - Load-time geometry tune of the draft chain's kernels (NEMOTRON_DRAFT_TUNE)
//
// The two draft steps of a round run serially, in small latency-bound launches whose geometry was picked on
// M1/M4-class hosts. `NemotronDraftTune` lets the ranked machine choose the geometry of eight of them during the
// load, each family rebuilt through its incumbent's own source pipeline:
// - fc1: the head's routed FC1 leader at R = 1 (tail-8 pair-dictionary leader; four outputs per SIMD group, two SIMD
//   groups per threadgroup, the slot on grid z, both main K loops unrolled by two once `NemotronU2` admitted DRAFT);
// - screen: the 2-bit screen partial pass with the paired upper rows (eight rows per SIMD group, two SIMD groups);
// - wo, up, down, eh: the R = 1 pair-dictionary multi-row QMVs of the attention wo (K 4096), the shared-expert up
//   (K 2688 -> 3712, ReLU2) and down (K 3712, its `dep` twin) and the second draft's eh_proj (K 5376) (four outputs
//   per SIMD group, two SIMD groups);
// - qkv: the R = 1 pair-dictionary fused Q/K/V (tiles of two SIMD groups of four outputs);
// - ehp: the first draft's R = 3 eh_proj over the post-verify rows (the plain prepared QMV: two outputs per SIMD
//   group, two SIMD groups; V0's twin and every variant carry the incumbent's closing store, the crown's vector
//   store once `NemotronTargetQKV.qualifyVectorStore` admitted it, else the lane-0 store).
// Every output keeps its arithmetic: one output (one screen row) is one SIMD group's work; its lanes hold the same
// values of each block, blocks run in K order with the same decode, products, term order and affine close, then the
// same closing simd_sum (the screen's paired rows: the same shuffle and sums), rounding and store. A variant moves
// only how many consecutive outputs (rows) one SIMD group owns, how many SIMD groups share a threadgroup, the slot
// axis (fc1), the unroll pragma on the main K loops (the same iterations in the same order) and, for ehp, the
// prepared QMV's main-loop load form (`nemotronPreparedVariantSource`: the same bits reach the same decode lines).
// Each tune kernel carries an unread `dep` input; production binds the launch's own input (the shared down: its
// `dep`) there.
// `tune` (load time, after the draft dictionaries, `NemotronU2` and the screen plan, before any draft graph is
// traced) checks every variant bit for bit against the production launch on the head's real weights with seeded
// inputs (fc1: six routes; screen: five rows, one through a view that is not 16-byte aligned, on both halves; the
// QMVs: plain, wide and ReLU2-like rows with -0 zeros), drops any mismatch, missing kernel or Metal error, checks the
// incumbent's own geometry through the tune's pipeline (V0's twin) the same way, then times V0's twin and the
// survivors round robin (`NemotronTuneAdoption.order`) as chains in which each launch binds the previous launch's
// output as `dep`, so they run one after another as in the round, and adopts the fastest variant only when
// `NemotronTuneAdoption.decide` admits it (median at least 3% below V0's twin and faster than V0's twin in at least
// 75% of the cycles, paired within each cycle).
// The chains read cold data as the round does: fc1 cycles 16 routes through the bank, the other families cycle copies
// of their weights (an integer add of zero, about 96 MB per pass). Load cap 4.5 s.
// NEMOTRON_DRAFT_TUNE=0 keeps every incumbent; NEMOTRON_DRAFT_FORCE=<n> adopts variant n (1-based, `variants`) of
// every family that qualified it, <family>:<n>[,<family>:<n>] of the named families; 0 keeps the incumbent.
enum NemotronDraftTune {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_DRAFT_TUNE"] != "0"
    static let forced: (all: Int?, families: [Family: Int]) = {
        let value = ProcessInfo.processInfo.environment["NEMOTRON_DRAFT_FORCE"] ?? ""
        if let n = Int(value) { return (n, [:]) }
        var out: [Family: Int] = [:]
        for item in value.split(whereSeparator: { $0 == "," || $0 == "+" }) {
            let parts = item.split(separator: ":")
            guard parts.count == 2, let f = Family(rawValue: String(parts[0])), let n = Int(parts[1]) else { continue }
            out[f] = n
        }
        return (nil, out)
    }()
    /// Soft cap on the tune's own load time: no family, variant (its first launch JITs its pipeline) or timing phase
    /// starts past it; what has not started keeps its incumbent. A forced run checks every family. Raised with the
    /// widened variant lists: the bit checks cost about a launch each and 76 variants now run where 54 did.
    static let budgetSeconds = forced.all != nil || !forced.families.isEmpty ? 60.0 : 4.5
    /// Wall time of one family's timed cycles (graph building included), sized from the warm-up cycle. `cycles` is
    /// `target / wall` and `wall` is one round of every candidate, so a round's cost and the cycle count move
    /// oppositely: the target is raised in proportion to the longest list (ehp, 7 -> 11 rows) to hold each
    /// candidate's sample count where it was, and `minCycles` still floors every family at 16.
    static let timedMilliseconds = 160.0

    /// In tune order (sdpa: `tuneSDPA`, after the draft attention's checks).
    enum Family: String, CaseIterable, Sendable {
        case fc1, screen, qkv, up, wo, down, ehp, eh, sdpa
    }

    struct Geometry: Hashable, Sendable, CustomStringConvertible {
        /// Outputs (screen: rows) per SIMD group.
        var rps: Int
        /// SIMD groups per threadgroup.
        var nsg: Int
        /// unroll_count of the main K loops (1: no pragma).
        var unroll = 1
        /// fc1: the slot on grid x instead of z.
        var slotX = false
        /// ehp: the prepared QMV's main-loop load form.
        var load = 0
        /// sdpa: the key offsets (0: the incumbent's 64-bit `nmp_at`, 1: 32-bit, 2: 32-bit from pages in lanes).
        var keys = 0
        /// sdpa: the key step's K and V kept as T until their products.
        var packed = false
        /// ehp: the vector-store closing loop (`nemotronQMVVectorStoreSource`); the incumbent's
        /// (`NemotronHeadPreparedActivation.projectVSVerified`).
        var vs = false
        var description: String {
            "r\(rps)n\(nsg)" + (slotX ? "x" : "") + "u\(unroll)" + (load != 0 ? "l\(load)" : "")
                + (keys != 0 ? "k\(keys)" : "") + (packed ? "p" : "") + (vs ? "vs" : "")
        }
    }

    struct ScreenPlan {
        let records2: MLXArray
        let groups: MLXArray
        let groupCount: Int
        let upperGroups: MLXArray?
        let upperGroupCount: Int
    }

    /// Adopted geometries by family, set once at load before any draft graph is traced.
    nonisolated(unsafe) private(set) static var adopted: [Family: Geometry] = [:]
    /// sdpa: the form `tuneSDPA` checked and timed (the fenced launches when their check passed, else the plain
    /// ones); only that form launches the adopted variant.
    nonisolated(unsafe) private(set) static var sdpaFenced = false

    /// The multi-row QMV family of a draft head R = 1 dictionary launch.
    static func qmvFamily(k: Int, n: Int, relu2: Bool) -> Family? {
        switch (k, n, relu2) {
        case (4096, 2688, false): return .wo
        case (2688, 3712, true): return .up
        case (3712, 2688, false): return .down
        case (5376, 2688, false): return .eh
        default: return nil
        }
    }

    /// Each family's variants in a fixed order (NEMOTRON_DRAFT_FORCE indexes them), resolved against the incumbent's
    /// unroll and store form; a variant equal to the incumbent is skipped.
    static func variants(_ f: Family, _ inc: Geometry) -> [(name: String, g: Geometry)] {
        func v(_ rps: Int, _ nsg: Int, unroll: Int? = nil, slotX: Bool = false, load: Int = 0, keys: Int = 0,
               packed: Bool = false) -> (name: String, g: Geometry) {
            let g = Geometry(rps: rps, nsg: nsg, unroll: unroll ?? inc.unroll, slotX: slotX, load: load, keys: keys,
                             packed: packed, vs: inc.vs)
            return (g.description, g)
        }
        // Each list ends with cross-products the family's siblings carry and it lacked: fc1's slot axis and
        // unroll at the wide row counts, the QMV/QKV families' r8n1 and their unrolled r8n2, the screen's
        // eight-group rows, and ehp's three load forms it never tried next to its rps/nsg crosses. Every one
        // is guarded by the same launch helpers as the rows above it, so a shape the kernel cannot express
        // drops as `n/a` at its bit check and the list costs only its check.
        switch f {
        case .fc1:
            return [v(2, 2), v(2, 4), v(4, 1), v(4, 4), v(8, 1), v(8, 2), v(4, 2, slotX: true),
                    v(4, 2, unroll: inc.unroll == 1 ? 2 : 1), v(4, 2, unroll: 4),
                    v(8, 2, slotX: true), v(4, 1, slotX: true),
                    v(8, 2, unroll: inc.unroll == 2 ? 4 : 2), v(2, 4, unroll: inc.unroll == 2 ? 4 : 2)]
        case .screen:
            return [v(4, 2), v(16, 2), v(8, 1), v(8, 4), v(16, 1), v(32, 1), v(4, 4), v(16, 4),
                    v(8, 8), v(16, 8), v(32, 4), v(4, 8)]
        case .wo, .up, .down, .eh, .qkv:
            return [v(2, 2), v(2, 4), v(4, 1), v(4, 4), v(8, 2), v(4, 2, unroll: 2), v(8, 1),
                    v(8, 2, unroll: 2)]
        case .ehp:
            return [v(2, 4), v(2, 1), v(4, 2), v(1, 4), v(2, 8), v(2, 2, unroll: 2), v(2, 2, load: 2),
                    v(2, 2, load: 1), v(2, 2, load: 3), v(2, 4, load: 2), v(4, 2, load: 3)]
        case .sdpa:
            return [v(1, 32, unroll: 8, keys: 2, packed: true), v(1, 32, unroll: 8, keys: 2),
                    v(1, 32, unroll: 8, keys: 1), v(1, 32, unroll: 4, keys: 2), v(1, 32, unroll: 4, keys: 1)]
        }
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var kernels: [String: MLXFast.MLXFastKernel?] = [:]
    static func cached(_ key: String, _ make: () -> MLXFast.MLXFastKernel?) -> MLXFast.MLXFastKernel? {
        lock.withLock {
            if let made = kernels[key] { return made }
            let made = make()
            kernels[key] = made
            return made
        }
    }

    /// The R = 1 dictionary multi-row body (the pipeline of `kernelHalfDict`: decode form, pair dictionary, vector
    /// store) with the SIMD groups per threadgroup from the template and `unroll` on its main K loop.
    static func qmvBody(unroll: Int) -> String? {
        nemotronCountedSwaps(nemotronMultiRowQMVSource, nemotronPreparedNSGSwaps)
            .flatMap { nemotronCountedSwaps($0, nemotronMultiRowHalfSwaps) }
            .flatMap(nemotronSBDictSource)
            .flatMap { nemotronUnrollSource($0, loop: NemotronU2.qmvLoop, count: 1, unroll: unroll) }
    }

    /// One R = 1 dictionary multi-row launch of geometry `g` (x: one row of K); nil when it cannot run.
    static func qmvLaunch(_ g: Geometry, w: MLXArray, index: MLXArray, table: MLXArray, x: MLXArray,
                          relu2: Bool, dep: MLXArray) -> MLXArray? {
        let k = x.dim(-1), n = w.dim(0)
        let form = NemotronMultiRowQMVKernel.r1HalfDecodeForm
        guard form != 0, x.size == k, n % g.rps == 0, (n / g.rps) % g.nsg == 0, g.rps <= 32, g.nsg <= 32,
              let kernel = cached("nemotron_draft_tune_qmv_u\(g.unroll)", {
                  qmvBody(unroll: g.unroll).map {
                      MLXFast.metalKernel(
                          name: "nemotron_draft_tune_qmv_u\(g.unroll)",
                          inputNames: ["w", "scales", "biases", "x", "K", "N", "dep"], outputNames: ["y"],
                          source: $0, header: nemotronMultiRowQMVHeader + nemotronHalfDecodeHeader)
                  }
              })
        else { return nil }
        let (kArray, nArray) = NemotronMultiRowQMV.scalarPair(k, n)
        return kernel(
            [w, index, table, x.reshaped(1, k), kArray, nArray, dep],
            template: [("T", x.dtype), ("R", 1), ("FAST", k % 512 == 0), ("RELU2", relu2), ("RPS", g.rps),
                       ("HF", form), ("NSG", g.nsg)],
            grid: (32, n / g.rps, 1), threadGroup: (32, g.nsg, 1),
            outputShapes: [[1, n]], outputDTypes: [x.dtype])[0]
    }

    /// Seeded BF16 rows [rows, k]: Gaussian at `scale`, or (relu2) the square of a shifted Gaussian's positive
    /// part with alternating +0 / -0 zeros.
    private static func seeded(_ rows: Int, _ k: Int, scale: Float, relu2: Bool = false, seed: UInt64) -> MLXArray {
        let x = MLXRandom.normal([rows, k], key: MLXRandom.key(seed)) * scale
        guard relu2 else { return x.asType(.bfloat16) }
        let p = MLX.maximum(x - 0.5, MLXArray(Float(0)))
        let odd = (MLXArray(Int32(0) ..< Int32(rows * k)).reshaped(rows, k) % 2) .== MLXArray(Int32(1))
        return which(p .> MLXArray(Float(0)), p * p, which(odd, MLXArray(Float(-0.0)), MLXArray(Float(0))))
            .asType(.bfloat16)
    }

    /// A new buffer with `a`'s bits (the timing chains' cold copies): an integer add of zero.
    private static func copy(_ a: MLXArray) -> MLXArray {
        switch a.dtype {
        case .uint32: return a + MLXArray(UInt32(0))
        case .uint16: return a + MLXArray(UInt16(0))
        default: return (a.view(dtype: .uint16) + MLXArray(UInt16(0))).view(dtype: a.dtype)
        }
    }

    /// Copies for one chain pass of about 96 MB.
    private static func copyCount(_ bytes: Int) -> Int { max(4, min(24, (96 << 20) / max(bytes, 1))) }

    private enum Check { case same, differs, unavailable, error }

    /// One family's check, timing and pick. `reference(i)`: case i's outputs through the production launch;
    /// `launch(g, i)`: through geometry g's tune kernel (nil: it cannot run); `link(g, previous, j)`: timing link j
    /// through g after `previous` (the link before's value, nil for the first), returning what the next link binds
    /// as `dep`. Returns the report.
    private static func run(
        _ family: Family, incumbent: Geometry, cases: Int, links: Int,
        reference: (Int) -> [MLXArray]?, launch: (Geometry, Int) -> [MLXArray]?,
        link: (Geometry, MLXArray?, Int) -> MLXArray?, elapsed: () -> Double, timed: Double = timedMilliseconds
    ) -> String {
        var references: [[MLXArray]] = []
        do {
            references = try withError { () -> [[MLXArray]] in
                let outs = (0 ..< cases).compactMap { reference($0) }
                guard outs.count == cases else { return [] }
                eval(outs.flatMap { $0 })
                return outs
            }
        } catch {
            return "off (incumbent: \(error))"
        }
        guard references.count == cases else { return "off (incumbent unavailable)" }
        func check(_ g: Geometry) -> Check {
            do {
                return try withError { () -> Check in
                    var ok = MLXArray(true)
                    for i in 0 ..< cases {
                        guard let out = launch(g, i), out.count == references[i].count else { return .unavailable }
                        for (a, b) in zip(out, references[i]) { ok = logicalAnd(ok, NemotronU2.equal(a, b)) }
                    }
                    return ok.item(Bool.self) ? .same : .differs
                }
            } catch {
                return .error
            }
        }
        let twin = check(incumbent)
        guard twin == .same else { return "off (V0 twin \(incumbent): \(twin))" }
        let all = variants(family, incumbent)
        var alive: [(name: String, g: Geometry, index: Int)] = []
        var dropped: [String] = []
        for (i, v) in all.enumerated() where v.g != incumbent {
            guard elapsed() < budgetSeconds else {
                dropped.append("\(v.name) budget")
                continue
            }
            switch check(v.g) {
            case .same: alive.append((v.name, v.g, i + 1))
            case .differs: dropped.append("\(v.name) differs")
            case .unavailable: dropped.append("\(v.name) n/a")
            case .error: dropped.append("\(v.name) error")
            }
        }

        // Time V0's twin and the survivors round robin (`NemotronTuneAdoption.order`): one chain of `links` launches
        // per sample, each binding the previous launch's output as `dep`, built before the clock starts; median ms per
        // launch.
        let checked = elapsed()
        let candidates = [incumbent] + alive.map(\.g)
        var medians: [Double] = []
        var verdict: NemotronTuneAdoption.Verdict? = nil
        var reps = 0
        if !alive.isEmpty, elapsed() >= budgetSeconds {
            dropped.append("timing skipped (load budget)")
        } else if !alive.isEmpty {
            var samples = Array(repeating: [Double](), count: candidates.count)
            do {
                try withError {
                    func build(_ c: Int) -> MLXArray? {
                        var previous: MLXArray? = nil
                        for j in 0 ..< links {
                            guard let out = link(candidates[c], previous, j) else { return nil }
                            previous = out
                        }
                        return previous
                    }
                    func round(_ order: [Int]) -> [Double?] {
                        let chains = order.map { build($0) }
                        return chains.map { last -> Double? in
                            guard let last else { return nil }
                            let start = DispatchTime.now().uptimeNanoseconds
                            eval(last)
                            Stream.gpu.synchronize()
                            return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6 / Double(links)
                        }
                    }
                    let start = DispatchTime.now().uptimeNanoseconds
                    _ = round(Array(candidates.indices))
                    let wall = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
                    reps = NemotronTuneAdoption.cycles(target: timed, wall: wall, cap: 58)
                    for rep in 0 ..< reps {
                        let order = NemotronTuneAdoption.order(candidates.count, cycle: rep)
                        for (i, ms) in zip(order, round(order)) {
                            if let ms { samples[i].append(ms) }
                        }
                    }
                }
            } catch {
                dropped.append("timing error")
            }
            if reps > 0, samples.allSatisfy({ $0.count == reps }) {
                verdict = NemotronTuneAdoption.decide(samples)
                medians = verdict?.medians ?? []
            } else {
                dropped.append("timing incomplete")
            }
        }

        var picked: (name: String, g: Geometry, index: Int)? = nil
        var why = "kept"
        if let f = forced.families[family] ?? forced.all {
            if let p = alive.first(where: { $0.index == f }) {
                picked = p
                why = "forced"
            } else {
                why = f == 0 ? "forced V0" : "forced \(f) not qualified, kept"
            }
        } else if medians.count == candidates.count, let best = verdict?.adopted {
            picked = alive[best - 1]
            why = "adopted"
        }
        if let picked { adopted[family] = picked.g }

        func ms(_ i: Int) -> String { i < medians.count ? String(format: "%.4f", medians[i]) : "n/a" }
        let pickedSlot = picked.flatMap { p in alive.firstIndex(where: { $0.index == p.index }).map { $0 + 1 } } ?? 0
        let times = alive.indices.map { "\(alive[$0].name) \(ms($0 + 1)) \(verdict?.won($0 + 1) ?? "n/a")" }
        let rule = verdict?.rule({ alive[$0 - 1].name }) ?? "no candidate timed"
        return "\(why) \(picked?.name ?? "V0") \(ms(pickedSlot)) ms vs V0 (\(incumbent)) \(ms(0)) ms per launch"
            + " (\(rule); median of \(reps) cycles, \(links) chained launches/sample;"
            + " \(alive.count)/\(all.count) bitwise over"
            + " \(cases) cases by \(String(format: "%.2f", checked)) s, V0 twin bitwise"
            + (times.isEmpty ? "" : "; [" + times.joined(separator: ", ") + "]")
            + (dropped.isEmpty ? "" : "; dropped: " + dropped.prefix(8).joined(separator: ", ")) + ")"
    }

    /// Load time, after `NemotronU2.qualify`, the draft dictionaries and the screen plan, before any draft graph is
    /// traced or warmed.
    static func tune(_ head: Module, screen: ScreenPlan?) {
        adopted = [:]
        let t0 = DispatchTime.now().uptimeNanoseconds
        func elapsed() -> Double { Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9 }
        func report(_ family: String, _ message: String) {
            FileHandle.standardError.write(Data(
                "nemotron draft tune \(family): \(message) (\(String(format: "%.2f", elapsed())) s)\n".utf8))
        }
        guard enabled else { return report("all", "off (setting)") }
        guard !NemotronMoELeader.selfTest, Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu
        else { return report("all", "off (not applicable)") }
        for family in Family.allCases where family != .sdpa {
            guard elapsed() < budgetSeconds else {
                report(family.rawValue, "skipped (load budget), incumbent kept")
                continue
            }
            let message: String
            switch family {
            case .fc1: message = tuneFC1(head, elapsed: elapsed)
            case .screen: message = screen.map { tuneScreen($0, elapsed: elapsed) } ?? "off (no paired 2-bit screen)"
            case .wo, .up, .down, .eh: message = tuneQMV(family, head, elapsed: elapsed)
            case .qkv: message = tuneQKV(head, elapsed: elapsed)
            case .ehp: message = tuneEhp(head, elapsed: elapsed)
            case .sdpa: continue
            }
            report(family.rawValue, message)
        }
    }

    private static func tensor(_ f: Family, _ head: Module) -> QuantizedLinear? {
        let attention = head.modules().lazy.compactMap { $0 as? NemotronH35MTPAttentionBlock }.first
        let shared = head.modules().lazy.compactMap { ($0 as? NemotronHMoE)?.sharedExperts }.first
        let linear: Linear?
        switch f {
        case .eh, .ehp: linear = attention?.projection
        case .wo: linear = attention?.mixer.wo
        case .up: linear = shared?.upProj
        case .down: linear = shared?.downProj
        default: linear = nil
        }
        guard let linear, type(of: linear) == QuantizedLinear.self else { return nil }
        return linear as? QuantizedLinear
    }

    private static func tuneFC1(_ head: Module, elapsed: () -> Double) -> String {
        guard NemotronMoELeader.enabled, NemotronMoELeader.fuseRelu2,
              let moe = head.modules().lazy.compactMap({ $0 as? NemotronHMoE }).first, moe.numExpertsPerTok == 6,
              type(of: moe.switchMLP.fc1) == QuantizedSwitchLinear.self,
              let bank = NemotronMoELeader.bank(moe.switchMLP.fc1, k: 2688, dtype: .bfloat16),
              bank.weight.shape == [128, 1856, 336], let dict = NemotronSBDict.draftDict(moe.switchMLP.fc1)
        else { return "off (no draft FC1 dictionary bank)" }
        let tail8 = NemotronMoELeader.leaderTail8
        var state: UInt64 = 0xDF1_7E5E
        func route() -> MLXArray {
            var r: [UInt32] = []
            while r.count < 6 {
                state &+= 0x9E37_79B9_7F4A_7C15
                var z = state
                z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
                z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
                let e = UInt32((z ^ (z >> 31)) % 128)
                if !r.contains(e) { r.append(e) }
            }
            return MLXArray(r)
        }
        let scales: [Float] = [0.5, 2, 1, 8, 1, 0.5]
        let xs = scales.enumerated().map { seeded(1, 2688, scale: $0.element, relu2: $0.offset == 2,
                                                  seed: UInt64(0xDF10 + $0.offset)) }
        let routes = xs.map { _ in route() }
        let timed = (0 ..< 16).map { _ in route() }
        eval(xs + routes + timed)
        let incumbent = Geometry(rps: NemotronMoELeader.leaderResultsPerSimdgroup(rows: 1), nsg: 2,
                                 unroll: NemotronU2.draft ? 2 : 1)
        return run(
            .fc1, incumbent: incumbent, cases: xs.count, links: timed.count,
            reference: { i in
                [NemotronMoELeader.apply(xs[i], routes[i], rows: 1, topK: 6, bank: bank, perSlot: false,
                                         relu2: true, tail8: tail8, dict: dict)]
            },
            launch: { g, i in
                NemotronMoELeader.draftTuneLaunch(g, xs[i], routes[i], bank: bank, dict: dict, relu2: true,
                                                  tail8: tail8, dep: xs[i]).map { [$0] }
            },
            link: { g, previous, j in
                NemotronMoELeader.draftTuneLaunch(g, xs[0], timed[j], bank: bank, dict: dict, relu2: true,
                                                  tail8: tail8, dep: previous ?? xs[0])
            }, elapsed: elapsed)
    }

    private static func tuneScreen(_ p: ScreenPlan, elapsed: () -> Double) -> String {
        var hs: [MLXArray] = [Float(0.5), 1, 2, 4].enumerated().map {
            seeded(1, 2688, scale: $0.element, seed: UInt64(0x5C7E + $0.offset)).reshaped(1, 1, 2688)
        }
        let padded = concatenated([MLXArray.zeros([1], dtype: .bfloat16), hs[0].reshaped(2688)], axis: 0)
        hs.append(padded[1 ..< 2689].reshaped(1, 1, 2688))
        eval(hs)
        let halves: [Bool] = p.upperGroups == nil ? [false] : [false, true]
        func scores(_ h: MLXArray, upper: Bool, records: MLXArray, g: Geometry, dep: MLXArray?) -> MLXArray? {
            NemotronHeadScreen.record2Scores(
                h, records2: records, groups: p.groups, groupCount: p.groupCount,
                upperGroups: upper ? p.upperGroups : nil, upperGroupCount: upper ? p.upperGroupCount : 0,
                rps: g.rps, nsg: g.nsg, dep: dep)
        }
        let incumbent = Geometry(rps: 8, nsg: 2)
        let c = copyCount(p.records2.nbytes)
        let copies = (0 ..< c).map { _ in copy(p.records2) }
        eval(copies)
        return run(
            .screen, incumbent: incumbent, cases: hs.count * halves.count, links: max(8, c),
            reference: { i in
                scores(hs[i % hs.count], upper: halves[i / hs.count], records: p.records2, g: incumbent, dep: nil)
                    .map { [$0] }
            },
            launch: { g, i in
                scores(hs[i % hs.count], upper: halves[i / hs.count], records: p.records2, g: g,
                       dep: hs[i % hs.count]).map { [$0] }
            },
            link: { g, previous, j in
                scores(hs[0], upper: halves[j % halves.count], records: copies[j % c], g: g,
                       dep: previous?.view(dtype: .bfloat16) ?? hs[0])
            }, elapsed: elapsed)
    }

    private static func tuneQMV(_ family: Family, _ head: Module, elapsed: () -> Double) -> String {
        guard NemotronMultiRowQMV.enabled, NemotronMultiRowQMVKernel.r1HalfDecodeForm != 0,
              let q = tensor(family, head), q.bias == nil, let dict = NemotronSBDict.draftDict(q)
        else { return "off (no draft dictionary tensor)" }
        let k = q.weight.dim(1) * 8, n = q.weight.dim(0)
        let relu2 = family == .up, dep = family == .down
        guard qmvFamily(k: k, n: n, relu2: relu2) == family else { return "off (shape)" }
        let xs = [seeded(1, k, scale: 0.5, relu2: dep, seed: UInt64(0x9D10 + k)),
                  seeded(1, k, scale: 4, seed: UInt64(0x9D20 + k)),
                  seeded(1, k, scale: 1, relu2: true, seed: UInt64(0x9D30 + k))]
        eval(xs)
        let incumbent = Geometry(rps: NemotronMultiRowQMV.resultsPerSimdgroup(rows: 1, k: k, n: n), nsg: 2)
        let c = copyCount(q.weight.nbytes + dict.index.nbytes)
        let copies = (0 ..< c).map { _ in (copy(q.weight), copy(dict.index)) }
        eval(copies.flatMap { [$0.0, $0.1] })
        return run(
            family, incumbent: incumbent, cases: xs.count, links: max(16, c),
            reference: { i in NemotronMultiRowQMV.apply(xs[i], q, relu2: relu2, dep: dep ? xs[i] : nil).map { [$0] } },
            launch: { g, i in
                qmvLaunch(g, w: q.weight, index: dict.index, table: dict.table, x: xs[i], relu2: relu2, dep: xs[i])
                    .map { [$0] }
            },
            link: { g, previous, j in
                qmvLaunch(g, w: copies[j % c].0, index: copies[j % c].1, table: dict.table, x: xs[0], relu2: relu2,
                          dep: previous ?? xs[0])
            }, elapsed: elapsed)
    }

    private static func tuneQKV(_ head: Module, elapsed: () -> Double) -> String {
        guard NemotronMTPDraftTail.fusedQKV, NemotronTargetQKV.r1HalfDecodeForm != 0,
              let attention = head.modules().lazy.compactMap({ $0 as? NemotronH35MTPAttentionBlock }).first
        else { return "off (no fused draft Q/K/V)" }
        let mixer = attention.mixer
        let linears: [Linear] = [mixer.wq, mixer.wk, mixer.wv]
        let qs = linears.compactMap { type(of: $0) == QuantizedLinear.self ? $0 as? QuantizedLinear : nil }
        let dicts = linears.compactMap { NemotronSBDict.draftDict($0) }
        guard qs.count == 3, dicts.count == 3 else { return "off (no draft Q/K/V dictionaries)" }
        let xs = [Float(0.5), 2, 4].enumerated().map {
            seeded(1, 2688, scale: $0.element, seed: UInt64(0x9C40 + $0.offset)).reshaped(1, 1, 2688)
        }
        eval(xs)
        let incumbent = Geometry(rps: NemotronTargetQKV.resultsPerSimdgroup(rows: 1), nsg: 2)
        let bytes = qs.map(\.weight.nbytes).reduce(0, +) + dicts.map(\.index.nbytes).reduce(0, +)
        let c = copyCount(bytes)
        let copies = (0 ..< c).map { _ in (qs.map { copy($0.weight) }, dicts.map { copy($0.index) }) }
        eval(copies.flatMap { $0.0 + $0.1 })
        let tables = dicts.map(\.table)
        return run(
            .qkv, incumbent: incumbent, cases: xs.count, links: max(16, c),
            reference: { i in
                NemotronTargetQKV.apply(xs[i], query: mixer.wq, key: mixer.wk, value: mixer.wv, captureMTP: false)
                    .map { [$0.0, $0.1, $0.2] }
            },
            launch: { g, i in
                NemotronTargetQKV.tuneLaunch(g, weights: qs.map(\.weight), indices: dicts.map(\.index),
                                             tables: tables, x: xs[i], dep: xs[i])
            },
            link: { g, previous, j in
                NemotronTargetQKV.tuneLaunch(g, weights: copies[j % c].0, indices: copies[j % c].1, tables: tables,
                                             x: xs[0], dep: previous ?? xs[0])?[0]
            }, elapsed: elapsed)
    }

    private static func tuneEhp(_ head: Module, elapsed: () -> Double) -> String {
        guard NemotronMTPExecution.batchedM1, NemotronMultiRowQMV.enabled, NemotronMultiRowQMV.narrowEnabled,
              let q = tensor(.ehp, head), q.bias == nil, q.weight.shape == [2688, 672], q.weight.dtype == .uint32,
              q.mode == .affine, q.bits == 4, q.groupSize == 64,
              let biases = q.biases, q.scales.dtype == .bfloat16, biases.dtype == .bfloat16,
              NemotronHeadPreparedActivation.draftTuneServes(q)
        else { return "off (no plain prepared eh_proj)" }
        let xs = [seeded(3, 5376, scale: 0.5, seed: 0x9E10), seeded(3, 5376, scale: 4, seed: 0x9E20),
                  seeded(3, 5376, scale: 1, relu2: true, seed: 0x9E30)]
        let packed = xs.map { NemotronHeadPreparedActivation.pack($0) }
        eval(xs + packed)
        // The production launch's store form: the crown's vector store once `qualifyVectorStore` admitted it.
        let incumbent = Geometry(rps: NemotronMultiRowQMV.headRPS(n: 2688), nsg: 2,
                                 vs: NemotronHeadPreparedActivation.projectVSVerified)
        let c = copyCount(q.weight.nbytes + q.scales.nbytes + biases.nbytes)
        let copies = (0 ..< c).map { _ in (copy(q.weight), copy(q.scales), copy(biases)) }
        eval(copies.flatMap { [$0.0, $0.1, $0.2] })
        return run(
            .ehp, incumbent: incumbent, cases: xs.count, links: max(16, c),
            reference: { i in
                [NemotronHeadPreparedActivation.apply(xs[i], q, relu2: false, rps: incumbent.rps, prepared: packed[i])]
            },
            launch: { g, i in
                NemotronHeadPreparedActivation.draftTuneLaunch(
                    g, packed: packed[i], w: q.weight, scales: q.scales, biases: biases, dep: packed[i]).map { [$0] }
            },
            link: { g, previous, j in
                NemotronHeadPreparedActivation.draftTuneLaunch(
                    g, packed: packed[0], w: copies[j % c].0, scales: copies[j % c].1, biases: copies[j % c].2,
                    dep: previous?.view(dtype: .float32) ?? packed[0])
            }, elapsed: elapsed)
    }

    /// sdpa (load time, after `NemotronMTPDraftSDPA.qualifyFence`: the family needs the paged draft attention
    /// qualified): the first and second drafts' attention over the head cache's pages at R = 1
    /// (`NemotronMTPDraftSDPA.tuneSource`; incumbent r1n32u4: one 1024-thread threadgroup per query head, four keys
    /// per step, 64-bit key offsets). The production form (fenced once its check passed, else plain) of each variant
    /// and of V0's twin against the production launch, bit for bit (output and emitted fence), on seeded paged
    /// storage laid out as the head cache's segment ([2, pages, 2, 16, 128] BF16, scattered local pages, first slots
    /// 0 ... 15): key counts across the 32-key, four- and eight-key-step and 16-slot page boundaries up to 1022 and the
    /// decode window's counts, with and without the own row, masks with a -inf tail and scattered -inf keys, query
    /// and key scales 1 to 16, zero and -0 queries (every score ties), one repeated key row with -0 values (ties
    /// between keys) and scales where all but the largest exponentials underflow. Timing chains alternate the first
    /// draft's launch over a cold copy of the storage and the second's (own row) over the same copy, key counts 491
    /// ... 619, 128 copies (~90 MB), each launch binding the previous output as `dep`.
    static func tuneSDPA(_ block: NemotronH35MTPAttentionBlock) {
        adopted[.sdpa] = nil
        let t0 = DispatchTime.now().uptimeNanoseconds
        func elapsed() -> Double { Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9 }
        func report(_ message: String) {
            FileHandle.standardError.write(Data(
                "nemotron draft tune sdpa: \(message) (\(String(format: "%.2f", elapsed())) s)\n".utf8))
        }
        guard enabled else { return report("off (setting)") }
        guard !NemotronMoELeader.selfTest, Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu
        else { return report("off (not applicable)") }
        guard NemotronMTPDraftSDPA.pagedEnabled, NemotronMTPDraftSDPA.pagedVerified, NemotronMTPDraftSDPA.verified
        else { return report("off (the paged draft launch is off)") }
        let fenced = NemotronMTPDraftSDPA.fenceEnabled && NemotronMTPDraftSDPA.fenceVerified
        sdpaFenced = fenced
        let scale = block.mixer.scale
        typealias Own = (keys: MLXArray, values: MLXArray)
        func ownRow(_ kvScale: Float, _ key: MLXArray) -> Own {
            let k = MLXRandom.split(key: key, into: 2)
            return (MLXRandom.normal([1, 256], dtype: .bfloat16, scale: kvScale, key: k[0])
                        .reshaped(1, 1, 2, 128).transposed(0, 2, 1, 3),
                    MLXRandom.normal([1, 256], dtype: .bfloat16, scale: kvScale, key: k[1])
                        .reshaped(1, 1, 2, 128).transposed(0, 2, 1, 3))
        }
        /// A seeded segment holding `count` keys from `firstSlot` (spare pages; the range's pages a scattered
        /// permutation, never page 0). `ties`: one repeated key row and -0 for the small values.
        func segment(_ count: Int, firstSlot: Int, kvScale: Float, ties: Bool, key: MLXArray) -> CBv2PagedDirectRead {
            let k = MLXRandom.split(key: key, into: 3)
            let pageCount = (firstSlot + count + 15) / 16
            let segmentPages = pageCount + 9
            var storage = MLXRandom.normal([2, segmentPages, 2, 16, 128], dtype: .bfloat16, scale: kvScale, key: k[0])
            if ties {
                let row = MLXRandom.normal([1, 1, 1, 1, 128], dtype: .bfloat16, scale: kvScale, key: k[2])
                let values = storage[1 ..< 2]
                storage = concatenated([
                    broadcast(row, to: [1, segmentPages, 2, 16, 128]),
                    which(abs(values) .< MLXArray(0.5 * kvScale), MLXArray(-Float(0)).asType(.bfloat16), values),
                ], axis: 0)
            }
            let order = argSort(MLXRandom.uniform(Float(0) ..< 1, [segmentPages - 1], key: k[1]))
            return CBv2PagedDirectRead(
                storage: storage, valueOffset: segmentPages * 2 * 16 * 128,
                pages: (order[..<pageCount] + Int32(1)).asType(.int32), firstSlot: firstSlot, count: count,
                kvHeads: 2, headDim: 128, pageSize: 16, dtype: .bfloat16, offset: count,
                writeFence: MLXArray.zeros([1], dtype: .int32))
        }
        func mask(_ count: Int, own: Bool, tail: Int, drop: MLXArray?) -> MLXArray {
            let n = count + (own ? 1 : 0)
            let position = MLXArray(Int32(0) ..< Int32(n))
            var keep = position .< Int32(count - min(tail, count - 1))
            if own { keep = keep .|| (position .== Int32(count)) }
            if let drop {
                let dropped = MLXRandom.uniform(Float(0) ..< 1, [n], key: drop) .< MLXArray(Float(0.2))
                keep = keep .&& (logicalNot(dropped) .|| (position .== Int32(0)))
            }
            return which(keep, MLXArray(Float(0)), MLXArray(-Float.infinity)).asType(.bfloat16).reshaped(1, 1, 1, n)
        }

        // Check cases: count index c, own row off / on (case 2c / 2c + 1); special c % 6 == 1 zero / -0 queries,
        // 3 repeated key row with -0 values, 5 underflow scales.
        let counts = [1, 2, 3, 15, 16, 17, 31, 32, 33, 63, 64, 65, 96, 97, 127, 128, 129, 224, 225, 255, 256, 257,
                      352, 481, 491, 511, 512, 513, 555, 600, 619, 1000, 1021, 1022]
        let inputScales: [(Float, Float)] = [(1, 1), (4, 4), (16, 8)]
        struct Case {
            let queries: MLXArray
            let read: CBv2PagedDirectRead
            let own: Own?
            let mask: MLXArray
            let live: MLXArray?
        }
        var cases: [Case] = []
        for (c, count) in counts.enumerated() {
            for own in [false, true] {
                let i = 2 * c + (own ? 1 : 0)
                let k = MLXRandom.split(key: MLXRandom.key(UInt64(0xD5DD_0000 + i)), into: 5)
                let special = c % 6
                let (qScale, kvScale): (Float, Float) = special == 5 ? (64, 8) : inputScales[(c / 2) % inputScales.count]
                var q = MLXRandom.normal([1, 4096], dtype: .bfloat16, scale: qScale, key: k[0])
                if special == 1 {
                    let odd = (MLXArray(Int32(0) ..< Int32(4096)) % 2) .== MLXArray(Int32(1))
                    q = which(odd, MLXArray(-Float(0)), MLXArray(Float(0))).asType(.bfloat16).reshaped(1, 4096)
                }
                cases.append(Case(
                    queries: q.reshaped(1, 1, 32, 128).transposed(0, 2, 1, 3),
                    read: segment(count, firstSlot: (c * 5 + (own ? 3 : 0)) % 16, kvScale: kvScale, ties: special == 3,
                                  key: k[1]),
                    own: own ? ownRow(kvScale, k[2]) : nil,
                    mask: mask(count, own: own, tail: c % 4, drop: c % 5 == 4 ? k[3] : nil),
                    live: fenced ? MLXArray([Int32(100 + i)]) : nil))
            }
        }
        eval(cases.flatMap { [$0.queries, $0.read.storage, $0.read.pages, $0.mask] + ($0.own.map { [$0.keys, $0.values] } ?? []) })

        // Timing copies: one segment per copy at the decode window's key counts, the first draft's mask (one -inf
        // tail key) and the second's (plus its own row).
        let copies = 128
        let timedCounts = [491, 523, 555, 587, 619]
        let timedMasks = timedCounts.map { (mask($0, own: false, tail: 1, drop: nil), mask($0, own: true, tail: 1, drop: nil)) }
        let timed = (0 ..< copies).map { c -> (read: CBv2PagedDirectRead, own: Own, masks: (MLXArray, MLXArray)) in
            let k = MLXRandom.split(key: MLXRandom.key(UInt64(0xD5DE_0000 + c)), into: 2)
            return (segment(timedCounts[c % timedCounts.count], firstSlot: (c * 7) % 16, kvScale: 1, ties: false,
                            key: k[0]),
                    ownRow(1, k[1]), timedMasks[c % timedCounts.count])
        }
        let timedQueries = MLXRandom.normal([1, 4096], dtype: .bfloat16, scale: 2, key: MLXRandom.key(0xD5DF))
            .reshaped(1, 1, 32, 128).transposed(0, 2, 1, 3)
        let timedLive: MLXArray? = fenced ? MLXArray([Int32(0)]) : nil
        eval(timed.flatMap { [$0.read.storage, $0.read.pages, $0.own.keys, $0.own.values] } + [timedQueries]
            + timedMasks.flatMap { [$0.0, $0.1] })
        let inputsReady = elapsed()
        let links = 64
        var nextCopy = 0
        var chainStart = 0
        let message = run(
            .sdpa, incumbent: Geometry(rps: 1, nsg: 32, unroll: 4), cases: cases.count, links: links,
            reference: { i in
                let c = cases[i]
                return NemotronMTPDraftSDPA.tuneOutputs(nil, queries: c.queries, read: c.read, own: c.own, scale: scale,
                                                        mask: c.mask, live: c.live, dep: c.queries)
            },
            launch: { g, i in
                let c = cases[i]
                return NemotronMTPDraftSDPA.tuneOutputs(g, queries: c.queries, read: c.read, own: c.own, scale: scale,
                                                        mask: c.mask, live: c.live, dep: c.queries)
            },
            link: { g, previous, j in
                // Each chain takes the next links / 2 copies (every copy's reuse ~90 MB of reads apart).
                if j == 0 {
                    chainStart = nextCopy
                    nextCopy = (nextCopy + links / 2) % copies
                }
                let t = timed[(chainStart + j / 2) % copies]
                let own = j % 2 == 1
                return NemotronMTPDraftSDPA.tuneOutputs(
                    g, queries: timedQueries, read: t.read, own: own ? t.own : nil, scale: scale,
                    mask: own ? t.masks.1 : t.masks.0, live: timedLive, dep: previous ?? timedQueries)?[0]
            }, elapsed: elapsed, timed: 200)
        report((fenced ? "fenced" : "plain") + " forms, inputs by \(String(format: "%.2f", inputsReady)) s, " + message)
    }
}

/// `source` through counted swaps (each `old` must occur exactly `count` times), or nil when a statement moved.
func nemotronCountedSwaps(_ source: String, _ swaps: [(String, String, Int)]) -> String? {
    var s = source
    for (old, new, count) in swaps {
        guard s.components(separatedBy: old).count - 1 == count else { return nil }
        s = s.replacingOccurrences(of: old, with: new)
    }
    return s
}

/// `source` with `#pragma clang loop unroll_count(unroll)` ahead of each of exactly `count` occurrences of `loop`
/// (unroll 1: no pragma), or nil when the text moved.
func nemotronUnrollSource(_ source: String, loop: String, count: Int, unroll: Int) -> String? {
    guard source.components(separatedBy: loop).count == count + 1 else { return nil }
    guard unroll > 1 else { return source }
    return source.replacingOccurrences(of: loop, with: "#pragma clang loop unroll_count(\(unroll))\n" + loop)
}

extension NemotronTargetQKV {
    /// `NemotronDraftTune`'s R = 1 dictionary Q/K/V launch of geometry `g`: the dictionary tile prefix with
    /// NSG SIMD groups per tile and the R = 1 dictionary multi-row body (`NemotronDraftTune.qmvBody`); unread
    /// `dep` input last. nil when it cannot run.
    static func tuneLaunch(_ g: NemotronDraftTune.Geometry, weights: [MLXArray], indices: [MLXArray],
                           tables: [MLXArray], x: MLXArray, dep: MLXArray) -> [MLXArray]? {
        let key = "nemotron_draft_tune_qkv_u\(g.unroll)"
        guard weights.count == 3, indices.count == 3, tables.count == 3, x.ndim == 3, x.dim(1) == 1,
              r1HalfDecodeForm != 0, 256 % (g.rps * g.nsg) == 0, g.rps <= 32, g.nsg <= 32,
              let kernel = NemotronDraftTune.cached(key, {
                  guard let prefix = nemotronCountedSwaps(dictTilePrefix, [("(2 * RPS)", "(NSG * RPS)", 2)]),
                        let body = NemotronDraftTune.qmvBody(unroll: g.unroll) else { return nil }
                  return MLXFast.metalKernel(
                      name: key,
                      inputNames: ["qw", "qs", "qb", "kw", "ks", "kb", "vw", "vs", "vb", "x", "K", "dep"],
                      outputNames: ["qy", "ky", "vy"], source: prefix + body + "\n}",
                      header: nemotronMultiRowQMVHeader + nemotronHalfDecodeHeader)
              })
        else { return nil }
        let bound = (0 ..< 3).flatMap { [weights[$0], indices[$0], tables[$0]] }
        return kernel(
            bound + [x, NemotronMultiRowQMV.scalarPair(2688, 4096).0, dep],
            template: [("T", x.dtype), ("R", 1), ("FAST", false), ("RELU2", false), ("RPS", g.rps),
                       ("HF", r1HalfDecodeForm), ("NSG", g.nsg)],
            grid: (32, 4608 / g.rps, 1), threadGroup: (32, g.nsg, 1),
            outputShapes: [[1, 1, 4096], [1, 1, 256], [1, 1, 256]], outputDTypes: [x.dtype, x.dtype, x.dtype])
    }
}

extension NemotronHeadScreen {
    private static let record2PairedDepKernel = record2PairedSource.map {
        MLXFast.metalKernel(
            name: "nemotron_head_screen_partial_record2_paired_dep_affine2",
            inputNames: ["rec", "x", "groups", "groups2", "dep"],
            outputNames: ["scores"], source: $0)
    }
    /// `record2PairedDepKernel` on the 32-bit twin's text (`partialOffset32`).
    private static let record2PairedOffset32DepKernel = record2PairedOffset32Source.map {
        MLXFast.metalKernel(
            name: "nemotron_head_screen_partial_record2_paired_o32_dep_affine2",
            inputNames: ["rec", "x", "groups", "groups2", "dep"],
            outputNames: ["scores"], source: $0)
    }

    /// `run`'s paired 2-bit partial pass at `rps` rows per SIMD group and `nsg` SIMD groups: the production kernel
    /// (`dep` nil) or `NemotronDraftTune`'s (the same source with an unread `dep` input), each on the 32-bit twin's
    /// text where `partialOffset32` applies (`offset32` nil: the load check's verdict). nil when it cannot run.
    static func record2Scores(
        _ hidden: MLXArray, records2: MLXArray, groups: MLXArray, groupCount: Int,
        upperGroups: MLXArray?, upperGroupCount: Int, rps: Int, nsg: Int, dep: MLXArray?, offset32: Bool? = nil
    ) -> MLXArray? {
        let rows = upperGroups == nil ? Self.rows : headRows
        let twin = partialOffset32(fullHead: upperGroups != nil, records2: records2, offset32: offset32)
        guard pairable(groupCount: groupCount, upperGroupCount: upperGroups == nil ? nil : upperGroupCount,
                       rowsPerSimdgroup: rps),
              rps <= 32, nsg <= 32, rows % rps == 0, (rows / rps) % nsg == 0,
              let kernel = twin
                ? (dep == nil ? record2PairedOffset32Kernel : record2PairedOffset32DepKernel)
                : (dep == nil ? record2PairedKernel : record2PairedDepKernel)
        else { return nil }
        let template: [(String, any KernelTemplateArg)] = [
            ("T", hidden.dtype), ("S", groupCount), ("RPS", rps), ("NSG", nsg), ("LO", Self.rows),
            ("S2", upperGroups == nil ? groupCount : upperGroupCount)]
        return kernel(
            [records2, hidden, groups, upperGroups ?? groups] + (dep.map { [$0] } ?? []), template: template,
            grid: (32, rows / rps, 1), threadGroup: (32, nsg, 1),
            outputShapes: [[rows]], outputDTypes: [.float32])[0]
    }
}

extension NemotronHeadPreparedActivation {
    /// The R = 3 eh_proj launch `apply` makes (K 5376, plain scales and biases, crown body, no load form or
    /// geometry override), which `NemotronDraftTune` (ehp) replaces.
    static func draftTuneServes(_ q: QuantizedLinear) -> Bool {
        (!NemotronSBDict.enabled || NemotronSBDict.dict(q) == nil) && decodeForm(rows: 3, fast: false) == nil
            && loadForm(k: 5376, n: 2688) == 0 && preparedGeometry[[5376, 2688]] == nil
            && !NemotronU2.prepared(rows: 3, n: 2688)
    }

    /// `NemotronDraftTune`'s R = 3 plain prepared launch of geometry `g` (the crown body, with the vector-store
    /// closing loop of `nemotronQMVVectorStoreSource` when `g.vs`, through the load-form and NSG swaps, `g.unroll` on
    /// the main K loop; unread `dep` input last). nil when it cannot run.
    static func draftTuneLaunch(_ g: NemotronDraftTune.Geometry, packed: MLXArray, w: MLXArray, scales: MLXArray,
                                biases: MLXArray, dep: MLXArray) -> MLXArray? {
        let k = w.dim(1) * 8, n = w.dim(0)
        let key = "nemotron_draft_tune_prep_l\(g.load)_u\(g.unroll)" + (g.vs ? "_vs" : "")
        let loads: [Int: [(String, String, Int)]] = [
            1: nemotronPreparedW32Swaps, 2: nemotronPreparedHoistSwaps, 3: nemotronPreparedPrefetchSwaps]
        guard packed.dim(0) == 3, g.load == 0 || loads[g.load] != nil, n % g.rps == 0, (n / g.rps) % g.nsg == 0,
              g.rps <= 32, g.nsg <= 32,
              let kernel = NemotronDraftTune.cached(key, {
                  (g.vs ? nemotronQMVVectorStoreSource(nemotronHeadPreparedQMVSource) : nemotronHeadPreparedQMVSource)
                      .flatMap { nemotronCountedSwaps($0, (loads[g.load] ?? []) + nemotronPreparedNSGSwaps) }
                      .flatMap { nemotronUnrollSource($0, loop: NemotronU2.qmvLoop, count: 1, unroll: g.unroll) }
                      .map {
                          MLXFast.metalKernel(
                              name: key, inputNames: ["w", "scales", "biases", "prepared", "K", "N", "dep"],
                              outputNames: ["y"], source: $0, header: variantHeader(half: false, load: g.load))
                      }
              })
        else { return nil }
        let args = NemotronMultiRowQMV.scalarPair(k, n)
        return kernel(
            [w, scales, biases, packed, args.0, args.1, dep],
            template: [("T", DType.bfloat16), ("R", 3), ("FAST", k % 512 == 0), ("RELU2", false),
                       ("RPS", g.rps), ("NSG", g.nsg)],
            grid: (32, n / g.rps, 1), threadGroup: (32, g.nsg, 1),
            outputShapes: [[3, n]], outputDTypes: [.bfloat16])[0]
    }
}
