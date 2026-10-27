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
private let nemotronMambaInProjConvDtEpilogue = """

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

    /// x: [3, 2688] BF16 rows (or their prepared activation), the Mamba
    /// in_proj [z | xBC | dt]. Returns the projection [3, N] plus the window's
    /// convolution [1, 3, convDim], padded history [1, 6, convDim] and FP32
    /// timestep [1, 3, heads], or nil for any other shape.
    static func apply(
        _ x: MLXArray, _ q: QuantizedLinear, prepared: MLXArray?,
        oldState: MLXArray, convWeight: MLXArray, convBias: MLXArray,
        dtBias: MLXArray, timeStepLimit: (Float, Float),
        convStart: Int, convDim: Int, heads: Int, sbDict: Bool = true
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
        let outputs = (dict != nil ? dictKernel! : kernel)(
            [q.weight, dict?.index ?? q.scales, dict?.table ?? biases, packed, args.0, args.1,
             oldState, convWeight, convBias, dtBias,
             MLXArray(timeStepLimit.0), MLXArray(timeStepLimit.1)],
            template: [("T", DType.bfloat16), ("R", rows), ("FAST", false), ("RELU2", false),
                       ("RPS", 2), ("CS", convStart), ("CC", convDim),
                       ("DS", convStart + convDim), ("NH", heads)],
            grid: (32, n / 2, 1), threadGroup: (32, 2, 1),
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
}

enum NemotronHeadPreparedActivation {
    static let prepare = MLXFast.metalKernel(
        name: "nemotron_head_prepare_activation", inputNames: ["x"],
        outputNames: ["prepared"], source: nemotronHeadPrepareSource,
        header: nemotronMultiRowQMVHeader)
    static let project = MLXFast.metalKernel(
        name: "nemotron_head_prepared_qmv", inputNames: ["w", "scales", "biases", "prepared", "K", "N"],
        outputNames: ["y"], source: nemotronHeadPreparedQMVSource,
        header: nemotronMultiRowQMVHeader)
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

    /// Kernel, extra template arguments, RPS and SIMD groups for one prepared
    /// launch of `rows` x [K, N] at the caller's RPS.
    private static func projectLaunch(rows: Int, k: Int, n: Int, rps: Int)
        -> (MLXFast.MLXFastKernel, [(String, any KernelTemplateArg)], Int, Int) {
        let (kernel, hf) = projectKernel(rows: rows, fast: k % 512 == 0)
        guard rows == 3 else { return (kernel, hf, rps, 2) }
        let load = loadForm(k: k, n: n)
        let geometry = preparedGeometry[[k, n]] ?? (rps: rps, nsg: 2)
        guard load != 0 || geometry.nsg != 2 else { return (kernel, hf, geometry.rps, 2) }
        let nsg = geometry.nsg != 2
        let extra: [(String, any KernelTemplateArg)] = nsg ? [("NSG", geometry.nsg)] : []
        return (variantKernel(half: !hf.isEmpty, load: load, nsg: nsg), hf + extra, geometry.rps, geometry.nsg)
    }

    /// projectLaunch's R = 3 launch with its kernel's dictionary twin, or nil.
    private static func dictLaunch(k: Int, n: Int, rps: Int)
        -> (MLXFast.MLXFastKernel, [(String, any KernelTemplateArg)], Int, Int)? {
        let (_, hf) = projectKernel(rows: 3, fast: k % 512 == 0)
        let load = loadForm(k: k, n: n)
        let geometry = preparedGeometry[[k, n]] ?? (rps: rps, nsg: 2)
        let nsg = geometry.nsg != 2
        let extra: [(String, any KernelTemplateArg)] = nsg ? [("NSG", geometry.nsg)] : []
        guard let kernel = dictKernel(half: !hf.isEmpty, load: load, nsg: nsg) else { return nil }
        return (kernel, hf + extra, geometry.rps, geometry.nsg)
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
        if NemotronSBDict.enabled, let d = NemotronSBDict.dict(q),
           case let (kernel, hf, rps, nsg)? = dictLaunch(k: k, n: n, rps: 2) {
            return kernel([q.weight, d.index, d.table, packed[0], args.0, args.1],
                template: [("T", DType.bfloat16), ("R", 3), ("FAST", k % 512 == 0),
                           ("RELU2", false), ("RPS", rps)] + hf,
                grid: (32, n / rps, 1), threadGroup: (32, nsg, 1),
                outputShapes: [[3, n]], outputDTypes: [.bfloat16])[0]
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

    static func apply(_ x: MLXArray, _ q: QuantizedLinear, relu2: Bool, rps: Int,
                      prepared: MLXArray? = nil, sbDict: Bool = true) -> MLXArray {
        let rows = x.dim(0), k = x.dim(1), n = q.weight.dim(0)
        let pv = k % 512 == 0 ? 16 : 8
        let packed = compatible(prepared, rows: rows, k: k)
            ?? prepare([x], template: [("T", x.dtype), ("K", k), ("PV", pv)],
            grid: (k / pv, rows, 1), threadGroup: (32, 1, 1),
            outputShapes: [[rows, (k / pv + 31) / 32, (pv + 1) * 32]], outputDTypes: [.float32])[0]
        let args = NemotronMultiRowQMV.scalarPair(k, n)
        if sbDict, rows == 3, NemotronSBDict.enabled, let d = NemotronSBDict.dict(q),
           case let (kernel, hf, launchRPS, nsg)? = dictLaunch(k: k, n: n, rps: rps) {
            return kernel([q.weight, d.index, d.table, packed, args.0, args.1],
                template: [("T", x.dtype), ("R", rows), ("FAST", k % 512 == 0), ("RELU2", relu2), ("RPS", launchRPS)] + hf,
                grid: (32, n / launchRPS, 1), threadGroup: (32, nsg, 1),
                outputShapes: [[rows, n]], outputDTypes: [x.dtype])[0]
        }
        let (kernel, hf, launchRPS, nsg) = projectLaunch(rows: rows, k: k, n: n, rps: rps)
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
    }
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
                                 dict: NemotronSBDict.Dict? = nil) -> MLXArray {
        let rows = x.dim(0), k = x.dim(1), n = w.dim(0)
        let (kArray, nArray) = scalarPair(k, n)
        let form = NemotronMultiRowQMVKernel.r1HalfDecodeForm
        let half = rows == 1 && form != 0
        let kernels = NemotronMultiRowQMVKernel.shared
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
                      prepared: MLXArray? = nil, sbDict: Bool = true) -> MLXArray? {
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
        let y = dispatch(q.weight, q.scales, biases, x, relu2: relu2, rps: rps, dict: dict)
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
        _ queries: MLXArray, _ keys: MLXArray, _ values: MLXArray, _ scale: Float
    ) -> MLXArray? {
        let rows = queries.ndim == 4 ? queries.dim(2) : 0
        guard enabled, Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              queries.ndim == 4, queries.dim(0) == 1, queries.dim(1) == 32,
              (2...4).contains(rows), queries.dim(3) == 128,
              keys.ndim == 4, keys.dim(0) == 1, keys.dim(1) == 2, keys.dim(3) == 128,
              values.shape == keys.shape, (rows ..< 1024).contains(keys.dim(2)),
              queries.dtype == .bfloat16, keys.dtype == .bfloat16, values.dtype == .bfloat16
        else { return nil }
        return kernel(
            [queries, keys, values, scaleArray(scale)],
            template: [("T", DType.bfloat16), ("D", 128), ("V", 128)],
            grid: (32 * 1024, 1, rows), threadGroup: (1024, 1, 1),
            outputShapes: [[rows, 32, 128]], outputDTypes: [DType.bfloat16])[0]
    }

    /// `merged` plus the R = 3 prepared activation from the same launch.
    private static func mergedPrepared(
        _ queries: MLXArray, _ keys: MLXArray, _ values: MLXArray, _ scale: Float
    ) -> (MLXArray, MLXArray)? {
        guard queries.ndim == 4, queries.dim(2) == 3,
              merged(queries, keys, values, scale) != nil else { return nil }
        let outputs = preparedKernel(
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
        if preparedVerified, let (out, prepared) = mergedPrepared(queries, keys, values, scale) {
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
// separate launches bit for bit at load. NEMOTRON_MTP_ROUTER_UP_COLAUNCH=0 keeps the separate launches (engaged by
// default, as the graded 2e6aaa2c tree ran it; a layer that misses keeps them).
enum NemotronRouterSharedUpColaunch {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_MTP_ROUTER_UP_COLAUNCH"] != "0"
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
    private static let preparedKernel = MLXFast.metalKernel(
        name: "nemotron_target_qkv_prepared_affine4_gs64",
        inputNames: ["qw", "qs", "qb", "kw", "ks", "kb", "vw", "vs", "vb", "prepared", "K"],
        outputNames: ["qy", "ky", "vy"], source: tilePrefix + nemotronHeadPreparedQMVSource + "\n}",
        header: nemotronMultiRowQMVHeader)
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
                  cache[((size_t)(n_out / 128) * (size_t)kvpos[1]
                         + (size_t)(kvpos[0] + r)) * 128 + (n_out % 128)] = h;
                }
        """
    private static let preparedWritingKVKernel: MLXFast.MLXFastKernel = {
        precondition(nemotronHeadPreparedQMVSource.components(separatedBy: storeLine).count == 2,
                     "NemotronTargetQKV: prepared body store not unique")
        return MLXFast.metalKernel(
            name: "nemotron_target_qkv_prepared_kv_affine4_gs64",
            inputNames: ["qw", "qs", "qb", "kw", "ks", "kb", "vw", "vs", "vb", "prepared", "K",
                         "kcache", "vcache", "kvpos"],
            outputNames: ["qy", "ky", "vy"],
            source: tilePrefix
                + nemotronHeadPreparedQMVSource.replacingOccurrences(of: storeLine, with: storeAndCache)
                + "\n}",
            header: nemotronMultiRowQMVHeader, mutableInputs: ["kcache", "vcache"])
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
    /// The paired pass needs an upper group set of at most 8 groups, even
    /// group counts (8-byte code words) and an even RPS.
    static func pairable(groupCount: Int, upperGroupCount: Int?, rowsPerSimdgroup: Int) -> Bool {
        record2PairedKernel != nil && groupCount % 2 == 0 && rowsPerSimdgroup % 2 == 0
            && upperGroupCount.map { $0 % 2 == 0 && 2 * $0 <= 16 } ?? true
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
        rowsPerSimdgroup: Int = 8, simdgroups: Int = 2
    ) -> Result {
        let rows = upperGroups == nil ? Self.rows : headRows
        let template: [(String, any KernelTemplateArg)] = [
            ("T", hidden.dtype), ("S", groupCount), ("RPS", rowsPerSimdgroup),
            ("NSG", simdgroups), ("LO", Self.rows),
            ("S2", upperGroups == nil ? groupCount : upperGroupCount)]
        let partialHalf = partialHalfDecodeForm != 0
        let partialTemplate = template + nemotronHalfTemplate(partialHalf, partialHalfDecodeForm)
        let record2Kernel = paired
            && pairable(groupCount: groupCount, upperGroupCount: upperGroups == nil ? nil : upperGroupCount,
                        rowsPerSimdgroup: rowsPerSimdgroup)
            ? record2PairedKernel ?? record2PartialKernel : record2PartialKernel
        let scores = (records2.map {
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
        if let counter {
            let picked = (selectHalf ? fusedSelectHalfKernel : fusedSelectKernel)(
                [scores, weight, scales, biases, hidden, counter],
                template: [("T", hidden.dtype), ("CH", chunk), ("C", keep), ("NSG", selectGroups),
                           ("NCH", chunks)] + nemotronHalfTemplate(selectHalf, selectHalfDecodeForm),
                grid: (32, selectGroups * chunks, 1), threadGroup: (32, selectGroups, 1),
                outputShapes: [[chunks * keep], [chunks * keep], [chunks], [chunks], [1]],
                outputDTypes: [.int32, .float32, .float32, .int32, .int32])
            return Result(id: picked[4], scores: scores, candidateIds: picked[0], candidateValues: picked[1])
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
