import Foundation
import MLX
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

func nemotronMTPHeadRows(_ x: MLXArray, _ linear: Linear) -> MLXArray {
    guard NemotronMTPExecution.exactMultirowHead, x.ndim == 3,
          x.dim(0) == 1, (2...8).contains(x.dim(1)), x.dim(2) == 2688,
          let q = linear as? QuantizedLinear, q.weight.dim(0) == 131072,
          q.groupSize == 64, q.bits == 4, q.mode == .affine,
          q.scales.dtype == x.dtype, q.biases?.dtype == x.dtype
    else { return nemotronMTPLinearRows(x, linear) }
    if let rows = NemotronMultiRowQMV.apply(x.reshaped(x.dim(1), x.dim(2)), q) {
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

private let nemotronMultiRowQMVSource = """
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
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
      for (int r = 0; r < R; r++) {
        result[r][row] += nmr_qdot<U, values_per_thread>(wl, x_thread[r], s, b, sum[r]);
      }
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
        for (int r = 0; r < R; r++) {
          result[r][row] += nmr_qdot_safe<U, values_per_thread>(
              wl, x_thread[r], s, b, sum[r], remaining);
        }
      }
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    for (int r = 0; r < R; r++) {
      U v = simd_sum(result[r][row]);
      if (simd_lid == 0) {
        yp[r * out_vec_size + row] = static_cast<T>(v);
      }
    }
  }
"""

private final class NemotronMultiRowQMVKernel: Sendable {
    static let shared = NemotronMultiRowQMVKernel()
    let kernel: MLXFast.MLXFastKernel
    private init() {
        kernel = MLXFast.metalKernel(
            name: "nemotron_multirow_qmv_affine4_gs64",
            inputNames: ["w", "scales", "biases", "x", "K", "N"],
            outputNames: ["y"],
            source: nemotronMultiRowQMVSource,
            header: nemotronMultiRowQMVHeader)
    }
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

    /// x: [rows, K] contiguous. Returns [rows, N] or nil when the shape is not
    /// one the stock M=1 dispatch would send to qmv / qmv_fast.
    static func apply(_ x: MLXArray, _ q: QuantizedLinear) -> MLXArray? {
        guard enabled, x.ndim == 2, (2...8).contains(x.dim(0)),
              q.mode == .affine, q.bits == 4, q.groupSize == 64,
              let biases = q.biases, x.dtype == .bfloat16,
              q.scales.dtype == x.dtype, biases.dtype == x.dtype,
              q.weight.ndim == 2, q.weight.dtype == .uint32
        else { return nil }
        let rows = x.dim(0), k = x.dim(1), n = q.weight.dim(0)
        // Stock dispatch_qmv sends K == 64/128 to qmv_quad; skip those.
        guard k % 64 == 0, k > 128, n % 8 == 0, q.weight.dim(1) * 8 == k else { return nil }
        let fast = k % 512 == 0
        let (kArray, nArray) = scalarPair(k, n)
        return NemotronMultiRowQMVKernel.shared.kernel(
            [q.weight, q.scales, biases, x, kArray, nArray],
            template: [("T", x.dtype), ("R", rows), ("FAST", fast)],
            grid: (32, n / 4, 1),
            threadGroup: (32, 2, 1),
            outputShapes: [[rows, n]],
            outputDTypes: [x.dtype])[0]
    }
}

/// Put verification tokens on the native *batch* axis, keeping matrix M=1.
/// A singleton leading weight axis prevents MLX's flatten-to-M optimization
/// and therefore avoids qmv_wide's different accumulation. The underlying
/// matrix/scales/offset storage is shared; no weight bank is duplicated.
func nemotronMTPLinearRows(_ x: MLXArray, _ linear: Linear) -> MLXArray {
    guard NemotronMTPExecution.batchedM1, x.ndim == 3, x.dim(0) == 1, x.dim(1) > 1,
        let q = linear as? QuantizedLinear, q.mode == .affine,
        q.scales.dtype == x.dtype, let biases = q.biases, biases.dtype == x.dtype
    else { return nemotronMTPMapRows(x) { linear($0) } }
    if let rows = NemotronMultiRowQMV.apply(x.reshaped(x.dim(1), x.dim(2)), q) {
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


private let nemotronQKVMultiRowSource = """
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int packs_per_thread = FAST ? 2 : 1;
  constexpr int pack_factor = 8;
  constexpr int bytes_per_pack = 4;
  constexpr int group_size = 64;
  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * 32;
  constexpr int scale_step_per_thread = group_size / values_per_thread;
  typedef float U;

  const int in_vec_size = K;
  const int out_vec_total = NQ + NK + NV;
  const uint3 tid = threadgroup_position_in_grid;
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const uint simd_lid = thread_index_in_simdgroup;

  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  if (out_row >= out_vec_total) {
    return;
  }

  // Block-aligned matrix select: every 8 out rows (two simdgroup blocks)
  // stay inside one matrix because NQ and NK are multiples of 8.
  const int is_k = out_row >= NQ;
  const int is_v = out_row >= NQ + NK;
  const int msel = is_v ? 2 : (is_k ? 1 : 0);
  const int local_row = is_v ? out_row - NQ - NK : (is_k ? out_row - NQ : out_row);
  const int out_vec_size = is_v ? NV : (is_k ? NK : NQ);
  const device uint8_t* wsel =
      msel == 0 ? (const device uint8_t*)wq
                : (msel == 1 ? (const device uint8_t*)wk
                             : (const device uint8_t*)wv);
  const device T* ssel = msel == 0 ? sq : (msel == 1 ? sk : sv);
  const device T* bsel = msel == 0 ? bq : (msel == 1 ? bk : bv);
  device T* yp = msel == 0 ? yq : (msel == 1 ? yk : yv);

  thread U x_thread[R][values_per_thread];
  thread U result[R][results_per_simdgroup];
  for (int r = 0; r < R; r++) {
    for (int row = 0; row < results_per_simdgroup; row++) {
      result[r][row] = 0;
    }
  }

  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;

  const device uint8_t* ws =
      wsel + local_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
  const device T* sc = ssel + local_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  const device T* bs = bsel + local_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  const device T* xp = x + simd_lid * values_per_thread;

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
      for (int r = 0; r < R; r++) {
        result[r][row] += nmr_qdot<U, values_per_thread>(wl, x_thread[r], s, b, sum[r]);
      }
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
        for (int r = 0; r < R; r++) {
          result[r][row] += nmr_qdot_safe<U, values_per_thread>(
              wl, x_thread[r], s, b, sum[r], remaining);
        }
      }
    }
  }

  for (int row = 0; row < results_per_simdgroup; row++) {
    for (int r = 0; r < R; r++) {
      U v = simd_sum(result[r][row]);
      if (simd_lid == 0) {
        yp[r * out_vec_size + local_row + row] = static_cast<T>(v);
      }
    }
  }
"""

private final class NemotronQKVMultiRowKernel: Sendable {
    static let shared = NemotronQKVMultiRowKernel()
    let kernel: MLXFast.MLXFastKernel
    private init() {
        kernel = MLXFast.metalKernel(
            name: "nemotron_qkv_multirow_qmv_affine4_gs64",
            inputNames: ["wq", "sq", "bq", "wk", "sk", "bk", "wv", "sv", "bv",
                         "x", "K", "NQ", "NK", "NV"],
            outputNames: ["yq", "yk", "yv"],
            source: nemotronQKVMultiRowSource,
            header: nemotronMultiRowQMVHeader)
    }
}

enum NemotronQKVMultiRowQMV {
    nonisolated(unsafe) private static var scalars: [Int: (MLXArray, MLXArray, MLXArray, MLXArray)] = [:]
    private static let scalarsLock = NSLock()
    static let enabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_QKV_MULTIROW_QMV"] != "0"

    static func apply(_ x: MLXArray, _ wq: QuantizedLinear, _ wk: QuantizedLinear,
                      _ wv: QuantizedLinear) -> (MLXArray, MLXArray, MLXArray)? {
        guard enabled, x.ndim == 2, (2...8).contains(x.dim(0)),
              wq.mode == .affine, wk.mode == .affine, wv.mode == .affine,
              wq.bits == 4, wk.bits == 4, wv.bits == 4,
              wq.groupSize == 64, wk.groupSize == 64, wv.groupSize == 64,
              wq.bias == nil, wk.bias == nil, wv.bias == nil,
              let bq = wq.biases, let bk = wk.biases, let bv = wv.biases,
              x.dtype == .bfloat16,
              wq.scales.dtype == x.dtype, wk.scales.dtype == x.dtype,
              wv.scales.dtype == x.dtype, bq.dtype == x.dtype,
              bk.dtype == x.dtype, bv.dtype == x.dtype,
              wq.weight.ndim == 2, wk.weight.ndim == 2, wv.weight.ndim == 2,
              wq.weight.dtype == .uint32, wk.weight.dtype == .uint32,
              wv.weight.dtype == .uint32
        else { return nil }
        let rows = x.dim(0), k = x.dim(1)
        let nq = wq.weight.dim(0), nk = wk.weight.dim(0), nv = wv.weight.dim(0)
        guard k % 64 == 0, k > 128, nq % 8 == 0, nk % 8 == 0, nv % 8 == 0,
              wq.weight.dim(1) * 8 == k, wk.weight.dim(1) * 8 == k,
              wv.weight.dim(1) * 8 == k
        else { return nil }
        let fast = k % 512 == 0
        scalarsLock.lock()
        let key = (k &* 1_000_003 &+ nq) &* 1_000_003 &+ nk &* 1_000_003 &+ nv
        let sc4 = scalars[key] ?? {
            let quad = (MLXArray(Int32(k)), MLXArray(Int32(nq)),
                        MLXArray(Int32(nk)), MLXArray(Int32(nv)))
            scalars[key] = quad
            return quad
        }()
        scalarsLock.unlock()
        let outputs = NemotronQKVMultiRowKernel.shared.kernel(
            [wq.weight, wq.scales, bq, wk.weight, wk.scales, bk,
             wv.weight, wv.scales, bv, x, sc4.0, sc4.1, sc4.2, sc4.3],
            template: [("T", x.dtype), ("R", rows), ("FAST", fast)],
            grid: (32, (nq + nk + nv) / 4, 1),
            threadGroup: (32, 2, 1),
            outputShapes: [[rows, nq], [rows, nk], [rows, nv]],
            outputDTypes: [x.dtype, x.dtype, x.dtype])
        return (outputs[0], outputs[1], outputs[2])
    }
}

func nemotronMTPQKVRows(_ x: MLXArray, _ wq: Linear, _ wk: Linear, _ wv: Linear)
    -> (MLXArray, MLXArray, MLXArray)?
{
    guard x.ndim == 3, x.dim(0) == 1, (2...8).contains(x.dim(1)),
          let qq = wq as? QuantizedLinear, let qk = wk as? QuantizedLinear,
          let qv = wv as? QuantizedLinear,
          let (q, k, v) = NemotronQKVMultiRowQMV.apply(
              x.reshaped(x.dim(1), x.dim(2)), qq, qk, qv)
    else { return nil }
    return (q.reshaped(1, x.dim(1), qq.weight.dim(0)),
            k.reshaped(1, x.dim(1), qk.weight.dim(0)),
            v.reshaped(1, x.dim(1), qv.weight.dim(0)))
}
