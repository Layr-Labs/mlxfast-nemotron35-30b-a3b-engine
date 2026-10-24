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
    /// Width one is used by the exact BF16 ReLU2 epilogue, not a GEMM.
    static func apply(_ x: MLXArray, _ q: QuantizedLinear, relu2: Bool = false) -> MLXArray? {
        guard enabled, x.ndim == 2, (1...8).contains(x.dim(0)),
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
            [nemotronWeightAnchor(q.weight), nemotronWeightAnchor(q.scales),
             nemotronWeightAnchor(biases), x, kArray, nArray],
            template: [("T", x.dtype), ("R", rows), ("FAST", fast), ("RELU2", relu2)],
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

// The epilogue must first round to BF16, exactly as the separate projection.
func nemotronMTPLinearRelu2Rows(_ x: MLXArray, _ linear: Linear) -> MLXArray {
    if NemotronMTPExecution.batchedM1, x.ndim == 3, x.dim(0) == 1,
       x.dim(1) >= 1, let q = linear as? QuantizedLinear, q.bias == nil,
       let rows = NemotronMultiRowQMV.apply(x.reshaped(x.dim(1), x.dim(2)), q, relu2: true) {
        return rows.reshaped(1, x.dim(1), q.weight.dim(0))
    }
    let projected = nemotronMTPLinearRows(x, linear)
    let positive = maximum(projected, MLXArray(0))
    return positive * positive
}

// Target Q/K/V share one dispatch; every tile keeps the existing row body.
enum NemotronTargetQKV {
    private static let source = """
      const uint tile = threadgroup_position_in_grid.y;
      const uint local_tile = tile < 512 ? tile : (tile < 544 ? tile - 512 : tile - 544);
      {
        const uint3 threadgroup_position_in_grid = uint3(0, local_tile, 0);
        const device uint32_t* w = tile < 512 ? qw : (tile < 544 ? kw : vw);
        const device T* scales = tile < 512 ? qs : (tile < 544 ? ks : vs);
        const device T* biases = tile < 512 ? qb : (tile < 544 ? kb : vb);
        device T* y = tile < 512 ? qy : (tile < 544 ? ky : vy);
        const int N = tile < 512 ? 4096 : 256;
    """ + nemotronMultiRowQMVSource + "\n}"
    private static let kernel = MLXFast.metalKernel(
        name: "nemotron_target_qkv_affine4_gs64",
        inputNames: ["qw", "qs", "qb", "kw", "ks", "kb", "vw", "vs", "vb", "x", "K"],
        outputNames: ["qy", "ky", "vy"], source: source, header: nemotronMultiRowQMVHeader)

    private static func buffers(_ linear: Linear, rows: Int) -> [MLXArray]? {
        guard type(of: linear) == QuantizedLinear.self, let q = linear as? QuantizedLinear,
              q.mode == .affine, q.bits == 4, q.groupSize == 64, q.bias == nil,
              q.weight.dtype == .uint32, q.weight.shape == [rows, 336],
              q.scales.dtype == .bfloat16, q.scales.shape == [rows, 42],
              let offsets = q.biases, offsets.dtype == .bfloat16, offsets.shape == [rows, 42]
        else { return nil }
        return [q.weight, q.scales, offsets]
    }

    static func apply(
        _ x: MLXArray, query: Linear, key: Linear, value: Linear, captureMTP: Bool
    ) -> (MLXArray, MLXArray, MLXArray)? {
        guard Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              x.ndim == 3, x.dim(0) == 1, x.dim(2) == 2688, x.dtype == .bfloat16,
              x.dim(1) == 1 || (captureMTP && [3, 4].contains(x.dim(1))
                && NemotronMTPExecution.batchedM1 && NemotronMultiRowQMV.enabled),
              let q = buffers(query, rows: 4096), let k = buffers(key, rows: 256),
              let v = buffers(value, rows: 256)
        else { return nil }
        let rows = x.dim(1)
        let outputs = kernel(q + k + v + [x, NemotronMultiRowQMV.scalarPair(2688, 4096).0],
            template: [("T", x.dtype), ("R", rows), ("FAST", false), ("RELU2", false)],
            grid: (32, 1152, 1), threadGroup: (32, 2, 1),
            outputShapes: [[1, rows, 4096], [1, rows, 256], [1, rows, 256]],
            outputDTypes: [x.dtype, x.dtype, x.dtype])
        return (outputs[0], outputs[1], outputs[2])
    }
}
