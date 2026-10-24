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
    /// NEMOTRON_MTP_QMV_NARROW=0 keeps four outputs per SIMD group everywhere.
    static let narrowEnabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_QMV_NARROW"] != "0"

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
                                 _ x: MLXArray, relu2: Bool, rps: Int) -> MLXArray {
        let rows = x.dim(0), k = x.dim(1), n = w.dim(0)
        let (kArray, nArray) = scalarPair(k, n)
        return NemotronMultiRowQMVKernel.shared.kernel(
            [w, scales, biases, x, kArray, nArray],
            template: [("T", x.dtype), ("R", rows), ("FAST", k % 512 == 0),
                       ("RELU2", relu2), ("RPS", rps)],
            grid: (32, n / rps, 1),
            threadGroup: (32, 2, 1),
            outputShapes: [[rows, n]],
            outputDTypes: [x.dtype])[0]
    }

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
        let k = x.dim(1), n = q.weight.dim(0)
        // Stock dispatch_qmv sends K == 64/128 to qmv_quad; skip those.
        guard k % 64 == 0, k > 128, n % 8 == 0, q.weight.dim(1) * 8 == k else { return nil }
        let rps = resultsPerSimdgroup(rows: x.dim(0), k: k, n: n)
        let y = dispatch(q.weight, q.scales, biases, x, relu2: relu2, rps: rps)
        guard rps != 4, NemotronMoELeader.selfTest else { return y }
        return nemotronBitwiseOrNaN(
            y, dispatch(q.weight, q.scales, biases, x, relu2: relu2, rps: 4))
    }

    /// JIT every narrow pipeline (R = 3, 4; FAST; RELU2) at load, so no
    /// timed window compiles one. Tiny dummy shapes; same template keys.
    static func warmNarrow() {
        guard enabled, narrowEnabled else { return }
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
        }
        outputs += NemotronTargetQKV.warmNarrow()
        eval(outputs)
    }
}

/// Self-test select: `y` where it is bitwise equal to `expected`, else NaN
/// everywhere, so the token gate fails loudly on any difference.
func nemotronBitwiseOrNaN(_ y: MLXArray, _ expected: MLXArray) -> MLXArray {
    let ok = (y.view(dtype: .uint16) .== expected.view(dtype: .uint16)).all()
    return which(ok, y, MLXArray(Float.nan).asType(y.dtype))
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
  constexpr int TM = 2;
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
            grid: (32 * (128 / 2), 8, 1),
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
    // only weights verified here at load take the fast path. Filled once by
    // warm(), before any compile trace, from the frozen loaded weights.
    nonisolated(unsafe) private static var rowMajor = Set<ObjectIdentifier>()

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
// A tile is one threadgroup of 2 * RPS output rows: Q tiles, then K, then V.
enum NemotronTargetQKV {
    private static let source = """
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
    """ + nemotronMultiRowQMVSource + "\n}"
    private static let kernel = MLXFast.metalKernel(
        name: "nemotron_target_qkv_affine4_gs64",
        inputNames: ["qw", "qs", "qb", "kw", "ks", "kb", "vw", "vs", "vb", "x", "K"],
        outputNames: ["qy", "ky", "vy"], source: source, header: nemotronMultiRowQMVHeader)

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

    private static func dispatch(_ weights: [MLXArray], _ x: MLXArray, rps: Int) -> [MLXArray] {
        let rows = x.dim(1)
        return kernel(weights + [x, NemotronMultiRowQMV.scalarPair(2688, 4096).0],
            template: [("T", x.dtype), ("R", rows), ("FAST", false), ("RELU2", false),
                       ("RPS", rps)],
            grid: (32, 4608 / rps, 1), threadGroup: (32, 2, 1),
            outputShapes: [[1, rows, 4096], [1, rows, 256], [1, rows, 256]],
            outputDTypes: [x.dtype, x.dtype, x.dtype])
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
        let rps = resultsPerSimdgroup(rows: x.dim(1))
        var outputs = dispatch(q + k + v, x, rps: rps)
        if rps != 4, NemotronMoELeader.selfTest {
            let expected = dispatch(q + k + v, x, rps: 4)
            outputs = zip(outputs, expected).map { nemotronBitwiseOrNaN($0, $1) }
        }
        return (outputs[0], outputs[1], outputs[2])
    }

    /// Narrow pipelines for the widths that use them (R = 3, 4).
    static func warmNarrow() -> [MLXArray] {
        let weights = [4096, 256, 256].flatMap { n in
            [MLXArray.zeros([n, 336], dtype: .uint32),
             MLXArray.zeros([n, 42], dtype: .bfloat16),
             MLXArray.zeros([n, 42], dtype: .bfloat16)]
        }
        return [3, 4].flatMap { rows in
            dispatch(weights, MLXArray.zeros([1, rows, 2688], dtype: .bfloat16), rps: 2)
        }
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
    static let width = 2688
    static let wordsPerRow = 336
    static let groupsPerRow = 42
    static let headRows = 131072
    static let rows = 65536

    // One simdgroup scores RPS rows. Lane l owns half-groups l, l + 32, ...
    // (32 values = one uint4 of packed words) of the selected groups.
    private static let partialSource = """
      // S <= 16: lane l owns half-group l (32 values, 16 bytes).
      constexpr int KW = 336;
      constexpr int KG = 42;
      constexpr int VPL = 32;
      constexpr int H = 2 * S;
      const int lane = int(thread_index_in_simdgroup);
      const int sg = int(threadgroup_position_in_grid.y) * NSG + int(simdgroup_index_in_threadgroup);
      const int row0 = sg * RPS;
      const bool on = lane < H;
      const int g = on ? int(groups[lane >> 1]) : 0;
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
    private static let selectSource = """
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
        inputNames: ["w", "scales", "biases", "x", "groups"],
        outputNames: ["scores"], source: partialSource)
    private static let selectKernel = MLXFast.metalKernel(
        name: "nemotron_head_screen_select_affine4_gs64",
        inputNames: ["scores", "w", "scales", "biases", "x"],
        outputNames: ["cand_ids", "cand_vals", "best_val", "best_id"],
        source: selectSource, header: nemotronMultiRowQMVHeader)
    private static let finalKernel = MLXFast.metalKernel(
        name: "nemotron_head_screen_final",
        inputNames: ["best_val", "best_id"], outputNames: ["out"], source: finalSource)

    struct Result {
        let id: MLXArray
        let scores: MLXArray
        let candidateIds: MLXArray
        let candidateValues: MLXArray
    }

    /// hidden [1, 1, 2688] bf16; the head arrays are the full shared head.
    static func run(
        _ hidden: MLXArray, weight: MLXArray, scales: MLXArray, biases: MLXArray,
        groups: MLXArray, groupCount: Int, chunk: Int, keep: Int,
        rowsPerSimdgroup: Int = 8, simdgroups: Int = 2
    ) -> Result {
        let scores = partialKernel(
            [weight, scales, biases, hidden, groups],
            template: [("T", hidden.dtype), ("S", groupCount), ("RPS", rowsPerSimdgroup),
                       ("NSG", simdgroups)],
            grid: (32, rows / rowsPerSimdgroup, 1), threadGroup: (32, simdgroups, 1),
            outputShapes: [[rows]], outputDTypes: [.float32])[0]
        let chunks = rows / chunk
        let selectGroups = 8
        let picked = selectKernel(
            [scores, weight, scales, biases, hidden],
            template: [("T", hidden.dtype), ("CH", chunk), ("C", keep), ("NSG", selectGroups)],
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
