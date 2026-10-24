import Foundation
import MLX
import MLXLMCommon
import MLXNN

extension NemotronHMLP {
    func mtpForwardRows(_ x: MLXArray, prepared: MLXArray? = nil) -> MLXArray {
        guard NemotronMTPExecution.batchedM1 else { return nemotronMTPMapRows(x) { self($0) } }
        let activated = nemotronMTPLinearRelu2Rows(x, upProj, prepared: prepared)
        return nemotronMTPLinearRows(activated, downProj)
    }
}

/// Stateless graph cache only. Request KV, recurrent state, offsets and
/// transactions must never be captured by this helper.
final class NemotronH35MTPGraphCache {
    private struct Entry {
        let identity: [ObjectIdentifier]
        let usesPrepared: Bool
        let function: @Sendable ([MLXArray]) -> [MLXArray]
    }
    private let lock = NSLock()
    private var entries: [Bool: Entry] = [:]
    private var generations = 0
    private var calls = 0
    var generationCount: Int { lock.withLock { generations } }
    var callCount: Int { lock.withLock { calls } }

    func callAsFunction(_ x: MLXArray, prepared: MLXArray? = nil, owner: NemotronHMoE) -> MLXArray {
        // Native parameter wrappers forbid replacing initialized array objects;
        // Module.update changes their contexts in place. Check every concrete
        // child identity cheaply, and collect array handles only on a miss.
        // Unknown subclasses retain the fully reflective fallback.
        let nativeIdentity = Self.nativeIdentity(owner)
        let parameters = nativeIdentity == nil ? owner.innerState() : nil
        let identity = nativeIdentity ?? (owner.modules().map(ObjectIdentifier.init)
            + (parameters ?? []).map(ObjectIdentifier.init))
        let usesPrepared = prepared != nil
        var retired: Entry?
        let current = lock.withLock { () -> Entry in
            calls += 1
            if let entry = entries[usesPrepared], entry.identity == identity { return entry }
            let state = parameters ?? owner.innerState()
            let function: @Sendable ([MLXArray]) -> [MLXArray] = compile(inputs: [state]) { [weak owner] arrays in
                guard let owner else { preconditionFailure("Released Nemotron MoE graph owner") }
                return [owner.mtpForwardRowsUncompiled(arrays[0],
                    prepared: usesPrepared ? arrays[1] : nil)]
            }
            let replacement = Entry(identity: identity, usesPrepared: usesPrepared, function: function)
            retired = entries[usesPrepared]
            entries[usesPrepared] = replacement
            generations += 1
            return replacement
        }
        // Destroy retired compiler entries outside this helper's lock; compiler
        // evaluation owns a separate global/per-function lock ordering.
        let result = withExtendedLifetime(retired) { current.function(prepared.map { [x, $0] } ?? [x]) }
        return result.isEmpty ? MLXArray(0) : result[0]
    }

    private static func nativeIdentity(_ owner: NemotronHMoE) -> [ObjectIdentifier]? {
        func linear(_ value: Linear) -> Bool {
            type(of: value) == Linear.self || type(of: value) == QuantizedLinear.self
        }
        func switched(_ value: SwitchLinear) -> Bool {
            type(of: value) == SwitchLinear.self || type(of: value) == QuantizedSwitchLinear.self
        }
        guard type(of: owner) == NemotronHMoE.self,
              type(of: owner.gate) == NemotronHMoEGate.self,
              type(of: owner.switchMLP) == NemotronHSwitchMLP.self,
              switched(owner.switchMLP.fc1), switched(owner.switchMLP.fc2) else { return nil }
        var modules: [Module] = [owner, owner.gate, owner.switchMLP,
                                 owner.switchMLP.fc1, owner.switchMLP.fc2]
        if let shared = owner.sharedExperts {
            guard type(of: shared) == NemotronHMLP.self,
                  linear(shared.upProj), linear(shared.downProj) else { return nil }
            modules += [shared, shared.upProj, shared.downProj]
        }
        return modules.map(ObjectIdentifier.init)
    }
}

extension NemotronHMoE {
    func mtpForwardRows(_ x: MLXArray, prepared: MLXArray? = nil) -> MLXArray {
        if NemotronMTPExecution.compiledMoE, NemotronMTPExecution.batchedM1,
            x.ndim == 3, x.dim(0) == 1, (1...8).contains(x.dim(1)) {
            return mtpCompiledRowsCache(x, prepared: prepared, owner: self)
        }
        return mtpForwardRowsUncompiled(x, prepared: prepared)
    }

    func mtpForwardRowsUncompiled(_ x: MLXArray, prepared: MLXArray? = nil) -> MLXArray {
        guard NemotronMTPExecution.batchedM1 else { return nemotronMTPMapRows(x) { self($0) } }
        let packed: MLXArray?
        if NemotronHResidualRMSNorm.preparedMoEEnabled, NemotronMoELeader.enabled,
           Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
           NemotronMoELeader.preparedGateUp, x.ndim == 3, x.dim(0) == 1, x.dim(2) == 2688,
           NemotronHeadPreparedActivation.supportsRows(x.dim(1)), x.dtype == .bfloat16 {
            // Share one dynamic operand buffer with both routed and shared up.
            packed = NemotronHeadPreparedActivation.compatible(prepared, rows: x.dim(1), k: 2688)
                ?? NemotronHeadPreparedActivation.prepareContiguousRows(x.reshaped(x.dim(1), 2688), pv: 8)
        } else {
            packed = nil
        }
        // Router matmul retains M=1. Gathered experts already have M=1 per
        // selected expert and can dispatch all assignments in one operation.
        let indices: MLXArray
        let scores: MLXArray
        if NemotronMTPExecution.batchedRouter {
            let routed = gate.mtpForwardRows(x)
            indices = routed.0
            scores = routed.1
        } else {
            let routed = (0..<x.dim(1)).map { gate(x[0..., $0..<($0 + 1), 0...]) }
            indices = concatenated(routed.map { $0.0 }, axis: 1)
            scores = concatenated(routed.map { $0.1 }, axis: 1)
        }
        var y: MLXArray
        if let reduced = NemotronMoELeader.switchRowsAndWeightedReduce(
            x, indices, scores, switchMLP, prepared: packed) {
            y = reduced
        } else {
            y = NemotronMoELeader.switchRows(x, indices, switchMLP) ?? switchMLP(x, indices)
            y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)
        }
        if let sharedExperts { y = y + sharedExperts.mtpForwardRows(x, prepared: packed) }
        return y
    }
}

// MARK: - Exact multi-row MoE verify (expert leader)
//
// Stock verify runs one affine_gather_qmv threadgroup per (row, slot). Here the
// first row that routed to an expert leads: it walks that expert's weights once
// for itself and the next row that routed to it (runs capped at 2; a third
// occurrence leads again). Later occurrences exit before any weight read. The
// leader keeps one accumulator, one K-loop order, one qdot and one closing
// simd_sum per (output, row), copied from qmv_impl (K = 2688 / 1856 are not
// multiples of 512, so stock dispatches gather_qmv, not gather_qmv_fast).
// Only loads are shared, so every row is bitwise equal to stock gather_qmv.
private let nemotronMoELeaderHeader = """
template <typename T, typename U, int values_per_thread>
inline U nml_load_vector(const device T* x, thread U* x_thread) {
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
inline U nml_load_vector_safe(const device T* x, thread U* x_thread, int N) {
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
inline U nml_qdot(
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
inline U nml_qdot_safe(
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

// RELU2 epilogue = stock relu2 on the stored bf16 value, op for op as MLX's
// fused elementwise kernel runs it: Maximum (NaN-propagating x > y ? x : y)
// against bf16(0), then a bf16 * bf16 multiply.
private let nemotronMoELeaderStore = """
template <typename T, bool RELU2>
inline T nml_store(float v) {
  T h = static_cast<T>(v);
  if (!RELU2) {
    return h;
  }
  T zero = static_cast<T>(0);
  T m = metal::isnan(h) ? h : (h > zero ? h : zero);
  return m * m;
}
"""

private let nemotronMoELeaderPreparedHeader = """
template <typename U, int values_per_thread>
inline U nml_prepared_vector(const device float* p, thread U* x_thread) {
  for (int i = 0; i < values_per_thread; i++) {
    x_thread[i] = p[i * 32];
  }
  return p[values_per_thread * 32];
}

template <typename U, int values_per_thread>
inline U nml_prepared_vector_safe(const device float* p, thread U* x_thread, int N) {
  for (int i = 0; i < N; i++) {
    x_thread[i] = p[i * 32];
  }
  for (int i = N; i < values_per_thread; i++) {
    x_thread[i] = 0;
  }
  return p[values_per_thread * 32];
}
"""

private func nmlPreparedSource(_ source: String, pointers: [(String, String)]) -> String? {
    var s = source
    for (from, to) in pointers {
        guard s.contains(from) else { return nil }
        s = s.replacingOccurrences(of: from, with: to)
    }
    for (from, to) in [
        ("nml_load_vector_safe<T, U, values_per_thread>(", "nml_prepared_vector_safe<U, values_per_thread>("),
        ("nml_load_vector<T, U, values_per_thread>(", "nml_prepared_vector<U, values_per_thread>("),
        ("x0 += block_size;", "x0 += PSTRIDE;"),
        ("x1 += block_size;", "x1 += PSTRIDE;"),
    ] {
        s = s.replacingOccurrences(of: from, with: to)
    }
    guard !s.contains("nml_load_vector"), !s.contains("block_size;\n"),
          !s.contains(" x + ") else { return nil }
    return "  constexpr int PSTRIDE = 9 * 32;\n  constexpr int PTILES = (K / 8 + 31) / 32;\n" + s
}

private let nemotronMoELeaderSource = """
  constexpr int results_per_simdgroup = RPS;
  constexpr int values_per_thread = 8;
  constexpr int block_size = values_per_thread * 32;
  constexpr int scale_step_per_thread = 64 / values_per_thread;
  constexpr int in_vec_size_w = K / 2;
  constexpr int in_vec_size_g = K / 64;
  typedef float U;

  const uint3 tid = threadgroup_position_in_grid;
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const uint simd_lid = thread_index_in_simdgroup;

  const int slot = tid.z;
  const int row = slot / TOPK;
  const uint expert = inds[slot];
  int before = 0;
  for (int r = 0; r < row; r++) {
    for (int j = 0; j < TOPK; j++) {
      before += inds[r * TOPK + j] == expert ? 1 : 0;
    }
  }
  if ((before & 1) != 0) {
    return;
  }
  int partner = -1;
  for (int r = R - 1; r > row; r--) {
    for (int j = 0; j < TOPK; j++) {
      if (inds[r * TOPK + j] == expert) {
        partner = r * TOPK + j;
      }
    }
  }

  const int out_row = tid.y * (2 * results_per_simdgroup) + simd_gid * results_per_simdgroup;
  const device uint8_t* ws = (const device uint8_t*)w +
      (ulong)expert * (N * in_vec_size_w) + out_row * in_vec_size_w +
      simd_lid * 4;
  const device T* sc = scales + (ulong)expert * (N * in_vec_size_g) +
      out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  const device T* bs = biases + (ulong)expert * (N * in_vec_size_g) +
      out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
  const device T* x0 = x + (XSLOT ? slot : row) * K + simd_lid * values_per_thread;
  device T* y0 = y + slot * N + out_row;

  if (partner < 0) {
    thread U x_thread[values_per_thread];
    thread U result[results_per_simdgroup] = {0};
    int k = 0;
    for (; k < K - block_size; k += block_size) {
      U sum = nml_load_vector<T, U, values_per_thread>(x0, x_thread);
      for (int r = 0; r < results_per_simdgroup; r++) {
        auto wl = (const device uint8_t*)(ws + r * in_vec_size_w);
        const device T* sl = sc + r * in_vec_size_g;
        const device T* bl = bs + r * in_vec_size_g;
        U s = sl[0];
        U b = bl[0];
        result[r] += nml_qdot<U, values_per_thread>(wl, x_thread, s, b, sum);
      }
      ws += block_size / 2;
      sc += block_size / 64;
      bs += block_size / 64;
      x0 += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(K - k - simd_lid * values_per_thread), 0, values_per_thread);
    if (remaining > 0) {
      U sum = nml_load_vector_safe<T, U, values_per_thread>(x0, x_thread, remaining);
      for (int r = 0; r < results_per_simdgroup; r++) {
        auto wl = (const device uint8_t*)(ws + r * in_vec_size_w);
        const device T* sl = sc + r * in_vec_size_g;
        const device T* bl = bs + r * in_vec_size_g;
        U s = sl[0];
        U b = bl[0];
        result[r] += nml_qdot_safe<U, values_per_thread>(
            wl, x_thread, s, b, sum, remaining);
      }
    }
    for (int r = 0; r < results_per_simdgroup; r++) {
      result[r] = simd_sum(result[r]);
      if (simd_lid == 0) {
        y0[r] = nml_store<T, RELU2>(result[r]);
      }
    }
  } else {
    const device T* x1 = x + (XSLOT ? partner : partner / TOPK) * K +
        simd_lid * values_per_thread;
    device T* y1 = y + partner * N + out_row;
    thread U x0_thread[values_per_thread];
    thread U x1_thread[values_per_thread];
    thread U result0[results_per_simdgroup] = {0};
    thread U result1[results_per_simdgroup] = {0};
    int k = 0;
    for (; k < K - block_size; k += block_size) {
      U sum0 = nml_load_vector<T, U, values_per_thread>(x0, x0_thread);
      U sum1 = nml_load_vector<T, U, values_per_thread>(x1, x1_thread);
      for (int r = 0; r < results_per_simdgroup; r++) {
        auto wl = (const device uint8_t*)(ws + r * in_vec_size_w);
        const device T* sl = sc + r * in_vec_size_g;
        const device T* bl = bs + r * in_vec_size_g;
        U s = sl[0];
        U b = bl[0];
        result0[r] += nml_qdot<U, values_per_thread>(wl, x0_thread, s, b, sum0);
        result1[r] += nml_qdot<U, values_per_thread>(wl, x1_thread, s, b, sum1);
      }
      ws += block_size / 2;
      sc += block_size / 64;
      bs += block_size / 64;
      x0 += block_size;
      x1 += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(K - k - simd_lid * values_per_thread), 0, values_per_thread);
    if (remaining > 0) {
      U sum0 = nml_load_vector_safe<T, U, values_per_thread>(x0, x0_thread, remaining);
      U sum1 = nml_load_vector_safe<T, U, values_per_thread>(x1, x1_thread, remaining);
      for (int r = 0; r < results_per_simdgroup; r++) {
        auto wl = (const device uint8_t*)(ws + r * in_vec_size_w);
        const device T* sl = sc + r * in_vec_size_g;
        const device T* bl = bs + r * in_vec_size_g;
        U s = sl[0];
        U b = bl[0];
        result0[r] += nml_qdot_safe<U, values_per_thread>(
            wl, x0_thread, s, b, sum0, remaining);
        result1[r] += nml_qdot_safe<U, values_per_thread>(
            wl, x1_thread, s, b, sum1, remaining);
      }
    }
    for (int r = 0; r < results_per_simdgroup; r++) {
      result0[r] = simd_sum(result0[r]);
      result1[r] = simd_sum(result1[r]);
      if (simd_lid == 0) {
        y0[r] = nml_store<T, RELU2>(result0[r]);
        y1[r] = nml_store<T, RELU2>(result1[r]);
      }
    }
  }
"""

private final class NemotronMoELeaderKernel: Sendable {
    static let shared = NemotronMoELeaderKernel()
    let kernel: MLXFast.MLXFastKernel
    private init() {
        kernel = MLXFast.metalKernel(
            name: "nemotron_moe_leader_qmv_affine4_gs64",
            inputNames: ["w", "scales", "biases", "x", "inds"],
            outputNames: ["y"],
            source: nemotronMoELeaderSource,
            header: nemotronMoELeaderHeader + nemotronMoELeaderStore)
    }
}

private final class NemotronMoELeaderPreparedKernels: Sendable {
    static let shared = NemotronMoELeaderPreparedKernels()
    let gateUp: MLXFast.MLXFastKernel?
    private init() {
        let header = nemotronMoELeaderHeader + nemotronMoELeaderStore + nemotronMoELeaderPreparedHeader
        gateUp = nmlPreparedSource(nemotronMoELeaderSource, pointers: [
            ("const device T* x0 = x + (XSLOT ? slot : row) * K + simd_lid * values_per_thread;",
             "const device float* x0 = prepared + row * (PTILES * PSTRIDE) + simd_lid;"),
            ("const device T* x1 = x + (XSLOT ? partner : partner / TOPK) * K +\n        simd_lid * values_per_thread;",
             "const device float* x1 = prepared + (partner / TOPK) * (PTILES * PSTRIDE) + simd_lid;"),
        ]).map {
            MLXFast.metalKernel(
                name: "nemotron_moe_leader_qmv_affine4_gs64_prepared",
                inputNames: ["w", "scales", "biases", "prepared", "inds"],
                outputNames: ["y"], source: $0, header: header)
        }
    }
}

// Keep the accepted leader/partner QMV trace, then reduce each token's six
// products after a shared-memory barrier. Each row has six distinct router
// indices. Experts can repeat across rows, as in the accepted leader path.
private let nemotronMoELeaderWeightedSource = """
  constexpr int results_per_simdgroup = 4;
  constexpr int values_per_thread = 8;
  constexpr int block_size = values_per_thread * 32;
  constexpr int scale_step_per_thread = 64 / values_per_thread;
  constexpr int in_vec_size_w = K / 2;
  constexpr int in_vec_size_g = K / 64;
  typedef float U;

  const uint3 tid = threadgroup_position_in_grid;
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const uint simd_lid = thread_index_in_simdgroup;
  const int out_row = tid.x * results_per_simdgroup;
  threadgroup float products[R * TOPK * results_per_simdgroup];

  for (int slot = simd_gid; slot < R * TOPK; slot += TOPK) {
    const int row = slot / TOPK;
    const uint expert = inds[slot];
    int before = 0;
    for (int r = 0; r < row; r++) {
      for (int j = 0; j < TOPK; j++) {
        before += inds[r * TOPK + j] == expert ? 1 : 0;
      }
    }
    if ((before & 1) != 0) {
      continue;
    }
    int partner = -1;
    for (int r = R - 1; r > row; r--) {
      for (int j = 0; j < TOPK; j++) {
        if (inds[r * TOPK + j] == expert) {
          partner = r * TOPK + j;
        }
      }
    }

    const device uint8_t* ws = (const device uint8_t*)w +
        (ulong)expert * (N * in_vec_size_w) + out_row * in_vec_size_w +
        simd_lid * 4;
    const device T* sc = scales + (ulong)expert * (N * in_vec_size_g) +
        out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    const device T* bs = biases + (ulong)expert * (N * in_vec_size_g) +
        out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    const device T* x0 = x + slot * K + simd_lid * values_per_thread;

    if (partner < 0) {
      thread U x_thread[values_per_thread];
      thread U result[results_per_simdgroup] = {0};
      int k = 0;
      for (; k < K - block_size; k += block_size) {
        U sum = nml_load_vector<T, U, values_per_thread>(x0, x_thread);
        for (int r = 0; r < results_per_simdgroup; r++) {
          auto wl = (const device uint8_t*)(ws + r * in_vec_size_w);
          const device T* sl = sc + r * in_vec_size_g;
          const device T* bl = bs + r * in_vec_size_g;
          U s = sl[0];
          U b = bl[0];
          result[r] += nml_qdot<U, values_per_thread>(wl, x_thread, s, b, sum);
        }
        ws += block_size / 2;
        sc += block_size / 64;
        bs += block_size / 64;
        x0 += block_size;
      }
      const int remaining = clamp(
          static_cast<int>(K - k - simd_lid * values_per_thread), 0, values_per_thread);
      if (remaining > 0) {
        U sum = nml_load_vector_safe<T, U, values_per_thread>(x0, x_thread, remaining);
        for (int r = 0; r < results_per_simdgroup; r++) {
          auto wl = (const device uint8_t*)(ws + r * in_vec_size_w);
          const device T* sl = sc + r * in_vec_size_g;
          const device T* bl = bs + r * in_vec_size_g;
          U s = sl[0];
          U b = bl[0];
          result[r] += nml_qdot_safe<U, values_per_thread>(
              wl, x_thread, s, b, sum, remaining);
        }
      }
      for (int r = 0; r < results_per_simdgroup; r++) {
        result[r] = simd_sum(result[r]);
        if (simd_lid == 0) {
          const T h = nml_store<T, false>(result[r]);
          products[slot * results_per_simdgroup + r] = float(h) * scores[slot] + 0.0f;
        }
      }
    } else {
      const device T* x1 = x + partner * K + simd_lid * values_per_thread;
      thread U x0_thread[values_per_thread];
      thread U x1_thread[values_per_thread];
      thread U result0[results_per_simdgroup] = {0};
      thread U result1[results_per_simdgroup] = {0};
      int k = 0;
      for (; k < K - block_size; k += block_size) {
        U sum0 = nml_load_vector<T, U, values_per_thread>(x0, x0_thread);
        U sum1 = nml_load_vector<T, U, values_per_thread>(x1, x1_thread);
        for (int r = 0; r < results_per_simdgroup; r++) {
          auto wl = (const device uint8_t*)(ws + r * in_vec_size_w);
          const device T* sl = sc + r * in_vec_size_g;
          const device T* bl = bs + r * in_vec_size_g;
          U s = sl[0];
          U b = bl[0];
          result0[r] += nml_qdot<U, values_per_thread>(wl, x0_thread, s, b, sum0);
          result1[r] += nml_qdot<U, values_per_thread>(wl, x1_thread, s, b, sum1);
        }
        ws += block_size / 2;
        sc += block_size / 64;
        bs += block_size / 64;
        x0 += block_size;
        x1 += block_size;
      }
      const int remaining = clamp(
          static_cast<int>(K - k - simd_lid * values_per_thread), 0, values_per_thread);
      if (remaining > 0) {
        U sum0 = nml_load_vector_safe<T, U, values_per_thread>(x0, x0_thread, remaining);
        U sum1 = nml_load_vector_safe<T, U, values_per_thread>(x1, x1_thread, remaining);
        for (int r = 0; r < results_per_simdgroup; r++) {
          auto wl = (const device uint8_t*)(ws + r * in_vec_size_w);
          const device T* sl = sc + r * in_vec_size_g;
          const device T* bl = bs + r * in_vec_size_g;
          U s = sl[0];
          U b = bl[0];
          result0[r] += nml_qdot_safe<U, values_per_thread>(
              wl, x0_thread, s, b, sum0, remaining);
          result1[r] += nml_qdot_safe<U, values_per_thread>(
              wl, x1_thread, s, b, sum1, remaining);
        }
      }
      for (int r = 0; r < results_per_simdgroup; r++) {
        result0[r] = simd_sum(result0[r]);
        result1[r] = simd_sum(result1[r]);
        if (simd_lid == 0) {
          const T h0 = nml_store<T, false>(result0[r]);
          const T h1 = nml_store<T, false>(result1[r]);
          products[slot * results_per_simdgroup + r] = float(h0) * scores[slot] + 0.0f;
          products[partner * results_per_simdgroup + r] = float(h1) * scores[partner] + 0.0f;
        }
      }
    }
  }

  // Every original slot is written by its leader or by its one partner.
  // All SIMD groups reach this barrier, including groups with no leader.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid < R && simd_lid == 0) {
    for (int r = 0; r < results_per_simdgroup; r++) {
      const int base = simd_gid * TOPK * results_per_simdgroup + r;
      float total = products[base];
      for (int j = 1; j < TOPK; j++) {
        total = products[base + j * results_per_simdgroup] + total;
      }
      y[simd_gid * N + out_row + r] = static_cast<T>(total);
    }
  }
"""

private final class NemotronMoELeaderWeightedKernel: Sendable {
    static let shared = NemotronMoELeaderWeightedKernel()
    let kernel: MLXFast.MLXFastKernel
    private init() {
        kernel = MLXFast.metalKernel(
            name: "nemotron_moe_leader_weighted_qmv_affine4_gs64",
            inputNames: ["w", "scales", "biases", "x", "inds", "scores"],
            outputNames: ["y"],
            source: nemotronMoELeaderWeightedSource,
            header: nemotronMoELeaderHeader + nemotronMoELeaderStore,
            ensureRowContiguous: true)
    }
}

/// Set to true only for a self-test build: every verify call also runs stock
/// gather_qmv and poisons the layer output with NaN on any bit difference, so
/// the benchd token gate fails loudly unless every row is bitwise stock.
let nemotronKernelSelfTestBuild = false

enum NemotronMoELeader {
    static let enabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_MOE_LEADER"] != "0"
    static let fuseRelu2 =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_MOE_FUSE_RELU2"] != "0"
    /// Exact winning98/E03 geometry; disable to retain four outputs per SIMD.
    static let narrowLeader =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_MOE_NARROW"] != "0"
    static func leaderResultsPerSimdgroup(rows: Int) -> Int {
        narrowLeader && rows == 3 ? 2 : 4
    }
    static let selfTest = nemotronKernelSelfTestBuild
        || ProcessInfo.processInfo.environment["NEMOTRON_KERNEL_SELFTEST"] == "1"

    struct Bank {
        let weight: MLXArray
        let scales: MLXArray
        let biases: MLXArray
        let k: Int
        let n: Int
    }

    static func bank(_ layer: SwitchLinear, k: Int, dtype: DType) -> Bank? {
        guard let q = layer as? QuantizedSwitchLinear, q.mode == .affine,
              q.bits == 4, q.groupSize == 64 else { return nil }
        var weight: MLXArray?, scales: MLXArray?, biases: MLXArray?
        for (key, value) in q.parameters().flattened() {
            switch key {
            case "weight": weight = value
            case "scales": scales = value
            case "biases": biases = value
            case "bias": return nil
            default: break
            }
        }
        guard let weight, let scales, let biases, weight.ndim == 3,
              weight.dtype == .uint32, weight.dim(2) * 8 == k,
              scales.dtype == dtype, biases.dtype == dtype,
              k % 64 == 0, k % 512 != 0, k > 256, weight.dim(1) % 8 == 0
        else { return nil }
        return Bank(weight: weight, scales: scales, biases: biases, k: k, n: weight.dim(1))
    }

    /// x: [R, K] (perSlot false) or [R * topK, K]; inds: [R * topK] uint32.
    /// Returns [R * topK, N].
    static func apply(_ x: MLXArray, _ inds: MLXArray, rows: Int, topK: Int,
                      bank: Bank, perSlot: Bool, relu2: Bool = false,
                      prepared: MLXArray? = nil) -> MLXArray {
        let rps = leaderResultsPerSimdgroup(rows: rows)
        if preparedGateUp, NemotronHeadPreparedActivation.supportsRows(rows),
           !perSlot, bank.k % 64 == 0, bank.k % 512 != 0,
           let kernel = NemotronMoELeaderPreparedKernels.shared.gateUp {
            let packed = NemotronHeadPreparedActivation.compatible(prepared, rows: rows, k: bank.k)
                ?? NemotronHeadPreparedActivation.prepareContiguousRows(x, pv: 8)
            return kernel(
                [bank.weight, bank.scales, bank.biases, packed, inds],
                template: [("T", x.dtype), ("R", rows), ("TOPK", topK),
                           ("K", bank.k), ("N", bank.n), ("XSLOT", perSlot),
                           ("RELU2", relu2), ("RPS", rps)],
                grid: (32, bank.n / rps, rows * topK),
                threadGroup: (32, 2, 1),
                outputShapes: [[rows * topK, bank.n]],
                outputDTypes: [x.dtype])[0]
        }
        return NemotronMoELeaderKernel.shared.kernel(
            [bank.weight, bank.scales, bank.biases, x, inds],
            template: [("T", x.dtype), ("R", rows), ("TOPK", topK),
                       ("K", bank.k), ("N", bank.n), ("XSLOT", perSlot),
                       ("RELU2", relu2), ("RPS", rps)],
            grid: (32, bank.n / rps, rows * topK),
            threadGroup: (32, 2, 1),
            outputShapes: [[rows * topK, bank.n]],
            outputDTypes: [x.dtype])[0]
    }

    static let preparedGateUp = true
    /// Same leader pairs as apply(), with separate FP32 products and the
    /// native top-six sum. Caller supplies six distinct indices per row.
    static func applyWeighted(_ x: MLXArray, _ inds: MLXArray, _ scores: MLXArray,
                              rows: Int, bank: Bank) -> MLXArray {
        if rows == 3 { return NMLBalanceWeighted.shared.run(x, inds, scores, bank) }
        return NemotronMoELeaderWeightedKernel.shared.kernel(
            [bank.weight, bank.scales, bank.biases, x, inds, scores],
            template: [("T", x.dtype), ("R", rows), ("TOPK", 6),
                       ("K", bank.k), ("N", bank.n)],
            grid: (32 * (bank.n / 4), 6, 1),
            threadGroup: (32, 6, 1),
            outputShapes: [[1, rows, bank.n]],
            outputDTypes: [x.dtype])[0]
    }

    /// The router produces six distinct indices per token. Keep its original
    /// slot order and the accepted cross-row leader pairs. Row-contiguous
    /// kernel copies support arbitrary input strides without weight changes.
    static func switchRowsAndWeightedReduce(
        _ x: MLXArray, _ indices: MLXArray, _ scores: MLXArray,
        _ mlp: NemotronHSwitchMLP, prepared: MLXArray? = nil
    ) -> MLXArray? {
        guard enabled, type(of: mlp) == NemotronHSwitchMLP.self,
              type(of: mlp.fc1) == QuantizedSwitchLinear.self,
              type(of: mlp.fc2) == QuantizedSwitchLinear.self,
              Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              x.ndim == 3, x.dim(0) == 1, (1...4).contains(x.dim(1)),
              x.dim(2) == 2688, x.dtype == .bfloat16,
              indices.dtype == .uint32, indices.shape == [1, x.dim(1), 6],
              scores.dtype == .float32, scores.shape == indices.shape,
              let up = bank(mlp.fc1, k: 2688, dtype: .bfloat16),
              let down = bank(mlp.fc2, k: 1856, dtype: .bfloat16),
              up.weight.shape == [128, 1856, 336],
              up.scales.shape == [128, 1856, 42], up.biases.shape == up.scales.shape,
              down.weight.shape == [128, 2688, 232],
              down.scales.shape == [128, 2688, 29], down.biases.shape == down.scales.shape
        else { return nil }
        let rows = x.dim(1)
        let flat = indices.reshaped(rows * 6)
        let xr = x.reshaped(rows, 2688)
        let act: MLXArray
        if fuseRelu2 {
            act = apply(xr, flat, rows: rows, topK: 6, bank: up, perSlot: false,
                relu2: true, prepared: prepared)
        } else {
            let h = apply(xr, flat, rows: rows, topK: 6, bank: up, perSlot: false,
                prepared: prepared)
            let a = maximum(h, MLXArray(0))
            act = a * a
        }
        var y = applyWeighted(act, flat, scores.reshaped(rows * 6), rows: rows, bank: down)
        if selfTest {
            let original = switchRows(x, indices, mlp) ?? mlp(x, indices)
            let expected = (original * scores[.ellipsis, .newAxis])
                .sum(axis: -2).asType(original.dtype)
            let ok = (y.view(dtype: .uint16) .== expected.view(dtype: .uint16)).all()
            y = which(ok, y, MLXArray(Float.nan).asType(y.dtype))
        }
        return y
    }

    /// Stock-equivalent `switchMLP(x, indices)` for x [1, R, H], indices
    /// [1, R, topK] (unsorted route, R * topK <= 64). Returns [1, R, topK, H].
    static func switchRows(_ x: MLXArray, _ indices: MLXArray,
                           _ mlp: NemotronHSwitchMLP) -> MLXArray? {
        guard enabled, x.ndim == 3, x.dim(0) == 1, (2...4).contains(x.dim(1)),
              x.dtype == .bfloat16, indices.ndim == 3, indices.dim(0) == 1,
              indices.dim(1) == x.dim(1), indices.dtype == .uint32,
              indices.size <= 64,
              let up = bank(mlp.fc1, k: x.dim(2), dtype: x.dtype),
              let down = bank(mlp.fc2, k: up.n, dtype: x.dtype), down.n == x.dim(2)
        else { return nil }
        let rows = x.dim(1), topK = indices.dim(2)
        let flat = indices.reshaped(rows * topK)
        let xr = x.reshaped(rows, x.dim(2))
        let act: MLXArray
        if fuseRelu2 {
            act = apply(xr, flat, rows: rows, topK: topK, bank: up, perSlot: false, relu2: true)
        } else {
            let h = apply(xr, flat, rows: rows, topK: topK, bank: up, perSlot: false)
            let a = maximum(h, MLXArray(0))
            act = a * a
        }
        var y = apply(act, flat, rows: rows, topK: topK, bank: down, perSlot: true)
        if selfTest {
            // Stock route on the same inputs: gather_qmv fc1, relu2 ops, gather_qmv fc2.
            let xe = MLX.expandedDimensions(x, axes: [-2, -3])
            let h0 = mlp.fc1(xe, indices, sortedIndices: false)
            let a0 = maximum(h0, MLXArray(0))
            let act0 = (a0 * a0).reshaped(rows * topK, up.n)
            let y0 = mlp.fc2(act.reshaped(1, rows, topK, 1, up.n), indices, sortedIndices: false)
                .reshaped(rows * topK, down.n)
            let ok = MLX.logicalAnd(
                (act.view(dtype: .uint16) .== act0.view(dtype: .uint16)).all(),
                (y.view(dtype: .uint16) .== y0.view(dtype: .uint16)).all())
            y = which(ok, y, MLXArray(Float.nan).asType(y.dtype))
        }
        return y.reshaped(1, rows, topK, down.n)
    }

    /// Compile every pipeline the verify window can request (R = 1...4) at load.
    static func warm(_ model: Module) {
        guard enabled else { return }
        var outputs: [MLXArray] = []
        var seen = Set<[Int]>()
        for module in model.modules() {
            guard let moe = module as? NemotronHMoE else { continue }
            let mlp = moe.switchMLP
            guard let up = bank(mlp.fc1, k: mlp.inputDims, dtype: .bfloat16),
                  let down = bank(mlp.fc2, k: up.n, dtype: .bfloat16) else { continue }
            let topK = moe.numExpertsPerTok
            for rows in 1...4 {
                guard seen.insert([rows, topK, up.k, up.n]).inserted else { continue }
                let values: [UInt32] = (0..<(rows * topK)).map { UInt32($0 % topK) }
                let inds = MLXArray(values)
                let x = MLXArray.zeros([rows, up.k], dtype: .bfloat16)
                let h = apply(x, inds, rows: rows, topK: topK, bank: up, perSlot: false,
                              relu2: fuseRelu2)
                if rows > 1 {
                    outputs.append(apply(h, inds, rows: rows, topK: topK, bank: down, perSlot: true))
                }
                if topK == 6, up.weight.shape == [128, 1856, 336],
                    down.weight.shape == [128, 2688, 232] {
                    let scores = MLXArray.ones([rows * topK], dtype: .float32)
                    outputs.append(applyWeighted(h, inds, scores, rows: rows, bank: down))
                }
            }
        }
        if !outputs.isEmpty { eval(outputs) }
    }
}

private let nmlBalancePlanSource = """
  const uint lane = thread_position_in_grid.x;
  threadgroup uint ids[18];
  threadgroup int lead[18];
  threadgroup int leaders[18];
  threadgroup int costs[18];
  threadgroup int tplan[36];
  if (lane < 18) { ids[lane] = inds[lane]; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane < 18) {
    const int slot = int(lane);
    const int row = slot / 6;
    const uint expert = ids[slot];
    int before = 0;
    for (int r = 0; r < row; r++) {
      for (int j = 0; j < 6; j++) { before += ids[r*6+j] == expert ? 1 : 0; }
    }
    int partner = -1;
    for (int r = 2; r > row; r--) {
      for (int j = 0; j < 6; j++) {
        if (ids[r*6+j] == expert) { partner = r*6+j; }
      }
    }
    // -2 marks a slot whose earlier occurrence leads it.
    lead[slot] = (before & 1) != 0 ? -2 : partner;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) {
    int counts[6] = {0}, loads[6] = {0}, original[6] = {0};
    int number = 0;
    for (int i = 0; i < 36; i++) { tplan[i] = -1; }
    for (int slot = 0; slot < 18; slot++) {
      const int partner = lead[slot];
      if (partner == -2) { continue; }
      leaders[number] = slot;
      costs[number] = partner < 0 ? SOLO_COST : PAIR_COST;
      original[slot % 6] += costs[number];
      tplan[18+slot] = partner;
      number++;
    }
    // Stable largest-cost-first order; integer load, then group id break ties.
    for (int assigned = 0; assigned < number; assigned++) {
      int best = -1;
      for (int j = 0; j < number; j++) {
        if (costs[j] >= 0 && (best < 0 || costs[j] > costs[best])) { best = j; }
      }
      int group = -1;
      for (int g = 0; g < 6; g++) {
        if (counts[g] < 3 && (group < 0 || loads[g] < loads[group])) { group = g; }
      }
      tplan[group*3+counts[group]] = leaders[best];
      counts[group]++; loads[group] += costs[best]; costs[best] = -1;
    }
    int oldMax = 0, newMax = 0;
    for (int g = 0; g < 6; g++) { oldMax = max(oldMax,original[g]); newMax = max(newMax,loads[g]); }
    if (oldMax <= newMax) {
      for (int i = 0; i < 18; i++) { tplan[i] = -1; }
      for (int g = 0; g < 6; g++) { counts[g] = 0; }
      for (int j = 0; j < number; j++) {
        const int slot = leaders[j], g = slot % 6;
        tplan[g*3+counts[g]++] = slot;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = lane; i < 36; i += 32) { plan[i] = tplan[i]; }
"""
private let nmlBalanceWeightedSource = nemotronMoELeaderWeightedSource.replacingOccurrences(
    of: """
  for (int slot = simd_gid; slot < R * TOPK; slot += TOPK) {
    const int row = slot / TOPK;
    const uint expert = inds[slot];
    int before = 0;
    for (int r = 0; r < row; r++) {
      for (int j = 0; j < TOPK; j++) {
        before += inds[r * TOPK + j] == expert ? 1 : 0;
      }
    }
    if ((before & 1) != 0) {
      continue;
    }
    int partner = -1;
    for (int r = R - 1; r > row; r--) {
      for (int j = 0; j < TOPK; j++) {
        if (inds[r * TOPK + j] == expert) {
          partner = r * TOPK + j;
        }
      }
    }

""", with: """
  for (int job = 0; job < 3; job++) {
    const int slot = plan[simd_gid * 3 + job];
    if (slot < 0) { continue; }
    const uint expert = inds[slot];
    const int partner = plan[18 + slot];

""")
private final class NMLBalancePlan: Sendable {
    static let shared = NMLBalancePlan()
    let kernel = MLXFast.metalKernel(name: "nml_fc2_current_call_plan",
        inputNames: ["inds"], outputNames: ["plan"], source: nmlBalancePlanSource)
    func make(_ ids: MLXArray, pairCost: Int = 1500) -> MLXArray {
        kernel([ids], template: [("PAIR_COST",pairCost), ("SOLO_COST",1000)],
            grid: (32,1,1), threadGroup: (32,1,1),
            outputShapes: [[36]], outputDTypes: [.int32])[0]
    }
}
private final class NMLBalanceWeighted: Sendable {
    static let shared = NMLBalanceWeighted()
    let kernel = MLXFast.metalKernel(name: "nml_fc2_original_jobs_balanced",
        inputNames: ["w","scales","biases","x","inds","scores","plan"],
        outputNames: ["y"], source: nmlBalanceWeightedSource,
        header: nemotronMoELeaderHeader + nemotronMoELeaderStore,
        ensureRowContiguous: true)
    func run(_ x: MLXArray, _ ids: MLXArray, _ scores: MLXArray,
             _ bank: NemotronMoELeader.Bank) -> MLXArray {
        let plan = NMLBalancePlan.shared.make(ids)
        return kernel([bank.weight,bank.scales,bank.biases,x,ids,scores,plan],
            template: [("T",x.dtype),("R",3),("TOPK",6),("K",bank.k),("N",bank.n)],
            grid: (32*(bank.n/4),6,1), threadGroup: (32,6,1),
            outputShapes: [[1,3,bank.n]], outputDTypes: [x.dtype])[0]
    }
}
