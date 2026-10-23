import Foundation
import MLX
import MLXLMCommon
import MLXNN

extension NemotronHMLP {
    func mtpForwardRows(_ x: MLXArray) -> MLXArray {
        guard NemotronMTPExecution.batchedM1 else { return nemotronMTPMapRows(x) { self($0) } }
        if NemotronMoELeader.enabled, NemotronMoELeader.fuseRelu2,
           type(of: self) == NemotronHMLP.self,
           type(of: upProj) == QuantizedLinear.self,
           Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
           x.shape == [1, 1, 2688], x.dtype == .bfloat16,
           let q = upProj as? QuantizedLinear,
           q.mode == .affine, q.bits == 4, q.groupSize == 64, q.bias == nil,
           q.weight.dtype == .uint32, q.weight.shape == [3712, 336],
           q.scales.dtype == .bfloat16, q.scales.shape == [3712, 42],
           let offsets = q.biases, offsets.dtype == .bfloat16, offsets.shape == [3712, 42] {
            // A singleton expert axis shares the original packed buffers.
            // Keep the native QMV BF16 cast before the ReLU-squared epilogue.
            let bank = NemotronMoELeader.Bank(
                weight: q.weight.expandedDimensions(axis: 0),
                scales: q.scales.expandedDimensions(axis: 0),
                biases: offsets.expandedDimensions(axis: 0), k: 2688, n: 3712)
            let activated = NemotronMoELeader.apply(
                x.reshaped(1, 2688), MLXArray([UInt32(0)]), rows: 1, topK: 1,
                bank: bank, perSlot: false, relu2: true).reshaped(1, 1, 3712)
            return nemotronMTPLinearRows(activated, downProj)
        }
        let up = nemotronMTPLinearRows(x, upProj)
        let activated = maximum(up, MLXArray(0))
        return nemotronMTPLinearRows(activated * activated, downProj)
    }
}

/// Stateless graph cache only. Request KV, recurrent state, offsets and
/// transactions must never be captured by this helper.
final class NemotronH35MTPGraphCache {
    private struct Entry {
        let identity: [ObjectIdentifier]
        let function: @Sendable (MLXArray) -> MLXArray
    }
    private let lock = NSLock()
    private var entry: Entry?
    private var generations = 0
    private var calls = 0
    var generationCount: Int { lock.withLock { generations } }
    var callCount: Int { lock.withLock { calls } }

    func callAsFunction(_ x: MLXArray, owner: NemotronHMoE) -> MLXArray {
        // Native parameter wrappers forbid replacing initialized array objects;
        // Module.update changes their contexts in place. Check every concrete
        // child identity cheaply, and collect array handles only on a miss.
        // Unknown subclasses retain the fully reflective fallback.
        let nativeIdentity = Self.nativeIdentity(owner)
        let parameters = nativeIdentity == nil ? owner.innerState() : nil
        let identity = nativeIdentity ?? (owner.modules().map(ObjectIdentifier.init)
            + (parameters ?? []).map(ObjectIdentifier.init))
        var retired: Entry?
        let current = lock.withLock { () -> Entry in
            calls += 1
            if let entry, entry.identity == identity { return entry }
            let state = parameters ?? owner.innerState()
            let function: @Sendable (MLXArray) -> MLXArray = compile(inputs: [state]) { [weak owner] x in
                guard let owner else { preconditionFailure("Released Nemotron MoE graph owner") }
                return owner.mtpForwardRowsUncompiled(x)
            }
            let replacement = Entry(identity: identity, function: function)
            retired = entry
            entry = replacement
            generations += 1
            return replacement
        }
        // Destroy retired compiler entries outside this helper's lock; compiler
        // evaluation owns a separate global/per-function lock ordering.
        return withExtendedLifetime(retired) { current.function(x) }
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
    func mtpForwardRows(_ x: MLXArray) -> MLXArray {
        if NemotronMTPExecution.compiledMoE, NemotronMTPExecution.batchedM1,
            x.ndim == 3, x.dim(0) == 1, (1...8).contains(x.dim(1)) {
            return mtpCompiledRowsCache(x, owner: self)
        }
        return mtpForwardRowsUncompiled(x)
    }

    func mtpForwardRowsUncompiled(_ x: MLXArray) -> MLXArray {
        guard NemotronMTPExecution.batchedM1 else { return nemotronMTPMapRows(x) { self($0) } }
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
            x, indices, scores, switchMLP) {
            y = reduced
        } else {
            y = NemotronMoELeader.switchRows(x, indices, switchMLP) ?? switchMLP(x, indices)
            y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)
        }
        if let sharedExperts { y = y + sharedExperts.mtpForwardRows(x) }
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

private let nemotronMoELeaderSource = """
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

  const int out_row = tid.y * 8 + simd_gid * results_per_simdgroup;
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

// R3/R4 experiment: the first occurrence owns every matching row. Each row
// retains its own native eight-value load and four-result QMV trace.
private let nemotronMoESharedRowsHeader = """
template <int R, int TOPK>
inline bool nmsr_members(const device uint32_t* inds, int slot, thread int* members) {
  const uint expert = inds[slot];
  const int row = slot / TOPK;
  #pragma unroll
  for (int m = 0; m < R; m++) {
    members[m] = -1;
    for (int j = 0; j < TOPK; j++) {
      if (inds[m * TOPK + j] == expert) {
        members[m] = m * TOPK + j;
      }
    }
    if (m < row && members[m] >= 0) {
      return false;
    }
  }
  return true;
}

inline float nmsr_qdot(
    uint32_t packed, const thread float* x_thread, float scale, float bias, float sum) {
  const uint16_t lo = static_cast<uint16_t>(packed & 0xffff);
  const uint16_t hi = static_cast<uint16_t>(packed >> 16);
  float accum = 0;
  accum +=
      (x_thread[0] * (lo & 0x000f) +
       x_thread[1] * (lo & 0x00f0) +
       x_thread[2] * (lo & 0x0f00) +
       x_thread[3] * (lo & 0xf000));
  accum +=
      (x_thread[4] * (hi & 0x000f) +
       x_thread[5] * (hi & 0x00f0) +
       x_thread[6] * (hi & 0x0f00) +
       x_thread[7] * (hi & 0xf000));
  return scale * accum + sum * bias;
}

template <typename T, int R, int K, int N, bool XSLOT>
inline void nmsr_qmv(
    const device uint32_t* w, const device T* scales, const device T* biases,
    const device T* x, uint expert, const thread int* members,
    int out_row, uint simd_lid, thread float* result) {
  constexpr int values_per_thread = 8;
  constexpr int block_size = 256;
  constexpr int in_vec_size_w = K / 2;
  constexpr int in_vec_size_g = K / 64;
  const device uint8_t* ws = (const device uint8_t*)w +
      (ulong)expert * (N * in_vec_size_w) + out_row * in_vec_size_w + simd_lid * 4;
  const device T* sc = scales + (ulong)expert * (N * in_vec_size_g) +
      out_row * in_vec_size_g + simd_lid / 8;
  const device T* bs = biases + (ulong)expert * (N * in_vec_size_g) +
      out_row * in_vec_size_g + simd_lid / 8;
  thread float x_thread[R * values_per_thread];
  thread float sums[R];
  #pragma unroll
  for (int i = 0; i < R * 4; i++) {
    result[i] = 0;
  }
  int k = 0;
  for (; k < K - block_size; k += block_size) {
    #pragma unroll
    for (int m = 0; m < R; m++) {
      if (members[m] >= 0) {
        const device T* xm = x + (XSLOT ? members[m] : m) * K +
            k + simd_lid * values_per_thread;
        sums[m] = nml_load_vector<T, float, values_per_thread>(
            xm, x_thread + m * values_per_thread);
      }
    }
    for (int r = 0; r < 4; r++) {
      const device T* sl = sc + r * in_vec_size_g;
      const device T* bl = bs + r * in_vec_size_g;
      float s = sl[0];
      float b = bl[0];
      const uint32_t packed = ((const device uint32_t*)(ws + r * in_vec_size_w))[0];
      #pragma unroll
      for (int m = 0; m < R; m++) {
        if (members[m] >= 0) {
          result[m * 4 + r] += nmsr_qdot(
              packed, x_thread + m * values_per_thread, s, b, sums[m]);
        }
      }
    }
    ws += block_size / 2;
    sc += block_size / 64;
    bs += block_size / 64;
  }
  const int remaining = clamp(
      static_cast<int>(K - k - simd_lid * values_per_thread), 0, values_per_thread);
  if (remaining > 0) {
    #pragma unroll
    for (int m = 0; m < R; m++) {
      if (members[m] >= 0) {
        const device T* xm = x + (XSLOT ? members[m] : m) * K +
            k + simd_lid * values_per_thread;
        sums[m] = nml_load_vector_safe<T, float, values_per_thread>(
            xm, x_thread + m * values_per_thread, remaining);
      }
    }
    // Native K2688/K1856 gives remaining8 or0, so this u32 read is complete.
    for (int r = 0; r < 4; r++) {
      const device T* sl = sc + r * in_vec_size_g;
      const device T* bl = bs + r * in_vec_size_g;
      float s = sl[0];
      float b = bl[0];
      const uint32_t packed = ((const device uint32_t*)(ws + r * in_vec_size_w))[0];
      #pragma unroll
      for (int m = 0; m < R; m++) {
        if (members[m] >= 0) {
          result[m * 4 + r] += nmsr_qdot(
              packed, x_thread + m * values_per_thread, s, b, sums[m]);
        }
      }
    }
  }
}
"""

private final class NemotronMoESharedRowsKernels: Sendable {
    static let shared = NemotronMoESharedRowsKernels()
    let qmv: MLXFast.MLXFastKernel
    let weighted: MLXFast.MLXFastKernel
    private init() {
        let header = nemotronMoELeaderHeader + nemotronMoELeaderStore
            + nemotronMoESharedRowsHeader
        qmv = MLXFast.metalKernel(
            name: "nemotron_moe_shared_rows_qmv_affine4_gs64",
            inputNames: ["w", "scales", "biases", "x", "inds"], outputNames: ["y"],
            source: """
                const uint3 tid = threadgroup_position_in_grid;
                const uint simd_gid = simdgroup_index_in_threadgroup;
                const uint simd_lid = thread_index_in_simdgroup;
                const int slot = tid.z;
                thread int members[R];
                if (!nmsr_members<R, TOPK>(inds, slot, members)) {
                    return;
                }
                const int out_row = tid.y * 8 + simd_gid * 4;
                thread float result[R * 4];
                nmsr_qmv<T, R, K, N, XSLOT>(
                    w, scales, biases, x, inds[slot], members, out_row, simd_lid, result);
                for (int r = 0; r < 4; r++) {
                    #pragma unroll
                    for (int m = 0; m < R; m++) {
                        if (members[m] >= 0) {
                            result[m * 4 + r] = simd_sum(result[m * 4 + r]);
                            if (simd_lid == 0) {
                                y[members[m] * N + out_row + r] =
                                    nml_store<T, RELU2>(result[m * 4 + r]);
                            }
                        }
                    }
                }
                """,
            header: header, ensureRowContiguous: true)
        weighted = MLXFast.metalKernel(
            name: "nemotron_moe_shared_rows_weighted_qmv_affine4_gs64",
            inputNames: ["w", "scales", "biases", "x", "inds", "scores"], outputNames: ["y"],
            source: """
                const uint3 tid = threadgroup_position_in_grid;
                const uint simd_gid = simdgroup_index_in_threadgroup;
                const uint simd_lid = thread_index_in_simdgroup;
                const int out_row = tid.x * 4;
                threadgroup float products[R * TOPK * 4];
                for (int slot = simd_gid; slot < R * TOPK; slot += TOPK) {
                    thread int members[R];
                    if (!nmsr_members<R, TOPK>(inds, slot, members)) {
                        continue;
                    }
                    thread float result[R * 4];
                    nmsr_qmv<T, R, K, N, true>(
                        w, scales, biases, x, inds[slot], members, out_row, simd_lid, result);
                    for (int r = 0; r < 4; r++) {
                        #pragma unroll
                        for (int m = 0; m < R; m++) {
                            if (members[m] >= 0) {
                                result[m * 4 + r] = simd_sum(result[m * 4 + r]);
                                if (simd_lid == 0) {
                                    const T h = nml_store<T, false>(result[m * 4 + r]);
                                    const int member = members[m];
                                    products[member * 4 + r] = float(h) * scores[member] + 0.0f;
                                }
                            }
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_gid < R && simd_lid == 0) {
                    for (int r = 0; r < 4; r++) {
                        const int base = simd_gid * TOPK * 4 + r;
                        float total = products[base];
                        for (int j = 1; j < TOPK; j++) {
                            total = products[base + j * 4] + total;
                        }
                        y[simd_gid * N + out_row + r] = static_cast<T>(total);
                    }
                }
                """,
            header: header, ensureRowContiguous: true)
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
                      bank: Bank, perSlot: Bool, relu2: Bool = false) -> MLXArray {
        NemotronMoELeaderKernel.shared.kernel(
            [bank.weight, bank.scales, bank.biases, x, inds],
            template: [("T", x.dtype), ("R", rows), ("TOPK", topK),
                       ("K", bank.k), ("N", bank.n), ("XSLOT", perSlot),
                       ("RELU2", relu2)],
            grid: (32, bank.n / 4, rows * topK),
            threadGroup: (32, 2, 1),
            outputShapes: [[rows * topK, bank.n]],
            outputDTypes: [x.dtype])[0]
    }

    /// Same leader pairs as apply(), with separate FP32 products and the
    /// native top-six sum. Caller supplies six distinct indices per row.
    static func applyWeighted(_ x: MLXArray, _ inds: MLXArray, _ scores: MLXArray,
                              rows: Int, bank: Bank) -> MLXArray {
        NemotronMoELeaderWeightedKernel.shared.kernel(
            [bank.weight, bank.scales, bank.biases, x, inds, scores],
            template: [("T", x.dtype), ("R", rows), ("TOPK", 6),
                       ("K", bank.k), ("N", bank.n)],
            grid: (32 * (bank.n / 4), 6, 1),
            threadGroup: (32, 6, 1),
            outputShapes: [[1, rows, bank.n]],
            outputDTypes: [x.dtype])[0]
    }

    static func applySharedRows(_ x: MLXArray, _ inds: MLXArray, rows: Int, topK: Int,
                                bank: Bank, perSlot: Bool, relu2: Bool = false) -> MLXArray {
        NemotronMoESharedRowsKernels.shared.qmv(
            [bank.weight, bank.scales, bank.biases, x, inds],
            template: [("T", x.dtype), ("R", rows), ("TOPK", topK),
                       ("K", bank.k), ("N", bank.n), ("XSLOT", perSlot), ("RELU2", relu2)],
            grid: (32, bank.n / 4, rows * topK), threadGroup: (32, 2, 1),
            outputShapes: [[rows * topK, bank.n]], outputDTypes: [x.dtype])[0]
    }

    static func applyWeightedSharedRows(
        _ x: MLXArray, _ inds: MLXArray, _ scores: MLXArray, rows: Int, bank: Bank
    ) -> MLXArray {
        NemotronMoESharedRowsKernels.shared.weighted(
            [bank.weight, bank.scales, bank.biases, x, inds, scores],
            template: [("T", x.dtype), ("R", rows), ("TOPK", 6), ("K", bank.k), ("N", bank.n)],
            grid: (32 * (bank.n / 4), 6, 1), threadGroup: (32, 6, 1),
            outputShapes: [[1, rows, bank.n]], outputDTypes: [x.dtype])[0]
    }

    /// The router produces six distinct indices per token. Keep its original
    /// slot order and per-row arithmetic. Row-contiguous
    /// kernel copies support arbitrary input strides without weight changes.
    static func switchRowsAndWeightedReduce(
        _ x: MLXArray, _ indices: MLXArray, _ scores: MLXArray,
        _ mlp: NemotronHSwitchMLP
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
        if rows > 2 && fuseRelu2 {
            act = applySharedRows(xr, flat, rows: rows, topK: 6, bank: up, perSlot: false, relu2: true)
        } else if rows > 2 {
            let h = applySharedRows(xr, flat, rows: rows, topK: 6, bank: up, perSlot: false)
            let a = maximum(h, MLXArray(0))
            act = a * a
        } else if fuseRelu2 {
            act = apply(xr, flat, rows: rows, topK: 6, bank: up, perSlot: false, relu2: true)
        } else {
            let h = apply(xr, flat, rows: rows, topK: 6, bank: up, perSlot: false)
            let a = maximum(h, MLXArray(0))
            act = a * a
        }
        var y = rows > 2
            ? applyWeightedSharedRows(act, flat, scores.reshaped(rows * 6), rows: rows, bank: down)
            : applyWeighted(act, flat, scores.reshaped(rows * 6), rows: rows, bank: down)
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
                    if rows > 2 {
                        let sharedH = applySharedRows(x, inds, rows: rows, topK: topK,
                            bank: up, perSlot: false, relu2: fuseRelu2)
                        let sharedAct: MLXArray
                        if fuseRelu2 {
                            sharedAct = sharedH
                        } else {
                            let a = maximum(sharedH, MLXArray(0))
                            sharedAct = a * a
                        }
                        outputs.append(applyWeightedSharedRows(
                            sharedAct, inds, scores, rows: rows, bank: down))
                    }
                }
            }
        }
        if !outputs.isEmpty { eval(outputs) }
    }
}
