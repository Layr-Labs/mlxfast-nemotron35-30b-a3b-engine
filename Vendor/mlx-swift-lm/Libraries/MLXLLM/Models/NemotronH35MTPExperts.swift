import Foundation
import MLX
import MLXLMCommon
import MLXNN

extension NemotronHMLP {
    func mtpForwardRows(_ x: MLXArray, prepared: MLXArray? = nil) -> MLXArray {
        guard NemotronMTPExecution.batchedM1 else { return nemotronMTPMapRows(x) { self($0) } }
        if let rows = NemotronSharedUpPacking.apply(x, up: upProj, down: downProj, prepared: prepared) {
            return rows
        }
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
    /// Slot = usesPrepared + 2 * parts; every slot is its own compiled graph.
    private var entries: [Entry?] = [nil, nil, nil, nil]
    private var generations = 0
    private var calls = 0
    var generationCount: Int { lock.withLock { generations } }
    var callCount: Int { lock.withLock { calls } }

    func callAsFunction(_ x: MLXArray, prepared: MLXArray? = nil, owner: NemotronHMoE) -> MLXArray {
        let result = outputs(x, prepared: prepared, parts: false, owner: owner)
        return result.isEmpty ? MLXArray(0) : result[0]
    }

    /// parts: [routed, shared] before their Add (owner must have shared experts).
    func outputs(_ x: MLXArray, prepared: MLXArray?, parts: Bool,
                 owner: NemotronHMoE) -> [MLXArray] {
        // Native parameter wrappers forbid replacing initialized array objects;
        // Module.update changes their contexts in place. Check every concrete
        // child identity cheaply, and collect array handles only on a miss.
        // Unknown subclasses retain the fully reflective fallback.
        let nativeIdentity = Self.nativeIdentity(owner)
        let parameters = nativeIdentity == nil ? owner.innerState() : nil
        let identity = nativeIdentity ?? (owner.modules().map(ObjectIdentifier.init)
            + (parameters ?? []).map(ObjectIdentifier.init))
        let usesPrepared = prepared != nil
        let slot = (usesPrepared ? 1 : 0) + (parts ? 2 : 0)
        var retired: Entry?
        let current = lock.withLock { () -> Entry in
            calls += 1
            if let entry = entries[slot], entry.identity == identity { return entry }
            let state = parameters ?? owner.innerState()
            let function: @Sendable ([MLXArray]) -> [MLXArray] = compile(inputs: [state]) { [weak owner] arrays in
                guard let owner else { preconditionFailure("Released Nemotron MoE graph owner") }
                return owner.mtpForwardRowsUncompiledOutputs(arrays[0],
                    prepared: usesPrepared ? arrays[1] : nil, parts: parts)
            }
            let replacement = Entry(identity: identity, usesPrepared: usesPrepared, function: function)
            retired = entries[slot]
            entries[slot] = replacement
            generations += 1
            return replacement
        }
        // Destroy retired compiler entries outside this helper's lock; compiler
        // evaluation owns a separate global/per-function lock ordering.
        return withExtendedLifetime(retired) { current.function(prepared.map { [x, $0] } ?? [x]) }
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

    /// Captured-verify join only: (routed, shared) before `routed + shared`,
    /// which the join kernel then computes (same operand order, same BF16
    /// rounding) ahead of the residual Add. The graph is otherwise unchanged.
    func mtpForwardRowsParts(_ x: MLXArray, prepared: MLXArray? = nil) -> (MLXArray, MLXArray)? {
        guard sharedExperts != nil, NemotronMTPExecution.batchedM1,
              x.ndim == 3, x.dim(0) == 1, (1...8).contains(x.dim(1)) else { return nil }
        let outputs = NemotronMTPExecution.compiledMoE
            ? mtpCompiledRowsCache.outputs(x, prepared: prepared, parts: true, owner: self)
            : mtpForwardRowsUncompiledOutputs(x, prepared: prepared, parts: true)
        return outputs.count == 2 ? (outputs[0], outputs[1]) : nil
    }

    func mtpForwardRowsUncompiled(_ x: MLXArray, prepared: MLXArray? = nil) -> MLXArray {
        mtpForwardRowsUncompiledOutputs(x, prepared: prepared, parts: false)[0]
    }

    /// [routed + shared], or [routed, shared] when `parts` (shared experts
    /// present, batched M=1 route).
    func mtpForwardRowsUncompiledOutputs(
        _ x: MLXArray, prepared: MLXArray?, parts: Bool
    ) -> [MLXArray] {
        guard NemotronMTPExecution.batchedM1 else { return [nemotronMTPMapRows(x) { self($0) }] }
        let packed: MLXArray?
        if NemotronHResidualRMSNorm.preparedMoEEnabled, NemotronMoELeader.enabled,
           Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
           NemotronMoELeader.preparedGateUp, x.shape == [1, 3, 2688], x.dtype == .bfloat16 {
            // Share one dynamic operand buffer with both routed and shared up.
            packed = NemotronHeadPreparedActivation.compatible(prepared, rows: 3, k: 2688)
                ?? NemotronHeadPreparedActivation.prepareContiguousRows(x.reshaped(3, 2688), pv: 8)
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
            x, indices, scores, switchMLP, prepared: packed,
            products: parts && NemotronHResidualRMSNorm.plumbingAvailable) {
            y = reduced
        } else {
            y = NemotronMoELeader.switchRows(x, indices, switchMLP) ?? switchMLP(x, indices)
            y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)
        }
        guard let sharedExperts else { return [y] }
        let shared = sharedExperts.mtpForwardRows(x, prepared: packed)
        return parts ? [y, shared] : [y + shared]
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

// R = 3 verify (HALF): the same masked coefficients, decoded off the
// int->float pipe. n0 and n1 of a word pair come from one 32-bit AND/OR into
// half2 magic numbers 1024 + v, minus 1024; n2 and n3 through ushort -> half.
// Every value has <= 4 significant bits and is <= 61440, so each float operand,
// the FP32 terms, their order and the affine close match the record's helpers.
template <typename U, int values_per_thread>
inline U nml_qdot_half(
    const device uint8_t* w, const thread U* x_thread, U scale, U bias, U sum) {
  U accum = 0;
  const device uint16_t* ws = (const device uint16_t*)w;
  const device uint* ws2 = (const device uint*)w;
  for (int i = 0; i < (values_per_thread / 4); i++) {
    const uint w2 = ws2[i / 2];
    const half2 h0 = as_type<half2>((w2 & 0x000f000fu) | 0x64006400u) - half2(1024.0h);
    const half2 h1 = as_type<half2>((w2 & 0x00f000f0u) | 0x64006400u) - half2(1024.0h);
    const U a = U(h0[i % 2]);
    const U b1 = U(h1[i % 2]);
    const U c = U(half(ushort(ws[i] & 0x0f00)));
    const U d = U(half(ushort(ws[i] & 0xf000)));
    accum +=
        (x_thread[4 * i] * a +
         x_thread[4 * i + 1] * b1 +
         x_thread[4 * i + 2] * c +
         x_thread[4 * i + 3] * d);
  }
  return scale * accum + sum * bias;
}

template <typename U, int values_per_thread>
inline U nml_qdot_safe_half(
    const device uint8_t* w, const thread U* x_thread, U scale, U bias, U sum,
    int N) {
  U accum = 0;
  const device uint16_t* ws = (const device uint16_t*)w;
  const device uint* ws2 = (const device uint*)w;
  for (int i = 0; i < (values_per_thread / 4); i++) {
    if (i < N / 4) {
      const uint w2 = ws2[i / 2];
      const half2 h0 = as_type<half2>((w2 & 0x000f000fu) | 0x64006400u) - half2(1024.0h);
      const half2 h1 = as_type<half2>((w2 & 0x00f000f0u) | 0x64006400u) - half2(1024.0h);
      const U a = U(h0[i % 2]);
      const U b1 = U(h1[i % 2]);
      const U c = U(half(ushort(ws[i] & 0x0f00)));
      const U d = U(half(ushort(ws[i] & 0xf000)));
      accum +=
          (x_thread[4 * i] * a +
           x_thread[4 * i + 1] * b1 +
           x_thread[4 * i + 2] * c +
           x_thread[4 * i + 3] * d);
    }
  }
  return scale * accum + sum * bias;
}

template <typename U, int values_per_thread>
inline void nml_qdot_pair_half(
    const device uint8_t* w, const thread U* x0, const thread U* x1,
    U scale, U bias, U sum0, U sum1, int N, thread U& value0, thread U& value1) {
  U accum0 = 0;
  U accum1 = 0;
  const device uint16_t* ws = (const device uint16_t*)w;
  const device uint* ws2 = (const device uint*)w;
  for (int i = 0; i < (values_per_thread / 4); i++) {
    if (i < N / 4) {
      const uint w2 = ws2[i / 2];
      const half2 h0 = as_type<half2>((w2 & 0x000f000fu) | 0x64006400u) - half2(1024.0h);
      const half2 h1 = as_type<half2>((w2 & 0x00f000f0u) | 0x64006400u) - half2(1024.0h);
      const U a = U(h0[i % 2]);
      const U b1 = U(h1[i % 2]);
      const U c = U(half(ushort(ws[i] & 0x0f00)));
      const U d = U(half(ushort(ws[i] & 0xf000)));
      accum0 += (x0[4 * i] * a + x0[4 * i + 1] * b1 + x0[4 * i + 2] * c + x0[4 * i + 3] * d);
      accum1 += (x1[4 * i] * a + x1[4 * i + 1] * b1 + x1[4 * i + 2] * c + x1[4 * i + 3] * d);
    }
  }
  value0 = scale * accum0 + sum0 * bias;
  value1 = scale * accum1 + sum1 * bias;
}

template <typename U, int values_per_thread, bool HALF = false>
inline U nml_qdot(
    const device uint8_t* w, const thread U* x_thread, U scale, U bias, U sum) {
  if (HALF) {
    return nml_qdot_half<U, values_per_thread>(w, x_thread, scale, bias, sum);
  }
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

template <typename U, int values_per_thread, bool HALF = false>
inline U nml_qdot_safe(
    const device uint8_t* w, const thread U* x_thread, U scale, U bias, U sum,
    int N) {
  if (HALF) {
    return nml_qdot_safe_half<U, values_per_thread>(w, x_thread, scale, bias, sum, N);
  }
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

// A leader and its partner share one expert's weights. Each packed word and
// its four masked coefficients are decoded once and fed to two independent
// accumulators, each with the stock per-token term order and affine close.
template <typename U, int values_per_thread, bool HALF = false>
inline void nml_qdot_pair(
    const device uint8_t* w, const thread U* x0, const thread U* x1,
    U scale, U bias, U sum0, U sum1, int N, thread U& value0, thread U& value1) {
  if (HALF) {
    nml_qdot_pair_half<U, values_per_thread>(
        w, x0, x1, scale, bias, sum0, sum1, N, value0, value1);
    return;
  }
  U accum0 = 0;
  U accum1 = 0;
  const device uint16_t* ws = (const device uint16_t*)w;
  for (int i = 0; i < (N / 4); i++) {
    const uint16_t q = ws[i];
    const U a = q & 0x000f;
    const U b = q & 0x00f0;
    const U c = q & 0x0f00;
    const U d = q & 0xf000;
    accum0 += (x0[4 * i] * a + x0[4 * i + 1] * b + x0[4 * i + 2] * c + x0[4 * i + 3] * d);
    accum1 += (x1[4 * i] * a + x1[4 * i + 1] * b + x1[4 * i + 2] * c + x1[4 * i + 3] * d);
  }
  value0 = scale * accum0 + sum0 * bias;
  value1 = scale * accum1 + sum1 * bias;
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

/// Tail block in its constant 8-value form (NEMOTRON_MTP_FC1_TAIL8 /
/// NEMOTRON_MTP_FC2_TAIL8). With 8 values per lane and K % 8 == 0, a lane's
/// tail `remaining` is 0 or 8, and the tail body runs only when it is 8. At
/// N = 8 every `_safe` load equals the full load and every qdot with N = 8
/// equals the main-loop call: same loads, same terms, same order and rounding.
private func nmlTail8Source(_ source: String) -> String? {
    let typedef = "  typedef float U;\n"
    guard source.components(separatedBy: typedef).count == 2 else { return nil }
    var s = source.replacingOccurrences(of: typedef, with:
        "  static_assert(K % values_per_thread == 0, \"tail lanes hold 8 values\");\n" + typedef)
    for x in ["x0, x_thread", "x0, x0_thread", "x1, x1_thread", "x2, x2_thread"] {
        s = s.replacingOccurrences(
            of: "nml_load_vector_safe<T, U, values_per_thread>(\(x), remaining)",
            with: "nml_load_vector<T, U, values_per_thread>(\(x))")
    }
    for (from, to) in [
        ("sum, remaining)", "sum, values_per_thread)"),
        ("sum1, remaining, ", "sum1, values_per_thread, "),
        ("sum2,\n            remaining, ", "sum2,\n            values_per_thread, "),
    ] {
        s = s.replacingOccurrences(of: from, with: to)
    }
    // `remaining` survives only in its declaration and the tail guards.
    let rest = s.replacingOccurrences(of: "const int remaining = clamp(", with: "")
        .replacingOccurrences(of: "(remaining > 0)", with: "")
    guard !rest.contains("remaining"), !s.contains("nml_load_vector_safe") else { return nil }
    return s
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
        result[r] += nml_qdot<U, values_per_thread, true>(wl, x_thread, s, b, sum);
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
        result[r] += nml_qdot_safe<U, values_per_thread, true>(
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
        U value0, value1;
        nml_qdot_pair<U, values_per_thread, true>(
            wl, x0_thread, x1_thread, s, b, sum0, sum1, values_per_thread, value0, value1);
        result0[r] += value0;
        result1[r] += value1;
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
        U value0, value1;
        nml_qdot_pair<U, values_per_thread, true>(
            wl, x0_thread, x1_thread, s, b, sum0, sum1, remaining, value0, value1);
        result0[r] += value0;
        result1[r] += value1;
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
    /// Production-only copy with the constant 8-value tail (nmlTail8Source);
    /// `kernel` stays the reference the self-tests compare against.
    let tail8: MLXFast.MLXFastKernel?
    private init() {
        kernel = MLXFast.metalKernel(
            name: "nemotron_moe_leader_qmv_affine4_gs64",
            inputNames: ["w", "scales", "biases", "x", "inds"],
            outputNames: ["y"],
            source: nemotronMoELeaderSource,
            header: nemotronMoELeaderHeader + nemotronMoELeaderStore)
        tail8 = nmlTail8Source(nemotronMoELeaderSource).map {
            MLXFast.metalKernel(
                name: "nemotron_moe_leader_qmv_tail8_affine4_gs64",
                inputNames: ["w", "scales", "biases", "x", "inds"],
                outputNames: ["y"],
                source: $0,
                header: nemotronMoELeaderHeader + nemotronMoELeaderStore)
        }
    }
}

private final class NemotronMoELeaderPreparedKernels: Sendable {
    static let shared = NemotronMoELeaderPreparedKernels()
    let gateUp: MLXFast.MLXFastKernel?
    let gateUpAndPlan: MLXFast.MLXFastKernel?
    let tripleGateUpAndPlan: MLXFast.MLXFastKernel?
    /// The same triple FC1 with its tail block in the constant 8-value form.
    let tripleGateUpAndPlanTail8: MLXFast.MLXFastKernel?
    private init() {
        let header = nemotronMoELeaderHeader + nemotronMoELeaderStore + nemotronMoELeaderPreparedHeader
        let preparedSource = nmlPreparedSource(nemotronMoELeaderSource, pointers: [
            ("const device T* x0 = x + (XSLOT ? slot : row) * K + simd_lid * values_per_thread;",
             "const device float* x0 = prepared + row * (PTILES * PSTRIDE) + simd_lid;"),
            ("const device T* x1 = x + (XSLOT ? partner : partner / TOPK) * K +\n        simd_lid * values_per_thread;",
             "const device float* x1 = prepared + (partner / TOPK) * (PTILES * PSTRIDE) + simd_lid;"),
        ])
        gateUp = preparedSource.map {
            MLXFast.metalKernel(
                name: "nemotron_moe_leader_qmv_affine4_gs64_prepared",
                inputNames: ["w", "scales", "biases", "prepared", "inds"],
                outputNames: ["y"], source: $0, header: header)
        }
        gateUpAndPlan = preparedSource.map {
            MLXFast.metalKernel(
                name: "nemotron_moe_leader_prepared_fc1_with_balance_plan",
                inputNames: ["w", "scales", "biases", "prepared", "inds"],
                outputNames: ["y", "plan"], source: nmlFC1WithBalancePlanSource($0),
                header: header)
        }
        // Select only the source's R3 FC1 traversal; weighted FC2 retains
        // the complete winner's paired body and original 36-word plan.
        let tripleSource = nemotronMoETripleGateUpSource.replacingOccurrences(
            of: "x2 += block_size;", with: "x2 += PSTRIDE;")
        let triplePointers = [
            ("const device T* x0 = x + (XSLOT ? slot : row) * K + simd_lid * values_per_thread;",
             "const device float* x0 = prepared + row * (PTILES * PSTRIDE) + simd_lid;"),
            ("const device T* x1 = x + (XSLOT ? partner : partner / TOPK) * K +\n        simd_lid * values_per_thread;",
             "const device float* x1 = prepared + (partner / TOPK) * (PTILES * PSTRIDE) + simd_lid;"),
            ("const device T* x2 = x + (XSLOT ? third : third / TOPK) * K +\n        simd_lid * values_per_thread;",
             "const device float* x2 = prepared + (third / TOPK) * (PTILES * PSTRIDE) + simd_lid;"),
        ]
        tripleGateUpAndPlan = nmlPreparedSource(tripleSource, pointers: triplePointers).map {
            MLXFast.metalKernel(
                name: "nemotron_moe_triple_prepared_fc1_with_pair_fc2_plan",
                inputNames: ["w", "scales", "biases", "prepared", "inds"],
                outputNames: ["y", "plan"], source: nmlFC1WithBalancePlanSource($0),
                header: header + nemotronMoETripleDecodeHeader)
        }
        tripleGateUpAndPlanTail8 = nmlTail8Source(tripleSource)
            .flatMap { nmlPreparedSource($0, pointers: triplePointers) }.map {
            MLXFast.metalKernel(
                name: "nemotron_moe_triple_prepared_fc1_tail8_with_pair_fc2_plan",
                inputNames: ["w", "scales", "biases", "prepared", "inds"],
                outputNames: ["y", "plan"], source: nmlFC1WithBalancePlanSource($0),
                header: header + nemotronMoETripleDecodeHeader)
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
          result[r] += nml_qdot<U, values_per_thread, true>(wl, x_thread, s, b, sum);
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
          result[r] += nml_qdot_safe<U, values_per_thread, true>(
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
          U value0, value1;
          nml_qdot_pair<U, values_per_thread, true>(
              wl, x0_thread, x1_thread, s, b, sum0, sum1, values_per_thread, value0, value1);
          result0[r] += value0;
          result1[r] += value1;
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
          U value0, value1;
          nml_qdot_pair<U, values_per_thread, true>(
              wl, x0_thread, x1_thread, s, b, sum0, sum1, remaining, value0, value1);
          result0[r] += value0;
          result1[r] += value1;
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
    /// Production-only copy with the constant 8-value tail (nmlTail8Source).
    let tail8: MLXFast.MLXFastKernel?
    private init() {
        kernel = MLXFast.metalKernel(
            name: "nemotron_moe_leader_weighted_qmv_affine4_gs64",
            inputNames: ["w", "scales", "biases", "x", "inds", "scores"],
            outputNames: ["y"],
            source: nemotronMoELeaderWeightedSource,
            header: nemotronMoELeaderHeader + nemotronMoELeaderStore,
            ensureRowContiguous: true)
        tail8 = nmlTail8Source(nemotronMoELeaderWeightedSource).map {
            MLXFast.metalKernel(
                name: "nemotron_moe_leader_weighted_qmv_tail8_affine4_gs64",
                inputNames: ["w", "scales", "biases", "x", "inds", "scores"],
                outputNames: ["y"],
                source: $0,
                header: nemotronMoELeaderHeader + nemotronMoELeaderStore,
                ensureRowContiguous: true)
        }
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
    /// tail8: production calls only (the self-test references keep false).
    static func apply(_ x: MLXArray, _ inds: MLXArray, rows: Int, topK: Int,
                      bank: Bank, perSlot: Bool, relu2: Bool = false,
                      prepared: MLXArray? = nil, tail8: Bool = false) -> MLXArray {
        let rps = leaderResultsPerSimdgroup(rows: rows)
        if preparedGateUp, rows == 3, !perSlot, bank.k % 64 == 0, bank.k % 512 != 0,
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
        let kernel = (tail8 ? NemotronMoELeaderKernel.shared.tail8 : nil)
            ?? NemotronMoELeaderKernel.shared.kernel
        return kernel(
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
    static let fuseBalancePlan =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_FC1_BALANCE_PLAN"] != "0"
    /// 7968e3f5 (c486095a grouping): NEMOTRON_MTP_TRIPLE_FC1_ONLY=0 keeps the
    /// pair-only prepared R3 FC1; on, one leader also serves an expert's third
    /// row with one shared coefficient decode.
    static let tripleGateUp =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_TRIPLE_FC1_ONLY"] != "0"
    /// Tail block of the R3 triple FC1 / balanced FC2 in its constant 8-value
    /// form (nmlTail8Source); =0 keeps the runtime-`remaining` tail.
    static let fc1Tail8 =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_FC1_TAIL8"] != "0"
    static let fc2Tail8 =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_FC2_TAIL8"] != "0"
    /// The same constant tail for the R != 3 (draft tail) leader and weighted
    /// kernels, through production-only kernel copies.
    static let leaderTail8 =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_LEADER_TAIL8"] != "0"

    /// The ordinary prepared R3/top-six FC1 arithmetic plus the original
    /// integer FC2 plan, emitted by just the first leader/output tile.
    /// Both outputs are fresh current-call arrays; no planner state persists.
    static func applyWithBalancePlan(
        _ x: MLXArray, _ inds: MLXArray, bank: Bank, relu2: Bool,
        prepared: MLXArray?
    ) -> (activation: MLXArray, plan: MLXArray)? {
        guard fuseBalancePlan, preparedGateUp,
              x.shape == [3, 2688], x.dtype == .bfloat16,
              inds.shape == [18], inds.dtype == .uint32,
              bank.weight.shape == [128, 1856, 336], bank.k == 2688, bank.n == 1856,
              let kernel = (tripleGateUp
                  ? (fc1Tail8 ? NemotronMoELeaderPreparedKernels.shared.tripleGateUpAndPlanTail8 : nil)
                      ?? NemotronMoELeaderPreparedKernels.shared.tripleGateUpAndPlan : nil)
                  ?? NemotronMoELeaderPreparedKernels.shared.gateUpAndPlan
        else { return nil }
        let packed = NemotronHeadPreparedActivation.compatible(prepared, rows: 3, k: bank.k)
            ?? NemotronHeadPreparedActivation.prepareContiguousRows(x, pv: 8)
        let rps = leaderResultsPerSimdgroup(rows: 3)
        let result = kernel([bank.weight, bank.scales, bank.biases, packed, inds],
            template: [("T", x.dtype), ("R", 3), ("TOPK", 6),
                       ("K", bank.k), ("N", bank.n), ("XSLOT", false),
                       ("RELU2", relu2), ("RPS", rps)],
            grid: (32, bank.n / rps, 18), threadGroup: (32, 2, 1),
            outputShapes: [[18, bank.n], [36]], outputDTypes: [x.dtype, .int32])
        guard selfTest else { return (result[0], result[1]) }
        let expected = apply(x, inds, rows: 3, topK: 6, bank: bank,
            perSlot: false, relu2: relu2, prepared: packed)
        let referencePlan = NMLBalancePlan.shared.make(inds)
        let ok = MLX.logicalAnd(
            (result[0].view(dtype: .uint16) .== expected.view(dtype: .uint16)).all(),
            (result[1] .== referencePlan).all())
        return (which(ok, result[0], MLXArray(Float.nan).asType(x.dtype)),
            which(ok, result[1], referencePlan))
    }

    /// Same leader pairs as apply(), with separate FP32 products and the
    /// native top-six sum. Caller supplies six distinct indices per row.
    static func applyWeighted(_ x: MLXArray, _ inds: MLXArray, _ scores: MLXArray,
                              rows: Int, bank: Bank, plan: MLXArray? = nil,
                              tail8: Bool = false) -> MLXArray {
        if rows == 3 { return NMLBalanceWeighted.shared.run(x, inds, scores, bank, plan: plan) }
        let kernel = (tail8 ? NemotronMoELeaderWeightedKernel.shared.tail8 : nil)
            ?? NemotronMoELeaderWeightedKernel.shared.kernel
        return kernel(
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
        _ mlp: NemotronHSwitchMLP, prepared: MLXArray? = nil, products: Bool = false
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
        let balancePlan: MLXArray?
        if fuseRelu2 {
            if rows == 3, let fused = applyWithBalancePlan(
                xr, flat, bank: up, relu2: true, prepared: prepared) {
                act = fused.activation
                balancePlan = fused.plan
            } else {
                act = apply(xr, flat, rows: rows, topK: 6, bank: up, perSlot: false,
                    relu2: true, prepared: prepared, tail8: leaderTail8)
                balancePlan = nil
            }
        } else {
            let h = apply(xr, flat, rows: rows, topK: 6, bank: up, perSlot: false,
                prepared: prepared, tail8: leaderTail8)
            let a = maximum(h, MLXArray(0))
            act = a * a
            balancePlan = nil
        }
        let weights = scores.reshaped(rows * 6)
        // `products`: the unreduced slot products for the captured verify's
        // join kernel, which sums them itself (same expressions and order).
        if products, rows == 3,
           let prod = NemotronZSkip.weightedProducts(act, flat, weights, owner: mlp.fc2) {
            guard selfTest else { return prod }
            let original = switchRows(x, indices, mlp) ?? mlp(x, indices)
            let expected = (original * scores[.ellipsis, .newAxis])
                .sum(axis: -2).asType(original.dtype)
            let ok = (NemotronZSkip.reduceProducts(prod, rows: 3).view(dtype: .uint16)
                .== expected.view(dtype: .uint16)).all()
            return which(ok, prod, MLXArray(Float.nan))
        }
        var y = NemotronZSkip.weighted(act, flat, weights, rows: rows, owner: mlp.fc2)
            ?? applyWeighted(act, flat, weights, rows: rows, bank: down, plan: balancePlan,
                tail8: leaderTail8)
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
                let fused = rows == 3 && topK == 6
                    ? applyWithBalancePlan(x, inds, bank: up, relu2: fuseRelu2, prepared: nil) : nil
                let h = fused?.activation
                    ?? apply(x, inds, rows: rows, topK: topK, bank: up, perSlot: false,
                             relu2: fuseRelu2)
                if let fused { outputs.append(fused.plan) }
                if fused == nil, leaderTail8 {
                    outputs.append(apply(x, inds, rows: rows, topK: topK, bank: up,
                        perSlot: false, relu2: fuseRelu2, tail8: true))
                }
                if rows > 1 {
                    outputs.append(apply(h, inds, rows: rows, topK: topK, bank: down, perSlot: true))
                }
                if topK == 6, up.weight.shape == [128, 1856, 336],
                    down.weight.shape == [128, 2688, 232] {
                    let scores = MLXArray.ones([rows * topK], dtype: .float32)
                    outputs.append(applyWeighted(h, inds, scores, rows: rows, bank: down,
                        plan: fused?.plan, tail8: leaderTail8))
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
    /// The same body with its tail block in the constant 8-value form.
    let tail8 = nmlTail8Source(nmlBalanceWeightedSource).map {
        MLXFast.metalKernel(name: "nml_fc2_original_jobs_balanced_tail8",
            inputNames: ["w","scales","biases","x","inds","scores","plan"],
            outputNames: ["y"], source: $0,
            header: nemotronMoELeaderHeader + nemotronMoELeaderStore,
            ensureRowContiguous: true)
    }
    func run(_ x: MLXArray, _ ids: MLXArray, _ scores: MLXArray,
             _ bank: NemotronMoELeader.Bank, plan: MLXArray? = nil) -> MLXArray {
        let plan = plan ?? NMLBalancePlan.shared.make(ids)
        let kernel = (NemotronMoELeader.fc2Tail8 ? tail8 : nil) ?? self.kernel
        return kernel([bank.weight,bank.scales,bank.biases,x,ids,scores,plan],
            template: [("T",x.dtype),("R",3),("TOPK",6),("K",bank.k),("N",bank.n)],
            grid: (32*(bank.n/4),6,1), threadGroup: (32,6,1),
            outputShapes: [[1,3,bank.n]], outputDTypes: [x.dtype])[0]
    }
}


/// Reuse the entire accepted prepared FC1 body verbatim. The extra work runs
/// in exactly one complete 64-thread TG: row-zero/slot-zero is always a leader,
/// so its original follower return cannot skip the plan. FC2 already depends
/// on FC1; reading the plan adds no dependency on a different producer.
private func nmlFC1WithBalancePlanSource(_ preparedFC1: String) -> String {
    let lane = "const uint lane = thread_position_in_grid.x;"
    let store = "for (uint i = lane; i < 36; i += 32)"
    precondition(nmlBalancePlanSource.components(separatedBy: lane).count == 2)
    precondition(nmlBalancePlanSource.components(separatedBy: store).count == 2)
    let plan = nmlBalancePlanSource
        .replacingOccurrences(of: lane, with: "const uint lane = simd_lid + 32 * simd_gid;")
        .replacingOccurrences(of: store, with: "for (uint i = lane; i < 36; i += 64)")
    return preparedFC1 + """

      // tid is a TG coordinate, so both SIMD groups take this branch together.
      if (tid.y == 0 && tid.z == 0) {
        constexpr int PAIR_COST = 1500;
        constexpr int SOLO_COST = 1000;
        \(plan)
      }
    """
}

// MARK: - Exact-zero skip for the routed fc2 verify (R = 3)
//
// relu2 makes most routed fc2 inputs exactly +0. A load-time copy of each fc2
// bank holds the same 4-bit codes, scales and biases in K-major order: codes
// as [expert, k, n / 8] words of eight outputs' nibbles, scales and biases as
// [expert, group, n]. A k whose input is zero goes unread. Every output keeps
// the record kernel's arithmetic: lane l of the 32-wide reduction owns chunks
// l + 32j (and 224 + l), each chunk is the same qdot of eight k with the same
// scale, bias and sum, chunks add into the lane partial in the same order, and
// the closing simd_sum sees the same partials in the same lanes. A skipped
// chunk is one whose qdot is scale * (+0) + (+0) * bias = +-0, which leaves a
// partial that is never -0 unchanged; a skipped k inside a chunk feeds (+0) * 0
// instead of (+0) * q, +0 either way. The six weighted products then reduce in
// the record's order, and their + 0.0f still maps a -0 result to +0.
private let nemotronZSkipHeader = """
inline void nmz_swap(thread uint& a, thread uint& b, uint shift, uint mask) {
  const uint t = ((a >> shift) ^ b) & mask;
  b ^= t;
  a ^= t << shift;
}

template <typename U>
inline U nmz_qdot(
    const thread uint16_t* ws, const thread U* x_thread, U scale, U bias, U sum) {
  U accum = 0;
  for (int i = 0; i < 2; i++) {
    accum +=
        (x_thread[4 * i] * (ws[i] & 0x000f) +
         x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
         x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
         x_thread[4 * i + 3] * (ws[i] & 0xf000));
  }
  return scale * accum + sum * bias;
}
"""

// wk[e][k][q] nibble o = code of output 8q + o at k.
private let nemotronZSkipLayoutSource = """
  const uint q = thread_position_in_grid.x;
  const uint k = thread_position_in_grid.y;
  const uint e = thread_position_in_grid.z;
  const device uint* src = w + ((ulong)e * N + q * 8) * (K / 8) + k / 8;
  const uint shift = 4 * (k % 8);
  uint v = 0;
  for (uint o = 0; o < 8; o++) {
    v |= ((src[o * (K / 8)] >> shift) & 0xfu) << (4 * o);
  }
  wk[((ulong)e * K + k) * (N / 8) + q] = v;
"""

// One threadgroup per (128-output tile, slot), every slot on its own: SIMD
// group s owns lane partials l = s, s + SGS, ... of all 128 outputs, so the
// whole group meets the same chunks and skips a zero chunk together. Lane i
// reads the 16-bit K-major words of outputs 4i..4i+3 and rebuilds each
// output's packed chunk word with two nibble-swap stages. The partials meet
// their lanes for the closing simd_sum through threadgroup memory.
private let nemotronZSkipSource = """
  constexpr int OT = 128;
  constexpr int LPS = 32 / SGS;
  constexpr int chunks = K / 8;
  constexpr int tail = chunks - 224;
  constexpr int halves = N / 4;
  constexpr int groups = K / 64;
  constexpr int PS = OT + 1;
  typedef float U;
  threadgroup float part[32 * PS];

  const int slot = int(threadgroup_position_in_grid.y);
  const int n0 = int(threadgroup_position_in_grid.x) * OT;
  const int lid = int(thread_index_in_simdgroup);
  const int sg = int(simdgroup_index_in_threadgroup);
  const uint expert = inds[slot];
  const device T* xs = x + slot * K;
  const device uint16_t* wq = (const device uint16_t*)wk + (ulong)expert * K * halves + n0 / 4 + lid;
  const ulong sbase = (ulong)expert * groups * N + n0 + 4 * lid;
  const device T* sq = sk + sbase;
  const device T* bq = bk + sbase;
  for (int ll = 0; ll < LPS; ll++) {
    const int l = sg + SGS * ll;
    thread U p[4];
    for (int o = 0; o < 4; o++) {
      p[o] = 0;
    }
    const int count = l < tail ? 8 : 7;
    for (int j = 0; j < count; j++) {
      const int c = j < 7 ? l + 32 * j : 224 + l;
      const device uint16_t* xb = (const device uint16_t*)(xs + 8 * c);
      uint m = 0;
      for (int i = 0; i < 8; i++) {
        m |= (xb[i] & 0x7fff) != 0 ? (1u << i) : 0u;
      }
      if (m == 0) {
        continue;
      }
      thread U xt[8];
      const U sum = nml_load_vector<T, U, 8>(xs + 8 * c, xt);
      uint u[4];
      for (int i = 0; i < 4; i++) {
        const uint lo = (m & (1u << i)) != 0 ? uint(wq[(8 * c + i) * halves]) : 0u;
        const uint hi = (m & (16u << i)) != 0 ? uint(wq[(8 * c + i + 4) * halves]) : 0u;
        u[i] = lo | (hi << 16);
      }
      nmz_swap(u[0], u[1], 4, 0x0f0f0f0fu);
      nmz_swap(u[2], u[3], 4, 0x0f0f0f0fu);
      nmz_swap(u[0], u[2], 8, 0x00ff00ffu);
      nmz_swap(u[1], u[3], 8, 0x00ff00ffu);
      const int gi = c / 8;
      for (int o = 0; o < 4; o++) {
        const uint16_t ws[2] = {uint16_t(u[o] & 0xffffu), uint16_t(u[o] >> 16)};
        const U s = sq[gi * N + o];
        const U b = bq[gi * N + o];
        p[o] += nmz_qdot<U>(ws, xt, s, b, sum);
      }
    }
    for (int o = 0; o < 4; o++) {
      part[l * PS + 4 * lid + o] = p[o];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int q = 0; q < OT / SGS; q++) {
    const int n = sg * (OT / SGS) + q;
    const U r = simd_sum(part[lid * PS + n]);
    if (lid == 0) {
      const T h = nml_store<T, false>(r);
      prod[slot * N + n0 + n] = float(h) * scores[slot] + 0.0f;
    }
  }
"""

// Staged twin of the slot kernel: the slot's activations and every chunk's
// zero mask go to threadgroup memory once per threadgroup (the same mask test),
// and a lane's four scales (and four biases), four consecutive BF16 values,
// load as one 8-byte word each and widen by the exact BF16 -> FP32 shift. The
// chunk sum, the pre-scaled x_thread values (nml_load_vector's expressions),
// code reads, nibble swaps, qdot, partial order and reduction are unchanged.
private let nemotronZSkipStagedSource = """
  constexpr int OT = 128;
  constexpr int LPS = 32 / SGS;
  constexpr int chunks = K / 8;
  constexpr int tail = chunks - 224;
  constexpr int halves = N / 4;
  constexpr int groups = K / 64;
  constexpr int PS = OT + 1;
  typedef float U;
  threadgroup float part[32 * PS];
  threadgroup T xsh[K];
  threadgroup uchar cmask[chunks];

  const int slot = int(threadgroup_position_in_grid.y);
  const int n0 = int(threadgroup_position_in_grid.x) * OT;
  const int lid = int(thread_index_in_simdgroup);
  const int sg = int(simdgroup_index_in_threadgroup);
  const uint expert = inds[slot];
  const device T* xs = x + slot * K;
  for (int c = sg * 32 + lid; c < chunks; c += SGS * 32) {
    const device uint16_t* xb = (const device uint16_t*)(xs + 8 * c);
    uint m = 0;
    for (int i = 0; i < 8; i++) {
      m |= (xb[i] & 0x7fff) != 0 ? (1u << i) : 0u;
      xsh[8 * c + i] = xs[8 * c + i];
    }
    cmask[c] = uchar(m);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const device uint16_t* wq = (const device uint16_t*)wk + (ulong)expert * K * halves + n0 / 4 + lid;
  const ulong sbase = (ulong)expert * groups * N + n0 + 4 * lid;
  const device T* sq = sk + sbase;
  const device T* bq = bk + sbase;
  for (int ll = 0; ll < LPS; ll++) {
    const int l = sg + SGS * ll;
    thread U p[4];
    for (int o = 0; o < 4; o++) {
      p[o] = 0;
    }
    const int count = l < tail ? 8 : 7;
    for (int j = 0; j < count; j++) {
      const int c = j < 7 ? l + 32 * j : 224 + l;
      const uint m = cmask[c];
      if (m == 0) {
        continue;
      }
      thread U xt[8];
      U sum = 0;
      const threadgroup T* xv = xsh + 8 * c;
      for (int i = 0; i < 8; i += 4) {
        sum += U(xv[i]) + U(xv[i + 1]) + U(xv[i + 2]) + U(xv[i + 3]);
        xt[i] = xv[i];
        xt[i + 1] = xv[i + 1] / 16.0f;
        xt[i + 2] = xv[i + 2] / 256.0f;
        xt[i + 3] = xv[i + 3] / 4096.0f;
      }
      uint u[4];
      for (int i = 0; i < 4; i++) {
        const uint lo = (m & (1u << i)) != 0 ? uint(wq[(8 * c + i) * halves]) : 0u;
        const uint hi = (m & (16u << i)) != 0 ? uint(wq[(8 * c + i + 4) * halves]) : 0u;
        u[i] = lo | (hi << 16);
      }
      nmz_swap(u[0], u[1], 4, 0x0f0f0f0fu);
      nmz_swap(u[2], u[3], 4, 0x0f0f0f0fu);
      nmz_swap(u[0], u[2], 8, 0x00ff00ffu);
      nmz_swap(u[1], u[3], 8, 0x00ff00ffu);
      const int gi = c / 8;
      const uint2 sr = *((const device uint2*)(sq + gi * N));
      const uint2 br = *((const device uint2*)(bq + gi * N));
      const uint sv[4] = {sr.x << 16, sr.x & 0xffff0000u, sr.y << 16, sr.y & 0xffff0000u};
      const uint bv[4] = {br.x << 16, br.x & 0xffff0000u, br.y << 16, br.y & 0xffff0000u};
      for (int o = 0; o < 4; o++) {
        const uint16_t ws[2] = {uint16_t(u[o] & 0xffffu), uint16_t(u[o] >> 16)};
        p[o] += nmz_qdot<U>(ws, xt, as_type<float>(sv[o]), as_type<float>(bv[o]), sum);
      }
    }
    for (int o = 0; o < 4; o++) {
      part[l * PS + 4 * lid + o] = p[o];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int q = 0; q < OT / SGS; q++) {
    const int n = sg * (OT / SGS) + q;
    const U r = simd_sum(part[lid * PS + n]);
    if (lid == 0) {
      const T h = nml_store<T, false>(r);
      prod[slot * N + n0 + n] = float(h) * scores[slot] + 0.0f;
    }
  }
"""

// Tile-major copy: for each expert and 128-output tile, row k's 32 uint16 words
// (four outputs' nibbles each, output 4h + o in bits 4o, as in the K-major
// halves) are contiguous, so a threadgroup's rows are 64-byte segments of one
// [K, 32] block. Scales and biases are [expert, tile, group, 128].
private let nemotronZSkipTiledLayoutSource = """
  const uint lid = thread_position_in_grid.x % 32;
  const uint t = thread_position_in_grid.x / 32;
  const uint k = thread_position_in_grid.y;
  const uint e = thread_position_in_grid.z;
  const uint n = t * 128 + 4 * lid;
  const uint shift = 4 * (k % 8);
  uint v = 0;
  for (uint o = 0; o < 4; o++) {
    v |= ((w[((ulong)e * N + n + o) * (K / 8) + k / 8] >> shift) & 0xfu) << (4 * o);
  }
  wk[(((ulong)e * (N / 128) + t) * K + k) * 32 + lid] = uint16_t(v);
"""

// The staged slot kernel on the tile-major copy: the same statements, with
// only the code-word, scale and bias addresses moved to that layout.
private let nemotronZSkipTiledSeqSource = """
  constexpr int OT = 128;
  constexpr int LPS = 32 / SGS;
  constexpr int chunks = K / 8;
  constexpr int tail = chunks - 224;
  constexpr int halves = N / 4;
  constexpr int groups = K / 64;
  constexpr int PS = OT + 1;
  typedef float U;
  threadgroup float part[32 * PS];
  threadgroup T xsh[K];
  threadgroup uchar cmask[chunks];

  const int slot = int(threadgroup_position_in_grid.y);
  const int n0 = int(threadgroup_position_in_grid.x) * OT;
  const int lid = int(thread_index_in_simdgroup);
  const int sg = int(simdgroup_index_in_threadgroup);
  const uint expert = inds[slot];
  const device T* xs = x + slot * K;
  for (int c = sg * 32 + lid; c < chunks; c += SGS * 32) {
    const device uint16_t* xb = (const device uint16_t*)(xs + 8 * c);
    uint m = 0;
    for (int i = 0; i < 8; i++) {
      m |= (xb[i] & 0x7fff) != 0 ? (1u << i) : 0u;
      xsh[8 * c + i] = xs[8 * c + i];
    }
    cmask[c] = uchar(m);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const ulong tb = (ulong)expert * (N / OT) + int(threadgroup_position_in_grid.x);
  const device uint16_t* wq = (const device uint16_t*)wk + tb * K * 32 + lid;
  const device T* sq = sk + tb * groups * OT + 4 * lid;
  const device T* bq = bk + tb * groups * OT + 4 * lid;
  for (int ll = 0; ll < LPS; ll++) {
    const int l = sg + SGS * ll;
    thread U p[4];
    for (int o = 0; o < 4; o++) {
      p[o] = 0;
    }
    const int count = l < tail ? 8 : 7;
    for (int j = 0; j < count; j++) {
      const int c = j < 7 ? l + 32 * j : 224 + l;
      const uint m = cmask[c];
      if (m == 0) {
        continue;
      }
      thread U xt[8];
      U sum = 0;
      const threadgroup T* xv = xsh + 8 * c;
      for (int i = 0; i < 8; i += 4) {
        sum += U(xv[i]) + U(xv[i + 1]) + U(xv[i + 2]) + U(xv[i + 3]);
        xt[i] = xv[i];
        xt[i + 1] = xv[i + 1] / 16.0f;
        xt[i + 2] = xv[i + 2] / 256.0f;
        xt[i + 3] = xv[i + 3] / 4096.0f;
      }
      uint u[4];
      for (int i = 0; i < 4; i++) {
        const uint lo = (m & (1u << i)) != 0 ? uint(wq[(8 * c + i) * 32]) : 0u;
        const uint hi = (m & (16u << i)) != 0 ? uint(wq[(8 * c + i + 4) * 32]) : 0u;
        u[i] = lo | (hi << 16);
      }
      nmz_swap(u[0], u[1], 4, 0x0f0f0f0fu);
      nmz_swap(u[2], u[3], 4, 0x0f0f0f0fu);
      nmz_swap(u[0], u[2], 8, 0x00ff00ffu);
      nmz_swap(u[1], u[3], 8, 0x00ff00ffu);
      const int gi = c / 8;
      const uint2 sr = *((const device uint2*)(sq + gi * OT));
      const uint2 br = *((const device uint2*)(bq + gi * OT));
      const uint sv[4] = {sr.x << 16, sr.x & 0xffff0000u, sr.y << 16, sr.y & 0xffff0000u};
      const uint bv[4] = {br.x << 16, br.x & 0xffff0000u, br.y << 16, br.y & 0xffff0000u};
      for (int o = 0; o < 4; o++) {
        const uint16_t ws[2] = {uint16_t(u[o] & 0xffffu), uint16_t(u[o] >> 16)};
        p[o] += nmz_qdot<U>(ws, xt, as_type<float>(sv[o]), as_type<float>(bv[o]), sum);
      }
    }
    for (int o = 0; o < 4; o++) {
      part[l * PS + 4 * lid + o] = p[o];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int q = 0; q < OT / SGS; q++) {
    const int n = sg * (OT / SGS) + q;
    const U r = simd_sum(part[lid * PS + n]);
    if (lid == 0) {
      const T h = nml_store<T, false>(r);
      prod[slot * N + n0 + n] = float(h) * scores[slot] + 0.0f;
    }
  }
"""

// The record's top-six reduce: products of one row in slot order.
private let nemotronZSkipReduceSource = """
  const uint n = thread_position_in_grid.x;
  const uint row = thread_position_in_grid.y;
  if (n >= N) {
    return;
  }
  float total = prod[(row * TOPK) * N + n];
  for (int j = 1; j < TOPK; j++) {
    total = prod[(row * TOPK + j) * N + n] + total;
  }
  y[row * N + n] = static_cast<T>(total);
"""

enum NemotronZSkip {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_MTP_ZSKIP"] != "0"
    /// Local hosts without room for the whole copy: zero-skip only the first N MoE layers.
    static let layerLimit = Int(ProcessInfo.processInfo.environment["NEMOTRON_MTP_ZSKIP_LAYERS"] ?? "") ?? Int.max

    struct Copy {
        let codes: MLXArray
        let scales: MLXArray
        let biases: MLXArray
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var copies: [ObjectIdentifier: Copy] = [:]

    private static let layout = MLXFast.metalKernel(
        name: "nemotron_zskip_kmajor_codes", inputNames: ["w"], outputNames: ["wk"],
        source: nemotronZSkipLayoutSource)
    private static let recordSlotKernel = MLXFast.metalKernel(
        name: "nemotron_zskip_fc2_slots", inputNames: ["wk", "sk", "bk", "x", "inds", "scores"],
        outputNames: ["prod"], source: nemotronZSkipSource,
        header: nemotronMoELeaderHeader + nemotronMoELeaderStore + nemotronZSkipHeader,
        ensureRowContiguous: true)
    private static let stagedSlotKernel = MLXFast.metalKernel(
        name: "nemotron_zskip_fc2_slots_staged", inputNames: ["wk", "sk", "bk", "x", "inds", "scores"],
        outputNames: ["prod"], source: nemotronZSkipStagedSource,
        header: nemotronMoELeaderHeader + nemotronMoELeaderStore + nemotronZSkipHeader,
        ensureRowContiguous: true)
    private static let tiledLayout = MLXFast.metalKernel(
        name: "nemotron_zskip_tiled_codes", inputNames: ["w"], outputNames: ["wk"],
        source: nemotronZSkipTiledLayoutSource)
    private static let tiledSlotKernel = MLXFast.metalKernel(
        name: "nemotron_zskip_fc2_slots_tiled", inputNames: ["wk", "sk", "bk", "x", "inds", "scores"],
        outputNames: ["prod"], source: nemotronZSkipTiledSeqSource,
        header: nemotronMoELeaderHeader + nemotronMoELeaderStore + nemotronZSkipHeader,
        ensureRowContiguous: true)
    /// The tile-major copy and its slot kernel unless NEMOTRON_MTP_ZSKIP_TILED=0.
    /// `prepare` falls back to the K-major copy and the staged kernel (and
    /// re-checks every layer) if the tiled kernel misses a bit check.
    nonisolated(unsafe) private static var tiled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_ZSKIP_TILED"] != "0"
    /// The staged slot kernel unless NEMOTRON_MTP_ZSKIP_STAGED=0. `prepare`
    /// also falls back to the original slot kernel (and re-checks every layer)
    /// if the staged one misses a bit check.
    nonisolated(unsafe) private static var staged =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_ZSKIP_STAGED"] != "0"
    private static var kernel: MLXFast.MLXFastKernel {
        tiled ? tiledSlotKernel : (staged ? stagedSlotKernel : recordSlotKernel)
    }
    private static let reduce = MLXFast.metalKernel(
        name: "nemotron_zskip_top6", inputNames: ["prod"], outputNames: ["y"],
        source: nemotronZSkipReduceSource)

    static func makeCopy(_ bank: NemotronMoELeader.Bank) -> Copy {
        let experts = bank.weight.dim(0)
        if tiled {
            let tiles = bank.n / 128, groups = bank.k / 64
            let codes = tiledLayout(
                [bank.weight], template: [("N", bank.n), ("K", bank.k)],
                grid: (32 * tiles, bank.k, experts), threadGroup: (32, 8, 1),
                outputShapes: [[experts, tiles, bank.k, 32]], outputDTypes: [.uint16])[0]
            func tileMajor(_ a: MLXArray) -> MLXArray {
                a.reshaped(experts, tiles, 128, groups).transposed(0, 1, 3, 2).contiguous()
            }
            return Copy(codes: codes, scales: tileMajor(bank.scales), biases: tileMajor(bank.biases))
        }
        let codes = layout(
            [bank.weight], template: [("N", bank.n), ("K", bank.k)],
            grid: (bank.n / 8, bank.k, experts), threadGroup: (32, 8, 1),
            outputShapes: [[experts, bank.k, bank.n / 8]], outputDTypes: [.uint32])[0]
        return Copy(codes: codes, scales: bank.scales.swappedAxes(1, 2).contiguous(),
                    biases: bank.biases.swappedAxes(1, 2).contiguous())
    }

    /// x: [rows * 6, 1856] relu2 rows per slot; inds, scores: [rows * 6].
    /// Returns [1, rows, 2688]. Every slot is its own threadgroup column, so
    /// the slot kernel is the same for any row count.
    static func run(_ x: MLXArray, _ inds: MLXArray, _ scores: MLXArray, copy: Copy,
                    rows: Int = 3) -> MLXArray {
        reduceProducts(products(x, inds, scores, copy: copy, rows: rows), rows: rows)
    }

    /// The slot kernel alone: [rows * 6 * 2688] FP32 weighted slot products.
    private static func products(_ x: MLXArray, _ inds: MLXArray, _ scores: MLXArray,
                                 copy: Copy, rows: Int) -> MLXArray {
        let slots = rows * 6
        return kernel(
            [copy.codes, copy.scales, copy.biases, x, inds, scores],
            template: [("T", x.dtype), ("K", 1856), ("N", 2688), ("SGS", 8)],
            grid: (256 * (2688 / 128), slots, 1), threadGroup: (256, 1, 1),
            outputShapes: [[slots * 2688]], outputDTypes: [.float32])[0]
    }

    /// The record's top-six reduce of `products` output: [1, rows, 2688] BF16.
    static func reduceProducts(_ prod: MLXArray, rows: Int) -> MLXArray {
        reduce(
            [prod], template: [("T", DType.bfloat16), ("TOPK", 6), ("N", 2688)],
            grid: (2688, rows, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1, rows, 2688]], outputDTypes: [.bfloat16])[0]
    }

    /// NEMOTRON_MTP_ZSKIP_JOIN=0 keeps the top-six reduce as its own launch in
    /// the captured verify; otherwise the MoE layer's join kernel sums the six
    /// slot products itself (`NemotronHResidualRMSNorm.applyPreparedRows`).
    static let joinReduce = ProcessInfo.processInfo.environment["NEMOTRON_MTP_ZSKIP_JOIN"] != "0"

    /// True when `owner` (a target layer's routed FC2) has a verified copy.
    static func covers(_ owner: SwitchLinear) -> Bool {
        lock.withLock { copies[ObjectIdentifier(owner)] != nil }
    }

    /// The R = 3 weighted slot products of a target layer ([18 * 2688] FP32,
    /// slot-major), or nil when `owner` has no verified copy.
    static func weightedProducts(_ x: MLXArray, _ inds: MLXArray, _ scores: MLXArray,
                                 owner: SwitchLinear) -> MLXArray? {
        guard joinReduce, NemotronHResidualRMSNorm.productsJoinVerified,
              x.shape == [18, 1856], x.dtype == .bfloat16,
              inds.shape == [18], scores.shape == [18], scores.dtype == .float32,
              let copy = lock.withLock({ copies[ObjectIdentifier(owner)] }) else { return nil }
        return products(x, inds, scores, copy: copy, rows: 3)
    }

    /// The R = 3 weighted fc2 of `owner`, or nil when it has no verified copy.
    static func weighted(_ x: MLXArray, _ inds: MLXArray, _ scores: MLXArray, rows: Int,
                         owner: SwitchLinear) -> MLXArray? {
        guard rows == 3, x.shape == [18, 1856], x.dtype == .bfloat16,
              inds.shape == [18], scores.shape == [18], scores.dtype == .float32,
              let copy = lock.withLock({ copies[ObjectIdentifier(owner)] }) else { return nil }
        return run(x, inds, scores, copy: copy)
    }

    /// Load time, before any verify graph is traced: copy every routed fc2 bank,
    /// then keep the copies only if every layer's scales and biases are finite
    /// (the skipped-chunk argument needs it) and its kernel equals the record
    /// kernel bit for bit on seeded relu2-like rows (sparse with an all-zero and
    /// a one-hot slot, tail chunks only, and dense; experts shared across rows).
    static func prepare(_ model: Module) {
        guard enabled, NemotronMoELeader.enabled, Device.defaultDevice().deviceType == .gpu else { return }
        var result = build(model)
        if case .kernelMiss = result, tiled {
            // The tiled kernel missed a check: the K-major copy, every layer again.
            tiled = false
            FileHandle.standardError.write(
                Data("nemotron zskip: tiled slot kernel differs; K-major copy\n".utf8))
            result = build(model)
        }
        if case .kernelMiss = result, staged {
            // The staged twin missed a check: the original slot kernel, every layer again.
            staged = false
            FileHandle.standardError.write(
                Data("nemotron zskip: staged slot kernel differs; original slot kernel\n".utf8))
            result = build(model)
        }
        switch result {
        case .built(let built):
            lock.withLock { copies = built }
            FileHandle.standardError.write(Data("nemotron zskip on: \(built.count) MoE layers\n".utf8))
        case .nonFinite(let layer):
            lock.withLock { copies = [:] }
            FileHandle.standardError.write(
                Data("nemotron zskip off: layer \(layer) has non-finite scales or biases\n".utf8))
        case .kernelMiss(let layer):
            lock.withLock { copies = [:] }
            FileHandle.standardError.write(
                Data("nemotron zskip off: layer \(layer) differs from the record kernel\n".utf8))
        }
    }

    private enum BuildResult {
        case built([ObjectIdentifier: Copy])
        case nonFinite(Int)
        case kernelMiss(Int)
    }

    private static func build(_ model: Module) -> BuildResult {
        var built: [ObjectIdentifier: Copy] = [:]
        for module in model.modules() {
            guard built.count < layerLimit, let moe = module as? NemotronHMoE,
                  moe.numExpertsPerTok == 6, type(of: moe.switchMLP.fc2) == QuantizedSwitchLinear.self,
                  let down = NemotronMoELeader.bank(moe.switchMLP.fc2, k: 1856, dtype: .bfloat16),
                  down.weight.shape == [128, 2688, 232], down.scales.shape == [128, 2688, 29],
                  down.biases.shape == down.scales.shape else { continue }
            let copy = makeCopy(down)
            eval(copy.codes, copy.scales, copy.biases)
            guard finite(down) else { return .nonFinite(built.count) }
            guard matches(down, copy, seed: built.count) else { return .kernelMiss(built.count) }
            built[ObjectIdentifier(moe.switchMLP.fc2)] = copy
        }
        return .built(built)
    }

    private static func finite(_ bank: NemotronMoELeader.Bank) -> Bool {
        let limit = MLXArray(Float.infinity).asType(.bfloat16)
        return logicalAnd((abs(bank.scales) .< limit).all(), (abs(bank.biases) .< limit).all()).item(Bool.self)
    }

    private static func matches(_ bank: NemotronMoELeader.Bank, _ copy: Copy, seed: Int) -> Bool {
        let offset = UInt32(seed) * 7
        var routes: [UInt32] = [3, 17, 42, 77, 100, 121, 17, 5, 64, 90, 3, 11, 3, 8, 50, 66, 101, 127]
        for i in routes.indices {
            routes[i] = (routes[i] + offset) % 128
        }
        let inds = MLXArray(routes)
        let keys = MLXRandom.split(key: MLXRandom.key(UInt64(0x25c0 + seed)), into: 3)
        let h = abs(MLXRandom.normal([18, 1856], key: keys[0])) + 1e-3
        let keep = MLXRandom.uniform(Float(0) ..< 1, [18, 1856], key: keys[1]) .< MLXArray(Float(0.08))
        let slot = MLXArray((0 ..< 18).map(Int32.init)).reshaped(18, 1)
        let column = MLXArray((0 ..< 1856).map(Int32.init)).reshaped(1, 1856)
        let zero = MLXArray(Float(0))
        let oneHot = which(column .== MLXArray(Int32(1800)), MLXArray(Float(1.5)), zero)
        let sparse = which(slot .== MLXArray(Int32(7)), zero,
                           which(slot .== MLXArray(Int32(8)), oneHot, which(keep, h, zero)))
        let tail = which(column .>= MLXArray(Int32(1792)), h, zero)
        let scores = MLXRandom.uniform(Float(0) ..< 1, [18], key: keys[2])
        var same = MLXArray(true)
        for a in [sparse.asType(.bfloat16), tail.asType(.bfloat16), h.asType(.bfloat16)] {
            let act = (a * a).asType(.bfloat16)
            let expected = NemotronMoELeader.applyWeighted(act, inds, scores, rows: 3, bank: bank)
            let got = run(act, inds, scores, copy: copy)
            same = logicalAnd(same, (got.view(dtype: .uint16) .== expected.view(dtype: .uint16)).all())
        }
        return same.item(Bool.self)
    }
}

// Selected only for native R3 prepared FC1. Public c486 grouping adapted to
// the winner's paired decode and a new triple decoder; paired FC2 stays intact.
private let nemotronMoETripleGateUpSource = """
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
  if ((R == 3 ? before : (before & 1)) != 0) {
    return;
  }
  int partner = -1;
  int third = -1;
  for (int r = R - 1; r > row; r--) {
    for (int j = 0; j < TOPK; j++) {
      if (inds[r * TOPK + j] == expert) {
        if (R == 3) { third = partner; }
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
        result[r] += nml_qdot<U, values_per_thread, R == 3>(wl, x_thread, s, b, sum);
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
        result[r] += nml_qdot_safe<U, values_per_thread, R == 3>(
            wl, x_thread, s, b, sum, remaining);
      }
    }
    for (int r = 0; r < results_per_simdgroup; r++) {
      result[r] = simd_sum(result[r]);
      if (simd_lid == 0) {
        y0[r] = nml_store<T, RELU2>(result[r]);
      }
    }
  } else if (third >= 0) {
    const device T* x1 = x + (XSLOT ? partner : partner / TOPK) * K +
        simd_lid * values_per_thread;
    const device T* x2 = x + (XSLOT ? third : third / TOPK) * K +
        simd_lid * values_per_thread;
    device T* y1 = y + partner * N + out_row;
    device T* y2 = y + third * N + out_row;
    thread U x0_thread[values_per_thread];
    thread U x1_thread[values_per_thread];
    thread U x2_thread[values_per_thread];
    thread U result0[results_per_simdgroup] = {0};
    thread U result1[results_per_simdgroup] = {0};
    thread U result2[results_per_simdgroup] = {0};
    int k = 0;
    for (; k < K - block_size; k += block_size) {
      U sum0 = nml_load_vector<T, U, values_per_thread>(x0, x0_thread);
      U sum1 = nml_load_vector<T, U, values_per_thread>(x1, x1_thread);
      U sum2 = nml_load_vector<T, U, values_per_thread>(x2, x2_thread);
      for (int r = 0; r < results_per_simdgroup; r++) {
        auto wl = (const device uint8_t*)(ws + r * in_vec_size_w);
        const device T* sl = sc + r * in_vec_size_g;
        const device T* bl = bs + r * in_vec_size_g;
        U s = sl[0];
        U b = bl[0];
        U value0, value1, value2;
        nml_qdot_triple<U, values_per_thread, R == 3>(
            wl, x0_thread, x1_thread, x2_thread, s, b, sum0, sum1, sum2,
            values_per_thread, value0, value1, value2);
        result0[r] += value0;
        result1[r] += value1;
        result2[r] += value2;
      }
      ws += block_size / 2;
      sc += block_size / 64;
      bs += block_size / 64;
      x0 += block_size;
      x1 += block_size;
      x2 += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(K - k - simd_lid * values_per_thread), 0, values_per_thread);
    if (remaining > 0) {
      U sum0 = nml_load_vector_safe<T, U, values_per_thread>(x0, x0_thread, remaining);
      U sum1 = nml_load_vector_safe<T, U, values_per_thread>(x1, x1_thread, remaining);
      U sum2 = nml_load_vector_safe<T, U, values_per_thread>(x2, x2_thread, remaining);
      for (int r = 0; r < results_per_simdgroup; r++) {
        auto wl = (const device uint8_t*)(ws + r * in_vec_size_w);
        const device T* sl = sc + r * in_vec_size_g;
        const device T* bl = bs + r * in_vec_size_g;
        U s = sl[0];
        U b = bl[0];
        U value0, value1, value2;
        nml_qdot_triple<U, values_per_thread, R == 3>(
            wl, x0_thread, x1_thread, x2_thread, s, b, sum0, sum1, sum2,
            remaining, value0, value1, value2);
        result0[r] += value0;
        result1[r] += value1;
        result2[r] += value2;
      }
    }
    for (int r = 0; r < results_per_simdgroup; r++) {
      result0[r] = simd_sum(result0[r]);
      result1[r] = simd_sum(result1[r]);
      result2[r] = simd_sum(result2[r]);
      if (simd_lid == 0) {
        y0[r] = nml_store<T, RELU2>(result0[r]);
        y1[r] = nml_store<T, RELU2>(result1[r]);
        y2[r] = nml_store<T, RELU2>(result2[r]);
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
        U value0, value1;
        nml_qdot_pair<U, values_per_thread, R == 3>(
            wl, x0_thread, x1_thread, s, b, sum0, sum1, values_per_thread, value0, value1);
        result0[r] += value0;
        result1[r] += value1;
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
        U value0, value1;
        nml_qdot_pair<U, values_per_thread, R == 3>(
            wl, x0_thread, x1_thread, s, b, sum0, sum1, remaining, value0, value1);
        result0[r] += value0;
        result1[r] += value1;
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


// Extend the new winner's shared coefficient decode to three independent
// FC1 rows; the singleton and pair branches retain its original helpers.
private let nemotronMoETripleDecodeHeader = """
template <typename U, int values_per_thread, bool HALF = false>
inline void nml_qdot_triple(
    const device uint8_t* w, const thread U* x0, const thread U* x1, const thread U* x2,
    U scale, U bias, U sum0, U sum1, U sum2, int N,
    thread U& value0, thread U& value1, thread U& value2) {
  U accum0 = 0;
  U accum1 = 0;
  U accum2 = 0;
  const device uint16_t* ws = (const device uint16_t*)w;
  if (HALF) {
    // nml_qdot_pair_half's decode and per-row term order, three rows.
    const device uint* ws2 = (const device uint*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      if (i < N / 4) {
        const uint w2 = ws2[i / 2];
        const half2 h0 = as_type<half2>((w2 & 0x000f000fu) | 0x64006400u) - half2(1024.0h);
        const half2 h1 = as_type<half2>((w2 & 0x00f000f0u) | 0x64006400u) - half2(1024.0h);
        const U a = U(h0[i % 2]);
        const U b1 = U(h1[i % 2]);
        const U c = U(half(ushort(ws[i] & 0x0f00)));
        const U d = U(half(ushort(ws[i] & 0xf000)));
        accum0 += (x0[4 * i] * a + x0[4 * i + 1] * b1 + x0[4 * i + 2] * c + x0[4 * i + 3] * d);
        accum1 += (x1[4 * i] * a + x1[4 * i + 1] * b1 + x1[4 * i + 2] * c + x1[4 * i + 3] * d);
        accum2 += (x2[4 * i] * a + x2[4 * i + 1] * b1 + x2[4 * i + 2] * c + x2[4 * i + 3] * d);
      }
    }
  } else {
    for (int i = 0; i < (N / 4); i++) {
      const uint16_t q = ws[i];
      const U a = q & 0x000f;
      const U b = q & 0x00f0;
      const U c = q & 0x0f00;
      const U d = q & 0xf000;
      accum0 += (x0[4 * i] * a + x0[4 * i + 1] * b + x0[4 * i + 2] * c + x0[4 * i + 3] * d);
      accum1 += (x1[4 * i] * a + x1[4 * i + 1] * b + x1[4 * i + 2] * c + x1[4 * i + 3] * d);
      accum2 += (x2[4 * i] * a + x2[4 * i + 1] * b + x2[4 * i + 2] * c + x2[4 * i + 3] * d);
    }
  }
  value0 = scale * accum0 + sum0 * bias;
  value1 = scale * accum1 + sum1 * bias;
  value2 = scale * accum2 + sum2 * bias;
}
"""
private let gauntletRedraw_35d0bcbe_20261025T151150Z: Int = 0
