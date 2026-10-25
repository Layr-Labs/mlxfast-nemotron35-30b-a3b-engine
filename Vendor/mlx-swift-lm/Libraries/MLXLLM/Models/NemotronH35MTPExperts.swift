import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Width-four verification is a separate experiment; depth remains declared
/// in the manifest. Disabling this restores the original generic R4 kernels.
enum NemotronMTPFourRows {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_MTP_FOUR_ROW_FUSION"] != "0"
}

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
        if let sharedExperts, let fused = NemotronRoutedSharedFourRows.apply(
            x, indices, scores, switchMLP, sharedExperts, prepared: prepared) {
            return parts ? [fused.0, fused.1] : [fused.0 + fused.1]
        }
        if let sharedExperts, let fused = NemotronRoutedSharedLaunch.apply(
            x, indices, scores, switchMLP, sharedExperts, prepared: packed) {
            return parts ? [fused.0, fused.1] : [fused.0 + fused.1]
        }
        var y: MLXArray
        if let reduced = NemotronMoELeader.switchRowsAndWeightedReduce(
            x, indices, scores, switchMLP, prepared: packed) {
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

private let nmlPreparedFC1Source = nmlPreparedSource(nemotronMoELeaderSource, pointers: [
    ("const device T* x0 = x + (XSLOT ? slot : row) * K + simd_lid * values_per_thread;",
     "const device float* x0 = prepared + row * (PTILES * PSTRIDE) + simd_lid;"),
    ("const device T* x1 = x + (XSLOT ? partner : partner / TOPK) * K +\n        simd_lid * values_per_thread;",
     "const device float* x1 = prepared + (partner / TOPK) * (PTILES * PSTRIDE) + simd_lid;"),
])


private final class NemotronMoELeaderPreparedKernels: Sendable {
    static let shared = NemotronMoELeaderPreparedKernels()
    let gateUp: MLXFast.MLXFastKernel?
    let gateUpAndPlan: MLXFast.MLXFastKernel?
    private init() {
        let header = nemotronMoELeaderHeader + nemotronMoELeaderStore + nemotronMoELeaderPreparedHeader
        let preparedSource = nmlPreparedFC1Source
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
    static let fuseBalancePlan =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_FC1_BALANCE_PLAN"] != "0"

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
              let kernel = NemotronMoELeaderPreparedKernels.shared.gateUpAndPlan
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
                              rows: Int, bank: Bank, plan: MLXArray? = nil) -> MLXArray {
        if rows == 3 { return NMLBalanceWeighted.shared.run(x, inds, scores, bank, plan: plan) }
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
        let balancePlan: MLXArray?
        if fuseRelu2 {
            if rows == 3, let fused = applyWithBalancePlan(
                xr, flat, bank: up, relu2: true, prepared: prepared) {
                act = fused.activation
                balancePlan = fused.plan
            } else {
                act = apply(xr, flat, rows: rows, topK: 6, bank: up, perSlot: false,
                    relu2: true, prepared: prepared)
                balancePlan = nil
            }
        } else {
            let h = apply(xr, flat, rows: rows, topK: 6, bank: up, perSlot: false,
                prepared: prepared)
            let a = maximum(h, MLXArray(0))
            act = a * a
            balancePlan = nil
        }
        var y = applyWeighted(act, flat, scores.reshaped(rows * 6),
            rows: rows, bank: down, plan: balancePlan)
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
                if rows > 1 {
                    outputs.append(apply(h, inds, rows: rows, topK: topK, bank: down, perSlot: true))
                }
                if topK == 6, up.weight.shape == [128, 1856, 336],
                    down.weight.shape == [128, 2688, 232] {
                    let scores = MLXArray.ones([rows * topK], dtype: .float32)
                    outputs.append(applyWeighted(h, inds, scores, rows: rows, bank: down,
                        plan: fused?.plan))
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
             _ bank: NemotronMoELeader.Bank, plan: MLXArray? = nil) -> MLXArray {
        let plan = plan ?? NMLBalancePlan.shared.make(ids)
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

// MARK: - Assistant-owned sparse FC2 and same-dispatch FC1 live mask
// The only caller is mtp.layers.1 after its verified weight load. Target MoE
// banks never enter this helper. The immutable neuron-major copy belongs to
// this assistant instance; current-call activations/masks are never cached.
private func nmlDraftFC1MaskSource() -> String {
    let store = "y0[r] = nml_store<T, RELU2>(result[r]);"
    precondition(nemotronMoELeaderSource.components(separatedBy: store).count == 2)
    let body = nemotronMoELeaderSource.replacingOccurrences(of: store, with: """
        const T stored = nml_store<T, RELU2>(result[r]);
        y0[r] = stored;
        live_bits |= (stored != static_cast<T>(0) ? 1u : 0u) << r;
        """)
    // R=1, RPS=4 and N divisible by8: both SIMD groups are always leaders,
    // own one four-output nibble each and reach the unconditional barrier.
    return """
      threadgroup uchar live_nibbles[2];
      uint live_bits = 0;
      """ + body + """

      if (simd_lid == 0) { live_nibbles[simd_gid] = uchar(live_bits); }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (simd_gid == 0 && simd_lid == 0) {
        mask[slot * (N / 8) + tid.y] = live_nibbles[0] | (live_nibbles[1] << 4);
      }
      """
}

private let nmlDraftNibbleQdotHeader = """
template <typename U, int values_per_thread>
inline U nml_qdot_nibbles(
    const thread uint* v, const thread U* x_thread, U scale, U bias, U sum) {
  U accum = 0;
  for (int i = 0; i < (values_per_thread / 4); i++) {
    const U a = U(v[4 * i]);
    const U b1 = U(v[4 * i + 1] << 4);
    const U c = U(v[4 * i + 2] << 8);
    const U d = U(v[4 * i + 3] << 12);
    accum +=
        (x_thread[4 * i] * a +
         x_thread[4 * i + 1] * b1 +
         x_thread[4 * i + 2] * c +
         x_thread[4 * i + 3] * d);
  }
  return scale * accum + sum * bias;
}
"""

// Adapted from838a38d4's sparse dot to the assistant's single row. Original
// lane/term/affine order, BF16 cast and top-six product order are retained.
// The producer mask skips an all-zero eight-input block BEFORE input, weight,
// scale or bias reads. NaNs compare nonzero in the producer and remain live.
private let nmlDraftSparseDownSource = """
  constexpr int values_per_thread = 8;
  constexpr int block_size = 256;
  constexpr int in_vec_size_g = K / 64;
  constexpr int wt_row = N / 8;
  constexpr int TILE = 64;
  typedef float U;
  const uint3 tid = threadgroup_position_in_grid;
  const uint t = thread_position_in_threadgroup.x;
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const uint simd_lid = thread_index_in_simdgroup;
  const int lane = int(t / 8);
  const int word = int(t % 8);
  const int tile_out = int(tid.x) * TILE;
  const int slot = int(tid.z);
  const uint expert = inds[slot];
  threadgroup float acc_tg[TILE * 32];
  const int out0 = tile_out + word * 8;
  const device uint32_t* wq = wt +
      ((ulong)expert * K + lane * values_per_thread) * wt_row + out0 / 8;
  const device T* sc = scales + ((ulong)expert * N + out0) * in_vec_size_g + lane / 8;
  const device T* bs = biases + ((ulong)expert * N + out0) * in_vec_size_g + lane / 8;
  const device T* x0 = x + slot * K + lane * values_per_thread;
  thread U result[8] = {0};
  for (int k = 0; k < K; k += block_size) {
    if (k + lane * values_per_thread < K) {
      const uchar active = mask[slot * (K / 8) + k / 8 + lane];
      if (active != 0) {
        thread U x_thread[values_per_thread];
        U sum = nml_load_vector<T, U, values_per_thread>(x0 + k, x_thread);
        uint q[values_per_thread];
        for (int j = 0; j < values_per_thread; j++) {
          q[j] = (active & (1u << j)) != 0 ? wq[(ulong)(k + j) * wt_row] : 0u;
        }
        for (int r = 0; r < 8; r++) {
          U s = sc[r * in_vec_size_g + k / 64];
          U b = bs[r * in_vec_size_g + k / 64];
          uint v[values_per_thread];
          for (int j = 0; j < values_per_thread; j++) {
            v[j] = (q[j] >> (4 * r)) & 0xfu;
          }
          result[r] += nml_qdot_nibbles<U, values_per_thread>(v, x_thread, s, b, sum);
        }
      }
    }
  }
  for (int r = 0; r < 8; r++) {
    acc_tg[(word * 8 + r) * 32 + lane] = result[r];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = 0; i < TILE / 8; i++) {
    const int o = int(simd_gid) * (TILE / 8) + i;
    U value = simd_sum(acc_tg[o * 32 + simd_lid]);
    if (simd_lid == 0) {
      const T h = nml_store<T, false>(value);
      products[slot * N + tile_out + o] = float(h) * scores[slot] + 0.0f;
    }
  }
"""

private let nmlDraftSparseReduceSource = """
  const uint o = thread_position_in_grid.x;
  if (o >= uint(N)) { return; }
  float total = products[o];
  for (int j = 1; j < 6; j++) { total = products[j * N + o] + total; }
  y[o] = static_cast<T>(total);
"""

/// Only the loaded assistant's FC1 may enter this private lossy copy. Keep
/// FC2 and its admitted 4-bit sparse/fold path intact instead of replacing
/// both banks and silently losing those optimizations.
private final class NemotronMTPDraftUp3: @unchecked Sendable {
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray
    private let inputRows = MLXArray.zeros([6], dtype: .uint32)
    var compilationState: [MLXArray] { [weight, scales, biases, inputRows] }

    private init(weight: MLXArray, scales: MLXArray, biases: MLXArray) {
        self.weight = weight
        self.scales = scales
        self.biases = biases
    }

    func project(_ x: MLXArray, _ ids: MLXArray) -> MLXArray {
        MLX.gatherQuantizedMM(x.reshaped(1, 2688), weight, scales: scales, biases: biases,
            lhsIndices: inputRows, rhsIndices: ids.reshaped(6), transpose: true,
            groupSize: 64, bits: 3, mode: .affine, sortedIndices: false).reshaped(6, 1856)
    }

    func referenceActivation(_ x: MLXArray, _ ids: MLXArray) -> MLXArray {
        let raw = project(x, ids)
        let positive = maximum(raw, MLXArray(0).asType(.bfloat16))
        return positive * positive
    }

    static func prepare(_ original: NemotronMoELeader.Bank) -> NemotronMTPDraftUp3? {
        guard (ProcessInfo.processInfo.environment["NEMOTRON_MTP_OWNED_UP_BITS"] ?? "3") == "3",
              original.k == 2688, original.n == 1856, original.weight.dtype == .uint32,
              original.weight.shape == [128, 1856, 336],
              original.scales.dtype == .bfloat16, original.biases.dtype == .bfloat16,
              original.scales.shape == [128, 1856, 42],
              original.biases.shape == original.scales.shape else { return nil }
        do {
            return try withError { () -> NemotronMTPDraftUp3? in
                var weights: [MLXArray] = [], scales: [MLXArray] = [], offsets: [MLXArray] = []
                // Bound dense temporaries to ONE expert, never the 128-expert
                // dense bank. Native packing handles 3-bit word crossings.
                for expert in 0..<128 {
                    let dense = dequantized(original.weight[expert],
                        scales: original.scales[expert], biases: original.biases[expert],
                        groupSize: 64, bits: 4, mode: .affine)
                    let next = MLX.quantized(dense, groupSize: 64, bits: 3, mode: .affine)
                    guard let bias = next.biases,
                          next.wq.dtype == .uint32, next.wq.shape == [1856, 252],
                          next.scales.dtype == .bfloat16, next.scales.shape == [1856, 42],
                          bias.dtype == .bfloat16, bias.shape == next.scales.shape else { return nil }
                    eval([next.wq, next.scales, bias])
                    let rebuilt = dequantized(next.wq, scales: next.scales, biases: bias,
                        groupSize: 64, bits: 3, mode: .affine)
                    let source = dense.asType(.float32)
                    let difference = rebuilt.asType(.float32) - source
                    let energy = (source * source).sum().item(Float.self)
                    let error = (difference * difference).sum().item(Float.self)
                    guard energy.isFinite, energy > 0, error.isFinite,
                          error <= energy * 0.0625, isFinite(rebuilt).all().item(Bool.self)
                    else { return nil }
                    weights.append(next.wq.expandedDimensions(axis: 0))
                    scales.append(next.scales.expandedDimensions(axis: 0))
                    offsets.append(bias.expandedDimensions(axis: 0))
                }
                let plan = NemotronMTPDraftUp3(weight: concatenated(weights, axis: 0),
                    scales: concatenated(scales, axis: 0), biases: concatenated(offsets, axis: 0))
                eval(plan.compilationState)
                return plan
            }
        } catch { return nil }
    }
}

private let nmlDraftNativeUpMaskSource = """
  const uint byte = thread_position_in_grid.x;
  if (byte >= uint(6 * (N / 8))) { return; }
  const uint base = byte * 8;
  uchar active = 0;
  for (uint j = 0; j < 8; j++) {
    const T stored = nml_store<T, true>(float(raw[base + j]));
    y[base + j] = stored;
    active |= uchar(stored != static_cast<T>(0) ? 1u << j : 0u);
  }
  mask[byte] = active;
"""

// Pinned native affine3 QMV helpers and dot order; assistant-only JIT.
private let nmlDraftUp3NativeHeader = """
// Copyright © 2023-2024 Apple Inc.
// Complete pinned native helpers; unique symbols only, arithmetic unchanged.
static constant constexpr const int NML3_SIMD_SIZE = 32;
template <int bits, int wsize = 8>
inline constexpr short nml3_get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

template <int bits, int wsize = 8>
inline constexpr short nml3_get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

template <typename T, typename U, int values_per_thread, int bits>
inline U nml3_load_vector(const device T* x, thread U* x_thread) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U sum = 0;

  if (bits == 2) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += U(x[i]) + U(x[i + 1]) + U(x[i + 2]) + U(x[i + 3]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += U(x[i]) + U(x[i + 1]) + U(x[i + 2]) + U(x[i + 3]) + U(x[i + 4]) +
          U(x[i + 5]) + U(x[i + 6]) + U(x[i + 7]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += U(x[i]) + U(x[i + 1]) + U(x[i + 2]) + U(x[i + 3]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += U(x[i]) + U(x[i + 1]) + U(x[i + 2]) + U(x[i + 3]) + U(x[i + 4]) +
          U(x[i + 5]) + U(x[i + 6]) + U(x[i + 7]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += U(x[i]) + U(x[i + 1]) + U(x[i + 2]) + U(x[i + 3]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }

  return sum;
}

template <typename T, typename U, int values_per_thread, int bits>
inline U nml3_load_vector_safe(const device T* x, thread U* x_thread, int N) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U sum = 0;

  if (bits == 2) {
    for (int i = 0; i < N; i += 4) {
      sum += U(x[i]) + U(x[i + 1]) + U(x[i + 2]) + U(x[i + 3]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < N; i += 8) {
      sum += U(x[i]) + U(x[i + 1]) + U(x[i + 2]) + U(x[i + 3]) + U(x[i + 4]) +
          U(x[i + 5]) + U(x[i + 6]) + U(x[i + 7]);

      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < N; i += 4) {
      sum += U(x[i]) + U(x[i + 1]) + U(x[i + 2]) + U(x[i + 3]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < N; i += 8) {
      sum += U(x[i]) + U(x[i + 1]) + U(x[i + 2]) + U(x[i + 3]) + U(x[i + 4]) +
          U(x[i + 5]) + U(x[i + 6]) + U(x[i + 7]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < N; i += 4) {
      sum += U(x[i]) + U(x[i + 1]) + U(x[i + 2]) + U(x[i + 3]);
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }

  for (int i = N; i < values_per_thread; i++) {
    x_thread[i] = 0;
  }

  return sum;
}

template <typename U, int values_per_thread, int bits>
inline U nml3_qdot(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U accum = 0;

  if (bits == 2) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (w[i] & 0x03) +
           x_thread[4 * i + 1] * (w[i] & 0x0c) +
           x_thread[4 * i + 2] * (w[i] & 0x30) +
           x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;

      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

template <typename U, int values_per_thread, int bits>
inline U nml3_qdot_safe(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum,
    int N) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U accum = 0;

  if (bits == 2) {
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * (w[i] & 0x03) +
           x_thread[4 * i + 1] * (w[i] & 0x0c) +
           x_thread[4 * i + 2] * (w[i] & 0x30) +
           x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (N / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (N / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;

      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (N / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

"""

private let nmlDraftUp3FusedSource = """
  constexpr int bits = 3;
  constexpr int group_size = 64;
  constexpr int in_vec_size = K;
  constexpr int out_vec_size = N;
  const uint3 tid = threadgroup_position_in_grid;
  const uint simd_gid = simdgroup_index_in_threadgroup;
  const uint simd_lid = thread_index_in_simdgroup;
  const uint slot = tid.z;
  const uint expert = inds[slot];
  w += (ulong)expert * N * (K * 3 / 32);
  scales += (ulong)expert * N * (K / 64);
  biases += (ulong)expert * N * (K / 64);
  y += slot * N;
  threadgroup uchar live_nibbles[2];
  uint live_bits = 0;
  constexpr int num_simdgroups = 2;
  constexpr int results_per_simdgroup = 4;
  constexpr int packs_per_thread = 1;
  constexpr int pack_factor = nml3_get_pack_factor<bits, 32>();
  constexpr int bytes_per_pack = nml3_get_bytes_per_pack<bits, 32>();

  constexpr int values_per_thread = pack_factor * packs_per_thread;
  constexpr int block_size = values_per_thread * NML3_SIMD_SIZE;
  constexpr int scale_step_per_thread = group_size / values_per_thread;

  const device uint8_t* ws = (const device uint8_t*)w;

  typedef float U;

  thread U x_thread[values_per_thread];
  thread U result[results_per_simdgroup] = {0};

  // Adjust positions
  const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
  const int in_vec_size_g = in_vec_size / group_size;
  const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) +
      simd_gid * results_per_simdgroup;
  const int used_out_row = min(out_vec_size - results_per_simdgroup, out_row);

  if (out_row >= out_vec_size) {
    return;
  }

  // In this case we need to properly guard all our reads because there isn't
  // even 1 tile in the matrix
  if (out_vec_size < (num_simdgroups * results_per_simdgroup)) {
    ws +=
        out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
    scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    x += tid.x * in_vec_size + simd_lid * values_per_thread;
    y += tid.x * out_vec_size + out_row;

    int k = 0;
    for (; k < in_vec_size - block_size; k += block_size) {
      U sum = nml3_load_vector<T, U, values_per_thread, bits>(x, x_thread);

      for (int row = 0;
           row < results_per_simdgroup && out_row + row < out_vec_size;
           row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] +=
            nml3_qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
      }

      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      biases += block_size / group_size;
      x += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
        0,
        values_per_thread);
    if (remaining > 0) {
      U sum = nml3_load_vector_safe<T, U, values_per_thread, bits>(
          x, x_thread, remaining);

      for (int row = 0;
           row < results_per_simdgroup && out_row + row < out_vec_size;
           row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] += nml3_qdot_safe<U, values_per_thread, bits>(
            wl, x_thread, s, b, sum, remaining);
      }
    }

    for (int row = 0;
         row < results_per_simdgroup && out_row + row < out_vec_size;
         row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        const T stored = nml_store<T, true>(result[row]);
        y[row] = stored;
        live_bits |= (stored != static_cast<T>(0) ? 1u : 0u) << row;
      }
    }
  }

  // In this case the last tile is moved back to redo some output values
  else {
    ws += used_out_row * in_vec_size_w +
        simd_lid * packs_per_thread * bytes_per_pack;
    scales += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    biases += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    x += tid.x * in_vec_size + simd_lid * values_per_thread;
    y += tid.x * out_vec_size + used_out_row;

    int k = 0;
    for (; k < in_vec_size - block_size; k += block_size) {
      U sum = nml3_load_vector<T, U, values_per_thread, bits>(x, x_thread);

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] +=
            nml3_qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
      }

      ws += block_size * bytes_per_pack / pack_factor;
      scales += block_size / group_size;
      biases += block_size / group_size;
      x += block_size;
    }
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - simd_lid * values_per_thread),
        0,
        values_per_thread);
    if (remaining > 0) {
      U sum = nml3_load_vector_safe<T, U, values_per_thread, bits>(
          x, x_thread, remaining);

      for (int row = 0; row < results_per_simdgroup; row++) {
        auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
        const device T* sl = scales + row * in_vec_size_g;
        const device T* bl = biases + row * in_vec_size_g;

        U s = sl[0];
        U b = bl[0];
        result[row] += nml3_qdot_safe<U, values_per_thread, bits>(
            wl, x_thread, s, b, sum, remaining);
      }
    }
    for (int row = 0; row < results_per_simdgroup; row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        const T stored = nml_store<T, true>(result[row]);
        y[row] = stored;
        live_bits |= (stored != static_cast<T>(0) ? 1u : 0u) << row;
      }
    }
  }
  // N=1856 is divisible by8: both SIMD groups always own four rows,
  // so no original out-of-range early return can bypass this TG barrier.
  if (simd_lid == 0) { live_nibbles[simd_gid] = uchar(live_bits); }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0 && simd_lid == 0) {
    mask[slot * (N / 8) + tid.y] = live_nibbles[0] | (live_nibbles[1] << 4);
  }
"""

private final class NemotronMTPDraftSparseKernels: Sendable {
    static let shared = NemotronMTPDraftSparseKernels()
    let fc1 = MLXFast.metalKernel(
        name: "nemotron_mtp_owned_fc1_relu2_live_mask",
        inputNames: ["w", "scales", "biases", "x", "inds"],
        outputNames: ["y", "mask"], source: nmlDraftFC1MaskSource(),
        header: nemotronMoELeaderHeader + nemotronMoELeaderStore)
    let nativeUpMask = MLXFast.metalKernel(
        name: "nemotron_mtp_native_up_relu2_live_mask",
        inputNames: ["raw"], outputNames: ["y", "mask"], source: nmlDraftNativeUpMaskSource,
        header: nemotronMoELeaderStore, ensureRowContiguous: true)
    let fusedUp3 = MLXFast.metalKernel(
        name: "nemotron_mtp_owned_up3_native_dot_relu2_live_mask",
        inputNames: ["w", "scales", "biases", "x", "inds"], outputNames: ["y", "mask"],
        source: nmlDraftUp3FusedSource, header: nmlDraftUp3NativeHeader + nemotronMoELeaderStore,
        ensureRowContiguous: true)
    let partials = MLXFast.metalKernel(
        name: "nemotron_mtp_owned_fc2_sparse_mask_partials",
        inputNames: ["wt", "scales", "biases", "x", "inds", "scores", "mask"],
        outputNames: ["products"], source: nmlDraftSparseDownSource,
        header: nemotronMoELeaderHeader + nemotronMoELeaderStore + nmlDraftNibbleQdotHeader,
        ensureRowContiguous: true)
    let reduce = MLXFast.metalKernel(
        name: "nemotron_mtp_owned_fc2_sparse_top6",
        inputNames: ["products"], outputNames: ["y"], source: nmlDraftSparseReduceSource,
        ensureRowContiguous: true)
}

/// Only immutable assistant parameters live here, never target parameters or
/// request outputs. Preparation and admission occur before expert-tail tracing.
final class NemotronMTPDraftSparseDown: @unchecked Sendable {
    static let enabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_OWNED_SPARSE_FC2"] != "0"
    private let ownerIdentity: [ObjectIdentifier]
    private let up: NemotronMoELeader.Bank
    private let down: NemotronMoELeader.Bank
    private let ownedUp3: NemotronMTPDraftUp3?
    let neuronMajor: MLXArray
    var compilationState: [MLXArray] {
        var arrays: [MLXArray] = [neuronMajor]
        if let ownedUp3 { arrays.append(contentsOf: ownedUp3.compilationState) }
        return arrays
    }
    var usesThreeBitUp: Bool { ownedUp3 != nil }
    // Finalized during load admission, before any compiler trace/publication.
    private(set) var fusedUp3Qualified = false
    // Written only during private load admission, before any compiled tail.
    private(set) var foldJoinQualified = false

    private init(_ mlp: NemotronHSwitchMLP, up: NemotronMoELeader.Bank,
                 down: NemotronMoELeader.Bank, neuronMajor: MLXArray,
                 ownedUp3: NemotronMTPDraftUp3? = nil) {
        self.ownerIdentity = [ObjectIdentifier(mlp), ObjectIdentifier(mlp.fc1), ObjectIdentifier(mlp.fc2)]
        self.up = up
        self.down = down
        self.ownedUp3 = ownedUp3
        self.neuronMajor = neuronMajor
    }

    func compatible(_ mlp: NemotronHSwitchMLP) -> Bool {
        ownerIdentity == [ObjectIdentifier(mlp), ObjectIdentifier(mlp.fc1), ObjectIdentifier(mlp.fc2)]
    }

    private func activate(_ x: MLXArray, _ ids: MLXArray) -> [MLXArray] {
        if let ownedUp3 {
            if fusedUp3Qualified { return fusedThreeBitActivation(x, ids, ownedUp3) }
            let raw = ownedUp3.project(x, ids)
            return NemotronMTPDraftSparseKernels.shared.nativeUpMask(
                [raw], template: [("T", x.dtype), ("N", up.n)],
                grid: (6 * (up.n / 8), 1, 1), threadGroup: (128, 1, 1),
                outputShapes: [[6, up.n], [6, up.n / 8]], outputDTypes: [x.dtype, .uint8])
        }
        return NemotronMTPDraftSparseKernels.shared.fc1(
            [up.weight, up.scales, up.biases, x, ids],
            template: [("T", x.dtype), ("R", 1), ("TOPK", 6), ("K", up.k),
                       ("N", up.n), ("XSLOT", false), ("RELU2", true), ("RPS", 4)],
            grid: (32, up.n / 4, 6), threadGroup: (32, 2, 1),
            outputShapes: [[6, up.n], [6, up.n / 8]], outputDTypes: [x.dtype, .uint8])
    }

    private func fusedThreeBitActivation(
        _ x: MLXArray, _ ids: MLXArray, _ bank: NemotronMTPDraftUp3
    ) -> [MLXArray] {
        NemotronMTPDraftSparseKernels.shared.fusedUp3(
            [bank.weight, bank.scales, bank.biases, x, ids],
            template: [("T", DType.bfloat16), ("K", 2688), ("N", 1856)],
            grid: (32, 1856 / 4, 6), threadGroup: (32, 2, 1),
            outputShapes: [[6, 1856], [6, 232]], outputDTypes: [.bfloat16, .uint8])
    }

    /// Same native3-bit reference as the admitted two-launch path. This is
    /// an implementation check, not equality to original4-bit proposals.
    private func qualifyFusedUp3() -> Bool {
        guard let ownedUp3,
              ProcessInfo.processInfo.environment["NEMOTRON_MTP_UP3_FUSED_MASK"] != "0"
        else { return false }
        do {
            return try withError {
                let probes: [MLXArray] = [
                    MLXArray.zeros([1, 2688], dtype: .bfloat16),
                    MLXArray((0..<2688).map { Float(($0 % 17) - 8) / 16 })
                        .asType(.bfloat16).reshaped(1, 2688),
                    MLXArray((0..<2688).map { $0 % 2 == 0 ? Float(0.125) : Float(-0.375) })
                        .asType(.bfloat16).reshaped(1, 2688),
                    MLXRandom.normal([1, 2688], dtype: .bfloat16,
                        key: MLXRandom.key(UInt64(0x3bf051))),
                ]
                for first in stride(from: 0, to: 128, by: 6) {
                    let ids = MLXArray((0..<6).map { UInt32((first + $0) % 128) })
                    for x in probes {
                        let expected = ownedUp3.referenceActivation(x, ids)
                        let got = fusedThreeBitActivation(x, ids, ownedUp3)
                        var expectedMask = MLXArray.zeros([6, 232], dtype: .uint32)
                        let live = (expected.reshaped(6, 232, 8) .!= MLXArray(0)).asType(.uint32)
                        for j in 0..<8 {
                            expectedMask = bitwiseOr(expectedMask,
                                live[.ellipsis, j] << MLXArray(UInt32(j)))
                        }
                        guard isFinite(expected).all().item(Bool.self),
                              (got[0].view(dtype: .uint16) .== expected.view(dtype: .uint16))
                                .all().item(Bool.self),
                              (got[1] .== expectedMask.asType(.uint8)).all().item(Bool.self)
                        else { return false }
                    }
                }
                return true
            }
        } catch { return false }
    }

    private func partials(_ activation: MLXArray, _ mask: MLXArray,
                          _ ids: MLXArray, _ scores: MLXArray) -> MLXArray {
        NemotronMTPDraftSparseKernels.shared.partials(
            [neuronMajor, down.scales, down.biases, activation, ids, scores, mask],
            template: [("T", activation.dtype), ("K", down.k), ("N", down.n)],
            grid: (256 * (down.n / 64), 1, 6), threadGroup: (256, 1, 1),
            outputShapes: [[6, down.n]], outputDTypes: [.float32])[0]
    }

    private func reduceProducts(_ products: MLXArray, dtype: DType) -> MLXArray {
        NemotronMTPDraftSparseKernels.shared.reduce(
            [products], template: [("T", dtype), ("N", down.n)],
            grid: (down.n, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [[1, 1, down.n]], outputDTypes: [dtype])[0]
    }

    private func reduce(_ activation: MLXArray, _ mask: MLXArray,
                        _ ids: MLXArray, _ scores: MLXArray) -> MLXArray {
        reduceProducts(partials(activation, mask, ids, scores), dtype: activation.dtype)
    }

    func productsForJoinedTail(_ h: MLXArray, _ indices: MLXArray, _ scores: MLXArray,
                               _ mlp: NemotronHSwitchMLP) -> MLXArray? {
        guard foldJoinQualified, compatible(mlp),
              h.shape == [1, 1, 2688], h.dtype == .bfloat16,
              indices.shape == [1, 1, 6], indices.dtype == .uint32,
              scores.shape == indices.shape, scores.dtype == .float32 else { return nil }
        let ids = indices.reshaped(6)
        let a = activate(h.reshaped(1, 2688), ids)
        return partials(a[0], a[1], ids, scores.reshaped(6))
    }

    func run(_ h: MLXArray, _ indices: MLXArray, _ scores: MLXArray,
             _ mlp: NemotronHSwitchMLP) -> MLXArray? {
        guard compatible(mlp), h.shape == [1, 1, 2688], h.dtype == .bfloat16,
              indices.shape == [1, 1, 6], indices.dtype == .uint32,
              scores.shape == indices.shape, scores.dtype == .float32 else { return nil }
        let ids = indices.reshaped(6)
        let a = activate(h.reshaped(1, 2688), ids)
        return reduce(a[0], a[1], ids, scores.reshaped(6))
    }

    /// Called exclusively on the loaded assistant's mtp.layers.1. All array
    /// execution below is official-worker load-time admission, never local.
    static func prepare(_ mlp: NemotronHSwitchMLP, finalNorm: RMSNorm)
        -> NemotronMTPDraftSparseDown? {
        guard enabled, NemotronMoELeader.enabled, NemotronMoELeader.fuseRelu2,
              !NemotronMoELeader.selfTest,
              Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              type(of: mlp) == NemotronHSwitchMLP.self,
              type(of: mlp.fc1) == QuantizedSwitchLinear.self,
              type(of: mlp.fc2) == QuantizedSwitchLinear.self,
              let up = NemotronMoELeader.bank(mlp.fc1, k: 2688, dtype: .bfloat16),
              let down = NemotronMoELeader.bank(mlp.fc2, k: 1856, dtype: .bfloat16),
              up.weight.shape == [128, 1856, 336], up.scales.shape == [128, 1856, 42],
              up.biases.shape == up.scales.shape,
              down.weight.shape == [128, 2688, 232], down.scales.shape == [128, 2688, 29],
              down.biases.shape == down.scales.shape else { return nil }
        do {
            return try withError { () -> NemotronMTPDraftSparseDown? in
                let shifts = MLXArray((0..<8).map { UInt32(4 * $0) })
                let nibbleMask = MLXArray(UInt32(15))
                var chunks: [MLXArray] = []
                for start in stride(from: 0, to: 128, by: 8) {
                    let nibbles = bitwiseAnd(
                        down.weight[start..<(start + 8)].expandedDimensions(axis: -1) >> shifts,
                        nibbleMask).asType(.uint8).reshaped(8, down.n, down.k)
                    let columns = nibbles.transposed(0, 2, 1)
                        .reshaped(8, down.k, down.n / 8, 8).asType(.uint32)
                    var packed = columns[.ellipsis, 0]
                    for j in 1..<8 {
                        packed = bitwiseOr(packed, columns[.ellipsis, j] << MLXArray(UInt32(4 * j)))
                    }
                    eval(packed)
                    chunks.append(packed)
                }
                let packed = concatenated(chunks, axis: 0)
                eval(packed)
                let candidate = NemotronMTPDraftSparseDown(mlp, up: up, down: down, neuronMajor: packed)
                guard candidate.qualify() else { return nil }
                candidate.foldJoinQualified = candidate.qualifyFoldJoin(norm: finalNorm)
                // Keep the complete admitted 4-bit plan available until the
                // optional lossy FC1 copy and its own implementation checks
                // pass. Any error/decline returns that original sparse plan.
                if let lowBitUp = NemotronMTPDraftUp3.prepare(up) {
                    do {
                        let experimental = try withError { () -> NemotronMTPDraftSparseDown? in
                            let plan = NemotronMTPDraftSparseDown(mlp, up: up, down: down,
                                neuronMajor: packed, ownedUp3: lowBitUp)
                            guard plan.qualify() else { return nil }
                            plan.foldJoinQualified = plan.qualifyFoldJoin(norm: finalNorm)
                            let nativeFold = plan.foldJoinQualified
                            plan.fusedUp3Qualified = plan.qualifyFusedUp3()
                            if plan.fusedUp3Qualified {
                                let fusedFold = plan.qualifyFoldJoin(norm: finalNorm)
                                if nativeFold && !fusedFold {
                                    plan.fusedUp3Qualified = false
                                } else {
                                    plan.foldJoinQualified = fusedFold
                                }
                            }
                            guard !candidate.foldJoinQualified || plan.foldJoinQualified
                            else { return nil }
                            return plan
                        }
                        if let experimental { return experimental }
                    } catch { /* retain candidate's original 4-bit sparse path */ }
                }
                return candidate
            }
        } catch {
            // A compiler/kernel/admission error keeps the complete original
            // exact one-token tail; no process-wide weight cache is installed.
            return nil
        }
    }

    /// An independent worker-only comparison of the fused fold/three-input
    /// join against the already admitted sparse reduce and original join.
    /// Failure leaves sparse FC2 plus the original separate join in use.
    private func qualifyFoldJoin(norm: RMSNorm) -> Bool {
        guard NemotronHResidualRMSNorm.draftExpertFold,
              type(of: norm) == RMSNorm.self,
              norm.weight.shape == [2688], norm.weight.dtype == .bfloat16 else { return false }
        do {
            return try withError {
                let residual = MLXArray((0..<2688).map { Float(($0 % 13) - 6) / 32 })
                    .asType(.bfloat16).reshaped(1, 1, 2688)
                let shared = MLXArray((0..<2688).map { Float(($0 % 19) - 9) / 64 })
                    .asType(.bfloat16).reshaped(1, 1, 2688)
                let scores = MLXArray([Float(0.05), 0.1, 0.15, 0.2, 0.225, 0.275])
                for first in stride(from: 0, to: 128, by: 6) {
                    let ids = MLXArray((0..<6).map { UInt32((first + $0) % 128) })
                    for probe in 0..<3 {
                        let x = MLXArray((0..<up.k).map { i -> Float in
                            Float(((i + 3 * probe) % 17) - 8) / 16
                        }).asType(.bfloat16).reshaped(1, up.k)
                        let a = activate(x, ids)
                        let products = partials(a[0], a[1], ids, scores)
                        let routed = reduceProducts(products, dtype: .bfloat16)
                        guard let expected = NemotronHResidualRMSNorm.applySum(
                            residual, routed, shared: shared, norm: norm),
                              let actual = NemotronHResidualRMSNorm.applyDraftExpertFold(
                            residual, products: products, shared: shared, norm: norm)
                        else { return false }
                        let same = MLX.logicalAnd(
                            (actual.0.view(dtype: .uint16) .== expected.0.view(dtype: .uint16)).all(),
                            (actual.1.view(dtype: .uint16) .== expected.1.view(dtype: .uint16)).all())
                        if !same.item(Bool.self) { return false }
                    }
                }
                let zero = MLXArray.zeros([6, 2688], dtype: .float32)
                let routed = reduceProducts(zero, dtype: .bfloat16)
                guard let expected = NemotronHResidualRMSNorm.applySum(
                    residual, routed, shared: shared, norm: norm),
                      let actual = NemotronHResidualRMSNorm.applyDraftExpertFold(
                    residual, products: zero, shared: shared, norm: norm) else { return false }
                return MLX.logicalAnd(
                    (actual.0.view(dtype: .uint16) .== expected.0.view(dtype: .uint16)).all(),
                    (actual.1.view(dtype: .uint16) .== expected.1.view(dtype: .uint16)).all()).item(Bool.self)
            }
        } catch {
            return false
        }
    }

    private func qualify() -> Bool {
        let scores = MLXArray([Float(0.05), 0.1, 0.15, 0.2, 0.225, 0.275])
        let probes: [MLXArray] = [
            MLXArray.zeros([1, up.k], dtype: .bfloat16),
            MLXArray((0..<up.k).map { $0 % 2 == 0 ? Float(0.125) : Float(-0.375) })
                .asType(.bfloat16).reshaped(1, up.k),
            MLXArray((0..<up.k).map { Float(($0 % 17) - 8) / 16 })
                .asType(.bfloat16).reshaped(1, up.k),
            MLXRandom.normal([1, up.k], dtype: .bfloat16,
                key: MLXRandom.key(UInt64(0x3bfc1))),
        ]
        // Every expert id is covered; duplicates in the final six slots are
        // harmless at R=1, whose FC1/sparse FC2 never deduplicate within a row.
        for first in stride(from: 0, to: 128, by: 6) {
            let ids = MLXArray((0..<6).map { UInt32((first + $0) % 128) })
            for x in probes {
                let referenceAct: MLXArray
                if let ownedUp3 {
                    // This admits implementation against the SAME lossy
                    // native 3-bit FC1, not equivalence to 4-bit proposals.
                    referenceAct = ownedUp3.referenceActivation(x, ids)
                    guard isFinite(referenceAct).all().item(Bool.self) else { return false }
                } else {
                    referenceAct = NemotronMoELeader.apply(
                        x, ids, rows: 1, topK: 6, bank: up, perSlot: false, relu2: true)
                }
                let got = activate(x, ids)
                var expectedMask = MLXArray.zeros([6, up.n / 8], dtype: .uint32)
                let live = (referenceAct.reshaped(6, up.n / 8, 8) .!= MLXArray(0)).asType(.uint32)
                for j in 0..<8 {
                    expectedMask = bitwiseOr(expectedMask, live[.ellipsis, j] << MLXArray(UInt32(j)))
                }
                let reference: MLXArray
                if ownedUp3 != nil {
                    let rows = MLXArray((0..<6).map { UInt32($0) })
                    let nativeDown = MLX.gatherQuantizedMM(
                        referenceAct.reshaped(6, 1, down.k), down.weight,
                        scales: down.scales, biases: down.biases, lhsIndices: rows,
                        rhsIndices: ids, transpose: true, groupSize: 64, bits: 4,
                        mode: .affine, sortedIndices: false).reshaped(1, 1, 6, down.n)
                    reference = (nativeDown * scores.reshaped(1, 1, 6, 1))
                        .sum(axis: -2).asType(.bfloat16)
                } else {
                    reference = NemotronMoELeader.applyWeighted(
                        referenceAct, ids, scores, rows: 1, bank: down)
                }
                let actual = reduce(got[0], got[1], ids, scores)
                let equal = MLX.logicalAnd(
                    (got[0].view(dtype: .uint16) .== referenceAct.view(dtype: .uint16)).all(),
                    MLX.logicalAnd((got[1] .== expectedMask.asType(.uint8)).all(),
                        (actual.view(dtype: .uint16) .== reference.view(dtype: .uint16)).all()))
                if !equal.item(Bool.self) { return false }
            }
        }
        // Exercise the completely dead mask independently of FC1's bias.
        let zero = MLXArray.zeros([6, down.k], dtype: .bfloat16)
        let mask = MLXArray.zeros([6, down.k / 8], dtype: .uint8)
        let ids = MLXArray((0..<6).map { UInt32($0) })
        let reference = NemotronMoELeader.applyWeighted(zero, ids, scores, rows: 1, bank: down)
        let actual = reduce(zero, mask, ids, scores)
        return (actual.view(dtype: .uint16) .== reference.view(dtype: .uint16)).all().item(Bool.self)
    }
}


// Shared/routed launch composition from authenticated pending61.
// Exact source3cc packing arithmetic; only output pointer is renamed.
private let nmlSharedPackedEpilogue = """
          threadgroup_barrier(mem_flags::mem_threadgroup);
          const uint local = simd_gid * 32 + simd_lid;
          if (local < R) {
            const int r = int(local);
            const int chunk = int(tid.y);
            const int tiles = (out_vec_size / 8 + 31) / 32;
            const int base = (r * tiles + chunk / 32) * 9 * 32 + chunk % 32;
            float total = 0;
            for (int i = 0; i < 8; i += 4) {
              total += float(values[r * 8 + i]) + float(values[r * 8 + i + 1])
                  + float(values[r * 8 + i + 2]) + float(values[r * 8 + i + 3]);
              spacked[base + i * 32] = float(values[r * 8 + i]);
              spacked[base + (i + 1) * 32] = float(values[r * 8 + i + 1]) / 16.0f;
              spacked[base + (i + 2) * 32] = float(values[r * 8 + i + 2]) / 256.0f;
              spacked[base + (i + 3) * 32] = float(values[r * 8 + i + 3]) / 4096.0f;
            }
            spacked[base + 8 * 32] = total;
          }
"""

private func nmlSharedExpertBranch(condition: String, tile: String, simdgroups: Int,
                                   prepared: String?, packSharedUp: Bool = false) -> String {
    let groups = "constexpr int num_simdgroups = 2;"
    let inSize = "const int in_vec_size = K;"
    let outSize = "const int out_vec_size = N;"
    for text in [groups, inSize, outSize] {
        precondition(nemotronHeadPreparedQMVSource.components(separatedBy: text).count == 2)
    }
    var body = nemotronHeadPreparedQMVSource
        .replacingOccurrences(of: groups, with: "constexpr int num_simdgroups = \(simdgroups);")
        .replacingOccurrences(of: inSize, with: "const int in_vec_size = SK;")
        .replacingOccurrences(of: outSize, with: "const int out_vec_size = SN;")
    if packSharedUp {
        // Routed FC1 still has TWO SIMD groups. Four shared outputs per
        // SIMD make a complete eight-value chunk without changing its TG.
        let changes = [
            ("constexpr int results_per_simdgroup = RPS;",
             "constexpr int results_per_simdgroup = 4;"),
            ("device T* yp = y + out_row;",
             "device T* yp = y + out_row;\n  threadgroup T values[R * 8];"),
            ("yp[r * out_vec_size + row] = h;",
             "yp[r * out_vec_size + row] = h;\n        values[r * 8 + simd_gid * 4 + row] = h;")
        ]
        for (old, new) in changes {
            precondition(body.components(separatedBy: old).count == 2)
            body = body.replacingOccurrences(of: old, with: new)
        }
        // SN3712 is divisible by8 and one added z-slice owns every chunk.
        // No thread returns before the shared-only local barrier.
        body += "\n" + nmlSharedPackedEpilogue
    }
    let input = prepared.map { "const device float* prepared = \($0);" } ?? ""
    return """
      if (\(condition)) {
        const uint shared_tile = \(tile);
        {
          const uint3 threadgroup_position_in_grid = uint3(0, shared_tile, 0);
          const device uint32_t* w = sw;
          const device T* scales = ss;
          const device T* biases = sb;
          device T* y = sy;
          \(input)
    \(body)
        }
        return;
      }

    """
}

private final class NMLRoutedSharedKernels: Sendable {
    static let shared = NMLRoutedSharedKernels()
    let up: MLXFast.MLXFastKernel?
    let packedUp: MLXFast.MLXFastKernel?
    let down: MLXFast.MLXFastKernel
    private init() {
        let upBranch = nmlSharedExpertBranch(
            condition: "threadgroup_position_in_grid.z >= uint(R * TOPK)",
            tile: "(threadgroup_position_in_grid.z - uint(R * TOPK)) * uint(N / (2 * RPS))"
                + " + threadgroup_position_in_grid.y",
            simdgroups: 2, prepared: nil)
        up = nmlPreparedFC1Source.map {
            MLXFast.metalKernel(
                name: "nml_fc1_plan_with_shared_up",
                inputNames: ["w", "scales", "biases", "prepared", "inds", "sw", "ss", "sb"],
                outputNames: ["y", "plan", "sy"],
                source: upBranch + nmlFC1WithBalancePlanSource($0),
                header: nemotronMoELeaderHeader + nemotronMoELeaderStore
                    + nemotronMoELeaderPreparedHeader)
        }
        let packedUpBranch = nmlSharedExpertBranch(
            condition: "threadgroup_position_in_grid.z >= uint(R * TOPK)",
            tile: "threadgroup_position_in_grid.y", simdgroups: 2, prepared: nil,
            packSharedUp: true)
        packedUp = nmlPreparedFC1Source.map {
            MLXFast.metalKernel(
                name: "nml_fc1_plan_shared_up_packed_epilogue_rps4",
                inputNames: ["w", "scales", "biases", "prepared", "inds", "sw", "ss", "sb"],
                outputNames: ["y", "plan", "sy", "spacked"],
                source: packedUpBranch + nmlFC1WithBalancePlanSource($0),
                header: nemotronMoELeaderHeader + nemotronMoELeaderStore
                    + nemotronMoELeaderPreparedHeader)
        }
        let downBranch = nmlSharedExpertBranch(
            condition: "threadgroup_position_in_grid.x >= uint(N / 4)",
            tile: "threadgroup_position_in_grid.x - uint(N / 4)",
            simdgroups: 6, prepared: "sprep")
        down = MLXFast.metalKernel(
            name: "nml_fc2_balanced_with_shared_down",
            inputNames: ["w", "scales", "biases", "x", "inds", "scores", "plan",
                         "sw", "ss", "sb", "sprep"],
            outputNames: ["y", "sy"], source: downBranch + nmlBalanceWeightedSource,
            header: nemotronMoELeaderHeader + nemotronMoELeaderStore,
            ensureRowContiguous: true)
    }
}

enum NemotronRoutedSharedLaunch {
    static let enabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_ROUTED_SHARED_LAUNCH"] != "0"
    nonisolated(unsafe) private static var verified = false
    static let packedEnabled = ProcessInfo.processInfo.environment[
        "NEMOTRON_MTP_ROUTED_SHARED_PREPARE"] != "0"
    nonisolated(unsafe) private static var packedVerified = false

    private static func buffers(_ linear: Linear, n: Int, k: Int) -> [MLXArray]? {
        guard type(of: linear) == QuantizedLinear.self, let q = linear as? QuantizedLinear,
              q.mode == .affine, q.bits == 4, q.groupSize == 64, q.bias == nil,
              q.weight.dtype == .uint32, q.weight.shape == [n, k / 8],
              q.scales.dtype == .bfloat16, q.scales.shape == [n, k / 64],
              let offsets = q.biases, offsets.dtype == .bfloat16, offsets.shape == q.scales.shape
        else { return nil }
        return [q.weight, q.scales, offsets]
    }

    private static func run(
        _ x: MLXArray, _ indices: MLXArray, _ scores: MLXArray,
        _ mlp: NemotronHSwitchMLP, _ sharedMLP: NemotronHMLP, prepared: MLXArray?,
        packSharedUp: Bool = false, inspectPacked: ((MLXArray, MLXArray) -> Void)? = nil
    ) -> (MLXArray, MLXArray)? {
        guard enabled, NemotronMoELeader.enabled, NemotronMoELeader.fuseRelu2,
              NemotronMoELeader.fuseBalancePlan, NemotronMoELeader.preparedGateUp,
              !NemotronMoELeader.selfTest,
              NemotronMoELeader.leaderResultsPerSimdgroup(rows: 3) == 2,
              NemotronMultiRowQMV.enabled, NemotronMultiRowQMV.narrowEnabled,
              NemotronMultiRowQMV.headRPS(n: 3712) == 2, NemotronMultiRowQMV.headRPS(n: 2688) == 2,
              NemotronMTPExecution.batchedM1,
              Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              type(of: mlp) == NemotronHSwitchMLP.self,
              type(of: mlp.fc1) == QuantizedSwitchLinear.self,
              type(of: mlp.fc2) == QuantizedSwitchLinear.self,
              x.shape == [1, 3, 2688], x.dtype == .bfloat16,
              indices.dtype == .uint32, indices.shape == [1, 3, 6],
              scores.dtype == .float32, scores.shape == [1, 3, 6],
              let up = NemotronMoELeader.bank(mlp.fc1, k: 2688, dtype: .bfloat16),
              let down = NemotronMoELeader.bank(mlp.fc2, k: 1856, dtype: .bfloat16),
              up.weight.shape == [128, 1856, 336],
              up.scales.shape == [128, 1856, 42], up.biases.shape == up.scales.shape,
              down.weight.shape == [128, 2688, 232],
              down.scales.shape == [128, 2688, 29], down.biases.shape == down.scales.shape,
              let sharedUp = buffers(sharedMLP.upProj, n: 3712, k: 2688),
              let sharedDown = buffers(sharedMLP.downProj, n: 2688, k: 3712),
              let upKernel = packSharedUp ? NMLRoutedSharedKernels.shared.packedUp
                : NMLRoutedSharedKernels.shared.up
        else { return nil }
        let flat = indices.reshaped(18)
        let packed = NemotronHeadPreparedActivation.compatible(prepared, rows: 3, k: 2688)
            ?? NemotronHeadPreparedActivation.prepareContiguousRows(x.reshaped(3, 2688), pv: 8)
        let first = upKernel([up.weight, up.scales, up.biases, packed, flat] + sharedUp,
            template: [("T", DType.bfloat16), ("R", 3), ("TOPK", 6),
                       ("K", 2688), ("N", 1856), ("XSLOT", false),
                       ("RELU2", true), ("RPS", 2), ("FAST", false),
                       ("SK", 2688), ("SN", 3712)],
            grid: (32, 1856 / 2, packSharedUp ? 19 : 20), threadGroup: (32, 2, 1),
            outputShapes: [[18, 1856], [36], [3, 3712]]
                + (packSharedUp ? [[3, 15, 9 * 32]] : []),
            outputDTypes: [.bfloat16, .int32, .bfloat16]
                + (packSharedUp ? [.float32] : []))
        let sharedPrepared = packSharedUp ? first[3]
            : NemotronHeadPreparedActivation.prepareContiguousRows(first[2], pv: 8)
        // Non-nil ONLY in official-worker load admission, never in inference.
        if packSharedUp { inspectPacked?(first[2], sharedPrepared) }
        let second = NMLRoutedSharedKernels.shared.down(
            [down.weight, down.scales, down.biases, first[0], flat, scores.reshaped(18), first[1]]
                + sharedDown + [sharedPrepared],
            template: [("T", DType.bfloat16), ("R", 3), ("TOPK", 6),
                       ("K", 1856), ("N", 2688),
                       ("RELU2", false), ("RPS", 2), ("FAST", false),
                       ("SK", 3712), ("SN", 2688)],
            grid: (32 * (2688 / 4 + 2688 / 12), 6, 1), threadGroup: (32, 6, 1),
            outputShapes: [[1, 3, 2688], [3, 2688]],
            outputDTypes: [.bfloat16, .bfloat16])
        return (second[0], second[1].reshaped(1, 3, 2688))
    }

    static func apply(
        _ x: MLXArray, _ indices: MLXArray, _ scores: MLXArray,
        _ mlp: NemotronHSwitchMLP, _ sharedMLP: NemotronHMLP, prepared: MLXArray?
    ) -> (MLXArray, MLXArray)? {
        guard verified else { return nil }
        if packedVerified, let fused = run(x, indices, scores, mlp, sharedMLP,
            prepared: prepared, packSharedUp: true) { return fused }
        return run(x, indices, scores, mlp, sharedMLP, prepared: prepared)
    }

    static func verify(_ model: Module) {
        verified = false
        packedVerified = false
        guard enabled,
              let moe = model.modules().lazy.compactMap({ $0 as? NemotronHMoE })
                .first(where: { $0.sharedExperts != nil }),
              let sharedMLP = moe.sharedExperts
        else { return }
        var matches = true
        for seed in 0 ..< 3 {
            let x = MLXRandom.normal(
                [1, 3, 2688], dtype: .bfloat16, key: MLXRandom.key(UInt64(0x5d0e + seed)))
            let scores = MLXRandom.normal(
                [1, 3, 6], key: MLXRandom.key(UInt64(0x5d1e + seed))).asType(.float32)
            let values: [UInt32] = (0 ..< 3).flatMap { row in
                (0 ..< 6).map { slot in UInt32((17 * (slot + row) + 9 * seed) % 128) }
            }
            let indices = MLXArray(values).reshaped(1, 3, 6)
            guard let fused = run(x, indices, scores, moe.switchMLP, sharedMLP, prepared: nil),
                  let routed = NemotronMoELeader.switchRowsAndWeightedReduce(
                      x, indices, scores, moe.switchMLP, prepared: nil)
            else { return }
            let shared = sharedMLP.mtpForwardRows(x, prepared: nil)
            let sameRouted = (fused.0.view(dtype: .uint16) .== routed.view(dtype: .uint16)).all()
            let sameShared = (fused.1.view(dtype: .uint16) .== shared.view(dtype: .uint16)).all()
            matches = matches && sameRouted.item(Bool.self) && sameShared.item(Bool.self)
        }
        verified = matches
        if matches && packedEnabled {
            packedVerified = qualifyPacked(moe, shared: sharedMLP)
        }
        print("[nemotron-routed-shared-packed] \(packedVerified ? "on (worker bitwise admission passed)" : "off")")
    }

    /// Official-worker load admission ONLY, before any MoE compiler trace.
    /// Compare valid packed FP32 bits, BF16 up and BOTH final outputs against
    /// original separate preparation. Padding is excluded just as in down.
    private static func qualifyPacked(_ moe: NemotronHMoE, shared: NemotronHMLP) -> Bool {
        do {
            return try withError {
                for seed in 0 ..< 3 {
                    let random = MLXRandom.normal([1, 3, 2688], dtype: .bfloat16,
                        key: MLXRandom.key(UInt64(0x8c30 + seed)))
                    // Keep the official-worker probe construction explicitly
                    // typed: nested MLXArray/flatMap overload inference exceeded
                    // the ranked Swift compiler's expression time budget.
                    var routeValues: [UInt32] = []
                    routeValues.reserveCapacity(18)
                    let seedOffset: Int = 9 * seed
                    for row in 0 ..< 3 {
                        for slot in 0 ..< 6 {
                            let expert: Int = (17 * (slot + row) + seedOffset) % 128
                            routeValues.append(UInt32(expert))
                        }
                    }
                    let routeArray: MLXArray = MLXArray(routeValues)
                    let ids: MLXArray = routeArray.reshaped(1, 3, 6)
                    let scores = MLXRandom.normal([1, 3, 6],
                        key: MLXRandom.key(UInt64(0x8c40 + seed))).asType(.float32)
                    for amplitude in [Float(0), Float(1), Float(8)] {
                        let x = (random * MLXArray(amplitude)).asType(.bfloat16)
                        let expectedUp = nemotronMTPLinearRelu2Rows(x, shared.upProj)
                            .reshaped(3, 3712)
                        let expectedPacked = NemotronHeadPreparedActivation
                            .prepareContiguousRows(expectedUp, pv: 8)
                        var packedOK = false
                        guard let got = run(x, ids, scores, moe.switchMLP, shared,
                            prepared: nil, packSharedUp: true, inspectPacked: { up, packed in
                                func valid(_ a: MLXArray) -> MLXArray {
                                    a.reshaped(3, 15, 9, 32).transposed(0, 1, 3, 2)
                                        .reshaped(3, 480, 9)[0..., 0 ..< 464, 0...]
                                }
                                let sameUp = (up.view(dtype: .uint16)
                                    .== expectedUp.view(dtype: .uint16)).all()
                                let samePacked = (valid(packed).view(dtype: .uint32)
                                    .== valid(expectedPacked).view(dtype: .uint32)).all()
                                packedOK = sameUp.item(Bool.self) && samePacked.item(Bool.self)
                            }),
                              let expected = run(x, ids, scores, moe.switchMLP, shared, prepared: nil)
                        else { return false }
                        let routedOK = (got.0.view(dtype: .uint16)
                            .== expected.0.view(dtype: .uint16)).all()
                        let sharedOK = (got.1.view(dtype: .uint16)
                            .== expected.1.view(dtype: .uint16)).all()
                        if !packedOK || !routedOK.item(Bool.self) || !sharedOK.item(Bool.self) {
                            return false
                        }
                    }
                }
                return true
            }
        } catch { return false }
    }
}

// MARK: - Width-four verifier: 24 routed assignments / 48-word balanced plan

/// Change only integer plan extent/capacity, preserving the original stable
/// greedy load balancing, pair cap, slot order and old-plan comparison.
private func nmlFourRowPlanSource() -> String {
    var value = nmlBalancePlanSource
    let changes: [(String, String, Int)] = [
        ("18", "24", 9), ("36", "48", 3),
        ("r = 2", "r = 3", 1), ("counts[g] < 3", "counts[g] < 4", 1),
        ("group*3", "group*4", 1), ("g*3", "g*4", 1)
    ]
    for (old, new, count) in changes {
        precondition(value.components(separatedBy: old).count == count + 1)
        value = value.replacingOccurrences(of: old, with: new)
    }
    return value
}

private func nmlFourRowFC1PlanSource(_ prepared: String) -> String {
    let lane = "const uint lane = thread_position_in_grid.x;"
    let store = "for (uint i = lane; i < 48; i += 32)"
    let original = nmlFourRowPlanSource()
    precondition(original.components(separatedBy: lane).count == 2)
    precondition(original.components(separatedBy: store).count == 2)
    let plan = original.replacingOccurrences(of: lane,
        with: "const uint lane = simd_lid + 32 * simd_gid;")
        .replacingOccurrences(of: store, with: "for (uint i = lane; i < 48; i += 64)")
    // Row0/slot0 is always a leader and both SIMD groups of tile0 are live.
    return prepared + """

      if (tid.y == 0 && tid.z == 0) {
        constexpr int PAIR_COST = 1500;
        constexpr int SOLO_COST = 1000;
        \(plan)
      }
    """
}

private func nmlFourRowWeightedSource() -> String {
    var value = nmlBalanceWeightedSource
    for (old, new) in [("job < 3", "job < 4"),
                       ("simd_gid * 3", "simd_gid * 4"),
                       ("plan[18 + slot]", "plan[24 + slot]")] {
        precondition(value.components(separatedBy: old).count == 2)
        value = value.replacingOccurrences(of: old, with: new)
    }
    return value
}

/// R4 keeps its original full-float coefficient path. Four independent
/// accumulators share each decoded word; each row retains the pair helper's
/// exact term grouping and affine close. No target representation changes.
private let nmlR4QuartetHeader = """
template <typename U, int values_per_thread>
inline void nml_qdot_quartet(
    const device uint8_t* w, const thread U* x0, const thread U* x1,
    const thread U* x2, const thread U* x3,
    U scale, U bias, U sum0, U sum1, U sum2, U sum3, int N,
    thread U& value0, thread U& value1, thread U& value2, thread U& value3) {
  U accum0 = 0;
  U accum1 = 0;
  U accum2 = 0;
  U accum3 = 0;
  const device uint16_t* ws = (const device uint16_t*)w;
  for (int i = 0; i < (N / 4); i++) {
    const uint16_t q = ws[i];
    const U a = q & 0x000f;
    const U b = q & 0x00f0;
    const U c = q & 0x0f00;
    const U d = q & 0xf000;
    accum0 += (x0[4 * i] * a + x0[4 * i + 1] * b + x0[4 * i + 2] * c + x0[4 * i + 3] * d);
    accum1 += (x1[4 * i] * a + x1[4 * i + 1] * b + x1[4 * i + 2] * c + x1[4 * i + 3] * d);
    accum2 += (x2[4 * i] * a + x2[4 * i + 1] * b + x2[4 * i + 2] * c + x2[4 * i + 3] * d);
    accum3 += (x3[4 * i] * a + x3[4 * i + 1] * b + x3[4 * i + 2] * c + x3[4 * i + 3] * d);
  }
  value0 = scale * accum0 + sum0 * bias;
  value1 = scale * accum1 + sum1 * bias;
  value2 = scale * accum2 + sum2 * bias;
  value3 = scale * accum3 + sum3 * bias;
}
"""

/// Only an expert occurring EXACTLY once in EACH of four rows is grouped.
/// Other patterns fall through to the original paired prepared FC1 body.
private let nmlR4QuartetFC1Source = """
  {
    const uint3 qtid = threadgroup_position_in_grid;
    const uint qgid = simdgroup_index_in_threadgroup;
    const uint qlane = thread_index_in_simdgroup;
    const int qslot = int(qtid.z);
    const int qrow = qslot / 6;
    const uint expert = inds[qslot];
    int members[4] = {-1, -1, -1, -1};
    bool complete = true;
    for (int row = 0; row < 4; row++) {
      int count = 0;
      for (int j = 0; j < 6; j++) {
        if (inds[row * 6 + j] == expert) { members[row] = row * 6 + j; count++; }
      }
      complete = complete && count == 1;
    }
    if (complete) {
      if (qrow != 0) { return; }
      constexpr int VP = 8;
      constexpr int BLOCK = 256;
      constexpr int PSTRIDE = 9 * 32;
      constexpr int PTILES = (K / 8 + 31) / 32;
      constexpr int WG = K / 64;
      const int out = int(qtid.y) * (2 * RPS) + int(qgid) * RPS;
      const device uint8_t* qw = (const device uint8_t*)w
          + (ulong)expert * N * (K / 2) + out * (K / 2) + qlane * 4;
      const device T* qs = scales + (ulong)expert * N * WG + out * WG + qlane / 8;
      const device T* qb = biases + (ulong)expert * N * WG + out * WG + qlane / 8;
      const device float* p0 = prepared + qlane;
      const device float* p1 = prepared + PTILES * PSTRIDE + qlane;
      const device float* p2 = prepared + 2 * PTILES * PSTRIDE + qlane;
      const device float* p3 = prepared + 3 * PTILES * PSTRIDE + qlane;
      float a0[RPS] = {0}, a1[RPS] = {0}, a2[RPS] = {0}, a3[RPS] = {0};
      float x0[VP], x1[VP], x2[VP], x3[VP];
      int k = 0;
      for (; k < K - BLOCK; k += BLOCK) {
        const float s0 = nml_prepared_vector<float, VP>(p0, x0);
        const float s1 = nml_prepared_vector<float, VP>(p1, x1);
        const float s2 = nml_prepared_vector<float, VP>(p2, x2);
        const float s3 = nml_prepared_vector<float, VP>(p3, x3);
        for (int r = 0; r < RPS; r++) {
          float v0, v1, v2, v3;
          nml_qdot_quartet<float, VP>(qw + r * (K / 2), x0, x1, x2, x3,
              qs[r * WG], qb[r * WG], s0, s1, s2, s3, VP, v0, v1, v2, v3);
          a0[r] += v0; a1[r] += v1; a2[r] += v2; a3[r] += v3;
        }
        qw += BLOCK / 2; qs += BLOCK / 64; qb += BLOCK / 64;
        p0 += PSTRIDE; p1 += PSTRIDE; p2 += PSTRIDE; p3 += PSTRIDE;
      }
      const int remaining = clamp(int(K - k - qlane * VP), 0, VP);
      if (remaining > 0) {
        const float s0 = nml_prepared_vector_safe<float, VP>(p0, x0, remaining);
        const float s1 = nml_prepared_vector_safe<float, VP>(p1, x1, remaining);
        const float s2 = nml_prepared_vector_safe<float, VP>(p2, x2, remaining);
        const float s3 = nml_prepared_vector_safe<float, VP>(p3, x3, remaining);
        for (int r = 0; r < RPS; r++) {
          float v0, v1, v2, v3;
          nml_qdot_quartet<float, VP>(qw + r * (K / 2), x0, x1, x2, x3,
              qs[r * WG], qb[r * WG], s0, s1, s2, s3, remaining, v0, v1, v2, v3);
          a0[r] += v0; a1[r] += v1; a2[r] += v2; a3[r] += v3;
        }
      }
      for (int r = 0; r < RPS; r++) {
        a0[r] = simd_sum(a0[r]); a1[r] = simd_sum(a1[r]);
        a2[r] = simd_sum(a2[r]); a3[r] = simd_sum(a3[r]);
        if (qlane == 0) {
          y[members[0] * N + out + r] = nml_store<T, RELU2>(a0[r]);
          y[members[1] * N + out + r] = nml_store<T, RELU2>(a1[r]);
          y[members[2] * N + out + r] = nml_store<T, RELU2>(a2[r]);
          y[members[3] * N + out + r] = nml_store<T, RELU2>(a3[r]);
        }
      }
      return;
    }
  }
"""

private func nmlFourRowQuartetFC1Source(_ prepared: String) -> String {
    // Produce the SAME48-word FC2 pair plan before quartet follower returns.
    // Row0/slot0/tile0 is one complete TG; the plan's barriers stay uniform.
    var value = """
      {
        const uint3 tid = threadgroup_position_in_grid;
        const uint simd_gid = simdgroup_index_in_threadgroup;
        const uint simd_lid = thread_index_in_simdgroup;
    """
    value += nmlFourRowFC1PlanSource("")
    value += "\n  }\n"
    value += nmlR4QuartetFC1Source
    value += prepared
    return value
}

private final class NMLRoutedSharedFourKernels: Sendable {
    static let shared = NMLRoutedSharedFourKernels()
    let up: MLXFast.MLXFastKernel?
    let upQuartet: MLXFast.MLXFastKernel?
    let down: MLXFast.MLXFastKernel
    private init() {
        let sharedUp = nmlSharedExpertBranch(
            condition: "threadgroup_position_in_grid.z >= uint(R * TOPK)",
            tile: "threadgroup_position_in_grid.y", simdgroups: 2,
            prepared: nil, packSharedUp: true)
        up = nmlPreparedFC1Source.map {
            MLXFast.metalKernel(name: "nml_r4_fc1_plan_shared_up_prepared",
                inputNames: ["w", "scales", "biases", "prepared", "inds", "sw", "ss", "sb"],
                outputNames: ["y", "plan", "sy", "spacked"],
                source: sharedUp + nmlFourRowFC1PlanSource($0),
                header: nemotronMoELeaderHeader + nemotronMoELeaderStore
                    + nemotronMoELeaderPreparedHeader)
        }
        upQuartet = nmlPreparedFC1Source.map {
            MLXFast.metalKernel(name: "nml_r4_fc1_quartet_plan_shared_up_prepared",
                inputNames: ["w", "scales", "biases", "prepared", "inds", "sw", "ss", "sb"],
                outputNames: ["y", "plan", "sy", "spacked"],
                source: sharedUp + nmlFourRowQuartetFC1Source($0),
                header: nemotronMoELeaderHeader + nemotronMoELeaderStore
                    + nemotronMoELeaderPreparedHeader + nmlR4QuartetHeader)
        }
        let sharedDown = nmlSharedExpertBranch(
            condition: "threadgroup_position_in_grid.x >= uint(N / 4)",
            tile: "threadgroup_position_in_grid.x - uint(N / 4)",
            simdgroups: 6, prepared: "sprep")
        down = MLXFast.metalKernel(name: "nml_r4_fc2_balanced_with_shared_down",
            inputNames: ["w", "scales", "biases", "x", "inds", "scores", "plan",
                         "sw", "ss", "sb", "sprep"],
            outputNames: ["y", "sy"], source: sharedDown + nmlFourRowWeightedSource(),
            header: nemotronMoELeaderHeader + nemotronMoELeaderStore,
            ensureRowContiguous: true)
    }
}

enum NemotronRoutedSharedFourRows {
    nonisolated(unsafe) private static var verified = false
    nonisolated(unsafe) private static var quartetVerified = false

    private static func buffers(_ layer: Linear, n: Int, k: Int) -> [MLXArray]? {
        guard type(of: layer) == QuantizedLinear.self, let q = layer as? QuantizedLinear,
              q.mode == .affine, q.bits == 4, q.groupSize == 64, q.bias == nil,
              q.weight.dtype == .uint32, q.weight.shape == [n, k / 8],
              q.scales.dtype == .bfloat16, q.scales.shape == [n, k / 64],
              let biases = q.biases, biases.dtype == .bfloat16,
              biases.shape == q.scales.shape else { return nil }
        return [q.weight, q.scales, biases]
    }

    private static func run(
        _ x: MLXArray, _ ids: MLXArray, _ scores: MLXArray,
        _ mlp: NemotronHSwitchMLP, _ sharedMLP: NemotronHMLP, prepared: MLXArray?,
        quartet: Bool = false,
        inspect: ((MLXArray, MLXArray) -> Void)? = nil,
        inspectUpPlan: ((MLXArray, MLXArray) -> Void)? = nil
    ) -> (MLXArray, MLXArray)? {
        guard NemotronMTPFourRows.enabled, NemotronRoutedSharedLaunch.enabled,
              NemotronRoutedSharedLaunch.packedEnabled,
              NemotronMoELeader.enabled, NemotronMoELeader.fuseRelu2,
              NemotronMoELeader.preparedGateUp, NemotronMoELeader.fuseBalancePlan,
              !NemotronMoELeader.selfTest, NemotronMTPExecution.batchedM1,
              NemotronMultiRowQMV.enabled, NemotronMultiRowQMV.narrowEnabled,
              Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              type(of: mlp) == NemotronHSwitchMLP.self,
              type(of: mlp.fc1) == QuantizedSwitchLinear.self,
              type(of: mlp.fc2) == QuantizedSwitchLinear.self,
              x.shape == [1, 4, 2688], x.dtype == .bfloat16,
              ids.shape == [1, 4, 6], ids.dtype == .uint32,
              scores.shape == ids.shape, scores.dtype == .float32,
              let up = NemotronMoELeader.bank(mlp.fc1, k: 2688, dtype: .bfloat16),
              let down = NemotronMoELeader.bank(mlp.fc2, k: 1856, dtype: .bfloat16),
              up.weight.shape == [128, 1856, 336], up.scales.shape == [128, 1856, 42],
              up.biases.shape == up.scales.shape,
              down.weight.shape == [128, 2688, 232], down.scales.shape == [128, 2688, 29],
              down.biases.shape == down.scales.shape,
              let sharedUp = buffers(sharedMLP.upProj, n: 3712, k: 2688),
              let sharedDown = buffers(sharedMLP.downProj, n: 2688, k: 3712),
              let kernel = quartet ? NMLRoutedSharedFourKernels.shared.upQuartet
                : NMLRoutedSharedFourKernels.shared.up else { return nil }
        let flat = ids.reshaped(24)
        let packed = NemotronHeadPreparedActivation.compatible(prepared, rows: 4, k: 2688)
            ?? NemotronHeadPreparedActivation.prepareContiguousRows(x.reshaped(4, 2688), pv: 8)
        let first = kernel([up.weight, up.scales, up.biases, packed, flat] + sharedUp,
            template: [("T", DType.bfloat16), ("R", 4), ("TOPK", 6),
                ("K", 2688), ("N", 1856), ("XSLOT", false), ("RELU2", true),
                ("RPS", 2), ("FAST", false), ("SK", 2688), ("SN", 3712)],
            grid: (32, 1856 / 2, 25), threadGroup: (32, 2, 1),
            outputShapes: [[24, 1856], [48], [4, 3712], [4, 15, 9 * 32]],
            outputDTypes: [.bfloat16, .int32, .bfloat16, .float32])
        // This callback is non-nil only during official-worker load admission.
        inspect?(first[2], first[3])
        inspectUpPlan?(first[0], first[1])
        let result = NMLRoutedSharedFourKernels.shared.down(
            [down.weight, down.scales, down.biases, first[0], flat, scores.reshaped(24), first[1]]
                + sharedDown + [first[3]],
            template: [("T", DType.bfloat16), ("R", 4), ("TOPK", 6),
                ("K", 1856), ("N", 2688), ("RELU2", false), ("RPS", 2),
                ("FAST", false), ("SK", 3712), ("SN", 2688)],
            grid: (32 * (2688 / 4 + 2688 / 12), 6, 1), threadGroup: (32, 6, 1),
            outputShapes: [[1, 4, 2688], [4, 2688]], outputDTypes: [.bfloat16, .bfloat16])
        return (result[0], result[1].reshaped(1, 4, 2688))
    }

    static func apply(_ x: MLXArray, _ ids: MLXArray, _ scores: MLXArray,
                      _ mlp: NemotronHSwitchMLP, _ shared: NemotronHMLP,
                      prepared: MLXArray?) -> (MLXArray, MLXArray)? {
        guard verified else { return nil }
        return run(x, ids, scores, mlp, shared, prepared: prepared, quartet: quartetVerified)
    }

    /// Original generic R4 routed reduce + separate shared up/prepare/down
    /// remain the reference and the inference fallback. No R3 source changes.
    static func verify(_ model: Module) {
        verified = false
        quartetVerified = false
        guard NemotronMTPFourRows.enabled,
              let moe = model.modules().lazy.compactMap({ $0 as? NemotronHMoE })
                .first(where: { $0.sharedExperts != nil }),
              let shared = moe.sharedExperts else { return }
        do {
            verified = try withError {
                for seed in 0..<3 {
                    let random = MLXRandom.normal([1, 4, 2688], dtype: .bfloat16,
                        key: MLXRandom.key(UInt64(0x4a10 + seed)))
                    var routes: [UInt32] = []
                    for row in 0..<4 {
                        for slot in 0..<6 {
                            let expert: Int = (17 * (row + slot) + 9 * seed) % 128
                            routes.append(UInt32(expert))
                        }
                    }
                    let routeArray: MLXArray = MLXArray(routes)
                    let ids: MLXArray = routeArray.reshaped(1, 4, 6)
                    let scores = MLXRandom.normal([1, 4, 6],
                        key: MLXRandom.key(UInt64(0x4a20 + seed))).asType(.float32)
                    for amplitude in [Float(0), Float(1), Float(8)] {
                        let x = (random * MLXArray(amplitude)).asType(.bfloat16)
                        let expectedUp = nemotronMTPLinearRelu2Rows(x, shared.upProj).reshaped(4, 3712)
                        let expectedPacked = NemotronHeadPreparedActivation.prepareContiguousRows(
                            expectedUp, pv: 8)
                        var packedOK = false
                        guard let actual = run(x, ids, scores, moe.switchMLP, shared,
                            prepared: nil, inspect: { up, packed in
                                func valid(_ a: MLXArray) -> MLXArray {
                                    a.reshaped(4, 15, 9, 32).transposed(0, 1, 3, 2)
                                        .reshaped(4, 480, 9)[0..., 0..<464, 0...]
                                }
                                let sameUp = (up.view(dtype: .uint16)
                                    .== expectedUp.view(dtype: .uint16)).all()
                                let samePacked = (valid(packed).view(dtype: .uint32)
                                    .== valid(expectedPacked).view(dtype: .uint32)).all()
                                packedOK = sameUp.item(Bool.self) && samePacked.item(Bool.self)
                            }),
                              let expectedRouted = NemotronMoELeader.switchRowsAndWeightedReduce(
                                  x, ids, scores, moe.switchMLP, prepared: nil) else { return false }
                        let expectedShared = shared.mtpForwardRows(x)
                        let routedOK = (actual.0.view(dtype: .uint16)
                            .== expectedRouted.view(dtype: .uint16)).all()
                        let sharedOK = (actual.1.view(dtype: .uint16)
                            .== expectedShared.view(dtype: .uint16)).all()
                        if !packedOK || !routedOK.item(Bool.self) || !sharedOK.item(Bool.self) {
                            return false
                        }
                    }
                }
                return true
            }
        } catch { verified = false }
        if verified { quartetVerified = qualifyQuartet(model) }
        print("[nemotron-r4-routed-shared-packed] \(verified ? "on (worker bitwise admission passed)" : "off")")
        print("[nemotron-r4-fc1-quartet] \(quartetVerified ? "on (all-layer worker rawbit admission passed)" : "off (original pairs)")")
    }

    private static func qualifyQuartet(_ model: Module) -> Bool {
        guard ProcessInfo.processInfo.environment["NEMOTRON_MTP_R4_FC1_QUARTET"] != "0"
        else { return false }
        do {
            return try withError {
                var checked = 0
                for module in model.modules() {
                    guard let moe = module as? NemotronHMoE,
                          let shared = moe.sharedExperts else { continue }
                    for seed in 0..<3 {
                        let random = MLXRandom.normal([1, 4, 2688], dtype: .bfloat16,
                            key: MLXRandom.key(UInt64(0x4a410 + seed)))
                        let amplitude: Float = seed == 0 ? 0 : (seed == 1 ? 1 : 8)
                        let x: MLXArray = (random * MLXArray(amplitude)).asType(.bfloat16)
                        let scores = MLXRandom.normal([1, 4, 6],
                            key: MLXRandom.key(UInt64(0x4a420 + seed))).asType(.float32)
                        for pattern in 0..<5 {
                            var routeValues: [UInt32] = []
                            for row in 0..<4 {
                                for slot in 0..<6 {
                                    let expert: Int
                                    if pattern == 0 {
                                        expert = (slot * 17 + seed * 7) % 128
                                    } else if pattern == 1 {
                                        expert = (((slot + row) % 6) * 17 + seed * 7) % 128
                                    } else if pattern == 2 {
                                        expert = slot < 2 ? slot * 17
                                            : (32 + row * 11 + slot + seed) % 128
                                    } else if pattern == 3 {
                                        expert = (row / 2) * 32 + slot + seed * 7
                                    } else {
                                        // Triple occurrence plus a disjoint fourth row:
                                        // exercises the untouched pair/solo fallback.
                                        expert = (row == 3 ? 64 : 0) + slot + seed * 7
                                    }
                                    routeValues.append(UInt32(expert))
                                }
                            }
                            let routeArray: MLXArray = MLXArray(routeValues)
                            let ids: MLXArray = routeArray.reshaped(1, 4, 6)
                            var originalUp: MLXArray?, originalPlan: MLXArray?
                            var candidateUp: MLXArray?, candidatePlan: MLXArray?
                            var originalSharedUp: MLXArray?, originalPacked: MLXArray?
                            var candidateSharedUp: MLXArray?, candidatePacked: MLXArray?
                            guard let original = run(x, ids, scores, moe.switchMLP, shared,
                                prepared: nil, inspect: { originalSharedUp = $0; originalPacked = $1 },
                                inspectUpPlan: { originalUp = $0; originalPlan = $1 }),
                                  let candidate = run(x, ids, scores, moe.switchMLP, shared,
                                prepared: nil, quartet: true,
                                inspect: { candidateSharedUp = $0; candidatePacked = $1 },
                                inspectUpPlan: { candidateUp = $0; candidatePlan = $1 }),
                                  let originalUp, let originalPlan, let candidateUp, let candidatePlan,
                                  let originalSharedUp, let originalPacked,
                                  let candidateSharedUp, let candidatePacked else { return false }
                            func same(_ a: MLXArray, _ b: MLXArray, _ dtype: DType) -> Bool {
                                (a.view(dtype: dtype) .== b.view(dtype: dtype)).all().item(Bool.self)
                            }
                            func valid(_ a: MLXArray) -> MLXArray {
                                a.reshaped(4, 15, 9, 32).transposed(0, 1, 3, 2)
                                    .reshaped(4, 480, 9)[0..., 0..<464, 0...]
                            }
                            guard isFinite(originalUp).all().item(Bool.self),
                                  isFinite(original.0).all().item(Bool.self),
                                  isFinite(original.1).all().item(Bool.self),
                                  same(originalUp, candidateUp, .uint16),
                                  (originalPlan .== candidatePlan).all().item(Bool.self),
                                  same(originalSharedUp, candidateSharedUp, .uint16),
                                  same(valid(originalPacked), valid(candidatePacked), .uint32),
                                  same(original.0, candidate.0, .uint16),
                                  same(original.1, candidate.1, .uint16) else { return false }
                        }
                    }
                    checked += 1
                }
                return checked == 23
            }
        } catch { return false }
    }
}
