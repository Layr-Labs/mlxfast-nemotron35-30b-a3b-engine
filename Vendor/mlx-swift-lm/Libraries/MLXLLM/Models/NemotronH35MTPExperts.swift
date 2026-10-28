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
    private var entries: [Entry?] = Array(repeating: nil, count: 16)
    private var generations = 0
    private var calls = 0
    private var joinedChecks = 0
    private var joinTimed = false
    var generationCount: Int { lock.withLock { generations } }
    var callCount: Int { lock.withLock { calls } }

    func callAsFunction(_ x: MLXArray, prepared: MLXArray? = nil, owner: NemotronHMoE) -> MLXArray {
        let result = outputs(x, prepared: prepared, parts: false, owner: owner)
        return result.isEmpty ? MLXArray(0) : result[0]
    }

    /// parts: [routed, shared] before their Add (owner must have shared experts).
    func outputs(_ x: MLXArray, prepared: MLXArray?, parts: Bool,
                 owner: NemotronHMoE, residual: MLXArray? = nil, nextNorm: RMSNorm? = nil) -> [MLXArray] {
        // Native parameter wrappers forbid replacing initialized array objects;
        // Module.update changes their contexts in place. Check every concrete
        // child identity cheaply, and collect array handles only on a miss.
        // Unknown subclasses retain the fully reflective fallback.
        let nativeIdentity = Self.nativeIdentity(owner)
        let parameters = nativeIdentity == nil ? owner.innerState() : nil
        var identity = nativeIdentity ?? (owner.modules().map(ObjectIdentifier.init)
            + (parameters ?? []).map(ObjectIdentifier.init))
        let joins = residual != nil && nextNorm != nil
        if let nextNorm, joins {
            identity += [ObjectIdentifier(nextNorm), ObjectIdentifier(nextNorm.weight)]
        }
        let usesPrepared = prepared != nil
        let slot = (usesPrepared ? 1 : 0) + (parts ? 2 : 0)
            + (NemotronZSkip.needsOriginal(owner.switchMLP.fc2) ? 4 : 0) + (joins ? 8 : 0)
        var retired: Entry?
        let current = lock.withLock { () -> Entry in
            calls += 1
            if let entry = entries[slot], entry.identity == identity { return entry }
            let state = (parameters ?? owner.innerState()) + (joins ? [nextNorm!.weight] : [])
            let function: @Sendable ([MLXArray]) -> [MLXArray] = compile(inputs: [state]) { [weak owner] arrays in
                guard let owner else { preconditionFailure("Released Nemotron MoE graph owner") }
                let outputs = owner.mtpForwardRowsUncompiledOutputs(arrays[0],
                    prepared: usesPrepared ? arrays[1] : nil, parts: parts)
                guard joins, let nextNorm else { return outputs }
                let lhs = arrays[usesPrepared ? 2 : 1]
                guard outputs.count == 2, let joined = NemotronHResidualRMSNorm.applyPreparedRows(
                    lhs, outputs[0], shared: outputs[1], norm: nextNorm) else {
                    preconditionFailure("Qualified MoE compiled join became ineligible")
                }
                return [joined.0, joined.1, joined.2]
            }
            let replacement = Entry(identity: identity, usesPrepared: usesPrepared, function: function)
            retired = entries[slot]
            entries[slot] = replacement
            generations += 1
            return replacement
        }
        // Destroy retired compiler entries outside this helper's lock; compiler
        // evaluation owns a separate global/per-function lock ordering.
        var arguments = prepared.map { [x, $0] } ?? [x]
        if let residual, joins { arguments.append(residual) }
        return withExtendedLifetime(retired) { current.function(arguments) }
    }

    func recordJoinCheck() {
        lock.withLock {
            joinedChecks += 1
            if joinedChecks == 1 {
                fputs("nemotron compiled MoE join: residual, norm, prepared live elements bitwise exact\n", stderr)
            }
        }
    }

    func benchmarkJoin(owner: NemotronHMoE, x: MLXArray, prepared: MLXArray?,
                       residual: MLXArray, nextNorm: RMSNorm, layerIndex: Int,
                       directory: String) {
        guard lock.withLock({ () -> Bool in
            if joinTimed { return false }; joinTimed = true; return true
        }) else { return }
        eval([x, residual] + (prepared.map { [$0] } ?? []))
        func make(_ joined: Bool) -> [MLXArray] {
            if joined {
                let y = owner.mtpForwardRowsJoined(x, prepared: prepared,
                    residual: residual, nextNorm: nextNorm, layerIndex: layerIndex)!
                return [y.0, y.1, y.2]
            }
            let parts = owner.mtpForwardRowsParts(x, prepared: prepared)!
            let y = NemotronHResidualRMSNorm.applyPreparedRows(
                residual, parts.0, shared: parts.1, norm: nextNorm)!
            return [y.0, y.1, y.2]
        }
        for i in 0..<16 { eval(make(i % 2 == 0)) }
        var blocks: [[String: Any]] = []
        for block in 0..<13 {
            var measurements: [String: Any] = ["block": block]
            for joined in (block % 2 == 0 ? [false, true] : [true, false]) {
                var times: [Double] = []
                for _ in 0..<64 {
                    let start = DispatchTime.now().uptimeNanoseconds
                    let y = make(joined)
                    let elapsed = DispatchTime.now().uptimeNanoseconds - start
                    times.append(Double(elapsed) / 1000.0)
                    eval(y)
                }
                measurements[joined ? "candidate_us" : "frontier_us"] = times
            }
            blocks.append(measurements)
        }
        let record: [String: Any] = ["layer": layerIndex, "blocks": blocks,
            "needs_original": NemotronZSkip.needsOriginal(owner.switchMLP.fc2),
            "scope": "host graph construction only; GPU eval excluded",
            "prepared": prepared != nil, "rows": x.shape]
        let data = try! JSONSerialization.data(withJSONObject: record, options: [.sortedKeys])
        try! data.write(to: URL(fileURLWithPath: directory)
            .appendingPathComponent("layer-\(layerIndex).json"), options: .atomic)
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

    private static let joinCheck = ProcessInfo.processInfo.environment["NEMOTRON_MTP_JOIN_CHECK"] == "1"
    /// Timing records are not written in this build (no file output).
    private static let joinTiming: String? = nil

    func mtpForwardRowsJoined(_ x: MLXArray, prepared: MLXArray?, residual: MLXArray,
                              nextNorm: RMSNorm, layerIndex: Int = -1) -> (MLXArray, MLXArray, MLXArray)? {
        guard NemotronMTPExecution.compiledMoE, NemotronMTPExecution.batchedM1,
              sharedExperts != nil, NemotronHResidualRMSNorm.plumbingAvailable,
              x.shape == [1, 3, 2688], x.dtype == .bfloat16,
              residual.shape == x.shape, residual.dtype == .bfloat16,
              type(of: nextNorm) == RMSNorm.self,
              nextNorm.weight.shape == [2688], nextNorm.weight.dtype == .bfloat16
        else { return nil }
        let outputs = mtpCompiledRowsCache.outputs(x, prepared: prepared, parts: true,
            owner: self, residual: residual, nextNorm: nextNorm)
        precondition(outputs.count == 3)
        if Self.joinCheck {
            let parts = mtpForwardRowsParts(x, prepared: prepared)!
            let expected = NemotronHResidualRMSNorm.applyPreparedRows(
                residual, parts.0, shared: parts.1, norm: nextNorm)!
            let got = outputs[2].reshaped([3, 11, 9, 32]).view(dtype: .uint32)
            let want = expected.2.reshaped([3, 11, 9, 32]).view(dtype: .uint32)
            let preparedEqual = MLX.logicalAnd(
                (got[0..., 0..<10, 0..., 0...] .== want[0..., 0..<10, 0..., 0...]).all(),
                (got[0..., 10, 0..., 0..<16] .== want[0..., 10, 0..., 0..<16]).all())
            let equal = MLX.logicalAnd(
                (outputs[0].view(dtype: .uint16) .== expected.0.view(dtype: .uint16)).all(),
                (outputs[1].view(dtype: .uint16) .== expected.1.view(dtype: .uint16)).all())
            precondition(MLX.logicalAnd(equal, preparedEqual).item(Bool.self),
                "Compiled MoE join differs from eager join")
            mtpCompiledRowsCache.recordJoinCheck()
        }
        if let directory = Self.joinTiming {
            mtpCompiledRowsCache.benchmarkJoin(owner: self, x: x, prepared: prepared,
                residual: residual, nextNorm: nextNorm, layerIndex: layerIndex, directory: directory)
        }
        return (outputs[0], outputs[1], outputs[2])
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
        var y: MLXArray
        let products = parts && NemotronHResidualRMSNorm.plumbingAvailable
        let colaunch = products && NemotronFC2SharedColaunch.enabled
            ? sharedExperts.map { NemotronFC2SharedColaunch.Shared(mlp: $0, x: x, prepared: packed) } : nil
        // Router matmul retains M=1. Gathered experts already have M=1 per
        // selected expert and can dispatch all assignments in one operation.
        let indices: MLXArray
        let scores: MLXArray
        if NemotronMTPExecution.batchedRouter, let colaunch, let packed,
           NemotronSharedUpPacking.unpackedApplies(colaunch.x, up: colaunch.mlp.upProj, down: colaunch.mlp.downProj),
           let fused = NemotronRouterSharedUpColaunch.run(x, gate: gate, up: colaunch.mlp.upProj, prepared: packed) {
            // The router and the shared up in one launch; the shared up rows
            // feed the shared down launch below.
            indices = fused.indices
            scores = fused.weights
            colaunch.up = fused.up
        } else if NemotronMTPExecution.batchedRouter {
            let routed = gate.mtpForwardRows(x)
            indices = routed.0
            scores = routed.1
        } else {
            let routed = (0..<x.dim(1)).map { gate(x[0..., $0..<($0 + 1), 0...]) }
            indices = concatenated(routed.map { $0.0 }, axis: 1)
            scores = concatenated(routed.map { $0.1 }, axis: 1)
        }
        if let reduced = NemotronMoELeader.switchRowsAndWeightedReduce(
            x, indices, scores, switchMLP, prepared: packed, products: products, colaunch: colaunch) {
            y = reduced
        } else {
            y = NemotronMoELeader.switchRows(x, indices, switchMLP) ?? switchMLP(x, indices)
            y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)
        }
        guard let sharedExperts else { return [y] }
        let shared = colaunch?.rows ?? sharedExperts.mtpForwardRows(x, prepared: packed)
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

/// The triple FC1 with its three main K loops (one, two and three routed
/// rows) unrolled by two (NEMOTRON_MTP_FC1_UNROLL): `#pragma clang loop
/// unroll_count(2)` ahead of each `for (; k < K - block_size; ...)`. The
/// unrolled loop runs the same block bodies in the same order (block k, then
/// block k + 256, with the tail block after the loop as before), so every
/// load, product, term order, accumulation, affine close, closing simd_sum
/// and store is the record's; only the instruction stream of two consecutive
/// blocks is laid out together. Applied to the triple source before the
/// tail-8, prepared and dictionary transforms; nil when the record text has
/// moved (the record kernels stay).
private func nmlFC1UnrollSource(_ source: String) -> String? {
    let loop = "    for (; k < K - block_size; k += block_size) {\n"
    guard source.components(separatedBy: loop).count == 4 else { return nil }
    return source.replacingOccurrences(of: loop, with: "    #pragma clang loop unroll_count(2)\n" + loop)
}

/// The (scale, bias) dictionary form of a leader-family or prepared QMV body
/// (`NemotronSBDict`). `scales` carries one uint16 pair index per group and
/// `biases` the tensor's pair table as FP32 (scale, bias), each entry the exact
/// BF16 -> FP32 widening of a pair the tensor holds. Every group's `s` and `b`
/// are the FP32 values the BF16 loads produced, so each qdot, sum, order and
/// rounding is unchanged; only the group-metadata bytes halve.
func nemotronSBDictSource(_ source: String) -> String? {
    let typedef = "  typedef float U;\n"
    guard source.components(separatedBy: typedef).count == 2 else { return nil }
    var s = source.replacingOccurrences(of: typedef, with:
        typedef + "  const device float2* sbt = (const device float2*)biases;\n")
    for (from, to) in [
        ("const device T* sc = scales + ", "const device uint16_t* sc = scales + "),
        ("const device T* bs = biases + ", "const device uint16_t* bs = scales + "),
        ("const device T* sl = sc + ", "const device uint16_t* sl = sc + "),
        ("const device T* bl = bs + ", "const device uint16_t* bl = bs + "),
        ("U s = sl[0];", "const float2 sbp = sbt[sl[0]];\n        U s = sbp.x;"),
        ("U b = bl[0];", "U b = sbp.y;"),
    ] {
        guard s.contains(from) else { return nil }
        s = s.replacingOccurrences(of: from, with: to)
    }
    // The hoisted and next-block register loads of the prepared load forms.
    for (from, to) in [
        ("sv[row] = sc[row * in_vec_size_g];", "sv[row] = sbt[sc[row * in_vec_size_g]].x;"),
        ("bv[row] = bs[row * in_vec_size_g];", "bv[row] = sbt[bs[row * in_vec_size_g]].y;"),
        ("sn[row] = sc[block_size / group_size + row * in_vec_size_g];",
         "sn[row] = sbt[sc[block_size / group_size + row * in_vec_size_g]].x;"),
        ("bn[row] = bs[block_size / group_size + row * in_vec_size_g];",
         "bn[row] = sbt[bs[block_size / group_size + row * in_vec_size_g]].y;"),
    ] {
        s = s.replacingOccurrences(of: from, with: to)
    }
    let rest = s.replacingOccurrences(of: "sbt[sc[", with: "").replacingOccurrences(of: "sbt[bs[", with: "")
    guard !rest.contains("sc["), !rest.contains("bs["), !s.contains("const device T* sc"),
          !s.contains("const device T* bs"), !s.contains("const device T* sl"),
          !s.contains("const device T* bl"), !s.contains("bl[0]"),
          s.components(separatedBy: "biases").count == 2 else { return nil }
    return nemotronSBDVectorStoreSource(s)
}

/// NEMOTRON_SBD_VSTORE=0 keeps the lane-0 stores of the dictionary kernels.
let nemotronSBDVectorStore = ProcessInfo.processInfo.environment["NEMOTRON_SBD_VSTORE"] != "0"

/// The closing store loops of a dictionary kernel (`nemotronSBDictSource`) with a vector store: each SIMD group
/// runs the same `simd_sum` of the same lane partials, lane q keeps the sum whose output it stores (a select, no
/// branch in the loop), and lanes 0 ... count - 1 then round and store their outputs at once with the loop's
/// statements, where lane 0 rounded and stored each output in turn. Every output gets the same sum, rounding and
/// address; only the storing lane changes. Forms: the prepared / multi-row QMV loop over (row, r) (lane
/// r * RPS + row), the in_proj register-epilogue loop (lane row * R + t stores the value it already keeps for its
/// epilogue), and the routed FC1 leader / triple loops over one, two or three rows (lane j * RPS + r stores row j's
/// output r). Each form is used only when its lanes fit in one SIMD group (else the lane-0 loop runs); a loop that
/// does not match keeps its lane-0 stores.
func nemotronSBDVectorStoreSource(_ source: String) -> String {
    guard nemotronSBDVectorStore else { return source }
    var s = source
    let qmvLoop = """
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
    s = s.replacingOccurrences(of: qmvLoop, with: """
      if (R * results_per_simdgroup <= 32) {
        U nmv_kept = 0;
        for (int row = 0; row < results_per_simdgroup; row++) {
          for (int r = 0; r < R; r++) {
            U v = simd_sum(result[r][row]);
            nmv_kept = simd_lid == uint(r * results_per_simdgroup + row) ? v : nmv_kept;
          }
        }
        if (simd_lid < uint(R * results_per_simdgroup)) {
          const int r = int(simd_lid) / results_per_simdgroup;
          const int row = int(simd_lid) % results_per_simdgroup;
          T h = static_cast<T>(nmv_kept);
          if (RELU2) {
            T zero = static_cast<T>(0);
            T m = metal::isnan(h) ? h : (h > zero ? h : zero);
            h = m * m;
          }
          yp[r * out_vec_size + row] = h;
        }
      } else {

    """ + qmvLoop + "\n  }")
    // In_proj register epilogue: the lane-0 store leaves the loop; lane l < RPS * R stores its (row, t) value
    // `yt` (the loop's `hc` for that output) before its conv / timestep work.
    let regsStore = """
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

    """
    let regsPick = """
            yt = t == s ? nmc_y[s] : yt;
          }

    """
    if s.components(separatedBy: regsStore).count == 2, s.components(separatedBy: regsPick).count == 2 {
        s = s.replacingOccurrences(of: regsStore, with: "      if (nmc_row == row) {\n")
            .replacingOccurrences(of: regsPick, with: regsPick + "      yp[t * out_vec_size + row] = yt;\n")
    }
    // Routed FC1 leader / triple: one, two and three rows.
    for rows in 1 ... 3 {
        let names = rows == 1 ? ["result"] : (0 ..< rows).map { "result\($0)" }
        let sums = names.map { "      \($0)[r] = simd_sum(\($0)[r]);\n" }.joined()
        let stores = names.enumerated()
            .map { "        y\($0.offset)[r] = nml_store<T, RELU2>(\($0.element)[r]);\n" }.joined()
        let loop = "    for (int r = 0; r < results_per_simdgroup; r++) {\n" + sums
            + "      if (simd_lid == 0) {\n" + stores + "      }\n    }\n"
        let keeps = names.enumerated().map {
            "      nmv_kept = simd_lid == uint(\($0.offset) * results_per_simdgroup + r) ? \($0.element)[r] : nmv_kept;\n"
        }.joined()
        var target = "y0"
        for j in 1 ..< rows {
            target = "simd_lid < uint(\(j) * results_per_simdgroup) ? (\(target)) : y\(j)"
        }
        let vector = "    if (\(rows) * results_per_simdgroup <= 32) {\n"
            + "    U nmv_kept = 0;\n"
            + "    for (int r = 0; r < results_per_simdgroup; r++) {\n" + sums + keeps + "    }\n"
            + "    if (simd_lid < uint(\(rows) * results_per_simdgroup)) {\n"
            + "      device T* nmv_y = \(target);\n"
            + "      nmv_y[int(simd_lid) % results_per_simdgroup] = nml_store<T, RELU2>(nmv_kept);\n"
            + "    }\n"
            + "    } else {\n" + loop + "    }\n"
        s = s.replacingOccurrences(of: loop, with: vector)
    }
    return s
}

/// NEMOTRON_MORE_VSTORE=0 keeps the lane-0 closing stores of the BF16 (non-dictionary) prepared QMV launches
/// that `nemotronQMVVectorStoreSource` rewrites (the verify's fused Q/K/V with its KV-cache stores, the MTP
/// head's post-verify Q/K/V rows and the crown prepared QMV its front projections take). Each twin is used
/// only after `NemotronTargetQKV.qualifyVectorStore` matched it to its lane-0 launch bit for bit.
let nemotronMoreVectorStore = ProcessInfo.processInfo.environment["NEMOTRON_MORE_VSTORE"] != "0"

/// The prepared / multi-row QMV closing store loop over (row, r) with whatever statements a launch stores
/// after the record `T h = static_cast<T>(v);` (the plain `yp` store, or that store plus a fused launch's
/// KV-cache stores) as a vector store, the form `nemotronSBDVectorStoreSource` gives the dictionary kernels:
/// each SIMD group runs the same `simd_sum` of the same lane partials in the same order, lane
/// r * RPS + row keeps the sum it stores (a select), and lanes 0 ... R * RPS - 1 then run the loop body's
/// statements at once for their (r, row), where lane 0 ran them for each output in turn. Every output gets the
/// same sum, rounding, ReLU² and addresses; only the storing lane changes. Used only when R * RPS <= 32 (else
/// the lane-0 loop runs). nil when no loop matches.
func nemotronQMVVectorStoreSource(_ source: String) -> String? {
    let pattern = "(?m)^( *)for \\(int row = 0; row < results_per_simdgroup; row\\+\\+\\) \\{\\n"
        + "\\1  for \\(int r = 0; r < R; r\\+\\+\\) \\{\\n"
        + "\\1    U v = simd_sum\\(result\\[r\\]\\[row\\]\\);\\n"
        + "\\1    if \\(simd_lid == 0\\) \\{\\n"
        + "(\\1      T h = static_cast<T>\\(v\\);\\n(?:\\1      [^\\n]*\\n)*?)"
        + "\\1    \\}\\n"
        + "\\1  \\}\\n"
        + "\\1\\}(?:\\n|$)"
    guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
    let matches = regex.matches(in: source, range: NSRange(source.startIndex..., in: source))
    guard !matches.isEmpty else { return nil }
    var s = source
    // Last match first, so the earlier ranges stay valid.
    for match in matches.reversed() {
        guard let whole = Range(match.range, in: s), let indent = Range(match.range(at: 1), in: s),
              let bodyRange = Range(match.range(at: 2), in: s) else { return nil }
        let i = String(s[indent])
        let loop = s[whole].hasSuffix("\n") ? String(s[whole]) : String(s[whole]) + "\n"
        let body = String(s[bodyRange])
        let vector = "\(i)if (R * results_per_simdgroup <= 32) {\n"
            + "\(i)  U nmq_kept = 0;\n"
            + "\(i)  for (int row = 0; row < results_per_simdgroup; row++) {\n"
            + "\(i)    for (int r = 0; r < R; r++) {\n"
            + "\(i)      U v = simd_sum(result[r][row]);\n"
            + "\(i)      nmq_kept = simd_lid == uint(r * results_per_simdgroup + row) ? v : nmq_kept;\n"
            + "\(i)    }\n"
            + "\(i)  }\n"
            + "\(i)  if (simd_lid < uint(R * results_per_simdgroup)) {\n"
            + "\(i)    const int r = int(simd_lid) / results_per_simdgroup;\n"
            + "\(i)    const int row = int(simd_lid) % results_per_simdgroup;\n"
            + "\(i)    const U v = nmq_kept;\n"
            + body
            + "\(i)  }\n"
            + "\(i)} else {\n"
            + loop
            + "\(i)}\n"
        s.replaceSubrange(whole, with: vector)
    }
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
    /// `kernel` and `tail8` reading (scale, bias) through the draft head's
    /// routed FC1 pair dictionary (`NemotronSBDict.draftDict`), or nil.
    let kernelDict: MLXFast.MLXFastKernel?
    let tail8Dict: MLXFast.MLXFastKernel?
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
        kernelDict = nemotronSBDictSource(nemotronMoELeaderSource).map {
            MLXFast.metalKernel(
                name: "nemotron_moe_leader_qmv_affine4_gs64_sbd",
                inputNames: ["w", "scales", "biases", "x", "inds"],
                outputNames: ["y"],
                source: $0,
                header: nemotronMoELeaderHeader + nemotronMoELeaderStore)
        }
        tail8Dict = nmlTail8Source(nemotronMoELeaderSource).flatMap(nemotronSBDictSource).map {
            MLXFast.metalKernel(
                name: "nemotron_moe_leader_qmv_tail8_affine4_gs64_sbd",
                inputNames: ["w", "scales", "biases", "x", "inds"],
                outputNames: ["y"],
                source: $0,
                header: nemotronMoELeaderHeader + nemotronMoELeaderStore)
        }
        // The two dictionary twins with both main K loops unrolled by two
        // (NEMOTRON_U2_DRAFT; the draft head's one-row routed FC1 only).
        kernelDictU2 = nemotronU2Source(nemotronSBDictSource(nemotronMoELeaderSource),
                                        loop: NemotronU2.leaderLoop, count: 2).map {
            MLXFast.metalKernel(
                name: "nemotron_moe_leader_qmv_affine4_gs64_sbd_u2",
                inputNames: ["w", "scales", "biases", "x", "inds"],
                outputNames: ["y"],
                source: $0,
                header: nemotronMoELeaderHeader + nemotronMoELeaderStore)
        }
        tail8DictU2 = nemotronU2Source(nmlTail8Source(nemotronMoELeaderSource).flatMap(nemotronSBDictSource),
                                       loop: NemotronU2.leaderLoop, count: 2).map {
            MLXFast.metalKernel(
                name: "nemotron_moe_leader_qmv_tail8_affine4_gs64_sbd_u2",
                inputNames: ["w", "scales", "biases", "x", "inds"],
                outputNames: ["y"],
                source: $0,
                header: nemotronMoELeaderHeader + nemotronMoELeaderStore)
        }
    }
    let kernelDictU2: MLXFast.MLXFastKernel?
    let tail8DictU2: MLXFast.MLXFastKernel?
}

private final class NemotronMoELeaderPreparedKernels: Sendable {
    static let shared = NemotronMoELeaderPreparedKernels()
    let gateUp: MLXFast.MLXFastKernel?
    let gateUpAndPlan: MLXFast.MLXFastKernel?
    let tripleGateUpAndPlan: MLXFast.MLXFastKernel?
    /// The same triple FC1 with its tail block in the constant 8-value form.
    let tripleGateUpAndPlanTail8: MLXFast.MLXFastKernel?
    /// The two triple FC1 kernels reading (scale, bias) through `NemotronSBDict`.
    let tripleGateUpAndPlanDict: MLXFast.MLXFastKernel?
    let tripleGateUpAndPlanTail8Dict: MLXFast.MLXFastKernel?
    let tripleGateUp: MLXFast.MLXFastKernel?
    let tripleGateUpTail8: MLXFast.MLXFastKernel?
    let tripleGateUpDict: MLXFast.MLXFastKernel?
    let tripleGateUpTail8Dict: MLXFast.MLXFastKernel?
    /// The four plan-free triple FC1 kernels with their main K loops unrolled
    /// by two (nmlFC1UnrollSource; NEMOTRON_MTP_FC1_UNROLL).
    let tripleGateUpUnrolled: MLXFast.MLXFastKernel?
    let tripleGateUpTail8Unrolled: MLXFast.MLXFastKernel?
    let tripleGateUpDictUnrolled: MLXFast.MLXFastKernel?
    let tripleGateUpTail8DictUnrolled: MLXFast.MLXFastKernel?
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
        tripleGateUpAndPlanDict = nmlPreparedSource(tripleSource, pointers: triplePointers)
            .map(nmlFC1WithBalancePlanSource).flatMap(nemotronSBDictSource).map {
            MLXFast.metalKernel(
                name: "nemotron_moe_triple_prepared_fc1_with_pair_fc2_plan_sbd",
                inputNames: ["w", "scales", "biases", "prepared", "inds"],
                outputNames: ["y", "plan"], source: $0,
                header: header + nemotronMoETripleDecodeHeader)
        }
        tripleGateUpAndPlanTail8Dict = nmlTail8Source(tripleSource)
            .flatMap { nmlPreparedSource($0, pointers: triplePointers) }
            .map(nmlFC1WithBalancePlanSource).flatMap(nemotronSBDictSource).map {
            MLXFast.metalKernel(
                name: "nemotron_moe_triple_prepared_fc1_tail8_with_pair_fc2_plan_sbd",
                inputNames: ["w", "scales", "biases", "prepared", "inds"],
                outputNames: ["y", "plan"], source: $0,
                header: header + nemotronMoETripleDecodeHeader)
        }
        tripleGateUp = nmlPreparedSource(tripleSource, pointers: triplePointers).map {
            MLXFast.metalKernel(
                name: "nemotron_moe_triple_prepared_fc1",
                inputNames: ["w", "scales", "biases", "prepared", "inds"],
                outputNames: ["y"], source: $0,
                header: header + nemotronMoETripleDecodeHeader)
        }
        tripleGateUpTail8 = nmlTail8Source(tripleSource)
            .flatMap { nmlPreparedSource($0, pointers: triplePointers) }.map {
            MLXFast.metalKernel(
                name: "nemotron_moe_triple_prepared_fc1_tail8",
                inputNames: ["w", "scales", "biases", "prepared", "inds"],
                outputNames: ["y"], source: $0,
                header: header + nemotronMoETripleDecodeHeader)
        }
        tripleGateUpDict = nmlPreparedSource(tripleSource, pointers: triplePointers)
            .flatMap(nemotronSBDictSource).map {
            MLXFast.metalKernel(
                name: "nemotron_moe_triple_prepared_fc1_sbd",
                inputNames: ["w", "scales", "biases", "prepared", "inds"],
                outputNames: ["y"], source: $0,
                header: header + nemotronMoETripleDecodeHeader)
        }
        tripleGateUpTail8Dict = nmlTail8Source(tripleSource)
            .flatMap { nmlPreparedSource($0, pointers: triplePointers) }
            .flatMap(nemotronSBDictSource).map {
            MLXFast.metalKernel(
                name: "nemotron_moe_triple_prepared_fc1_tail8_sbd",
                inputNames: ["w", "scales", "biases", "prepared", "inds"],
                outputNames: ["y"], source: $0,
                header: header + nemotronMoETripleDecodeHeader)
        }
        // The same four plan-free kernels from the unrolled main loops (the
        // tail-8, prepared and dictionary transforms follow it unchanged).
        let unrolledSource = nmlFC1UnrollSource(tripleSource)
        tripleGateUpUnrolled = unrolledSource
            .flatMap { nmlPreparedSource($0, pointers: triplePointers) }.map {
            MLXFast.metalKernel(
                name: "nemotron_moe_triple_prepared_fc1_u2",
                inputNames: ["w", "scales", "biases", "prepared", "inds"],
                outputNames: ["y"], source: $0,
                header: header + nemotronMoETripleDecodeHeader)
        }
        tripleGateUpTail8Unrolled = unrolledSource.flatMap(nmlTail8Source)
            .flatMap { nmlPreparedSource($0, pointers: triplePointers) }.map {
            MLXFast.metalKernel(
                name: "nemotron_moe_triple_prepared_fc1_tail8_u2",
                inputNames: ["w", "scales", "biases", "prepared", "inds"],
                outputNames: ["y"], source: $0,
                header: header + nemotronMoETripleDecodeHeader)
        }
        tripleGateUpDictUnrolled = unrolledSource
            .flatMap { nmlPreparedSource($0, pointers: triplePointers) }
            .flatMap(nemotronSBDictSource).map {
            MLXFast.metalKernel(
                name: "nemotron_moe_triple_prepared_fc1_sbd_u2",
                inputNames: ["w", "scales", "biases", "prepared", "inds"],
                outputNames: ["y"], source: $0,
                header: header + nemotronMoETripleDecodeHeader)
        }
        tripleGateUpTail8DictUnrolled = unrolledSource.flatMap(nmlTail8Source)
            .flatMap { nmlPreparedSource($0, pointers: triplePointers) }
            .flatMap(nemotronSBDictSource).map {
            MLXFast.metalKernel(
                name: "nemotron_moe_triple_prepared_fc1_tail8_sbd_u2",
                inputNames: ["w", "scales", "biases", "prepared", "inds"],
                outputNames: ["y"], source: $0,
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
    /// dict: the draft head's routed FC1 pair dictionary (one row only).
    static func apply(_ x: MLXArray, _ inds: MLXArray, rows: Int, topK: Int,
                      bank: Bank, perSlot: Bool, relu2: Bool = false,
                      prepared: MLXArray? = nil, tail8: Bool = false,
                      dict: NemotronSBDict.Dict? = nil, u2: Bool? = nil) -> MLXArray {
        let rps = leaderResultsPerSimdgroup(rows: rows)
        if preparedGateUp, rows == 3, !perSlot, bank.k % 64 == 0, bank.k % 512 != 0,
           let kernel = NemotronMoELeaderPreparedKernels.shared.gateUp {
            let packed = NemotronHeadPreparedActivation.compatible(prepared, rows: rows, k: bank.k)
                ?? NemotronHeadPreparedActivation.prepareContiguousRows(x, pv: 8)
            return kernel(
                [NemotronBankBinding.bind(bank.weight), NemotronBankBinding.bind(bank.scales),
                 NemotronBankBinding.bind(bank.biases), packed, inds],
                template: [("T", x.dtype), ("R", rows), ("TOPK", topK),
                           ("K", bank.k), ("N", bank.n), ("XSLOT", perSlot),
                           ("RELU2", relu2), ("RPS", rps)],
                grid: (32, bank.n / rps, rows * topK),
                threadGroup: (32, 2, 1),
                outputShapes: [[rows * topK, bank.n]],
                outputDTypes: [x.dtype])[0]
        }
        let kernels = NemotronMoELeaderKernel.shared
        // `NemotronDraftTune` (fc1): the geometry adopted at load for the draft head's one-row dictionary launch
        // (unread `dep`: the input row).
        if let dict, rows == 1, u2 == nil, !perSlot, relu2, topK == 6, bank.k == 2688, bank.n == 1856,
           let g = NemotronDraftTune.adopted[.fc1],
           let y = draftTuneLaunch(g, x, inds, bank: bank, dict: dict, relu2: relu2,
                                   tail8: tail8 && kernels.tail8 != nil, dep: x) {
            return y
        }
        let kernel = (tail8 ? kernels.tail8 : nil) ?? kernels.kernel
        // `dict`: the selected kernel's twin, reading the bank's pair indices
        // (addressed from their base like the scales) and its FP32 pair table.
        // `u2` (nil: NEMOTRON_U2_DRAFT once admitted): that twin with its main loops unrolled by two.
        let unrolled = (u2 ?? NemotronU2.draft)
            ? (tail8 && kernels.tail8 != nil ? kernels.tail8DictU2 : kernels.kernelDictU2) : nil
        if let dict, rows == 1,
           let twin = unrolled ?? (tail8 && kernels.tail8 != nil ? kernels.tail8Dict : kernels.kernelDict) {
            return twin(
                [NemotronBankBinding.bind(bank.weight), NemotronBankBinding.bind(dict.index),
                 dict.table, x, inds],
                template: [("T", x.dtype), ("R", rows), ("TOPK", topK),
                           ("K", bank.k), ("N", bank.n), ("XSLOT", perSlot),
                           ("RELU2", relu2), ("RPS", rps)],
                grid: (32, bank.n / rps, rows * topK),
                threadGroup: (32, 2, 1),
                outputShapes: [[rows * topK, bank.n]],
                outputDTypes: [x.dtype])[0]
        }
        return kernel(
            [NemotronBankBinding.bind(bank.weight), NemotronBankBinding.bind(bank.scales),
             NemotronBankBinding.bind(bank.biases), x, inds],
            template: [("T", x.dtype), ("R", rows), ("TOPK", topK),
                       ("K", bank.k), ("N", bank.n), ("XSLOT", perSlot),
                       ("RELU2", relu2), ("RPS", rps)],
            grid: (32, bank.n / rps, rows * topK),
            threadGroup: (32, 2, 1),
            outputShapes: [[rows * topK, bank.n]],
            outputDTypes: [x.dtype])[0]
    }

    /// Geometry of the plan-free triple FC1: one SIMD group per threadgroup
    /// with two outputs each (NEMOTRON_MTP_FC1_NSG, NEMOTRON_MTP_FC1_RPS;
    /// NEMOTRON_MTP_FC1_RPS=4 restores four outputs, NSG 2 with the leader's
    /// narrow two the older threadgroups). Only the output-to-SIMD-group map
    /// moves: every output keeps its lane split, K order, qdot and closing
    /// simd_sum.
    static let fc1SimdGroups = max(1, Int(ProcessInfo.processInfo.environment["NEMOTRON_MTP_FC1_NSG"] ?? "") ?? 1)
    static let fc1Results: Int? = Int(ProcessInfo.processInfo.environment["NEMOTRON_MTP_FC1_RPS"] ?? "") ?? 2
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
    /// The plan-free triple FC1 with its main K loops unrolled by two
    /// (nmlFC1UnrollSource); =0 keeps the record loops. Selected only after
    /// `qualifyFC1Unroll` found the unrolled kernel bit for bit equal to the
    /// record kernel on the real weights of every routed FC1 bank.
    static let fc1Unroll =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_FC1_UNROLL"] != "0"
    nonisolated(unsafe) private(set) static var fc1UnrollQualified = false
    /// The same constant tail for the R != 3 (draft tail) leader and weighted
    /// kernels, through production-only kernel copies.
    static let leaderTail8 =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_LEADER_TAIL8"] != "0"

    /// The ordinary prepared R3/top-six FC1 arithmetic plus the original
    /// integer FC2 plan, emitted by just the first leader/output tile.
    /// Both outputs are fresh current-call arrays; no planner state persists.
    /// The dictionary twin of the triple FC1 this process selects, if built.
    static var dictFC1Available: Bool {
        let kernels = NemotronMoELeaderPreparedKernels.shared
        return tripleGateUp && (fc1Tail8 && kernels.tripleGateUpAndPlanTail8 != nil
            ? kernels.tripleGateUpAndPlanTail8Dict : kernels.tripleGateUpAndPlanDict) != nil
    }
    /// The dictionary twin of the R = 1 leader `apply` selects with
    /// `tail8: leaderTail8` (the draft tail's FC1), if built.
    static var draftDictFC1Available: Bool {
        let kernels = NemotronMoELeaderKernel.shared
        return (leaderTail8 && kernels.tail8 != nil ? kernels.tail8Dict : kernels.kernelDict) != nil
    }

    static func applyWithBalancePlan(
        _ x: MLXArray, _ inds: MLXArray, bank: Bank, relu2: Bool,
        prepared: MLXArray?, dict: NemotronSBDict.Dict? = nil
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
        // The selected triple kernel's dictionary twin, when `dict` is given.
        let kernels = NemotronMoELeaderPreparedKernels.shared
        let twin = dict == nil || !tripleGateUp ? nil
            : (fc1Tail8 && kernels.tripleGateUpAndPlanTail8 != nil
                ? kernels.tripleGateUpAndPlanTail8Dict
                : (kernels.tripleGateUpAndPlan != nil ? kernels.tripleGateUpAndPlanDict : nil))
        let result = (twin != nil && dict != nil ? twin! : kernel)(
            twin != nil && dict != nil
                ? [NemotronBankBinding.bind(bank.weight), NemotronBankBinding.bind(dict!.index), dict!.table, packed, inds]
                : [NemotronBankBinding.bind(bank.weight), NemotronBankBinding.bind(bank.scales),
                   NemotronBankBinding.bind(bank.biases), packed, inds],
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

    static let elideUnreadPlan =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_FC1_PLAN_ELIDE"] != "0"

    /// unroll: the unrolled main loops (nil: `fc1Unroll` once qualified); the
    /// record kernel stands in whenever its unrolled twin is missing.
    /// tuned: launch this `NemotronFC1Tune` geometry (nil when it cannot run);
    /// otherwise, with `unroll` nil, the geometry the load-time tune adopted.
    /// dep: the tune geometry's unread `dep` binding (nil: its prepared activation).
    static func applyTripleWithoutPlan(
        _ x: MLXArray, _ inds: MLXArray, bank: Bank, relu2: Bool,
        prepared: MLXArray?, dict: NemotronSBDict.Dict? = nil, unroll: Bool? = nil,
        tuned: NemotronFC1Tune.Variant? = nil, dep: MLXArray? = nil
    ) -> MLXArray? {
        let kernels = NemotronMoELeaderPreparedKernels.shared
        guard elideUnreadPlan, fuseBalancePlan, preparedGateUp, tripleGateUp,
              x.shape == [3, 2688], x.dtype == .bfloat16,
              inds.shape == [18], inds.dtype == .uint32,
              bank.weight.shape == [128, 1856, 336], bank.k == 2688, bank.n == 1856
        else { return nil }
        let tail8 = fc1Tail8 && kernels.tripleGateUpAndPlanTail8 != nil
        guard tail8 || kernels.tripleGateUpAndPlan != nil else { return nil }
        let useDict = dict != nil
            && (tail8 ? kernels.tripleGateUpAndPlanTail8Dict : kernels.tripleGateUpAndPlanDict) != nil
        guard let record = useDict
            ? (tail8 ? kernels.tripleGateUpTail8Dict : kernels.tripleGateUpDict)
            : (tail8 ? kernels.tripleGateUpTail8 : kernels.tripleGateUp)
        else { return nil }
        let unrolled = unroll ?? (fc1Unroll && fc1UnrollQualified)
        let kernel = (unrolled
            ? (useDict
                ? (tail8 ? kernels.tripleGateUpTail8DictUnrolled : kernels.tripleGateUpDictUnrolled)
                : (tail8 ? kernels.tripleGateUpTail8Unrolled : kernels.tripleGateUpUnrolled))
            : nil) ?? record
        let packed = NemotronHeadPreparedActivation.compatible(prepared, rows: 3, k: bank.k)
            ?? NemotronHeadPreparedActivation.prepareContiguousRows(x, pv: 8)
        let rps = fc1Results ?? leaderResultsPerSimdgroup(rows: 3)
        let nsg = (bank.n / rps) % fc1SimdGroups == 0 ? fc1SimdGroups : 2
        var retuned: MLXArray? = nil
        if unroll == nil, let v = tuned ?? NemotronFC1Tune.choice {
            retuned = NemotronFC1Tune.launch(v, packed: packed, inds: inds, bank: bank,
                dict: useDict ? dict : nil, relu2: relu2, tail8: tail8, explicit: tuned != nil, dep: dep)
            if retuned == nil, tuned != nil { return nil }
        }
        let activation = retuned ?? kernel(
            useDict
                ? [NemotronBankBinding.bind(bank.weight), NemotronBankBinding.bind(dict!.index), dict!.table, packed, inds]
                : [NemotronBankBinding.bind(bank.weight), NemotronBankBinding.bind(bank.scales),
                   NemotronBankBinding.bind(bank.biases), packed, inds],
            template: [("T", x.dtype), ("R", 3), ("TOPK", 6),
                       ("K", bank.k), ("N", bank.n), ("XSLOT", false),
                       ("RELU2", relu2), ("RPS", rps)],
            grid: (32, bank.n / rps, 18), threadGroup: (32, nsg, 1),
            outputShapes: [[18, bank.n]], outputDTypes: [x.dtype])[0]
        guard selfTest else { return activation }
        let expected = apply(x, inds, rows: 3, topK: 6, bank: bank,
            perSlot: false, relu2: relu2, prepared: packed)
        let ok = (activation.view(dtype: .uint16) .== expected.view(dtype: .uint16)).all()
        return which(ok, activation, MLXArray(Float.nan).asType(x.dtype))
    }

    /// Load time, after the pair dictionaries: the unrolled plan-free triple
    /// FC1 against the record kernel, bit for bit, on the real weights of
    /// every routed FC1 bank, with routes that give one expert all three
    /// rows, one expert two rows and the rest one row each (both ReLU2 forms).
    /// Any difference, or a missing twin, keeps the record kernel.
    static func qualifyFC1Unroll(_ model: Module) {
        guard enabled else { return }
        guard fc1Unroll else {
            FileHandle.standardError.write(Data("nemotron fc1 unroll: off (setting)\n".utf8))
            return
        }
        let kernels = NemotronMoELeaderPreparedKernels.shared
        guard kernels.tripleGateUpUnrolled != nil, kernels.tripleGateUpTail8Unrolled != nil,
              kernels.tripleGateUpDictUnrolled != nil, kernels.tripleGateUpTail8DictUnrolled != nil
        else {
            FileHandle.standardError.write(
                Data("nemotron fc1 unroll off: the record main loop moved, no unrolled kernel\n".utf8))
            return
        }
        let keys = MLXRandom.split(key: MLXRandom.key(0x5bd4), into: 2)
        let routes: [[UInt32]] = [
            [3, 17, 42, 77, 100, 121, 17, 5, 64, 90, 3, 11, 3, 8, 50, 66, 101, 127],
            [96, 30, 1, 65, 127, 33, 65, 2, 96, 40, 12, 7, 33, 96, 9, 19, 64, 65],
        ]
        var same = MLXArray(true)
        var layers = 0
        for module in model.modules() {
            guard let moe = module as? NemotronHMoE,
                  type(of: moe.switchMLP.fc1) == QuantizedSwitchLinear.self,
                  let up = bank(moe.switchMLP.fc1, k: 2688, dtype: .bfloat16),
                  up.weight.shape == [128, 1856, 336] else { continue }
            let dict = NemotronSBDict.enabled ? NemotronSBDict.dict(moe.switchMLP.fc1) : nil
            var ran = false
            for (r, route) in routes.enumerated() {
                let x = (MLXRandom.normal([3, 2688], key: keys[r]) * 0.5).asType(.bfloat16)
                let inds = MLXArray(route)
                for relu2 in [false, true] {
                    guard let a = applyTripleWithoutPlan(
                              x, inds, bank: up, relu2: relu2, prepared: nil, dict: dict, unroll: false),
                          let b = applyTripleWithoutPlan(
                              x, inds, bank: up, relu2: relu2, prepared: nil, dict: dict, unroll: true)
                    else { continue }
                    same = logicalAnd(same, (a.view(dtype: .uint16) .== b.view(dtype: .uint16)).all())
                    ran = true
                }
            }
            if ran { layers += 1 }
        }
        guard layers > 0, same.item(Bool.self) else {
            FileHandle.standardError.write(Data(
                "nemotron fc1 unroll off: \(layers > 0 ? "differs from the record kernel" : "no routed FC1 bank ran")\n".utf8))
            return
        }
        fc1UnrollQualified = true
        FileHandle.standardError.write(Data(
            "nemotron fc1 unroll on: \(layers) routed FC1 banks, plan-free triple FC1 bitwise equal to the record kernel\n".utf8))
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
            [NemotronBankBinding.bind(bank.weight), NemotronBankBinding.bind(bank.scales),
             NemotronBankBinding.bind(bank.biases), x, inds, scores],
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
        _ mlp: NemotronHSwitchMLP, prepared: MLXArray? = nil, products: Bool = false,
        colaunch: NemotronFC2SharedColaunch.Shared? = nil, after: MLXArray? = nil,
        draftDep: MLXArray? = nil, activation: ((MLXArray) -> Void)? = nil
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
        // One row (the draft tail): the draft head bank's admitted pair dictionary, if any.
        let draftDict = rows == 1 ? NemotronSBDict.draftDict(mlp.fc1) : nil
        if fuseRelu2 {
            let dict = rows == 3 && NemotronSBDict.enabled ? NemotronSBDict.dict(mlp.fc1) : nil
            if rows == 3, !NemotronZSkip.needsOriginal(mlp.fc2), NemotronZSkip.covers(mlp.fc2),
               let activation = applyTripleWithoutPlan(
                   xr, flat, bank: up, relu2: true, prepared: prepared, dict: dict) {
                act = activation
                balancePlan = nil
            } else if rows == 3, let fused = applyWithBalancePlan(
                xr, flat, bank: up, relu2: true, prepared: prepared, dict: dict) {
                act = fused.activation
                balancePlan = fused.plan
            } else {
                act = apply(xr, flat, rows: rows, topK: 6, bank: up, perSlot: false,
                    relu2: true, prepared: prepared, tail8: leaderTail8, dict: draftDict)
                balancePlan = nil
            }
        } else {
            let h = apply(xr, flat, rows: rows, topK: 6, bank: up, perSlot: false,
                prepared: prepared, tail8: leaderTail8, dict: draftDict)
            let a = maximum(h, MLXArray(0))
            act = a * a
            balancePlan = nil
        }
        let weights = scores.reshaped(rows * 6)
        // `NemotronDraftColaunch` (draft tail): the FC1 rows, as the shared down's tape-order input.
        activation?(act)
        // `products`: the unreduced slot products for the captured verify's
        // (R = 3) or the draft tail's (R = 1) join kernel, which sums them
        // itself (same expressions and order).
        if products, rows == 3, let colaunch,
           let pair = NemotronFC2SharedColaunch.run(act, flat, weights, fc2: mlp.fc2, shared: colaunch) {
            return pair
        }
        if products, let prod = rows == 3
            ? NemotronZSkip.weightedProducts(act, flat, weights, owner: mlp.fc2)
            : (rows == 1 ? NemotronZSkip.draftProducts(
                after.map { depends(input: act, dependencies: [$0]) } ?? act, flat, weights, owner: mlp.fc2,
                dep: draftDep) : nil) {
            guard selfTest else { return prod }
            let original = switchRows(x, indices, mlp) ?? mlp(x, indices)
            let expected = (original * scores[.ellipsis, .newAxis])
                .sum(axis: -2).asType(original.dtype)
            let ok = (NemotronZSkip.reduceProducts(prod, rows: rows).view(dtype: .uint16)
                .== expected.view(dtype: .uint16)).all()
            return which(ok, prod, MLXArray(Float.nan))
        }
        var y = NemotronZSkip.weighted(act, flat, weights, rows: rows, owner: mlp.fc2)
            ?? applyWeighted(act, flat, weights, rows: rows, bank: down, plan: NemotronZSkip.needsOriginal(mlp.fc2) ? nil : balancePlan,
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
        return kernel([NemotronBankBinding.bind(bank.weight), NemotronBankBinding.bind(bank.scales),
                       NemotronBankBinding.bind(bank.biases), x, ids, scores, plan],
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

// The tiled slot kernel with exact fast paths for chunks with one or two
// non-zero activations (the relu2 rows are ~84% exact zeros). A chunk's mask
// is uniform across the SIMD group (all 32 lanes meet the same chunk), so the
// branches do not diverge. Every term x_thread[i] * code is exact in FP32 (a
// BF16 value scaled by a power of two times a nibble shifted by a multiple of
// four: at most 12 significant bits), so the record qdot's zero terms add +0
// and its sum is the in-order sum of the non-zero terms: x1 * c1 (+ x2 * c2).
// The chunk sum likewise reduces to x1 (+ x2), and the closing
// scale * accum + sum * bias keeps the record's form. Chunks with three or
// more non-zero values take the tiled kernel's statements unchanged.
// In the sparse paths the nibble shift cancels the activation's power-of-two
// scale exactly when the BF16 magnitude is at least 2^-110 and finite. Smaller
// or non-finite values retain the complete tiled arithmetic below, avoiding
// flush-to-zero differences from cancelling a subnormal intermediate.
private let nemotronZSkipTiledFastSource = """
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
    uint m = 0, unsafe = 0;
    for (int i = 0; i < 8; i++) {
      const uint ab = uint(xb[i]) & 0x7fffu;
      m |= ab != 0 ? (1u << i) : 0u;
      unsafe |= ab != 0 && (ab < 0x0880u || ab >= 0x7f80u) ? 1u : 0u;
      xsh[8 * c + i] = xs[8 * c + i];
    }
    // A full-mask sentinel sends tiny/nonfinite chunks to the unchanged dense
    // arithmetic, without a second shared array or per-output exponent tests.
    cmask[c] = unsafe != 0 ? uchar(255) : uchar(m);
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
      if (popcount(m) == 1) {
        const int j1 = ctz(m);
        const T xvj = xsh[8 * c + j1];
        const U xtj = U(xvj);
        const U sum1 = U(xvj);
        const uint w = uint(wq[(8 * c + j1) * 32]);
        const int gi = c / 8;
        const uint2 sr = *((const device uint2*)(sq + gi * OT));
        const uint2 br = *((const device uint2*)(bq + gi * OT));
        const uint sv[4] = {sr.x << 16, sr.x & 0xffff0000u, sr.y << 16, sr.y & 0xffff0000u};
        const uint bv[4] = {br.x << 16, br.x & 0xffff0000u, br.y << 16, br.y & 0xffff0000u};
        for (int o = 0; o < 4; o++) {
          const uint code = ((w >> (4 * o)) & 0xfu);
          U accum = 0;
          accum += xtj * U(int(code));
          p[o] += as_type<float>(sv[o]) * accum + sum1 * as_type<float>(bv[o]);
        }
        continue;
      }
      if (popcount(m) == 2) {
        const int j1 = ctz(m);
        const int j2 = ctz(m & (m - 1));
        const uint w1 = uint(wq[(8 * c + j1) * 32]);
        const uint w2 = uint(wq[(8 * c + j2) * 32]);
        const T xv1 = xsh[8 * c + j1];
        const T xv2 = xsh[8 * c + j2];
        const U xt1 = U(xv1);
        const U xt2 = U(xv2);
        const U sum2 = U(xv1) + U(xv2);
        const int gi = c / 8;
        const uint2 sr = *((const device uint2*)(sq + gi * OT));
        const uint2 br = *((const device uint2*)(bq + gi * OT));
        const uint sv[4] = {sr.x << 16, sr.x & 0xffff0000u, sr.y << 16, sr.y & 0xffff0000u};
        const uint bv[4] = {br.x << 16, br.x & 0xffff0000u, br.y << 16, br.y & 0xffff0000u};
        for (int o = 0; o < 4; o++) {
          const U c1 = U(int(((w1 >> (4 * o)) & 0xfu)));
          const U c2 = U(int(((w2 >> (4 * o)) & 0xfu)));
          const U accum = xt1 * c1 + xt2 * c2;
          p[o] += as_type<float>(sv[o]) * accum + sum2 * as_type<float>(bv[o]);
        }
        continue;
      }
      if (popcount(m) == 3) {
        const int j1 = ctz(m);
        const uint m2 = m & (m - 1);
        const int j2 = ctz(m2);
        const int j3 = ctz(m2 & (m2 - 1));
        const uint w1 = uint(wq[(8 * c + j1) * 32]);
        const uint w2 = uint(wq[(8 * c + j2) * 32]);
        const uint w3 = uint(wq[(8 * c + j3) * 32]);
        const T xv1 = xsh[8 * c + j1];
        const T xv2 = xsh[8 * c + j2];
        const T xv3 = xsh[8 * c + j3];
        const U xt1 = U(xv1);
        const U xt2 = U(xv2);
        const U xt3 = U(xv3);
        // The record adds each half (values 0-3, then 4-7) left to right before adding the second half to
        // the first, so one value in the first half and two in the second associate as a + (b + c).
        const bool split = j1 < 4 && j2 >= 4;
        const U sum3 = split ? U(xv1) + (U(xv2) + U(xv3)) : (U(xv1) + U(xv2)) + U(xv3);
        const int gi = c / 8;
        const uint2 sr = *((const device uint2*)(sq + gi * OT));
        const uint2 br = *((const device uint2*)(bq + gi * OT));
        const uint sv[4] = {sr.x << 16, sr.x & 0xffff0000u, sr.y << 16, sr.y & 0xffff0000u};
        const uint bv[4] = {br.x << 16, br.x & 0xffff0000u, br.y << 16, br.y & 0xffff0000u};
        for (int o = 0; o < 4; o++) {
          const U c1 = U(int(((w1 >> (4 * o)) & 0xfu)));
          const U c2 = U(int(((w2 >> (4 * o)) & 0xfu)));
          const U c3 = U(int(((w3 >> (4 * o)) & 0xfu)));
          const U t1 = xt1 * c1;
          const U t2 = xt2 * c2;
          const U t3 = xt3 * c3;
          const U accum = split ? t1 + (t2 + t3) : (t1 + t2) + t3;
          p[o] += as_type<float>(sv[o]) * accum + sum3 * as_type<float>(bv[o]);
        }
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

// A tiled slot kernel's source reading each group's (scale, bias) through the
// routed FC2 bank's `NemotronSBDict` pair dictionary: `sk` holds the
// tile-major uint16 pair index (one per group and output) and `bk` the bank's
// FP32 pair table. A lane's four indices load as one 8-byte word; each table
// entry is the exact BF16 -> FP32 widening of the pair, so `sv` and `bv` hold
// the same bits as the shifted BF16 words and every product is unchanged.
// Nil when the source no longer has the expected statements.
private func nemotronZSkipDictSource(_ source: String) -> String? {
    var s = source
    let pointers = "  const device T* sq = sk + tb * groups * OT + 4 * lid;\n"
        + "  const device T* bq = bk + tb * groups * OT + 4 * lid;\n"
    guard s.components(separatedBy: pointers).count == 2 else { return nil }
    s = s.replacingOccurrences(of: pointers, with:
        "  const device uint16_t* sq = (const device uint16_t*)sk + tb * groups * OT + 4 * lid;\n"
        + "  const device float2* sbt = (const device float2*)bk;\n")
    var replaced = 0
    for indent in ["      ", "        "] {
        let from = indent + "const uint2 sr = *((const device uint2*)(sq + gi * OT));\n"
            + indent + "const uint2 br = *((const device uint2*)(bq + gi * OT));\n"
            + indent + "const uint sv[4] = {sr.x << 16, sr.x & 0xffff0000u, sr.y << 16, sr.y & 0xffff0000u};\n"
            + indent + "const uint bv[4] = {br.x << 16, br.x & 0xffff0000u, br.y << 16, br.y & 0xffff0000u};\n"
        let to = indent + "const uint2 ir = *((const device uint2*)(sq + gi * OT));\n"
            + indent + "const float2 e0 = sbt[ir.x & 0xffffu];\n"
            + indent + "const float2 e1 = sbt[ir.x >> 16];\n"
            + indent + "const float2 e2 = sbt[ir.y & 0xffffu];\n"
            + indent + "const float2 e3 = sbt[ir.y >> 16];\n"
            + indent + "const uint sv[4] = {as_type<uint>(e0.x), as_type<uint>(e1.x), as_type<uint>(e2.x), as_type<uint>(e3.x)};\n"
            + indent + "const uint bv[4] = {as_type<uint>(e0.y), as_type<uint>(e1.y), as_type<uint>(e2.y), as_type<uint>(e3.y)};\n"
        replaced += s.components(separatedBy: from).count - 1
        s = s.replacingOccurrences(of: from, with: to)
    }
    guard replaced > 0, !s.contains("bq"), !s.contains("sr."), !s.contains("br.") else { return nil }
    return s
}

// The closing loop of a zero-skip slot kernel (and of the wide body's two halves) with a vector store: lane q of
// each SIMD group keeps the sum of iteration q, the same simd_sum of the same lane partials, and lanes
// 0 ... OT / SGS - 1 then round and store their sums at once, with the loop's nml_store and store statement for
// output n = sg * (OT / SGS) + lane, where lane 0 rounded and stored each sum in turn. Every loop of `source` is
// rewritten; nil when none matches or a lane-0 store remains.
// The general chunk path of a fast tiled slot kernel (a chunk with four or more non-zero activations, or a tiny or
// non-finite one) loading all eight code words of its chunk. The masked form skips the words of the chunk's +-0
// activations; such a word only enters the qdot as x * (code & mask) with x = +-0 and a non-negative finite code,
// the same signed zero as x * 0, so every term, sum, rounding and store is unchanged. nil unless the masked loads
// occur exactly once.
private func nemotronZSkipUnmaskedLoadsSource(_ source: String) -> String? {
    let masked = "        const uint lo = (m & (1u << i)) != 0 ? uint(wq[(8 * c + i) * 32]) : 0u;\n"
        + "        const uint hi = (m & (16u << i)) != 0 ? uint(wq[(8 * c + i + 4) * 32]) : 0u;\n"
    guard source.components(separatedBy: masked).count == 2 else { return nil }
    return source.replacingOccurrences(of: masked, with: "        const uint lo = uint(wq[(8 * c + i) * 32]);\n"
        + "        const uint hi = uint(wq[(8 * c + i + 4) * 32]);\n")
}

private func nemotronZSkipVectorStoreSource(_ source: String) -> String? {
    let pattern = "(?m)^( *)for \\(int q = 0; q < OT / SGS; q\\+\\+\\) \\{\\n"
        + "\\1  const int n = sg \\* \\(OT / SGS\\) \\+ q;\\n"
        + "\\1  const U r = simd_sum\\(part\\[lid \\* PS \\+ n\\]\\);\\n"
        + "\\1  if \\(lid == 0\\) \\{\\n"
        + "\\1    const T h = nml_store<T, false>\\(r\\);\\n"
        + " *([^\\n]*;)\\n"
        + "\\1  \\}\\n"
        + "\\1\\}"
    let template = "$1U nmz_kept = 0;\n"
        + "$1for (int q = 0; q < OT / SGS; q++) {\n"
        + "$1  const int n = sg * (OT / SGS) + q;\n"
        + "$1  const U r = simd_sum(part[lid * PS + n]);\n"
        + "$1  nmz_kept = lid == q ? r : nmz_kept;\n"
        + "$1}\n"
        + "$1if (lid < OT / SGS) {\n"
        + "$1  const int n = sg * (OT / SGS) + lid;\n"
        + "$1  const T h = nml_store<T, false>(nmz_kept);\n"
        + "$1  $2\n"
        + "$1}"
    guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
    let range = NSRange(source.startIndex..., in: source)
    guard regex.numberOfMatches(in: source, range: range) > 0 else { return nil }
    let s = regex.stringByReplacingMatches(in: source, range: range, withTemplate: template)
    guard !s.contains("nml_store<T, false>(r)") else { return nil }
    return s
}

// MARK: Wide zero-skip tiles (NEMOTRON_ZSKIP_WIDE)
//
// The shared-down zero-skip copy only (the routed FC2 copies keep the tile-major layout: the request prefill
// reads it, and wide routed tiles measured slower in the co-launch). The wide copy puts the code rows of two
// neighbouring 128-output tiles 2W and 2W + 1 of the tile-major copy side by side: row k of wide tile W is one
// 128-byte line of 32 uint32 words whose low half is tile 2W's word and whose high half is tile 2W + 1's word
// for the same lane. A lane loads one uint32 for a non-zero k, so one load serves eight outputs and the
// per-chunk work (mask, fast-path dispatch, activation reads) is shared by both tiles. The 128-output tail tile
// (2688 = 10 * 256 + 128) keeps its rows, at the same addresses as in the tile-major copy, and runs the
// tile-major body verbatim. The (scale, bias) arrays are the tile-major ones, unchanged.
//
// In a wide tile, lane i owns outputs 4i .. 4i + 3 of both tiles, as in each tile's own threadgroup: the low
// half's partials p[] and the high half's partials r[] take the tile-major body's statements, chunk order,
// zero tests and fast paths, with the high half's codes shifted down by 16 bits. The partials meet their lanes
// through the same `part` array twice (low tile, then high tile), so each output's closing simd_sum sees the
// same 32 lane partials in the same lanes, and the store is unchanged. Every output therefore carries the bits
// of the tile-major kernel.
private let nemotronZSkipWideBodySource = """
    constexpr int OT = 128;
    constexpr int LPS = 32 / SGS;
    constexpr int chunks = K / 8;
    constexpr int full = chunks / 32;
    constexpr int tail = chunks - 32 * full;
    constexpr int groups = K / 64;
    constexpr int PS = OT + 1;
    typedef float U;

    const int wt = int(threadgroup_position_in_grid.x);
    const int lid = int(thread_index_in_simdgroup);
    const int sg = int(simdgroup_index_in_threadgroup);
    const uint expert = inds[slot];
    const device T* xs = x + slot * K;
    for (int c = sg * 32 + lid; c < chunks; c += SGS * 32) {
      const device uint16_t* xb = (const device uint16_t*)(xs + 8 * c);
      uint m = 0, unsafe = 0;
      for (int i = 0; i < 8; i++) {
        const uint ab = uint(xb[i]) & 0x7fffu;
        m |= ab != 0 ? (1u << i) : 0u;
        unsafe |= ab != 0 && (ab < 0x0880u || ab >= 0x7f80u) ? 1u : 0u;
        xsh[8 * c + i] = xs[8 * c + i];
      }
      cmask[c] = unsafe != 0 ? uchar(255) : uchar(m);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ulong tb = (ulong)expert * (N / OT) + 2 * wt;
    const device uint* wq = (const device uint*)(wk + tb * K * 32) + lid;
  NMW_SB_SETUP
    thread U rh[LPS * 4];
    for (int ll = 0; ll < LPS; ll++) {
      const int l = sg + SGS * ll;
      thread U p[4];
      thread U r[4];
      for (int o = 0; o < 4; o++) {
        p[o] = 0;
        r[o] = 0;
      }
      const int count = l < tail ? full + 1 : full;
      for (int j = 0; j < count; j++) {
        const int c = j < full ? l + 32 * j : 32 * full + l;
        const uint m = cmask[c];
        if (m == 0) {
          continue;
        }
        if (popcount(m) == 1) {
          const int j1 = ctz(m);
          const T xvj = xsh[8 * c + j1];
          const U xtj = U(xvj);
          const U sum1 = U(xvj);
          const uint w = wq[(8 * c + j1) * 32];
          const int gi = c / 8;
          uint sv[4], bv[4];
          nmw_sb(gi, 0, sv, bv);
          for (int o = 0; o < 4; o++) {
            const uint code = ((w >> (4 * o)) & 0xfu);
            U accum = 0;
            accum += xtj * U(int(code));
            p[o] += as_type<float>(sv[o]) * accum + sum1 * as_type<float>(bv[o]);
          }
          nmw_sb(gi, 1, sv, bv);
          for (int o = 0; o < 4; o++) {
            const uint code = ((w >> (16 + 4 * o)) & 0xfu);
            U accum = 0;
            accum += xtj * U(int(code));
            r[o] += as_type<float>(sv[o]) * accum + sum1 * as_type<float>(bv[o]);
          }
          continue;
        }
        if (popcount(m) == 2) {
          const int j1 = ctz(m);
          const int j2 = ctz(m & (m - 1));
          const uint w1 = wq[(8 * c + j1) * 32];
          const uint w2 = wq[(8 * c + j2) * 32];
          const T xv1 = xsh[8 * c + j1];
          const T xv2 = xsh[8 * c + j2];
          const U xt1 = U(xv1);
          const U xt2 = U(xv2);
          const U sum2 = U(xv1) + U(xv2);
          const int gi = c / 8;
          uint sv[4], bv[4];
          nmw_sb(gi, 0, sv, bv);
          for (int o = 0; o < 4; o++) {
            const U c1 = U(int(((w1 >> (4 * o)) & 0xfu)));
            const U c2 = U(int(((w2 >> (4 * o)) & 0xfu)));
            const U accum = xt1 * c1 + xt2 * c2;
            p[o] += as_type<float>(sv[o]) * accum + sum2 * as_type<float>(bv[o]);
          }
          nmw_sb(gi, 1, sv, bv);
          for (int o = 0; o < 4; o++) {
            const U c1 = U(int(((w1 >> (16 + 4 * o)) & 0xfu)));
            const U c2 = U(int(((w2 >> (16 + 4 * o)) & 0xfu)));
            const U accum = xt1 * c1 + xt2 * c2;
            r[o] += as_type<float>(sv[o]) * accum + sum2 * as_type<float>(bv[o]);
          }
          continue;
        }
        if (popcount(m) == 3) {
          const int j1 = ctz(m);
          const uint m2 = m & (m - 1);
          const int j2 = ctz(m2);
          const int j3 = ctz(m2 & (m2 - 1));
          const uint w1 = wq[(8 * c + j1) * 32];
          const uint w2 = wq[(8 * c + j2) * 32];
          const uint w3 = wq[(8 * c + j3) * 32];
          const T xv1 = xsh[8 * c + j1];
          const T xv2 = xsh[8 * c + j2];
          const T xv3 = xsh[8 * c + j3];
          const U xt1 = U(xv1);
          const U xt2 = U(xv2);
          const U xt3 = U(xv3);
          const bool split = j1 < 4 && j2 >= 4;
          const U sum3 = split ? U(xv1) + (U(xv2) + U(xv3)) : (U(xv1) + U(xv2)) + U(xv3);
          const int gi = c / 8;
          uint sv[4], bv[4];
          nmw_sb(gi, 0, sv, bv);
          for (int o = 0; o < 4; o++) {
            const U c1 = U(int(((w1 >> (4 * o)) & 0xfu)));
            const U c2 = U(int(((w2 >> (4 * o)) & 0xfu)));
            const U c3 = U(int(((w3 >> (4 * o)) & 0xfu)));
            const U t1 = xt1 * c1;
            const U t2 = xt2 * c2;
            const U t3 = xt3 * c3;
            const U accum = split ? t1 + (t2 + t3) : (t1 + t2) + t3;
            p[o] += as_type<float>(sv[o]) * accum + sum3 * as_type<float>(bv[o]);
          }
          nmw_sb(gi, 1, sv, bv);
          for (int o = 0; o < 4; o++) {
            const U c1 = U(int(((w1 >> (16 + 4 * o)) & 0xfu)));
            const U c2 = U(int(((w2 >> (16 + 4 * o)) & 0xfu)));
            const U c3 = U(int(((w3 >> (16 + 4 * o)) & 0xfu)));
            const U t1 = xt1 * c1;
            const U t2 = xt2 * c2;
            const U t3 = xt3 * c3;
            const U accum = split ? t1 + (t2 + t3) : (t1 + t2) + t3;
            r[o] += as_type<float>(sv[o]) * accum + sum3 * as_type<float>(bv[o]);
          }
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
        uint u[4], v[4];
        for (int i = 0; i < 4; i++) {
          const uint lo = (m & (1u << i)) != 0 ? wq[(8 * c + i) * 32] : 0u;
          const uint hi = (m & (16u << i)) != 0 ? wq[(8 * c + i + 4) * 32] : 0u;
          u[i] = (lo & 0xffffu) | (hi << 16);
          v[i] = (lo >> 16) | (hi & 0xffff0000u);
        }
        nmz_swap(u[0], u[1], 4, 0x0f0f0f0fu);
        nmz_swap(u[2], u[3], 4, 0x0f0f0f0fu);
        nmz_swap(u[0], u[2], 8, 0x00ff00ffu);
        nmz_swap(u[1], u[3], 8, 0x00ff00ffu);
        nmz_swap(v[0], v[1], 4, 0x0f0f0f0fu);
        nmz_swap(v[2], v[3], 4, 0x0f0f0f0fu);
        nmz_swap(v[0], v[2], 8, 0x00ff00ffu);
        nmz_swap(v[1], v[3], 8, 0x00ff00ffu);
        const int gi = c / 8;
        uint sv[4], bv[4];
        nmw_sb(gi, 0, sv, bv);
        for (int o = 0; o < 4; o++) {
          const uint16_t ws[2] = {uint16_t(u[o] & 0xffffu), uint16_t(u[o] >> 16)};
          p[o] += nmz_qdot<U>(ws, xt, as_type<float>(sv[o]), as_type<float>(bv[o]), sum);
        }
        nmw_sb(gi, 1, sv, bv);
        for (int o = 0; o < 4; o++) {
          const uint16_t ws[2] = {uint16_t(v[o] & 0xffffu), uint16_t(v[o] >> 16)};
          r[o] += nmz_qdot<U>(ws, xt, as_type<float>(sv[o]), as_type<float>(bv[o]), sum);
        }
      }
      for (int o = 0; o < 4; o++) {
        part[l * PS + 4 * lid + o] = p[o];
        rh[4 * ll + o] = r[o];
      }
    }
    for (int hf = 0; hf < 2; hf++) {
      if (hf == 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int ll = 0; ll < LPS; ll++) {
          const int l = sg + SGS * ll;
          for (int o = 0; o < 4; o++) {
            part[l * PS + 4 * lid + o] = rh[4 * ll + o];
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const int n0 = (2 * wt + hf) * OT;
      for (int q = 0; q < OT / SGS; q++) {
        const int n = sg * (OT / SGS) + q;
        const U r = simd_sum(part[lid * PS + n]);
        if (lid == 0) {
          const T h = nml_store<T, false>(r);
  NMW_STORE
        }
      }
    }
"""

/// The (scale, bias) bits of a lane's four outputs of group `gi` in tile 2W + h from the tile-major BF16 arrays
/// (the tiled kernel's loads and shifts). Bit moves only.
private let nemotronZSkipWidePlainSetup = """
    const device T* sq = sk + tb * groups * OT + 4 * lid;
    const device T* bq = bk + tb * groups * OT + 4 * lid;
    auto nmw_sb = [&](int gi, int h, thread uint* sv, thread uint* bv) {
      const uint2 sr = *((const device uint2*)(sq + h * groups * OT + gi * OT));
      const uint2 br = *((const device uint2*)(bq + h * groups * OT + gi * OT));
      sv[0] = sr.x << 16;
      sv[1] = sr.x & 0xffff0000u;
      sv[2] = sr.y << 16;
      sv[3] = sr.y & 0xffff0000u;
      bv[0] = br.x << 16;
      bv[1] = br.x & 0xffff0000u;
      bv[2] = br.y << 16;
      bv[3] = br.y & 0xffff0000u;
    };
"""

/// A wide slot kernel's source: `narrow` (a tile-major fast slot source with BF16 scales and biases: the shared
/// down body) runs the 128-output tail tile, the wide body every other threadgroup. Both bodies share one
/// `part`/`xsh`/`cmask` set and one slot index, declared by the same statements as the tile-major source, so
/// `NemotronFC2SharedColaunch.source` co-launches a wide source with a tile-major one as it does two tile-major
/// ones. Grid x: N / 256 + 1 threadgroups (N % 256 == 128); threadgroups past the tail tile return at once (a
/// co-launch's grid is the tile-major body's N / 128). Nil when the narrow source no longer has the expected
/// statements.
private func nemotronZSkipWideSource(narrow: String, store: String) -> String? {
    let arrays = "  threadgroup float part[32 * PS];\n  threadgroup T xsh[K];\n  threadgroup uchar cmask[chunks];\n"
    let slot = "  const int slot = int(threadgroup_position_in_grid.y);\n"
    let grid = "int(threadgroup_position_in_grid.x)"
    guard narrow.components(separatedBy: arrays).count == 2, narrow.components(separatedBy: slot).count == 2,
          narrow.components(separatedBy: grid).count == 3, narrow.contains("constexpr int OT = 128;")
    else { return nil }
    let tailBody = narrow.replacingOccurrences(of: arrays, with: "")
        .replacingOccurrences(of: slot, with: "")
        .replacingOccurrences(of: grid, with: "(N / 128 - 1)")
    let wideBody = nemotronZSkipWideBodySource
        .replacingOccurrences(of: "  NMW_SB_SETUP\n", with: nemotronZSkipWidePlainSetup)
        .replacingOccurrences(of: "  NMW_STORE\n", with: "          " + store + "\n")
    guard !wideBody.contains("NMW_") else { return nil }
    return "  constexpr int PS = 128 + 1;\n  constexpr int chunks = K / 8;\n" + arrays + slot
        + "  static_assert(N % 256 == 128, \"wide zero-skip tiles need a 128-output tail tile\");\n"
        + "  if (int(threadgroup_position_in_grid.x) > N / 256) {\n    return;\n  }\n"
        + "  if (int(threadgroup_position_in_grid.x) == N / 256) {\n" + tailBody + "  } else {\n" + wideBody + "  }\n"
}

/// The wide copy's codes from a 4-bit bank [E, N, K / 8]: the tile-major layout kernel's code words, stored at
/// the wide copy's addresses (`nemotronZSkipWideBodySource`). Output [E, N / 128 * K * 32] uint16.
private let nemotronZSkipWideLayoutSource = """
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
  const bool paired = (t & 1u) != 0 || t + 1 < N / 128;
  const ulong base = ((ulong)e * (N / 128) + (t & ~1u)) * K * 32;
  wk[paired ? base + k * 64 + 2 * lid + (t & 1u) : base + k * 32 + lid] = uint16_t(v);
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

/// MLX commits a command buffer once the distinct input buffers bound in it
/// exceed its per-buffer budget (`max_mb_per_buffer`, 40-50 MB). An expert
/// bank is one buffer of hundreds of MB, so every launch that binds a whole
/// bank ends its command buffer although it reads only the routed experts.
/// The expert kernels address a bank from its base pointer with their own
/// constants (never through the bound array's shape or strides), so binding
/// the bank's first expert, a view of the same MTLBuffer at the same base,
/// reads the same memory; only MLX's byte count of the binding changes.
/// NEMOTRON_MTP_BANK_VIEWS=0 binds the whole banks.
enum NemotronBankBinding {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_MTP_BANK_VIEWS"] != "0"
    private static let lock = NSLock()
    nonisolated(unsafe) private static var views: [ObjectIdentifier: (MLXArray, MLXArray)] = [:]

    /// The cached `bank[0 ..< 1]` of a bank recorded by `warm`, else the bank.
    /// Never evaluates (callers may be inside a compile trace).
    static func bind(_ bank: MLXArray) -> MLXArray {
        guard enabled else { return bank }
        if let cached = lock.withLock({ views[ObjectIdentifier(bank)] }), cached.0 === bank {
            return cached.1
        }
        return bank
    }

    /// Load time, outside any trace: record evaluated first-expert views of
    /// every switch-linear bank of `model` and of `extra` banks.
    static func warm(_ model: Module, extra: [MLXArray] = []) {
        guard enabled else { return }
        var banks = extra
        for module in model.modules() {
            guard let q = module as? QuantizedSwitchLinear else { continue }
            for (key, value) in q.parameters().flattened()
            where key == "weight" || key == "scales" || key == "biases" {
                banks.append(value)
            }
        }
        var made: [ObjectIdentifier: (MLXArray, MLXArray)] = [:]
        for bank in banks where bank.ndim >= 2 && bank.dim(0) > 1 {
            let view = bank[0 ..< 1]
            made[ObjectIdentifier(bank)] = (bank, view)
        }
        eval(made.values.map { $0.1 })
        lock.withLock { views.merge(made) { _, new in new } }
    }
}

enum NemotronZSkip {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_MTP_ZSKIP"] != "0"
    /// Local hosts without room for the whole copy: zero-skip only the first N MoE layers.
    static let layerLimit = Int(ProcessInfo.processInfo.environment["NEMOTRON_MTP_ZSKIP_LAYERS"] ?? "") ?? Int.max

    struct Copy {
        let codes: MLXArray
        /// The tile-major scales, or with `dict` the tile-major uint16 pair index.
        let scales: MLXArray
        /// The tile-major biases, or with `dict` the bank's FP32 pair table.
        let biases: MLXArray
        var dict = false
    }

    fileprivate static let lock = NSLock()
    nonisolated(unsafe) fileprivate static var copies: [ObjectIdentifier: Copy] = [:]
    /// The load-time copies' arrays (for `NemotronBankBinding.warm`).
    static var copyArrays: [MLXArray] {
        // A dictionary's pair table is small and stays bound whole.
        lock.withLock { copies.values.flatMap { $0.dict ? [$0.codes, $0.scales] : [$0.codes, $0.scales, $0.biases] } }
    }

    nonisolated(unsafe) private static var copyReady: [ObjectIdentifier: Bool] = [:]
    nonisolated(unsafe) private static var roundOriginal: Set<ObjectIdentifier> = []
    static func beginCopyReadiness(_ owners: [ObjectIdentifier]) {
        lock.withLock {
            copyReady = Dictionary(uniqueKeysWithValues: owners.map { ($0, false) })
            roundOriginal = Set(owners)
        }
    }
    static func markCopyReady(_ owners: [ObjectIdentifier]) {
        lock.withLock { for owner in owners { copyReady[owner] = true } }
    }
    static func snapshotCopyReadiness() -> Int {
        lock.withLock {
            roundOriginal = Set(copyReady.compactMap { $0.value ? nil : $0.key })
            return roundOriginal.count
        }
    }
    static func endCopyReadiness() {
        lock.withLock { copyReady = [:]; roundOriginal = [] }
    }
    static func needsOriginal(_ owner: SwitchLinear) -> Bool {
        lock.withLock { roundOriginal.contains(ObjectIdentifier(owner)) }
    }
    static func decodeCopyOwners(_ model: Module) -> [ObjectIdentifier] {
        let supported = lock.withLock { Set(copies.keys) }
        return model.namedModules().compactMap { name, module -> (Int, ObjectIdentifier)? in
            let parts = name.split(separator: ".")
            guard let i = parts.firstIndex(of: "layers"), i + 1 < parts.count,
                  let n = Int(parts[i + 1]), let moe = module as? NemotronHMoE,
                  supported.contains(ObjectIdentifier(moe.switchMLP.fc2)) else { return nil }
            return (n, ObjectIdentifier(moe.switchMLP.fc2))
        }.sorted { $0.0 < $1.0 }.map { $0.1 }
    }

    /// A dictionary copy's FP32 pair table is read by every decode slot launch, so it joins its layer's
    /// residency set unless NEMOTRON_MTP_RESID=0.
    static let residentTables = ProcessInfo.processInfo.environment["NEMOTRON_MTP_RESID"] != "0"

    /// Numeric backbone order, independent of dictionary iteration order.
    static func decodeCopyLayers(_ model: Module) -> [[MLXArray]] {
        let layers = model.namedModules().compactMap { name, module -> (Int, NemotronHMoE)? in
            let parts = name.split(separator: ".")
            guard let i = parts.firstIndex(of: "layers"), i + 1 < parts.count,
                  let n = Int(parts[i + 1]), let moe = module as? NemotronHMoE else { return nil }
            return (n, moe)
        }.sorted { $0.0 < $1.0 }
        return lock.withLock {
            layers.compactMap { _, moe in
                guard let c = copies[ObjectIdentifier(moe.switchMLP.fc2)] else { return nil }
                return c.dict && !residentTables ? [c.codes, c.scales] : [c.codes, c.scales, c.biases]
            }
        }
    }

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
    private static let tiledFastSlotKernel = MLXFast.metalKernel(
        name: "nemotron_zskip_fc2_slots_tiled_fast", inputNames: ["wk", "sk", "bk", "x", "inds", "scores"],
        outputNames: ["prod"], source: nemotronZSkipTiledFastSource,
        header: nemotronMoELeaderHeader + nemotronMoELeaderStore + nemotronZSkipHeader,
        ensureRowContiguous: true)
    private static let tiledDictSlotKernel = nemotronZSkipDictSource(nemotronZSkipTiledSeqSource).map {
        MLXFast.metalKernel(
            name: "nemotron_zskip_fc2_slots_tiled_sbd", inputNames: ["wk", "sk", "bk", "x", "inds", "scores"],
            outputNames: ["prod"], source: $0,
            header: nemotronMoELeaderHeader + nemotronMoELeaderStore + nemotronZSkipHeader,
            ensureRowContiguous: true)
    }
    private static let tiledFastDictSlotKernel = nemotronZSkipDictSource(nemotronZSkipTiledFastSource).map {
        MLXFast.metalKernel(
            name: "nemotron_zskip_fc2_slots_tiled_fast_sbd", inputNames: ["wk", "sk", "bk", "x", "inds", "scores"],
            outputNames: ["prod"], source: $0,
            header: nemotronMoELeaderHeader + nemotronMoELeaderStore + nemotronZSkipHeader,
            ensureRowContiguous: true)
    }
    /// The slot kernel that reads a dictionary copy (the fast or plain tiled form).
    private static var dictKernel: MLXFast.MLXFastKernel? {
        fast ? (vstore ? (unmasked ? tiledFastDictSlotKernelUL : nil) ?? tiledFastDictSlotKernelVS : tiledFastDictSlotKernel)
            : tiledDictSlotKernel
    }
    /// The fast tiled slot kernels (plain and dictionary, and their `dep` twins) with the vector store
    /// (`nemotronZSkipVectorStoreSource`), used once `qualifyVectorStore` matched each to its lane-0 form.
    private static func vectorStoreKernel(_ name: String, _ source: String?, dep: Bool) -> MLXFast.MLXFastKernel? {
        source.flatMap(nemotronZSkipVectorStoreSource).map {
            MLXFast.metalKernel(
                name: name, inputNames: ["wk", "sk", "bk", "x", "inds", "scores"] + (dep ? ["dep"] : []),
                outputNames: ["prod"], source: $0,
                header: nemotronMoELeaderHeader + nemotronMoELeaderStore + nemotronZSkipHeader,
                ensureRowContiguous: true)
        }
    }
    private static let tiledFastSlotKernelVS = vectorStoreKernel(
        "nemotron_zskip_fc2_slots_tiled_fast_vs", nemotronZSkipTiledFastSource, dep: false)
    private static let tiledFastDictSlotKernelVS = vectorStoreKernel(
        "nemotron_zskip_fc2_slots_tiled_fast_sbd_vs", nemotronZSkipDictSource(nemotronZSkipTiledFastSource), dep: false)
    private static let tiledFastSlotKernelDepVS = vectorStoreKernel(
        "nemotron_zskip_fc2_slots_tiled_fast_dep_vs", nemotronZSkipTiledFastSource, dep: true)
    private static let tiledFastDictSlotKernelDepVS = vectorStoreKernel(
        "nemotron_zskip_fc2_slots_tiled_fast_sbd_dep_vs", nemotronZSkipDictSource(nemotronZSkipTiledFastSource),
        dep: true)
    /// The fast tiled slot kernels and the shared-down kernels store each SIMD group's sums from its first
    /// OT / SGS lanes at once unless NEMOTRON_ZSKIP_VSTORE=0, once `qualifyVectorStore` matched every
    /// vector-store form to its lane-0 form.
    static let vstoreRequested = ProcessInfo.processInfo.environment["NEMOTRON_ZSKIP_VSTORE"] != "0"
    nonisolated(unsafe) fileprivate static var vstoreVerified = false
    static var vstore: Bool { vstoreRequested && vstoreVerified }
    /// The vector-store slot kernels above (plain and dictionary, and their `dep` twins) whose general chunk path
    /// loads all eight code words (`nemotronZSkipUnmaskedLoadsSource`), used unless NEMOTRON_ZSKIP_UNMASKED_LOADS=0
    /// once `qualifyUnmaskedLoads` matched each to its masked form. The shared-down kernels keep their loads.
    private static func unmaskedKernel(_ name: String, _ source: String?, dep: Bool) -> MLXFast.MLXFastKernel? {
        source.flatMap(nemotronZSkipVectorStoreSource).flatMap(nemotronZSkipUnmaskedLoadsSource).map {
            MLXFast.metalKernel(
                name: name, inputNames: ["wk", "sk", "bk", "x", "inds", "scores"] + (dep ? ["dep"] : []),
                outputNames: ["prod"], source: $0,
                header: nemotronMoELeaderHeader + nemotronMoELeaderStore + nemotronZSkipHeader,
                ensureRowContiguous: true)
        }
    }
    private static let tiledFastSlotKernelUL = unmaskedKernel(
        "nemotron_zskip_fc2_slots_tiled_fast_vs_ul", nemotronZSkipTiledFastSource, dep: false)
    private static let tiledFastDictSlotKernelUL = unmaskedKernel(
        "nemotron_zskip_fc2_slots_tiled_fast_sbd_vs_ul", nemotronZSkipDictSource(nemotronZSkipTiledFastSource),
        dep: false)
    private static let tiledFastSlotKernelDepUL = unmaskedKernel(
        "nemotron_zskip_fc2_slots_tiled_fast_dep_vs_ul", nemotronZSkipTiledFastSource, dep: true)
    private static let tiledFastDictSlotKernelDepUL = unmaskedKernel(
        "nemotron_zskip_fc2_slots_tiled_fast_sbd_dep_vs_ul", nemotronZSkipDictSource(nemotronZSkipTiledFastSource),
        dep: true)
    static let unmaskedRequested = ProcessInfo.processInfo.environment["NEMOTRON_ZSKIP_UNMASKED_LOADS"] != "0"
    nonisolated(unsafe) fileprivate static var unmaskedVerified = false
    static var unmasked: Bool { vstore && unmaskedRequested && unmaskedVerified }
    /// `NemotronDraftColaunch`: the fast tiled slot kernels (plain and dictionary) with one more input, `dep`,
    /// bound last and never read. The body is the same source text; the extra buffer only orders the launch
    /// after `dep`'s producer in MLX's eval tape.
    private static let tiledFastSlotKernelDep = MLXFast.metalKernel(
        name: "nemotron_zskip_fc2_slots_tiled_fast_dep",
        inputNames: ["wk", "sk", "bk", "x", "inds", "scores", "dep"],
        outputNames: ["prod"], source: nemotronZSkipTiledFastSource,
        header: nemotronMoELeaderHeader + nemotronMoELeaderStore + nemotronZSkipHeader,
        ensureRowContiguous: true)
    private static let tiledFastDictSlotKernelDep = nemotronZSkipDictSource(nemotronZSkipTiledFastSource).map {
        MLXFast.metalKernel(
            name: "nemotron_zskip_fc2_slots_tiled_fast_sbd_dep",
            inputNames: ["wk", "sk", "bk", "x", "inds", "scores", "dep"],
            outputNames: ["prod"], source: $0,
            header: nemotronMoELeaderHeader + nemotronMoELeaderStore + nemotronZSkipHeader,
            ensureRowContiguous: true)
    }
    /// The `dep` twin of the slot kernel `products` selects for `copy`, or nil (not the fast tiled form).
    fileprivate static func depKernel(for copy: Copy) -> MLXFast.MLXFastKernel? {
        guard tiled, fast else { return nil }
        if vstore {
            return copy.dict ? (unmasked ? tiledFastDictSlotKernelDepUL : nil) ?? tiledFastDictSlotKernelDepVS
                : (unmasked ? tiledFastSlotKernelDepUL : nil) ?? tiledFastSlotKernelDepVS
        }
        return copy.dict ? tiledFastDictSlotKernelDep : tiledFastSlotKernelDep
    }
    /// The tile-major copy reads (scale, bias) through a pair dictionary of the
    /// bank (`NemotronSBDict.make`) unless NEMOTRON_MTP_ZSKIP_SB_DICT=0. `prepare`
    /// falls back to the BF16 tile-major copy (and re-checks every layer) if a
    /// dictionary kernel misses a bit check.
    nonisolated(unsafe) private static var sbDict =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_ZSKIP_SB_DICT"] != "0"
    /// The tiled slot kernel's one- and two-value fast paths unless
    /// NEMOTRON_MTP_ZSKIP_FAST=0. `prepare` falls back to the tiled kernel (and
    /// re-checks every layer) if the fast twin misses a bit check.
    nonisolated(unsafe) fileprivate static var fast =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_ZSKIP_FAST"] != "0"
    /// The tile-major copy and its slot kernel unless NEMOTRON_MTP_ZSKIP_TILED=0.
    /// `prepare` falls back to the K-major copy and the staged kernel (and
    /// re-checks every layer) if the tiled kernel misses a bit check.
    nonisolated(unsafe) fileprivate static var tiled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_ZSKIP_TILED"] != "0"
    /// SIMD groups per slot-kernel threadgroup: 16 halves each thread's lane
    /// partials (two instead of four) with the same lane-to-chunk map, so every
    /// partial, its order and the closing reduction are unchanged.
    /// NEMOTRON_MTP_ZSKIP_SGS=8 keeps the base's 256-thread threadgroups.
    static let sgs = Int(ProcessInfo.processInfo.environment["NEMOTRON_MTP_ZSKIP_SGS"] ?? "") == 8 ? 8 : 16
    /// The staged slot kernel unless NEMOTRON_MTP_ZSKIP_STAGED=0. `prepare`
    /// also falls back to the original slot kernel (and re-checks every layer)
    /// if the staged one misses a bit check.
    nonisolated(unsafe) private static var staged =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_ZSKIP_STAGED"] != "0"
    private static var kernel: MLXFast.MLXFastKernel {
        tiled
            ? (fast ? (vstore ? (unmasked ? tiledFastSlotKernelUL : nil) ?? tiledFastSlotKernelVS ?? tiledFastSlotKernel
                : tiledFastSlotKernel) : tiledSlotKernel)
            : (staged ? stagedSlotKernel : recordSlotKernel)
    }
    private static let reduce = MLXFast.metalKernel(
        name: "nemotron_zskip_top6", inputNames: ["prod"], outputNames: ["y"],
        source: nemotronZSkipReduceSource)

    /// NEMOTRON_ZSKIP_DICT_ORDER (default on): a dictionary copy's pair ids renumbered by descending use over the
    /// bank's index (ties by first appearance), so the pairs most groups read share the first lines of the table.
    /// `useOrdered` keeps the first-appearance ids unless every index entry of the renumbered dictionary decodes to
    /// the same (scale, bias) bits.
    static let dictOrderRequested = ProcessInfo.processInfo.environment["NEMOTRON_ZSKIP_DICT_ORDER"] != "0"
    nonisolated(unsafe) private static var dictOrdered = 0
    nonisolated(unsafe) private static var dictOrderMissed = 0

    private static func useOrdered(_ d: NemotronSBDict.Dict) -> NemotronSBDict.Dict? {
        let pairs = d.table.dim(0)
        let ids = d.index.reshaped(-1).asType(.int32)
        let uses = MLXArray.zeros([pairs], dtype: .int32).at[ids].add(MLXArray(Int32(1)))
        // Descending use, then ascending first-appearance id: one distinct int64 key per pair.
        let first = MLXArray((0 ..< pairs).map { Int32($0) }).asType(.int64)
        let order = argSort(-(uses.asType(.int64) * 65536 + (65535 - first)))
        let rank = argSort(order).asType(.int32)
        let index = take(rank, ids).asType(.uint16).reshaped(d.index.shape)
        let table = take(d.table, order, axis: 0)
        let same = (take(table, index.reshaped(-1).asType(.int32), axis: 0).view(dtype: .uint32)
            .== take(d.table, ids, axis: 0).view(dtype: .uint32)).all()
        eval(index, table, same)
        return same.item(Bool.self) ? NemotronSBDict.Dict(index: index, table: table) : nil
    }

    /// One stderr line on the renumbered dictionaries of the copies built since the counters were reset.
    private static func reportDictOrder(_ scope: String) {
        let (ordered, missed) = lock.withLock { (dictOrdered, dictOrderMissed) }
        FileHandle.standardError.write(Data(("nemotron \(scope) dict order: " + (!dictOrderRequested
            ? "off (setting)"
            : missed == 0
                ? "on (pair ids by descending use; \(ordered) dictionary copies, every entry decodes to the same (scale, bias) bits)"
                : "on in \(ordered) dictionary copies, off in \(missed) (a renumbered entry differs)") + "\n").utf8))
    }

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
            if sbDict, dictKernel != nil,
               let d = NemotronSBDict.make(scales: bank.scales, biases: bank.biases) {
                let ordered = dictOrderRequested ? useOrdered(d) : nil
                lock.withLock {
                    if dictOrderRequested { if ordered != nil { dictOrdered += 1 } else { dictOrderMissed += 1 } }
                }
                return Copy(codes: codes, scales: tileMajor(ordered?.index ?? d.index),
                            biases: ordered?.table ?? d.table, dict: true)
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
    fileprivate static func products(_ x: MLXArray, _ inds: MLXArray, _ scores: MLXArray,
                                 copy: Copy, rows: Int, dep: MLXArray? = nil) -> MLXArray {
        // `NemotronFC2Tune`: the geometry adopted at load for this launch (the verify's R = 3 launch or the draft
        // tail's R = 1 dep launch on a dictionary copy), if any; else the selection below.
        if let v = NemotronFC2Tune.adopted(rows: rows, dep: dep != nil, dict: copy.dict),
           let out = NemotronFC2Tune.launch(v, x, inds, scores, copy: copy, dep: dep ?? x) {
            return out
        }
        let slots = rows * 6
        let threads = 32 * sgs
        let twin = dep.flatMap { _ in depKernel(for: copy) }
        return (twin ?? (copy.dict ? dictKernel! : kernel))(
            [NemotronBankBinding.bind(copy.codes), NemotronBankBinding.bind(copy.scales),
             NemotronBankBinding.bind(copy.biases), x, inds, scores] + (twin != nil ? [dep!] : []),
            template: [("T", x.dtype), ("K", 1856), ("N", 2688), ("SGS", sgs)],
            grid: (threads * (2688 / 128), slots, 1), threadGroup: (threads, 1, 1),
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

    /// The verified copy of `owner` (a target layer's routed FC2), if any.
    static func copy(for owner: SwitchLinear) -> Copy? {
        lock.withLock { copies[ObjectIdentifier(owner)] }
    }

    /// True when `owner` (a target layer's routed FC2) has a verified copy.
    static func covers(_ owner: SwitchLinear) -> Bool {
        lock.withLock { copies[ObjectIdentifier(owner)] != nil }
    }

    /// The R = 3 weighted slot products of a target layer ([18 * 2688] FP32,
    /// slot-major), or nil when `owner` has no verified copy.
    static func weightedProducts(_ x: MLXArray, _ inds: MLXArray, _ scores: MLXArray,
                                 owner: SwitchLinear) -> MLXArray? {
        guard !needsOriginal(owner), joinReduce, NemotronHResidualRMSNorm.productsJoinVerified,
              x.shape == [18, 1856], x.dtype == .bfloat16,
              inds.shape == [18], scores.shape == [18], scores.dtype == .float32,
              let copy = lock.withLock({ copies[ObjectIdentifier(owner)] }) else { return nil }
        return products(x, inds, scores, copy: copy, rows: 3)
    }

    /// The R = 3 weighted fc2 of `owner`, or nil when it has no verified copy.
    static func weighted(_ x: MLXArray, _ inds: MLXArray, _ scores: MLXArray, rows: Int,
                         owner: SwitchLinear) -> MLXArray? {
        guard !needsOriginal(owner), rows == 3, x.shape == [18, 1856], x.dtype == .bfloat16,
              inds.shape == [18], scores.shape == [18], scores.dtype == .float32,
              let copy = lock.withLock({ copies[ObjectIdentifier(owner)] }) else { return nil }
        return run(x, inds, scores, copy: copy)
    }

    // MARK: Draft head (one row per draft step)
    //
    // The draft head's routed MoE runs on one row: each draft step's one-token
    // expert tail. The record runs its fc2 through the R = 1 weighted leader
    // kernel, whose per-slot arithmetic is the slot kernel's (same lane chunks,
    // coefficient values, qdot, simd_sum, store and weighted product); the
    // tail's join then sums the six products in the reduce's slot order. The
    // draft copy lives apart from the target's, with its own check and switch.
    static let draftEnabled = ProcessInfo.processInfo.environment["NEMOTRON_MTP_DRAFT_ZSKIP"] != "0"
    nonisolated(unsafe) fileprivate static var draftCopies: [ObjectIdentifier: Copy] = [:]

    /// The draft head's R = 1 weighted slot products ([6 * 2688] FP32), or nil
    /// when `owner` has no verified draft copy.
    /// The draft copies' arrays (decode-only backing for the residency lease).
    static var draftCopyArrays: [MLXArray] {
        lock.withLock { draftCopies.values.flatMap { [$0.codes, $0.scales, $0.biases] } }
    }
    /// `module`'s parameters except the original banks of switch linears whose draft copy is verified: the draft
    /// rounds read the copy, so those banks are not decode backing. NEMOTRON_MTP_DRAFT_BANK_RESIDENCY=1 keeps them.
    static func withoutCopiedDraftBanks(_ module: Module) -> [MLXArray] {
        let all = module.parameters().flattened().map { $0.1 }
        guard ProcessInfo.processInfo.environment["NEMOTRON_MTP_DRAFT_BANK_RESIDENCY"] != "1" else { return all }
        let copied = lock.withLock { Set(draftCopies.keys) }
        var skip = Set<ObjectIdentifier>()
        for (_, m) in module.namedModules() {
            guard let linear = m as? SwitchLinear, copied.contains(ObjectIdentifier(linear)) else { continue }
            for (_, a) in linear.parameters().flattened() { skip.insert(ObjectIdentifier(a)) }
        }
        let kept = all.filter { !skip.contains(ObjectIdentifier($0)) }
        FileHandle.standardError.write(Data(
            "nemotron head residency: \(kept.count) arrays, \(all.count - kept.count) copied draft-bank arrays left out\n".utf8))
        return kept
    }

    static func draftProducts(_ x: MLXArray, _ inds: MLXArray, _ scores: MLXArray,
                              owner: SwitchLinear, dep: MLXArray? = nil) -> MLXArray? {
        guard x.shape == [6, 1856], x.dtype == .bfloat16,
              inds.shape == [6], scores.shape == [6], scores.dtype == .float32,
              let copy = lock.withLock({ draftCopies[ObjectIdentifier(owner)] }) else { return nil }
        return products(x, inds, scores, copy: copy, rows: 1, dep: dep)
    }

    /// True when `owner` has a verified draft copy whose slot kernel has a `dep` twin.
    static func draftDepAvailable(_ owner: SwitchLinear) -> Bool {
        guard let copy = lock.withLock({ draftCopies[ObjectIdentifier(owner)] }) else { return false }
        return depKernel(for: copy) != nil
    }

    /// Load time, after `prepare` (same copy layout and slot kernel) and before
    /// any draft graph is traced: copy the draft head's routed fc2 bank and keep
    /// it only if its scales and biases are finite, the slot kernel plus the
    /// reduce equals the production R = 1 kernel bit for bit on seeded one-row
    /// inputs, and the one-row products join equals reduce + join. Any miss
    /// leaves the draft head on the record kernel; the target is untouched.
    static func prepareDraft(_ head: Module) {
        guard draftEnabled, NemotronMoELeader.enabled, Device.defaultDevice().deviceType == .gpu else { return }
        lock.withLock { dictOrdered = 0; dictOrderMissed = 0 }
        var built: [ObjectIdentifier: Copy] = [:]
        for module in head.modules() {
            guard let moe = module as? NemotronHMoE,
                  moe.numExpertsPerTok == 6, type(of: moe.switchMLP.fc2) == QuantizedSwitchLinear.self,
                  let down = NemotronMoELeader.bank(moe.switchMLP.fc2, k: 1856, dtype: .bfloat16),
                  down.weight.shape == [128, 2688, 232], down.scales.shape == [128, 2688, 29],
                  down.biases.shape == down.scales.shape else { continue }
            let copy = makeCopy(down)
            eval(copy.codes, copy.scales, copy.biases)
            guard finite(down), matchesOneRow(down, copy, seed: built.count) else {
                FileHandle.standardError.write(
                    Data("nemotron draft zskip off: the draft head differs from the record kernel\n".utf8))
                return
            }
            built[ObjectIdentifier(moe.switchMLP.fc2)] = copy
        }
        guard !built.isEmpty else {
            FileHandle.standardError.write(Data("nemotron draft zskip off: no 4-bit routed fc2 bank\n".utf8))
            return
        }
        guard NemotronHResidualRMSNorm.checkDraftProductsJoin(head) else {
            FileHandle.standardError.write(
                Data("nemotron draft zskip off: the products join differs from reduce + join\n".utf8))
            return
        }
        lock.withLock { draftCopies = built }
        let dicts = built.values.filter(\.dict).count
        FileHandle.standardError.write(
            Data("nemotron draft zskip on: \(built.count) MoE layers, \(dicts) with pair dictionaries\n".utf8))
        if dicts > 0 { reportDictOrder("draft zskip") }
    }

    /// Seeded one-row cases against the production R = 1 kernel: sparse rows
    /// with an all-zero slot, a one-hot slot and -0 in place of some zeros,
    /// tail chunks only, and dense.
    private static func matchesOneRow(_ bank: NemotronMoELeader.Bank, _ copy: Copy, seed: Int) -> Bool {
        let offset = UInt32(seed) * 7
        var routes: [UInt32] = [3, 17, 42, 77, 100, 121]
        for i in routes.indices {
            routes[i] = (routes[i] + offset) % 128
        }
        let inds = MLXArray(routes)
        let keys = MLXRandom.split(key: MLXRandom.key(UInt64(0x25d0 + seed)), into: 4)
        let h = abs(MLXRandom.normal([6, 1856], key: keys[0])) + 1e-3
        let keep = MLXRandom.uniform(Float(0) ..< 1, [6, 1856], key: keys[1]) .< MLXArray(Float(0.09))
        let slot = MLXArray((0 ..< 6).map(Int32.init)).reshaped(6, 1)
        let column = MLXArray((0 ..< 1856).map(Int32.init)).reshaped(1, 1856)
        let zero = MLXArray(Float(0))
        let oneHot = which(column .== MLXArray(Int32(1800)), MLXArray(Float(1.5)), zero)
        let sparse = which(slot .== MLXArray(Int32(2)), zero,
                           which(slot .== MLXArray(Int32(3)), oneHot, which(keep, h, zero)))
        let tail = which(column .>= MLXArray(Int32(1792)), h, zero)
        let scores = MLXRandom.uniform(Float(0) ..< 1, [6], key: keys[2])
        let negative = MLXRandom.uniform(Float(0) ..< 1, [6, 1856], key: keys[3]) .< MLXArray(Float(0.05))
        let negativeZero = MLXArray(Float(-0.0)).asType(.bfloat16)
        var cases: [MLXArray] = []
        for a in [sparse.asType(.bfloat16), tail.asType(.bfloat16), h.asType(.bfloat16)] {
            cases.append((a * a).asType(.bfloat16))
        }
        cases.append(which(logicalAnd(cases[0] .== MLXArray(Float(0)).asType(.bfloat16), negative),
                           negativeZero, cases[0]))
        var same = MLXArray(true)
        for act in cases {
            let expected = NemotronMoELeader.applyWeighted(act, inds, scores, rows: 1, bank: bank,
                                                           tail8: NemotronMoELeader.leaderTail8)
            let got = run(act, inds, scores, copy: copy, rows: 1)
            same = logicalAnd(same, (got.view(dtype: .uint16) .== expected.view(dtype: .uint16)).all())
        }
        return same.item(Bool.self)
    }

    /// Load time, at the start of `prepare` (before any copy is checked, and before the shared down's
    /// `prepare`): every vector-store form against its lane-0 form, bit for bit, on synthetic copies (random
    /// codes; scales, biases and pair-table entries with zeros, -0, infinities and NaNs among Gaussian values) and
    /// synthetic rows: ReLU2-like rows at the measured verify densities (routed 8.75%, shared 13.4%) with an
    /// all-zero and a one-hot slot, tail chunks only, dense rows, -0 for some zeros, an infinity, a NaN and a tiny
    /// value (the dense-path sentinel) in three slots, and one row in every slot; slot scores uniform, and with 0,
    /// -0, NaN, both infinities and a tie; experts shared between slots. Routed: the plain and dictionary slot
    /// kernels at R = 3 and their `dep` twins at R = 1. Shared down: the tile-major and wide kernels. The lane-0
    /// forms stay on any miss. One stderr line.
    fileprivate static func qualifyVectorStore() {
        vstoreVerified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron zskip vstore: \(message)\n".utf8))
        }
        guard vstoreRequested else { return report("off (setting)") }
        guard let plain = tiledFastSlotKernelVS, let dict = tiledFastDictSlotKernelVS,
              let plainDep = tiledFastSlotKernelDepVS, let dictDep = tiledFastDictSlotKernelDepVS,
              let dictBase = tiledFastDictSlotKernel, let dictDepBase = tiledFastDictSlotKernelDep,
              let sharedBase = NemotronSharedDownZSkip.kernel, let shared = NemotronSharedDownZSkip.kernelVS,
              let wideBase = NemotronSharedDownZSkip.wideKernel, let wide = NemotronSharedDownZSkip.wideKernelVS
        else { return report("off (a slot kernel source no longer matches)") }
        let experts = 8, k = 1856, n = 2688, tiles = n / 128, groups = k / 64
        let sharedK = NemotronSharedDownZSkip.k
        let zero = MLXArray(Float(0))
        let bf16 = { (v: Float) in MLXArray(v).asType(.bfloat16) }
        let keys = MLXRandom.split(key: MLXRandom.key(0x7653), into: 12)
        let specials = MLXArray([Float(0), -0.0, .infinity, -.infinity, .nan])
        // Gaussian values with specials where `u` is below `low` or above `high`.
        func values(_ shape: [Int], _ u: MLXArray, _ key: MLXArray, low: Float = -1, high: Float = 2) -> MLXArray {
            let special = logicalOr(u .< MLXArray(low), u .> MLXArray(high))
            return which(special, take(specials, (u * 100_000).asType(.int32) % MLXArray(Int32(5))),
                         MLXRandom.normal(shape, key: key) * 0.01).asType(.bfloat16)
        }
        let su = MLXRandom.uniform(Float(0) ..< 1, [experts, tiles, groups, 128], key: keys[0])
        let codes = MLXRandom.randInt(Int32(0) ..< 65536, [experts, tiles, k, 32], key: keys[1]).asType(.uint16)
        let scales = values([experts, tiles, groups, 128], su, keys[2], low: 0.0005)
        let biases = values([experts, tiles, groups, 128], su, keys[3], high: 0.9995)
        let index = MLXRandom.randInt(Int32(0) ..< 512, [experts, tiles, groups, 128], key: keys[4]).asType(.uint16)
        let table = concatenated([
            MLXArray([Float(0), -0.0, .infinity, 1, .nan, 0.5, -0.0, -.infinity]).reshaped(4, 2),
            MLXRandom.normal([508, 2], key: keys[5]) * 0.01,
        ]).asType(.bfloat16).asType(.float32)
        let wu = MLXRandom.uniform(Float(0) ..< 1, [1, tiles, sharedK / 64, 128], key: keys[6])
        let narrowCopy = NemotronSharedDownZSkip.Copy(
            codes: MLXRandom.randInt(Int32(0) ..< 65536, [1, tiles, sharedK, 32], key: keys[7]).asType(.uint16),
            scales: values([1, tiles, sharedK / 64, 128], wu, keys[8], low: 0.0005),
            biases: values([1, tiles, sharedK / 64, 128], wu, keys[9], high: 0.9995))
        var wideCopy = narrowCopy
        wideCopy.wideCodes = MLXRandom.randInt(Int32(0) ..< 65536, [1, tiles * sharedK * 32], key: keys[10])
            .asType(.uint16)
        // ReLU2-like rows [slots, kk] (see above).
        func rows(_ slots: Int, _ kk: Int, density: Float, seed: UInt64) -> [MLXArray] {
            let r = MLXRandom.split(key: MLXRandom.key(seed), into: 3)
            let h = abs(MLXRandom.normal([slots, kk], key: r[0])) * 0.5 + 1e-3
            let u = MLXRandom.uniform(Float(0) ..< 1, [slots, kk], key: r[1])
            let slot = MLXArray((0 ..< slots).map(Int32.init)).reshaped(slots, 1)
            let column = MLXArray((0 ..< kk).map(Int32.init)).reshaped(1, kk)
            let oneHot = which(column .== MLXArray(Int32(kk - 56)), MLXArray(Float(1.5)), zero)
            let sparse = which(slot .== MLXArray(Int32(1)), zero,
                               which(slot .== MLXArray(Int32(2)), oneHot, which(u .< MLXArray(density), h, zero)))
            let tail = which(column .>= MLXArray(Int32(kk - 64)), h, zero)
            var out = [sparse, tail, h].map { a -> MLXArray in
                let b = a.asType(.bfloat16)
                return (b * b).asType(.bfloat16)
            }
            let negative = MLXRandom.uniform(Float(0) ..< 1, [slots, kk], key: r[2]) .< MLXArray(Float(0.05))
            let signed = which(logicalAnd(out[0] .== bf16(0), negative), bf16(-0.0), out[0])
            func at(_ s: Int, _ c: Int) -> MLXArray {
                logicalAnd(slot .== MLXArray(Int32(s % slots)), column .== MLXArray(Int32(c)))
            }
            out.append(signed)
            out.append(which(at(3, 77), bf16(.infinity),
                             which(at(4, 901), bf16(.nan), which(at(5, 1300), bf16(1e-35), signed))))
            out.append(broadcast(out[0][0 ..< 1], to: [slots, kk]))
            return out
        }
        func scores(_ slots: Int, seed: UInt64, special: Bool) -> MLXArray {
            let s = MLXRandom.uniform(Float(0) ..< 1, [slots], key: MLXRandom.key(seed))
            guard special else { return s }
            let head: [Float] = [0, -0.0, .nan, .infinity, -.infinity, 0.25, 0.25]
            let fixed = MLXArray((0 ..< slots).map { $0 < head.count ? head[$0] : 0 })
            return which(MLXArray((0 ..< slots).map(Int32.init)) .< MLXArray(Int32(head.count)), fixed, s)
        }
        func launch(_ kernel: MLXFast.MLXFastKernel, _ x: MLXArray, _ inds: MLXArray, _ scores: MLXArray,
                    _ sk: MLXArray, _ bk: MLXArray, dep: Bool) -> MLXArray {
            kernel(
                [codes, sk, bk, x, inds, scores] + (dep ? [zero] : []),
                template: [("T", DType.bfloat16), ("K", k), ("N", n), ("SGS", sgs)],
                grid: (32 * sgs * tiles, x.dim(0), 1), threadGroup: (32 * sgs, 1, 1),
                outputShapes: [[x.dim(0) * n]], outputDTypes: [.float32])[0]
        }
        var cases = 0
        var admitted = false
        do {
            admitted = try withError { () -> Bool in
                var ok = MLXArray(true)
                let routed: [(MLXFast.MLXFastKernel, MLXFast.MLXFastKernel, MLXArray, MLXArray, Bool)] = [
                    (tiledFastSlotKernel, plain, scales, biases, false), (dictBase, dict, index, table, false),
                    (tiledFastSlotKernelDep, plainDep, scales, biases, true), (dictDepBase, dictDep, index, table, true),
                ]
                for (base, twin, sk, bk, dep) in routed {
                    let slots = dep ? 6 : 18
                    let inds = MLXArray((0 ..< slots).map { UInt32((5 * $0) % experts) })
                    for (i, x) in rows(slots, k, density: 0.0875, seed: UInt64(0x7660 + slots)).enumerated() {
                        for special in [false, true] {
                            let s = scores(slots, seed: UInt64(0x7670 + i), special: special)
                            let a = launch(base, x, inds, s, sk, bk, dep: dep)
                            let b = launch(twin, x, inds, s, sk, bk, dep: dep)
                            ok = ok .&& (a.view(dtype: .uint32) .== b.view(dtype: .uint32)).all()
                            cases += 1
                        }
                    }
                }
                let sinds = MLXArray.zeros([3], dtype: .uint32)
                for (base, twin, copy) in [(sharedBase, shared, narrowCopy), (wideBase, wide, wideCopy)] {
                    for x in rows(3, sharedK, density: 0.134, seed: 0x7680) {
                        let a = NemotronSharedDownZSkip.dispatch(x, copy, sinds, base)
                        let b = NemotronSharedDownZSkip.dispatch(x, copy, sinds, twin)
                        ok = ok .&& (a.view(dtype: .uint16) .== b.view(dtype: .uint16)).all()
                        cases += 1
                    }
                }
                return ok.item(Bool.self)
            }
        } catch {
            return report("off (\(error))")
        }
        vstoreVerified = admitted
        report(admitted
            ? "on (\(cases) cases bitwise equal to the lane-0 store: routed plain and pair-dictionary slot kernels "
                + "at R = 3 and their dep twins at R = 1, shared down tile-major and wide; zeros, -0, infinities, "
                + "NaNs and ties included)"
            : "off (differs from the lane-0 store)")
    }

    /// Load time, after `qualifyVectorStore` admitted the vector store: every unmasked-load slot kernel against its
    /// masked vector-store form, bit for bit, on synthetic copies (random codes; scales, biases and pair-table
    /// entries with zeros, -0, infinities and NaNs) and synthetic rows at the verify (8.75%), draft-head (45%) and
    /// full densities, with -0 for some zeros, an all-zero and a one-hot slot, and an infinity, a NaN and a tiny
    /// value (the dense-path sentinel) in three slots; slot scores uniform and with 0, -0, NaN, both infinities and
    /// a tie; experts shared between slots. The plain and dictionary kernels at R = 3, their `dep` twins at R = 1.
    /// The masked forms stay on any miss. One stderr line.
    fileprivate static func qualifyUnmaskedLoads() {
        unmaskedVerified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron zskip unmasked loads: \(message)\n".utf8))
        }
        guard unmaskedRequested else { return report("off (setting)") }
        guard vstore else { return report("off (vector store off)") }
        guard let plain = tiledFastSlotKernelVS, let dict = tiledFastDictSlotKernelVS,
              let plainDep = tiledFastSlotKernelDepVS, let dictDep = tiledFastDictSlotKernelDepVS,
              let plainU = tiledFastSlotKernelUL, let dictU = tiledFastDictSlotKernelUL,
              let plainDepU = tiledFastSlotKernelDepUL, let dictDepU = tiledFastDictSlotKernelDepUL
        else { return report("off (a slot kernel source no longer matches)") }
        let experts = 8, k = 1856, n = 2688, tiles = n / 128, groups = k / 64
        let zero = MLXArray(Float(0))
        let bf16 = { (v: Float) in MLXArray(v).asType(.bfloat16) }
        let keys = MLXRandom.split(key: MLXRandom.key(0x7553), into: 8)
        let specials = MLXArray([Float(0), -0.0, .infinity, -.infinity, .nan])
        func values(_ shape: [Int], _ u: MLXArray, _ key: MLXArray, low: Float = -1, high: Float = 2) -> MLXArray {
            let special = logicalOr(u .< MLXArray(low), u .> MLXArray(high))
            return which(special, take(specials, (u * 100_000).asType(.int32) % MLXArray(Int32(5))),
                         MLXRandom.normal(shape, key: key) * 0.01).asType(.bfloat16)
        }
        let su = MLXRandom.uniform(Float(0) ..< 1, [experts, tiles, groups, 128], key: keys[0])
        let codes = MLXRandom.randInt(Int32(0) ..< 65536, [experts, tiles, k, 32], key: keys[1]).asType(.uint16)
        let scales = values([experts, tiles, groups, 128], su, keys[2], low: 0.0005)
        let biases = values([experts, tiles, groups, 128], su, keys[3], high: 0.9995)
        let index = MLXRandom.randInt(Int32(0) ..< 512, [experts, tiles, groups, 128], key: keys[4]).asType(.uint16)
        let table = concatenated([
            MLXArray([Float(0), -0.0, .infinity, 1, .nan, 0.5, -0.0, -.infinity]).reshaped(4, 2),
            MLXRandom.normal([508, 2], key: keys[5]) * 0.01,
        ]).asType(.bfloat16).asType(.float32)
        // ReLU2-like rows [slots, k] at `density` (see above).
        func rows(_ slots: Int, density: Float, seed: UInt64) -> MLXArray {
            let r = MLXRandom.split(key: MLXRandom.key(seed), into: 3)
            let h = abs(MLXRandom.normal([slots, k], key: r[0])) * 0.5 + 1e-3
            let u = MLXRandom.uniform(Float(0) ..< 1, [slots, k], key: r[1])
            let slot = MLXArray((0 ..< slots).map(Int32.init)).reshaped(slots, 1)
            let column = MLXArray((0 ..< k).map(Int32.init)).reshaped(1, k)
            let oneHot = which(column .== MLXArray(Int32(k - 56)), MLXArray(Float(1.5)), zero)
            let picked = which(slot .== MLXArray(Int32(1)), zero,
                               which(slot .== MLXArray(Int32(2)), oneHot, which(u .< MLXArray(density), h, zero)))
            let squared = (picked.asType(.bfloat16) * picked.asType(.bfloat16)).asType(.bfloat16)
            let negative = MLXRandom.uniform(Float(0) ..< 1, [slots, k], key: r[2]) .< MLXArray(Float(0.05))
            let signed = which(logicalAnd(squared .== bf16(0), negative), bf16(-0.0), squared)
            func at(_ s: Int, _ c: Int) -> MLXArray {
                logicalAnd(slot .== MLXArray(Int32(s % slots)), column .== MLXArray(Int32(c)))
            }
            return which(at(3, 77), bf16(.infinity), which(at(4, 901), bf16(.nan), which(at(5, 1300), bf16(1e-35), signed)))
        }
        func scores(_ slots: Int, seed: UInt64, special: Bool) -> MLXArray {
            let s = MLXRandom.uniform(Float(0) ..< 1, [slots], key: MLXRandom.key(seed))
            guard special else { return s }
            let head: [Float] = [0, -0.0, .nan, .infinity, -.infinity, 0.25, 0.25]
            let fixed = MLXArray((0 ..< slots).map { $0 < head.count ? head[$0] : 0 })
            return which(MLXArray((0 ..< slots).map(Int32.init)) .< MLXArray(Int32(min(slots, head.count))), fixed, s)
        }
        func launch(_ kernel: MLXFast.MLXFastKernel, _ x: MLXArray, _ inds: MLXArray, _ scores: MLXArray,
                    _ sk: MLXArray, _ bk: MLXArray, dep: Bool) -> MLXArray {
            kernel(
                [codes, sk, bk, x, inds, scores] + (dep ? [zero] : []),
                template: [("T", DType.bfloat16), ("K", k), ("N", n), ("SGS", sgs)],
                grid: (32 * sgs * tiles, x.dim(0), 1), threadGroup: (32 * sgs, 1, 1),
                outputShapes: [[x.dim(0) * n]], outputDTypes: [.float32])[0]
        }
        var cases = 0
        var admitted = false
        do {
            admitted = try withError { () -> Bool in
                var ok = MLXArray(true)
                let forms: [(MLXFast.MLXFastKernel, MLXFast.MLXFastKernel, MLXArray, MLXArray, Bool)] = [
                    (plain, plainU, scales, biases, false), (dict, dictU, index, table, false),
                    (plainDep, plainDepU, scales, biases, true), (dictDep, dictDepU, index, table, true),
                ]
                for (base, twin, sk, bk, dep) in forms {
                    let slots = dep ? 6 : 18
                    let inds = MLXArray((0 ..< slots).map { UInt32((5 * $0) % experts) })
                    for (i, density) in [Float(0.0875), 0.45, 1].enumerated() {
                        let x = rows(slots, density: density, seed: UInt64(0x7560 + 16 * i + slots))
                        for special in [false, true] {
                            let s = scores(slots, seed: UInt64(0x7570 + 2 * i + (special ? 1 : 0)), special: special)
                            let a = launch(base, x, inds, s, sk, bk, dep: dep)
                            let b = launch(twin, x, inds, s, sk, bk, dep: dep)
                            ok = ok .&& (a.view(dtype: .uint32) .== b.view(dtype: .uint32)).all()
                            cases += 1
                        }
                    }
                }
                return ok.item(Bool.self)
            }
        } catch {
            return report("off (\(error))")
        }
        unmaskedVerified = admitted
        report(admitted
            ? "on (\(cases) cases bitwise equal to the masked loads: routed plain and pair-dictionary slot kernels at "
                + "R = 3 and their dep twins at R = 1, densities 8.75%, 45% and full; zeros, -0, infinities, NaNs and "
                + "ties included)"
            : "off (differs from the masked loads)")
    }

    /// Load time, before any verify graph is traced: copy every routed fc2 bank,
    /// then keep the copies only if every layer's scales and biases are finite
    /// (the skipped-chunk argument needs it) and its kernel equals the record
    /// kernel bit for bit on seeded relu2-like rows (sparse with an all-zero and
    /// a one-hot slot, tail chunks only, and dense; experts shared across rows).
    static func prepare(_ model: Module) {
        guard enabled, NemotronMoELeader.enabled, Device.defaultDevice().deviceType == .gpu else { return }
        qualifyVectorStore()
        qualifyUnmaskedLoads()
        var result = build(model)
        if case .kernelMiss = result, tiled, sbDict {
            // A dictionary kernel missed a check: the BF16 tile-major copy, every layer again.
            sbDict = false
            FileHandle.standardError.write(
                Data("nemotron zskip: dictionary slot kernel differs; BF16 tile-major copy\n".utf8))
            result = build(model)
        }
        if case .kernelMiss = result, tiled, fast {
            // The fast twin missed a check: the tiled kernel, every layer again.
            fast = false
            FileHandle.standardError.write(
                Data("nemotron zskip: fast slot kernel differs; tiled slot kernel\n".utf8))
            result = build(model)
        }
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
            let dicts = built.values.filter(\.dict).count
            FileHandle.standardError.write(
                Data("nemotron zskip on: \(built.count) MoE layers, \(dicts) with pair dictionaries\n".utf8))
            if dicts > 0 { reportDictOrder("zskip") }
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
        lock.withLock { dictOrdered = 0; dictOrderMissed = 0 }
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

//
enum NemotronSharedDownZSkip {
    static let enabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_SHARED_DOWN_ZSKIP"] != "0"
    fileprivate static let k = 3712
    fileprivate static let n = 2688

    private static let layout = MLXFast.metalKernel(
        name: "nemotron_zskip_shared_tiled_codes", inputNames: ["w"], outputNames: ["wk"],
        source: nemotronZSkipTiledLayoutSource)
    /// The shared-down body: the routed slot kernel's fast tiled source with K-generic chunk bounds and a BF16
    /// row store (also the shared half of `NemotronFC2SharedColaunch`). Nil when a swap no longer applies.
    fileprivate static let source: String? = {
        let swaps: [(String, String)] = [
            ("  constexpr int tail = chunks - 224;\n",
             "  constexpr int full = chunks / 32;\n  constexpr int tail = chunks - 32 * full;\n"),
            ("    const int count = l < tail ? 8 : 7;\n",
             "    const int count = l < tail ? full + 1 : full;\n"),
            ("      const int c = j < 7 ? l + 32 * j : 224 + l;\n",
             "      const int c = j < full ? l + 32 * j : 32 * full + l;\n"),
            ("      prod[slot * N + n0 + n] = float(h) * scores[slot] + 0.0f;\n",
             "      shared_y[slot * N + n0 + n] = h;\n"),
        ]
        var source = nemotronZSkipTiledFastSource
        for (old, new) in swaps {
            guard source.components(separatedBy: old).count == 2 else { return nil }
            source = source.replacingOccurrences(of: old, with: new)
        }
        return source
    }()
    fileprivate static let kernel: MLXFast.MLXFastKernel? = source.map { source in
        MLXFast.metalKernel(
            name: "nemotron_zskip_shared_down_tiled_fast",
            inputNames: ["wk", "sk", "bk", "x", "inds"], outputNames: ["shared_y"],
            source: source,
            header: nemotronMoELeaderHeader + nemotronMoELeaderStore + nemotronZSkipHeader,
            ensureRowContiguous: true)
    }

    /// Wide shared-down tiles (`nemotronZSkipWideSource`) unless NEMOTRON_ZSKIP_WIDE=0: a second code copy whose
    /// rows of two neighbouring 128-output tiles share one 128-byte line, so a lane's non-zero k is one 4-byte
    /// load for eight outputs; scales and biases stay the tile-major ones. The decode's shared-down kernel and
    /// the FC2 co-launch read the wide codes; the tile-major codes stay as they are for every other reader
    /// (`copyArrays(for:)`). `prepare` qualifies the wide kernels against the tile-major ones on the first and
    /// last layers' weights and drops the wide codes on any miss.
    static let wideEnabled = ProcessInfo.processInfo.environment["NEMOTRON_ZSKIP_WIDE"] != "0"
    /// The wide shared-down body (also the shared half of the wide `NemotronFC2SharedColaunch` kernels).
    fileprivate static let wideSource: String? = source.flatMap {
        nemotronZSkipWideSource(narrow: $0, store: "shared_y[slot * N + n0 + n] = h;")
    }
    fileprivate static let wideKernel: MLXFast.MLXFastKernel? = wideSource.map { source in
        MLXFast.metalKernel(
            name: "nemotron_zskip_shared_down_wide",
            inputNames: ["wk", "sk", "bk", "x", "inds"], outputNames: ["shared_y"],
            source: source,
            header: nemotronMoELeaderHeader + nemotronMoELeaderStore + nemotronZSkipHeader,
            ensureRowContiguous: true)
    }
    private static let wideLayout = MLXFast.metalKernel(
        name: "nemotron_zskip_shared_wide_codes", inputNames: ["w"], outputNames: ["wk"],
        source: nemotronZSkipWideLayoutSource)
    /// The tile-major and wide kernels with the vector store (`nemotronZSkipVectorStoreSource`; the wide source's
    /// tail-tile loop and both halves of the wide body), used once `NemotronZSkip.qualifyVectorStore` matched them
    /// to the lane-0 forms. `source` and `wideSource` (the co-launch's shared bodies) stay as they are.
    fileprivate static let kernelVS: MLXFast.MLXFastKernel? = source.flatMap(nemotronZSkipVectorStoreSource).map {
        MLXFast.metalKernel(
            name: "nemotron_zskip_shared_down_tiled_fast_vs",
            inputNames: ["wk", "sk", "bk", "x", "inds"], outputNames: ["shared_y"],
            source: $0,
            header: nemotronMoELeaderHeader + nemotronMoELeaderStore + nemotronZSkipHeader,
            ensureRowContiguous: true)
    }
    fileprivate static let wideKernelVS: MLXFast.MLXFastKernel? = wideSource.flatMap(nemotronZSkipVectorStoreSource)
        .map {
            MLXFast.metalKernel(
                name: "nemotron_zskip_shared_down_wide_vs",
                inputNames: ["wk", "sk", "bk", "x", "inds"], outputNames: ["shared_y"],
                source: $0,
                header: nemotronMoELeaderHeader + nemotronMoELeaderStore + nemotronZSkipHeader,
                ensureRowContiguous: true)
        }

    fileprivate struct Copy {
        let codes: MLXArray
        let scales: MLXArray
        let biases: MLXArray
        /// The same codes in the wide layout ([1, N / 128 * K * 32]), when the wide kernels are on; `codes`
        /// stays tile-major ([1, N / 128, K, 32]) either way.
        var wideCodes: MLXArray? = nil
        var wide: Bool { wideCodes != nil }
        /// The codes the decode's shared-down kernel and co-launch read.
        var decodeCodes: MLXArray { wideCodes ?? codes }
    }
    fileprivate static let lock = NSLock()
    nonisolated(unsafe) fileprivate static var copies: [ObjectIdentifier: Copy] = [:]
    nonisolated(unsafe) fileprivate static var inds: MLXArray? = nil
    /// The decode's backing: the codes its kernels read (the wide ones when on), scales and biases.
    static var copyArrays: [MLXArray] {
        lock.withLock { copies.values.flatMap { [$0.decodeCodes, $0.scales, $0.biases] } }
    }
    /// The verified copy of `down` (a target layer's shared down projection), if any.
    static func copyArrays(for down: Linear) -> (codes: MLXArray, scales: MLXArray, biases: MLXArray)? {
        lock.withLock { copies[ObjectIdentifier(down)].map { ($0.codes, $0.scales, $0.biases) } }
    }

    private static func admissible(_ q: QuantizedLinear) -> Bool {
        type(of: q) == QuantizedLinear.self && q.mode == .affine && q.bits == 4
            && q.groupSize == 64 && q.bias == nil
            && q.weight.shape == [n, k / 8] && q.weight.dtype == .uint32
            && q.scales.shape == [n, k / 64] && q.scales.dtype == .bfloat16
            && q.biases?.shape == q.scales.shape && q.biases?.dtype == .bfloat16
    }

    /// The tile-major copy of `q`, with the wide codes too when `wide`.
    private static func makeCopy(_ q: QuantizedLinear, wide: Bool = false) -> Copy {
        let tiles = n / 128, groups = k / 64
        let codes = layout(
            [q.weight.reshaped(1, n, k / 8)], template: [("N", n), ("K", k)],
            grid: (32 * tiles, k, 1), threadGroup: (32, 8, 1),
            outputShapes: [[1, tiles, k, 32]], outputDTypes: [.uint16])[0]
        func tileMajor(_ a: MLXArray) -> MLXArray {
            a.reshaped(1, tiles, 128, groups).transposed(0, 1, 3, 2).contiguous()
        }
        var copy = Copy(codes: codes, scales: tileMajor(q.scales), biases: tileMajor(q.biases!))
        if wide {
            copy.wideCodes = wideLayout(
                [q.weight.reshaped(1, n, k / 8)], template: [("N", n), ("K", k)],
                grid: (32 * tiles, k, 1), threadGroup: (32, 8, 1),
                outputShapes: [[1, tiles * k * 32]], outputDTypes: [.uint16])[0]
        }
        return copy
    }

    /// The shared-down kernel that reads `copy`'s decode codes (the vector-store form once qualified).
    fileprivate static func kernel(for copy: Copy) -> MLXFast.MLXFastKernel? {
        if NemotronZSkip.vstore {
            return copy.wide ? wideKernelVS : kernelVS
        }
        return copy.wide ? wideKernel : kernel
    }

    /// `copy` without its wide codes: the tile-major kernels' copy.
    fileprivate static func tileMajor(_ copy: Copy) -> Copy {
        var narrow = copy
        narrow.wideCodes = nil
        return narrow
    }

    fileprivate static func dispatch(_ x: MLXArray, _ copy: Copy, _ inds: MLXArray,
                                 _ kernel: MLXFast.MLXFastKernel) -> MLXArray {
        let sgs = NemotronZSkip.sgs
        let tiles = copy.wide ? n / 256 + 1 : n / 128
        return kernel(
            [copy.decodeCodes, copy.scales, copy.biases, x, inds],
            template: [("T", DType.bfloat16), ("K", k), ("N", n), ("SGS", sgs)],
            grid: (32 * sgs * tiles, 3, 1), threadGroup: (32 * sgs, 1, 1),
            outputShapes: [[1, 3, n]], outputDTypes: [.bfloat16])[0]
    }

    static func run(_ x: MLXArray, owner: Linear) -> MLXArray? {
        guard enabled, kernel != nil, x.shape == [3, k], x.dtype == .bfloat16 else { return nil }
        let found = lock.withLock { () -> (Copy, MLXArray)? in
            guard let copy = copies[ObjectIdentifier(owner)], let inds else { return nil }
            return (copy, inds)
        }
        guard let found, let kernel = kernel(for: found.0) else { return nil }
        return dispatch(x, found.0, found.1, kernel)
    }

    static func covers(_ owner: Linear) -> Bool {
        guard enabled, kernel != nil else { return false }
        return lock.withLock { copies[ObjectIdentifier(owner)] != nil && inds != nil }
    }

    static func prepare(_ model: Module) {
        guard enabled, kernel != nil, NemotronMoELeader.enabled,
              Device.defaultDevice().deviceType == .gpu else { return }
        let zeros = MLXArray.zeros([3], dtype: .uint32)
        eval(zeros)
        let wide = wideEnabled && wideKernel != nil && qualifyWide(model, zeros)
        let (accepted, checked, wideMissed) = build(model, zeros, wide: wide)
        if wideMissed {
            FileHandle.standardError.write(Data(
                "nemotron shared down zskip: wide kernel differs from the record; tile-major copy\n".utf8))
        }
        let built = wideMissed ? checked.mapValues(tileMajor) : checked
        lock.withLock {
            copies = accepted ? built : [:]
            inds = accepted ? zeros : nil
        }
        let form = built.values.contains(where: \.wide) ? " (wide)" : ""
        FileHandle.standardError.write(Data((accepted
            ? "nemotron shared down zskip on: \(built.count) layers\(form)\n"
            : "nemotron shared down zskip off\n").utf8))
    }

    /// Drops every copy's wide codes when any has them (true), for a consumer whose own check missed with the
    /// wide codes: the tile-major codes passed their record check in `build`.
    fileprivate static func dropWide(_ model: Module) -> Bool {
        let count = lock.withLock { () -> Int? in
            guard copies.values.contains(where: \.wide) else { return nil }
            copies = copies.mapValues(tileMajor)
            return copies.count
        }
        guard let count else { return false }
        FileHandle.standardError.write(Data("nemotron shared down zskip on: \(count) layers\n".utf8))
        return true
    }

    /// Every admissible layer's tile-major copy, checked against the record rows bit for bit with the tile-major
    /// kernel; with `wide`, also its wide codes, checked the same way with the wide kernel. (accepted, copies,
    /// a wide check missed); the copies keep their wide codes only when every wide check passed.
    private static func build(_ model: Module, _ zeros: MLXArray, wide: Bool)
        -> (Bool, [ObjectIdentifier: Copy], Bool)
    {
        var built: [ObjectIdentifier: Copy] = [:]
        var accepted = false
        var wideMissed = false
        do {
            accepted = try withError { () -> Bool in
                for module in model.modules() {
                    guard let moe = module as? NemotronHMoE, let shared = moe.sharedExperts,
                          let q = shared.downProj as? QuantizedLinear, admissible(q),
                          built[ObjectIdentifier(q)] == nil else { continue }
                    var copy = makeCopy(q, wide: wide && !wideMissed)
                    eval(copy.codes, copy.scales, copy.biases)
                    guard let narrowKernel = kernel(for: tileMajor(copy)), finite(q),
                          matches(q, tileMajor(copy), zeros, narrowKernel, seed: built.count)
                    else { return false }
                    if let wideCodes = copy.wideCodes {
                        eval(wideCodes)
                        let same = kernel(for: copy).map { matches(q, copy, zeros, $0, seed: built.count) } ?? false
                        if !same {
                            wideMissed = true
                            copy = tileMajor(copy)
                        }
                    }
                    built[ObjectIdentifier(q)] = copy
                }
                return !built.isEmpty
            }
        } catch {
            accepted = false
        }
        return (accepted, accepted ? built : [:], accepted && wideMissed)
    }

    /// Load time, before any copy is built: the wide layout against the tile-major one on real weights. On the
    /// first and last admissible layers, the wide and tile-major shared-down kernels (`shared_y`), and, when the
    /// layer's routed FC2 has a zero-skip copy, the FC2 co-launch with either shared copy (`prod` and `shared_y`),
    /// bit for bit on seeded relu2-like rows: the record checks' cases plus the measured verify densities.
    /// The transient copies are dropped. One stderr line; false on any miss.
    private static func qualifyWide(_ model: Module, _ zeros: MLXArray) -> Bool {
        guard let narrowKernel = kernel, let wideKernel else { return false }
        var layers: [NemotronHMoE] = []
        for module in model.modules() {
            guard let moe = module as? NemotronHMoE, let q = moe.sharedExperts?.downProj as? QuantizedLinear,
                  admissible(q) else { continue }
            layers.append(moe)
        }
        guard let first = layers.first, let last = layers.last else { return false }
        let picked = first === last ? [first] : [first, last]
        var cases = 0, colaunched = 0
        var same = true
        do {
            try withError {
                for (i, moe) in picked.enumerated() {
                    let q = moe.sharedExperts!.downProj as! QuantizedLinear
                    let wide = makeCopy(q, wide: true), narrow = tileMajor(wide)
                    eval(wide.codes, wide.scales, wide.biases, wide.decodeCodes)
                    var ok = MLXArray(true)
                    for sx in seededRows(seed: i) + realisticRows(seed: i) {
                        let a = dispatch(sx, narrow, zeros, narrowKernel)
                        let b = dispatch(sx, wide, zeros, wideKernel)
                        ok = logicalAnd(ok, (a.view(dtype: .uint16) .== b.view(dtype: .uint16)).all())
                        cases += 1
                    }
                    same = same && ok.item(Bool.self)
                    if let co = NemotronFC2SharedColaunch.qualifyWide(
                        fc2: moe.switchMLP.fc2, narrow: narrow, wide: wide, sinds: zeros, seed: i) {
                        same = same && co.0
                        cases += co.1
                        colaunched += 1
                    }
                }
            }
        } catch {
            same = false
        }
        let scope = picked.count == 2 ? "first and last" : "one"
        FileHandle.standardError.write(Data((same
            ? "nemotron zskip wide: shared down qualified (\(scope) MoE layers, \(cases) cases, \(colaunched) co-launch layers bit-identical)\n"
            : "nemotron zskip wide off: shared down differs from the tile-major kernel\n").utf8))
        return same
    }

    private static func finite(_ q: QuantizedLinear) -> Bool {
        let limit = MLXArray(Float.infinity).asType(.bfloat16)
        return logicalAnd((abs(q.scales) .< limit).all(), (abs(q.biases!) .< limit).all())
            .item(Bool.self)
    }

    /// The record check's seeded rows: sparse with an all-zero and a one-hot row, mixed, tail chunks only, dense,
    /// and -0 in place of some zeros.
    private static func seededRows(seed: Int) -> [MLXArray] {
        let keys = MLXRandom.split(key: MLXRandom.key(UInt64(0x5d0e + seed)), into: 3)
        let h = abs(MLXRandom.normal([3, k], key: keys[0])) + 1e-3
        let keep = MLXRandom.uniform(Float(0) ..< 1, [3, k], key: keys[1]) .< MLXArray(Float(0.16))
        let row = MLXArray((0 ..< 3).map(Int32.init)).reshaped(3, 1)
        let column = MLXArray((0 ..< k).map(Int32.init)).reshaped(1, k)
        let zero = MLXArray(Float(0))
        let oneHot = which(column .== MLXArray(Int32(k - 100)), MLXArray(Float(1.5)), zero)
        let sparse = which(row .== MLXArray(Int32(1)), zero,
                           which(row .== MLXArray(Int32(2)), oneHot, which(keep, h, zero)))
        let mixed = which(keep, h, zero)
        let tail = which(column .>= MLXArray(Int32(k - 128)), h, zero)
        let negative = MLXRandom.uniform(Float(0) ..< 1, [3, k], key: keys[2]) .< MLXArray(Float(0.05))
        let negativeZero = MLXArray(Float(-0.0)).asType(.bfloat16)
        var cases: [MLXArray] = []
        for a in [sparse.asType(.bfloat16), mixed.asType(.bfloat16), tail.asType(.bfloat16),
                  h.asType(.bfloat16)] {
            cases.append((a * a).asType(.bfloat16))
        }
        cases.append(which(logicalAnd(cases[1] .== MLXArray(Float(0)).asType(.bfloat16), negative),
                           negativeZero, cases[1]))
        return cases
    }

    /// relu2-like rows at the measured verify shared-down density (13.4% non-zero), and sparser and denser.
    fileprivate static func realisticRows(seed: Int) -> [MLXArray] {
        let keys = MLXRandom.split(key: MLXRandom.key(UInt64(0x5d1e + seed)), into: 2)
        let h = abs(MLXRandom.normal([3, k], key: keys[0])) * 0.5 + 1e-3
        let u = MLXRandom.uniform(Float(0) ..< 1, [3, k], key: keys[1])
        let zero = MLXArray(Float(0))
        return [Float(0.134), 0.05, 0.4].map { density in
            let a = which(u .< MLXArray(density), h, zero).asType(.bfloat16)
            return (a * a).asType(.bfloat16)
        }
    }

    private static func matches(_ q: QuantizedLinear, _ copy: Copy, _ inds: MLXArray,
                                _ kernel: MLXFast.MLXFastKernel, seed: Int) -> Bool {
        var same = MLXArray(true)
        for act in seededRows(seed: seed) {
            let expected = nemotronMTPLinearRows(act.reshaped(1, 3, k), q)
            let got = dispatch(act, copy, inds, kernel)
            same = logicalAnd(same, (got.view(dtype: .uint16) .== expected.view(dtype: .uint16)).all())
        }
        return same.item(Bool.self)
    }
}

// MARK: - Routed FC2 + shared down in one launch
//
// The captured verify's routed FC2 slot kernel (18 slot rows of 21 tiles) and the shared-expert down zero-skip
// kernel (3 rows of 21 tiles) are data independent: FC2 reads the routed relu2 rows and shared down the shared
// relu2 rows, and only the layer's join reads both results. One launch of 21 rows runs both: a threadgroup picks
// its body by `threadgroup_position_in_grid.y`, and each body is the separate kernel's source text, unchanged
// except that its threadgroup arrays are views of one maximal set and its slot index is the row minus the other
// body's rows. Every thread therefore runs the same loads, partials, order, closing simd_sum and store as in its
// own kernel, so `prod` and `shared_y` carry the same bits. `prepare` re-checks every layer against the two
// separate kernels (which keep their own record checks) and leaves the separate launches on any miss.
// The shared rows (twice the K of a slot row) come first. NEMOTRON_MTP_COLAUNCH=0 keeps the separate launches;
// NEMOTRON_MTP_COLAUNCH_SD_FIRST=0 puts the shared rows last.
enum NemotronFC2SharedColaunch {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_MTP_COLAUNCH"] != "0"
        && !NemotronMoELeader.selfTest
    private static let sharedFirst = ProcessInfo.processInfo.environment["NEMOTRON_MTP_COLAUNCH_SD_FIRST"] != "0"
    /// Unless NEMOTRON_MTP_SD_EARLY=0, a covered layer runs the two separate launches the co-launch was checked
    /// against, and the routed slot products take the shared rows as a dependency: MLX then encodes the shared
    /// down beside the routed FC1 and the shared up beside the router.
    private static let early = ProcessInfo.processInfo.environment["NEMOTRON_MTP_SD_EARLY"] != "0"
    private static let slots = 18
    private static let sharedRows = 3

    /// The layer's shared expert for one captured-verify MoE call; `rows` receives its output when the
    /// co-launch ran (the caller then skips the shared expert's own launches).
    final class Shared {
        let mlp: NemotronHMLP
        let x: MLXArray
        let prepared: MLXArray?
        var rows: MLXArray? = nil
        /// The shared up ReLU² rows when `NemotronRouterSharedUpColaunch` already produced them.
        var up: MLXArray? = nil
        init(mlp: NemotronHMLP, x: MLXArray, prepared: MLXArray?) {
            self.mlp = mlp
            self.x = x
            self.prepared = prepared
        }
    }

    /// The routed slot products, with `shared.rows` set, or nil (nothing set) when the co-launch does not apply.
    static func run(_ x: MLXArray, _ inds: MLXArray, _ scores: MLXArray, fc2: SwitchLinear,
                    shared: Shared) -> MLXArray? {
        let down = shared.mlp.downProj
        guard covers(fc2: fc2, down: down),
              let sx = shared.up ?? NemotronSharedUpPacking.unpackedActivation(
                  shared.x, up: shared.mlp.upProj, down: down, prepared: shared.prepared) else { return nil }
        if early, let rows = NemotronSharedDownZSkip.run(sx, owner: down),
           let prod = NemotronZSkip.weightedProducts(
               depends(input: x, dependencies: [rows]), inds, scores, owner: fc2) {
            shared.rows = rows
            return prod
        }
        guard let pair = run(x, inds, scores, fc2: fc2, sx: sx, down: down) else { return nil }
        shared.rows = pair.1
        return pair.0
    }

    /// `fc2`: the routed slot kernel's source (plain or dictionary); `shared`: `NemotronSharedDownZSkip.source`.
    private static func source(fc2: String, shared: String) -> String? {
        let arrays = "  threadgroup float part[32 * PS];\n  threadgroup T xsh[K];\n  threadgroup uchar cmask[chunks];\n"
        let views = "  threadgroup float* part = nmc_part;\n  threadgroup T* xsh = nmc_xsh;\n"
            + "  threadgroup uchar* cmask = nmc_cmask;\n"
        let slot = "  const int slot = int(threadgroup_position_in_grid.y);\n"
        guard fc2.components(separatedBy: arrays).count == 2, fc2.components(separatedBy: slot).count == 2,
              shared.components(separatedBy: arrays).count == 2, shared.components(separatedBy: slot).count == 2,
              fc2.contains("constexpr int OT = 128;"), shared.contains("constexpr int OT = 128;")
        else { return nil }
        let routed = fc2.replacingOccurrences(of: arrays, with: views)
            .replacingOccurrences(of: slot, with: "  const int slot = nmc_row - (SDFIRST ? SROWS : 0);\n")
        let down = shared.replacingOccurrences(of: arrays, with: views)
            .replacingOccurrences(of: slot, with: "  const int slot = nmc_row - (SDFIRST ? 0 : SLOTS);\n")
        return """
          threadgroup float nmc_part[32 * (128 + 1)];
          threadgroup T nmc_xsh[KS > KF ? KS : KF];
          threadgroup uchar nmc_cmask[(KS > KF ? KS : KF) / 8];
          const int nmc_row = int(threadgroup_position_in_grid.y);
          if (SDFIRST ? nmc_row >= SROWS : nmc_row < SLOTS) {
            constexpr int K = KF;
          \(routed)
          } else {
            constexpr int K = KS;
            const auto wk = swk;
            const auto sk = ssk;
            const auto bk = sbk;
            const auto x = sx;
            const auto inds = sinds;
          \(down)
          }
          """
    }

    private static func make(_ name: String, fc2: String?, shared: String?) -> MLXFast.MLXFastKernel? {
        guard let fc2, let shared, let source = source(fc2: fc2, shared: shared) else { return nil }
        return MLXFast.metalKernel(
            name: name, inputNames: ["wk", "sk", "bk", "x", "inds", "scores", "swk", "ssk", "sbk", "sx", "sinds"],
            outputNames: ["prod", "shared_y"], source: source,
            header: nemotronMoELeaderHeader + nemotronMoELeaderStore + nemotronZSkipHeader,
            ensureRowContiguous: true)
    }
    private static let plainKernel = make("nemotron_zskip_fc2_shared_down_tiled_fast",
                                          fc2: nemotronZSkipTiledFastSource, shared: NemotronSharedDownZSkip.source)
    private static let dictKernel = make("nemotron_zskip_fc2_shared_down_tiled_fast_sbd",
                                         fc2: nemotronZSkipDictSource(nemotronZSkipTiledFastSource),
                                         shared: NemotronSharedDownZSkip.source)
    /// The same routed bodies with the wide shared-down body (`NemotronSharedDownZSkip.wideSource`): the grid
    /// stays the routed body's N / 128 threadgroups per row, and shared threadgroups past the wide tail tile
    /// return at once.
    private static let plainWideKernel = make("nemotron_zskip_fc2_shared_down_wide",
                                              fc2: nemotronZSkipTiledFastSource,
                                              shared: NemotronSharedDownZSkip.wideSource)
    private static let dictWideKernel = make("nemotron_zskip_fc2_shared_down_wide_sbd",
                                             fc2: nemotronZSkipDictSource(nemotronZSkipTiledFastSource),
                                             shared: NemotronSharedDownZSkip.wideSource)

    private static let lock = NSLock()
    /// Routed FC2 owner -> its layer's shared down owner, for layers whose co-launch matched.
    nonisolated(unsafe) private static var pairs: [ObjectIdentifier: ObjectIdentifier] = [:]

    /// The kernel that runs `copy`'s slot body today with `shared`'s code layout, or nil when the routed side is
    /// not the fast tiled kernel.
    private static func kernel(for copy: NemotronZSkip.Copy,
                               shared: NemotronSharedDownZSkip.Copy) -> MLXFast.MLXFastKernel? {
        guard NemotronZSkip.tiled, NemotronZSkip.fast else { return nil }
        if shared.wide {
            return copy.dict ? dictWideKernel : plainWideKernel
        }
        return copy.dict ? dictKernel : plainKernel
    }

    private static func dispatch(
        _ kernel: MLXFast.MLXFastKernel, _ x: MLXArray, _ inds: MLXArray, _ scores: MLXArray,
        _ copy: NemotronZSkip.Copy, _ sx: MLXArray, _ shared: NemotronSharedDownZSkip.Copy, _ sinds: MLXArray
    ) -> (MLXArray, MLXArray) {
        let sgs = NemotronZSkip.sgs
        let outputs = kernel(
            [NemotronBankBinding.bind(copy.codes), NemotronBankBinding.bind(copy.scales),
             NemotronBankBinding.bind(copy.biases), x, inds, scores,
             shared.decodeCodes, shared.scales, shared.biases, sx, sinds],
            template: [("T", DType.bfloat16), ("KF", 1856), ("KS", NemotronSharedDownZSkip.k),
                       ("N", 2688), ("SGS", sgs), ("SLOTS", slots), ("SROWS", sharedRows),
                       ("SDFIRST", sharedFirst)],
            grid: (32 * sgs * (2688 / 128), slots + sharedRows, 1), threadGroup: (32 * sgs, 1, 1),
            outputShapes: [[slots * 2688], [1, sharedRows, NemotronSharedDownZSkip.n]],
            outputDTypes: [.float32, .bfloat16])
        return (outputs[0], outputs[1])
    }

    /// The captured verify's (R = 3 weighted slot products, shared down rows) of one MoE layer in one launch, or
    /// nil whenever `NemotronZSkip.weightedProducts` or `NemotronSharedDownZSkip.run` would not run (the caller
    /// then keeps the separate launches). `sx`: the shared relu2 rows [3, 3712].
    static func run(_ x: MLXArray, _ inds: MLXArray, _ scores: MLXArray, fc2: SwitchLinear,
                    sx: MLXArray, down: Linear) -> (MLXArray, MLXArray)? {
        guard enabled, !NemotronZSkip.needsOriginal(fc2), NemotronZSkip.joinReduce,
              NemotronHResidualRMSNorm.productsJoinVerified,
              x.shape == [18, 1856], x.dtype == .bfloat16,
              inds.shape == [18], scores.shape == [18], scores.dtype == .float32,
              sx.shape == [3, NemotronSharedDownZSkip.k], sx.dtype == .bfloat16,
              NemotronSharedDownZSkip.enabled,
              lock.withLock({ pairs[ObjectIdentifier(fc2)] == ObjectIdentifier(down) }),
              let copy = NemotronZSkip.lock.withLock({ NemotronZSkip.copies[ObjectIdentifier(fc2)] }),
              let shared = sharedCopy(down),
              let kernel = kernel(for: copy, shared: shared.0)
        else { return nil }
        return dispatch(kernel, x, inds, scores, copy, sx, shared.0, shared.1)
    }

    private static func sharedCopy(_ down: Linear) -> (NemotronSharedDownZSkip.Copy, MLXArray)? {
        NemotronSharedDownZSkip.lock.withLock { () -> (NemotronSharedDownZSkip.Copy, MLXArray)? in
            guard let c = NemotronSharedDownZSkip.copies[ObjectIdentifier(down)],
                  let i = NemotronSharedDownZSkip.inds else { return nil }
            return (c, i)
        }
    }

    /// True when the co-launch may serve the layer of `fc2` and `down`.
    static func covers(fc2: SwitchLinear, down: Linear) -> Bool {
        enabled && lock.withLock { pairs[ObjectIdentifier(fc2)] == ObjectIdentifier(down) }
    }

    /// Load time, after `NemotronZSkip.prepare` and `NemotronSharedDownZSkip.prepare`: every layer's co-launch
    /// against the two separate kernels it replaces, bit for bit on both outputs (the FP32 slot products and the
    /// BF16 shared rows) over seeded relu2-like rows. Any miss keeps every layer on the separate launches.
    static func prepare(_ model: Module) {
        lock.withLock { pairs = [:] }
        guard enabled, Device.defaultDevice().deviceType == .gpu else { return }
        var built: [ObjectIdentifier: ObjectIdentifier] = [:]
        var missed: Int? = nil
        do {
            try withError { () -> Void in
                for module in model.modules() {
                    guard missed == nil, let moe = module as? NemotronHMoE,
                          let down = moe.sharedExperts?.downProj,
                          let copy = NemotronZSkip.lock.withLock({
                              NemotronZSkip.copies[ObjectIdentifier(moe.switchMLP.fc2)] }),
                          let shared = sharedCopy(down),
                          let kernel = kernel(for: copy, shared: shared.0),
                          let sharedKernel = NemotronSharedDownZSkip.kernel(for: shared.0) else { continue }
                    if matches(kernel, copy, shared.0, shared.1, sharedKernel, seed: built.count) {
                        built[ObjectIdentifier(moe.switchMLP.fc2)] = ObjectIdentifier(down)
                    } else {
                        missed = built.count
                    }
                }
            }
        } catch {
            missed = missed ?? built.count
        }
        if let missed {
            if NemotronSharedDownZSkip.dropWide(model) {
                // A wide shared-down copy missed: every layer again on the tile-major copies.
                FileHandle.standardError.write(Data(
                    "nemotron colaunch: layer \(missed) differs with the wide shared down; tile-major copy\n".utf8))
                return prepare(model)
            }
            FileHandle.standardError.write(Data(
                "nemotron colaunch off: layer \(missed) differs from the separate launches\n".utf8))
            return
        }
        lock.withLock { pairs = built }
        FileHandle.standardError.write(Data("nemotron colaunch on: \(built.count) MoE layers\n".utf8))
    }

    /// `NemotronSharedDownZSkip`'s wide qualifier, co-launch half: the co-launch with the tile-major shared copy
    /// `narrow` against the co-launch with the wide shared copy `wide` of the same layer, bit for bit on both
    /// outputs, over the record check's seeded rows plus rows at the measured verify densities (routed 7.5%,
    /// shared 13.4%). Nil when the layer's routed FC2 has no copy or the co-launch is off; else (all equal,
    /// cases).
    fileprivate static func qualifyWide(fc2: SwitchLinear, narrow: NemotronSharedDownZSkip.Copy,
                                        wide: NemotronSharedDownZSkip.Copy, sinds: MLXArray,
                                        seed: Int) -> (Bool, Int)? {
        guard enabled, let copy = NemotronZSkip.lock.withLock({ NemotronZSkip.copies[ObjectIdentifier(fc2)] }),
              let narrowKernel = kernel(for: copy, shared: narrow),
              let wideKernel = kernel(for: copy, shared: wide) else { return nil }
        let cases = seededCases(seed: seed, realistic: true)
        var same = MLXArray(true)
        for (x, sx) in cases.rows {
            let a = dispatch(narrowKernel, x, cases.inds, cases.scores, copy, sx, narrow, sinds)
            let b = dispatch(wideKernel, x, cases.inds, cases.scores, copy, sx, wide, sinds)
            same = logicalAnd(same, logicalAnd(
                (a.0.view(dtype: .uint32) .== b.0.view(dtype: .uint32)).all(),
                (a.1.view(dtype: .uint16) .== b.1.view(dtype: .uint16)).all()))
        }
        return (same.item(Bool.self), cases.rows.count)
    }

    /// Seeded (routed rows [18, 1856], shared rows [3, 3712]) pairs: sparse with an all-zero and a one-hot
    /// slot/row, tail chunks only, dense, -0 for some zeros; with `realistic`, three more pairs of relu2-like rows
    /// at the measured verify densities (routed 7.5%, shared 13.4%) and sparser/denser ones. Routes repeat experts
    /// across the three rows.
    private static func seededCases(seed: Int, realistic: Bool)
        -> (inds: MLXArray, scores: MLXArray, rows: [(MLXArray, MLXArray)])
    {
        let k = NemotronSharedDownZSkip.k
        var routes: [UInt32] = [3, 17, 42, 77, 100, 121, 17, 5, 64, 90, 3, 11, 3, 8, 50, 66, 101, 127]
        for i in routes.indices {
            routes[i] = (routes[i] + UInt32(seed) * 7) % 128
        }
        let inds = MLXArray(routes)
        let keys = MLXRandom.split(key: MLXRandom.key(UInt64(0xc01a + seed)), into: 6)
        let zero = MLXArray(Float(0))
        let bf16Zero = MLXArray(Float(0)).asType(.bfloat16)
        let negativeZero = MLXArray(Float(-0.0)).asType(.bfloat16)
        // Routed rows: sparse with an all-zero and a one-hot slot, tail chunks only, dense, -0 for some zeros.
        let h = abs(MLXRandom.normal([18, 1856], key: keys[0])) + 1e-3
        let keep = MLXRandom.uniform(Float(0) ..< 1, [18, 1856], key: keys[1]) .< MLXArray(Float(0.08))
        let slot = MLXArray((0 ..< 18).map(Int32.init)).reshaped(18, 1)
        let column = MLXArray((0 ..< 1856).map(Int32.init)).reshaped(1, 1856)
        let oneHot = which(column .== MLXArray(Int32(1800)), MLXArray(Float(1.5)), zero)
        let sparse = which(slot .== MLXArray(Int32(7)), zero,
                           which(slot .== MLXArray(Int32(8)), oneHot, which(keep, h, zero)))
        let tail = which(column .>= MLXArray(Int32(1792)), h, zero)
        var routed = [sparse, tail, h].map { a -> MLXArray in
            let b = a.asType(.bfloat16)
            return (b * b).asType(.bfloat16)
        }
        let flip = MLXRandom.uniform(Float(0) ..< 1, [18, 1856], key: keys[5]) .< MLXArray(Float(0.05))
        routed.append(which(logicalAnd(routed[0] .== bf16Zero, flip), negativeZero, routed[0]))
        let scores = MLXRandom.uniform(Float(0) ..< 1, [18], key: keys[2])
        // Shared rows: sparse with an all-zero and a one-hot row, tail chunks only, dense, -0 for some zeros.
        let sh = abs(MLXRandom.normal([3, k], key: keys[3])) + 1e-3
        let skeep = MLXRandom.uniform(Float(0) ..< 1, [3, k], key: keys[4]) .< MLXArray(Float(0.16))
        let row = MLXArray((0 ..< 3).map(Int32.init)).reshaped(3, 1)
        let scolumn = MLXArray((0 ..< k).map(Int32.init)).reshaped(1, k)
        let soneHot = which(scolumn .== MLXArray(Int32(k - 100)), MLXArray(Float(1.5)), zero)
        let ssparse = which(row .== MLXArray(Int32(1)), zero,
                            which(row .== MLXArray(Int32(2)), soneHot, which(skeep, sh, zero)))
        let stail = which(scolumn .>= MLXArray(Int32(k - 128)), sh, zero)
        var sharedRows = [ssparse, stail, sh].map { a -> MLXArray in
            let b = a.asType(.bfloat16)
            return (b * b).asType(.bfloat16)
        }
        let sflip = logicalNot(skeep)
        sharedRows.append(which(logicalAnd(sharedRows[0] .== bf16Zero, sflip), negativeZero, sharedRows[0]))
        var rows = Array(zip(routed, sharedRows))
        if realistic {
            let rkeys = MLXRandom.split(key: MLXRandom.key(UInt64(0xc11a + seed)), into: 2)
            let rh = abs(MLXRandom.normal([18, 1856], key: rkeys[0])) * 0.5 + 1e-3
            let u = MLXRandom.uniform(Float(0) ..< 1, [18, 1856], key: rkeys[1])
            let shared = NemotronSharedDownZSkip.realisticRows(seed: seed)
            for (i, density) in [Float(0.075), 0.02, 0.3].enumerated() {
                let a = which(u .< MLXArray(density), rh, zero).asType(.bfloat16)
                rows.append(((a * a).asType(.bfloat16), shared[i]))
            }
        }
        return (inds, scores, rows)
    }

    private static func matches(_ kernel: MLXFast.MLXFastKernel, _ copy: NemotronZSkip.Copy,
                                _ shared: NemotronSharedDownZSkip.Copy, _ sinds: MLXArray,
                                _ sharedKernel: MLXFast.MLXFastKernel, seed: Int) -> Bool {
        let cases = seededCases(seed: seed, realistic: false)
        var same = MLXArray(true)
        for (x, sx) in cases.rows {
            let got = dispatch(kernel, x, cases.inds, cases.scores, copy, sx, shared, sinds)
            let products = NemotronZSkip.products(x, cases.inds, cases.scores, copy: copy, rows: 3)
            let down = NemotronSharedDownZSkip.dispatch(sx, shared, sinds, sharedKernel)
            same = logicalAnd(same, logicalAnd(
                (got.0.view(dtype: .uint32) .== products.view(dtype: .uint32)).all(),
                (got.1.view(dtype: .uint16) .== down.view(dtype: .uint16)).all()))
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

  const int out_row = tid.y * (int(threads_per_threadgroup.y) * results_per_simdgroup) + simd_gid * results_per_simdgroup;
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

// MARK: - Exact (scale, bias) pair dictionary for the R = 3 verify QMVs
//
// The pinned checkpoint's affine4/gs64 groups carry a BF16 scale and a BF16
// bias each: 4 bytes of metadata per 64 weights, 1/9 of a group's bytes. Per
// tensor the (scale, bias) pairs take only a few thousand distinct values
// (2,841-4,518 measured on whole routed FC1/FC2, shared up and in_proj
// tensors), so one uint16 pair index per group and a per-tensor table of the
// pairs, each entry the exact BF16 -> FP32 widening, hold the same numbers in
// half the metadata bytes. The decode-bound R = 3 verify QMVs (routed FC1 and
// the prepared projections) read their metadata through the index; their
// arithmetic, order and rounding are unchanged (`nemotronSBDictSource`).
// Target weights, codes, quantization and every other path keep the loaded
// arrays. Built at load from the loaded arrays; a tensor whose pairs need more
// than 16 bits or whose table does not decode back bit for bit gets none, and
// a kernel that differs from its record kernel on the real weights drops all.
// NEMOTRON_MTP_SB_DICT=0 keeps the record kernels.
enum NemotronSBDict {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_MTP_SB_DICT"] != "0"

    struct Dict {
        /// One uint16 pair index per group, in the scales' shape and order.
        let index: MLXArray
        /// [pairs, 2] FP32: each pair's scale and bias, widened exactly from BF16.
        let table: MLXArray
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var dicts: [ObjectIdentifier: Dict] = [:]

    /// The verified dictionary of `owner`'s scales and biases, if any.
    static func dict(_ owner: AnyObject) -> Dict? {
        lock.withLock { dicts[ObjectIdentifier(owner)] }
    }

    /// Every built dictionary's index (for `NemotronBankBinding.warm`: an FC1
    /// index is per expert and addressed from its base like the scales).
    static var indexArrays: [MLXArray] {
        lock.withLock { dicts.values.map { $0.index } }
    }

    /// Every built dictionary's index and table (decode-only backing for the residency lease).
    static var allArrays: [MLXArray] {
        lock.withLock { dicts.values.flatMap { [$0.index, $0.table] } }
    }

    /// One tensor's pair dictionary, or nil when its pairs need more than 16
    /// bits or the device decode of the table differs from the BF16 -> FP32
    /// widening of the loaded scales and biases in any bit.
    static func make(scales: MLXArray, biases: MLXArray) -> Dict? {
        guard scales.dtype == .bfloat16, biases.dtype == .bfloat16, scales.shape == biases.shape
        else { return nil }
        let s = scales.view(dtype: .uint16).asArray(UInt16.self)
        let b = biases.view(dtype: .uint16).asArray(UInt16.self)
        var scaleID = [Int32](repeating: -1, count: 65536)
        var biasID = [Int32](repeating: -1, count: 65536)
        var scaleCount = 0, biasCount = 0
        for v in s where scaleID[Int(v)] < 0 {
            scaleID[Int(v)] = Int32(scaleCount)
            scaleCount += 1
        }
        for v in b where biasID[Int(v)] < 0 {
            biasID[Int(v)] = Int32(biasCount)
            biasCount += 1
        }
        guard scaleCount * biasCount <= 1 << 26 else { return nil }
        var pairID = [Int32](repeating: -1, count: scaleCount * biasCount)
        var words: [UInt32] = []
        var index = [UInt16](repeating: 0, count: s.count)
        var fits = true
        // Raw buffers: ~10M groups per routed bank at load.
        s.withUnsafeBufferPointer { sp in b.withUnsafeBufferPointer { bp in
        scaleID.withUnsafeBufferPointer { si in biasID.withUnsafeBufferPointer { bi in
        pairID.withUnsafeMutableBufferPointer { pid in index.withUnsafeMutableBufferPointer { ip in
            for i in 0 ..< sp.count {
                let key = Int(si[Int(sp[i])]) * biasCount + Int(bi[Int(bp[i])])
                var p = pid[key]
                if p < 0 {
                    guard words.count < 2 * 65536 else { fits = false; return }
                    p = Int32(words.count / 2)
                    pid[key] = p
                    words.append(UInt32(sp[i]) << 16)
                    words.append(UInt32(bp[i]) << 16)
                }
                ip[i] = UInt16(p)
            }
        } } } } } }
        guard fits else { return nil }
        let indexArray = MLXArray(index, scales.shape)
        let table = MLXArray(words, [words.count / 2, 2]).view(dtype: .float32)
        let decoded = take(table, indexArray.reshaped(-1).asType(.int32), axis: 0)
        let widened = stacked(
            [scales.reshaped(-1).asType(.float32), biases.reshaped(-1).asType(.float32)], axis: 1)
        guard (decoded.view(dtype: .uint32) .== widened.view(dtype: .uint32)).all().item(Bool.self)
        else { return nil }
        eval(indexArray, table)
        return Dict(index: indexArray, table: table)
    }

    /// Load time, before any verify graph is traced or warmed: a dictionary for
    /// every routed FC1 bank and every 4-bit target projection, then each
    /// dictionary kernel against its record kernel, bit for bit, on the real
    /// weights of every layer it serves.
    static func prepare(_ model: Module) {
        guard enabled, Device.defaultDevice().deviceType == .gpu else { return }
        var built: [ObjectIdentifier: Dict] = [:]
        for module in model.modules() {
            if let moe = module as? NemotronHMoE {
                guard type(of: moe.switchMLP.fc1) == QuantizedSwitchLinear.self,
                      let up = NemotronMoELeader.bank(moe.switchMLP.fc1, k: 2688, dtype: .bfloat16),
                      up.weight.shape == [128, 1856, 336],
                      let d = make(scales: up.scales, biases: up.biases) else { continue }
                built[ObjectIdentifier(moe.switchMLP.fc1)] = d
            } else if let q = module as? QuantizedLinear, q.mode == .affine, q.bits == 4,
                      q.groupSize == 64, q.bias == nil, q.weight.dtype == .uint32, q.weight.ndim == 2,
                      let biases = q.biases, let d = make(scales: q.scales, biases: biases) {
                built[ObjectIdentifier(q)] = d
            }
        }
        lock.withLock { dicts = built }
        let (same, fc1, shapes) = checkKernels(model)
        guard same else {
            lock.withLock { dicts = [:] }
            FileHandle.standardError.write(
                Data("nemotron sb dict off: a dictionary kernel differs from its record kernel\n".utf8))
            return
        }
        FileHandle.standardError.write(Data(
            "nemotron sb dict on: \(built.count) tensors, \(fc1) FC1 banks, prepared shapes \(shapes)\n".utf8))
    }

    private static func checkKernels(_ model: Module) -> (Bool, Int, [[Int]]) {
        var same = MLXArray(true)
        let keys = MLXRandom.split(key: MLXRandom.key(0x5bd1), into: 2)
        // Expert 3 is in all three rows (the triple walk), 17 in two (a pair).
        let inds = MLXArray([3, 17, 42, 77, 100, 121, 17, 5, 64, 90, 3, 11, 3, 8, 50, 66, 101, 127] as [UInt32])
        let x = (MLXRandom.normal([3, 2688], key: keys[0]) * 0.5).asType(.bfloat16)
        var fc1 = 0
        var shapes: [[Int]] = []
        var inProj = false, sharedUp = false
        for module in model.modules() {
            if !inProj, let mixer = module as? NemotronHMamba2Mixer,
               let q = mixer.inProj as? QuantizedLinear, dict(q) != nil,
               NemotronMambaInProjConvDt.dictAvailable, let convBias = mixer.conv1d.bias {
                let xk = (MLXRandom.normal([3, 2688], key: keys[1]) * 0.5).asType(.bfloat16)
                let state = (MLXRandom.normal([1, 3, mixer.convDim], key: keys[0]) * 0.5).asType(.bfloat16)
                func run(_ on: Bool) -> [MLXArray]? {
                    NemotronMambaInProjConvDt.apply(
                        xk, q, prepared: nil, oldState: state, convWeight: mixer.conv1d.weight,
                        convBias: convBias, dtBias: mixer.dtBias, timeStepLimit: mixer.timeStepLimit,
                        convStart: mixer.intermediateSize, convDim: mixer.convDim, heads: mixer.numHeads,
                        sbDict: on).map { [$0.projected, $0.convolution, $0.padded, $0.dt] }
                }
                if let a = run(false), let b = run(true) {
                    for (u, v) in zip(a, b) {
                        same = logicalAnd(same, u.dtype == .float32
                            ? (u.view(dtype: .uint32) .== v.view(dtype: .uint32)).all()
                            : (u.view(dtype: .uint16) .== v.view(dtype: .uint16)).all())
                    }
                    inProj = true
                }
            }
            if !sharedUp, let q = module as? QuantizedLinear, dict(q) != nil,
               q.weight.shape == [3712, 336] {
                let xk = (MLXRandom.normal([3, 2688], key: keys[1]) * 0.5).asType(.bfloat16)
                if let ok = NemotronSharedUpPacking.dictMatches(xk, q) {
                    same = logicalAnd(same, ok)
                    sharedUp = true
                }
            }
            if let moe = module as? NemotronHMoE, let d = dict(moe.switchMLP.fc1),
               let up = NemotronMoELeader.bank(moe.switchMLP.fc1, k: 2688, dtype: .bfloat16) {
                guard NemotronMoELeader.dictFC1Available,
                      let a = NemotronMoELeader.applyWithBalancePlan(
                          x, inds, bank: up, relu2: true, prepared: nil),
                      let b = NemotronMoELeader.applyWithBalancePlan(
                          x, inds, bank: up, relu2: true, prepared: nil, dict: d)
                else { continue }
                same = logicalAnd(same, logicalAnd(
                    (a.activation.view(dtype: .uint16) .== b.activation.view(dtype: .uint16)).all(),
                    (a.plan .== b.plan).all()))
                fc1 += 1
            } else if let q = module as? QuantizedLinear, dict(q) != nil {
                let k = q.weight.dim(1) * 8, n = q.weight.dim(0)
                guard NemotronHeadPreparedActivation.dictLaunchAvailable(k: k, n: n),
                      !shapes.contains([k, n]) else { continue }
                shapes.append([k, n])
                let xk = (MLXRandom.normal([3, k], key: keys[1]) * 0.5).asType(.bfloat16)
                let rps = n == 131072 ? NemotronMultiRowQMV.headRPS(n: n) : 2
                for relu2 in [false, true] {
                    let a = NemotronHeadPreparedActivation.apply(xk, q, relu2: relu2, rps: rps, sbDict: false)
                    let b = NemotronHeadPreparedActivation.apply(xk, q, relu2: relu2, rps: rps)
                    same = logicalAnd(same, (a.view(dtype: .uint16) .== b.view(dtype: .uint16)).all())
                }
            }
        }
        if inProj { shapes.append([2688, 10304, 1]) }
        if sharedUp { shapes.append([2688, 3712, 1]) }
        return (same.item(Bool.self), fc1, shapes)
    }

    // MARK: Draft head (one row per draft step)
    //
    // The MTP head's own 4-bit tensors (eh_proj, Q/K/V, o_proj, shared up and
    // down, the routed FC1 bank) get pair dictionaries of their own, read only
    // by the head's R = 1 draft-step launches: the one-row QMV (eh_proj,
    // o_proj, shared up and down), the fused Q/K/V launch and the routed FC1
    // leader. Same tables and decode check as above, kept apart from the
    // target's: a target tensor never finds a dictionary here, and the head's
    // multi-row launches keep the loaded arrays. The routed FC2 reads its own
    // copy (`NemotronZSkip.prepareDraft`, with the bank's pair dictionary). A
    // tensor keeps its dictionary only if every R = 1 dictionary kernel that
    // reads it equals its record kernel bit for bit on the real weights.
    // NEMOTRON_MTP_DRAFT_SBDICT=0 (or NEMOTRON_MTP_SB_DICT=0) keeps the record kernels.
    static let draftEnabled = enabled
        && ProcessInfo.processInfo.environment["NEMOTRON_MTP_DRAFT_SBDICT"] != "0"
    nonisolated(unsafe) private static var draftDicts: [ObjectIdentifier: Dict] = [:]

    /// The admitted draft-head dictionary of `owner`'s scales and biases, if any.
    static func draftDict(_ owner: AnyObject) -> Dict? {
        lock.withLock { draftDicts[ObjectIdentifier(owner)] }
    }

    /// Every admitted draft-head index and table (decode-only backing for the head's residency set).
    static var draftArrays: [MLXArray] {
        lock.withLock { draftDicts.values.flatMap { [$0.index, $0.table] } }
    }

    /// `arrays` without the scales and biases of the head's routed banks that have an admitted dictionary: the
    /// head's routed experts run only one-row draft steps, whose dictionary twin binds the bank's codes, index
    /// and table, so those scales and biases are not decode backing. NEMOTRON_MTP_DRAFT_BANK_SB_RESIDENCY=1
    /// keeps them.
    static func withoutDictBankMetadata(_ arrays: [MLXArray], module: Module) -> [MLXArray] {
        guard ProcessInfo.processInfo.environment["NEMOTRON_MTP_DRAFT_BANK_SB_RESIDENCY"] != "1" else { return arrays }
        var skip = Set<ObjectIdentifier>()
        for (_, m) in module.namedModules() {
            guard let linear = m as? SwitchLinear, draftDict(linear) != nil else { continue }
            for (key, a) in linear.parameters().flattened() where key == "scales" || key == "biases" {
                skip.insert(ObjectIdentifier(a))
            }
        }
        let kept = arrays.filter { !skip.contains(ObjectIdentifier($0)) }
        FileHandle.standardError.write(Data(
            "nemotron head residency: \(arrays.count - kept.count) dictionary bank scale/bias arrays left out\n".utf8))
        return kept
    }

    /// The admitted routed-bank indices (for `NemotronBankBinding.warm`: addressed from their base).
    static var draftBankIndexArrays: [MLXArray] {
        lock.withLock { draftDicts.values.compactMap { $0.index.ndim == 3 ? $0.index : nil } }
    }

    /// Load time, before any draft graph is traced or warmed: a dictionary for
    /// the head's routed FC1 bank and every 4-bit head projection, then each
    /// R = 1 dictionary kernel against its record kernel (`checkDraftKernels`).
    /// The head's weights are untouched; a tensor whose check differs (or that
    /// no dictionary kernel reads) keeps its record kernels.
    static func prepareDraft(_ head: Module) {
        guard draftEnabled, Device.defaultDevice().deviceType == .gpu else { return }
        var built: [ObjectIdentifier: Dict] = [:]
        var names: [ObjectIdentifier: String] = [:]
        for (name, module) in head.namedModules() {
            if let moe = module as? NemotronHMoE {
                guard type(of: moe.switchMLP.fc1) == QuantizedSwitchLinear.self,
                      let up = NemotronMoELeader.bank(moe.switchMLP.fc1, k: 2688, dtype: .bfloat16),
                      up.weight.shape == [128, 1856, 336],
                      let d = make(scales: up.scales, biases: up.biases) else { continue }
                built[ObjectIdentifier(moe.switchMLP.fc1)] = d
                names[ObjectIdentifier(moe.switchMLP.fc1)] = name + ".switch_mlp.fc1"
            } else if let q = module as? QuantizedLinear, q.mode == .affine, q.bits == 4,
                      q.groupSize == 64, q.bias == nil, q.weight.dtype == .uint32, q.weight.ndim == 2,
                      let biases = q.biases, let d = make(scales: q.scales, biases: biases) {
                built[ObjectIdentifier(q)] = d
                names[ObjectIdentifier(q)] = name
            }
        }
        lock.withLock { draftDicts = built }
        let kept = checkDraftKernels(head)
        lock.withLock { draftDicts = built.filter { kept.contains($0.key) } }
        let on = built.keys.filter { kept.contains($0) }.compactMap { names[$0] }.sorted()
        let off = built.keys.filter { !kept.contains($0) }.compactMap { names[$0] }.sorted()
        guard !on.isEmpty else {
            FileHandle.standardError.write(Data(
                "nemotron draft sb dict off: no R = 1 dictionary kernel equals its record kernel \(off)\n".utf8))
            return
        }
        let groups = built.filter { kept.contains($0.key) }.values.reduce(0) { $0 + $1.index.size }
        let rest = off.isEmpty ? "" : "; record kernels kept for \(off)"
        FileHandle.standardError.write(Data(
            "nemotron draft sb dict on: \(on.count) tensors \(on), \(groups) groups, R = 1 kernels bitwise equal\(rest)\n".utf8))
    }

    /// Each draft dictionary's R = 1 kernels against their record kernels, bit
    /// for bit, on the head's real weights and seeded rows: the one-row QMV on
    /// every projection (ReLU2 off and on), the fused Q/K/V launch on Q, K and
    /// V together, and the routed FC1 leader on route sets covering all 128
    /// experts (ReLU2 off and on). Also JITs every R = 1 twin pipeline outside
    /// any timed window. Returns the owners every check of which ran and
    /// matched; a dictionary no kernel reads is not kept.
    private static func checkDraftKernels(_ head: Module) -> Set<ObjectIdentifier> {
        let keys = MLXRandom.split(key: MLXRandom.key(0x5bd2), into: 3)
        var same: [ObjectIdentifier: MLXArray] = [:]
        func record(_ owner: AnyObject, _ ok: MLXArray) {
            let id = ObjectIdentifier(owner)
            same[id] = logicalAnd(same[id] ?? MLXArray(true), ok)
        }
        func equal(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
            (a.view(dtype: .uint16) .== b.view(dtype: .uint16)).all()
        }
        var failed = Set<ObjectIdentifier>()
        for module in head.modules() {
            if let q = module as? QuantizedLinear, draftDict(q) != nil,
               NemotronMultiRowQMV.draftDictAvailable {
                let x = (MLXRandom.normal([1, q.weight.dim(1) * 8], key: keys[0]) * 0.5).asType(.bfloat16)
                for relu2 in [false, true] {
                    guard let a = NemotronMultiRowQMV.apply(x, q, relu2: relu2, sbDict: false),
                          let b = NemotronMultiRowQMV.apply(x, q, relu2: relu2) else { continue }
                    record(q, equal(a, b))
                }
            }
            if let attention = module as? NemotronHAttention, NemotronTargetQKV.draftDictAvailable,
               draftDict(attention.wq) != nil, draftDict(attention.wk) != nil,
               draftDict(attention.wv) != nil {
                let x = (MLXRandom.normal([1, 1, 2688], key: keys[1]) * 0.5).asType(.bfloat16)
                let owners: [AnyObject] = [attention.wq, attention.wk, attention.wv]
                if let a = NemotronTargetQKV.apply(
                       x, query: attention.wq, key: attention.wk, value: attention.wv,
                       captureMTP: false, sbDict: false),
                   let b = NemotronTargetQKV.apply(
                       x, query: attention.wq, key: attention.wk, value: attention.wv,
                       captureMTP: false) {
                    let ok = logicalAnd(equal(a.0, b.0), logicalAnd(equal(a.1, b.1), equal(a.2, b.2)))
                    for owner in owners { record(owner, ok) }
                } else {
                    // The launch did not run: Q, K and V share it, so none keeps a dictionary.
                    for owner in owners { failed.insert(ObjectIdentifier(owner)) }
                }
            }
            if let moe = module as? NemotronHMoE, let d = draftDict(moe.switchMLP.fc1),
               NemotronMoELeader.draftDictFC1Available,
               let up = NemotronMoELeader.bank(moe.switchMLP.fc1, k: 2688, dtype: .bfloat16) {
                // 22 sets of six distinct experts cover all 128 (one row each).
                let rows = (MLXRandom.normal([22, 2688], key: keys[2]) * 0.5).asType(.bfloat16)
                for route in 0 ..< 22 {
                    let inds = MLXArray((0 ..< 6).map { UInt32((route * 6 + $0) % 128) })
                    let x = rows[route ..< route + 1]
                    for relu2 in [false, true] {
                        let a = NemotronMoELeader.apply(x, inds, rows: 1, topK: 6, bank: up, perSlot: false,
                            relu2: relu2, tail8: NemotronMoELeader.leaderTail8)
                        let b = NemotronMoELeader.apply(x, inds, rows: 1, topK: 6, bank: up, perSlot: false,
                            relu2: relu2, tail8: NemotronMoELeader.leaderTail8, dict: d)
                        record(moe.switchMLP.fc1, equal(a, b))
                    }
                }
            }
        }
        eval(Array(same.values))
        return Set(same.compactMap { $0.value.item(Bool.self) && !failed.contains($0.key) ? $0.key : nil })
    }
}

// MARK: - Unroll by two: draft head and load-time admission (NEMOTRON_U2_*)

extension NemotronU2 {
    /// DRAFT: the draft head's one-row routed FC1 leader on its admitted pair dictionary (the tail-8 or plain
    /// twin the draft selects), over route sets covering all 128 experts, ReLU2 off and on, unrolled against
    /// record, bit for bit.
    static func draftCheck(_ head: Module) -> (MLXArray, Int)? {
        let kernels = NemotronMoELeaderKernel.shared
        guard (NemotronMoELeader.leaderTail8 && kernels.tail8 != nil ? kernels.tail8DictU2 : kernels.kernelDictU2)
                != nil else { return nil }
        var same = MLXArray(true)
        var cases = 0
        for module in head.modules() {
            if let moe = module as? NemotronHMoE, let d = NemotronSBDict.draftDict(moe.switchMLP.fc1),
               let up = NemotronMoELeader.bank(moe.switchMLP.fc1, k: 2688, dtype: .bfloat16) {
                let rows = (MLXRandom.normal([22, 2688], key: MLXRandom.key(0x52a7)) * 0.5).asType(.bfloat16)
                for route in 0 ..< 22 {
                    let inds = MLXArray((0 ..< 6).map { UInt32((route * 6 + $0) % 128) })
                    let x = rows[route ..< route + 1]
                    for relu2 in [false, true] {
                        let a = NemotronMoELeader.apply(x, inds, rows: 1, topK: 6, bank: up, perSlot: false,
                            relu2: relu2, tail8: NemotronMoELeader.leaderTail8, dict: d, u2: false)
                        let b = NemotronMoELeader.apply(x, inds, rows: 1, topK: 6, bank: up, perSlot: false,
                            relu2: relu2, tail8: NemotronMoELeader.leaderTail8, dict: d, u2: true)
                        same = logicalAnd(same, equal(a, b))
                        cases += 1
                    }
                }
                eval(same)
            }
        }
        return cases > 0 ? (same, cases) : nil
    }

    /// Load time, after the pair dictionaries (target and draft head) are built and before any verify or draft
    /// graph is traced: each family's twin against its record kernel on the real weights (`admit`). A family
    /// whose switch is 0, whose twin is missing or whose check misses keeps its record kernel.
    static func qualify(target: Module, head: Module) {
        inProj = admit(.inProj) { NemotronMambaInProjConvDt.u2Check(target) }
        self.head = admit(.head) { NemotronHeadPreparedActivation.u2CheckHead(target) }
        draft = admit(.draft) { draftCheck(head) }
    }
}

// MARK: - Load-time geometry tune of the verify's triple FC1 (NEMOTRON_FC1_TUNE)
//
// The plan-free triple FC1 (`applyTripleWithoutPlan`: R = 3, top six, K 2688,
// N 1856, prepared FP32 activations, tail-8, pair dictionary, unroll by two)
// launches one 32-lane SIMD group per threadgroup with two outputs each, the
// slot on grid z and the 928 output tiles on grid y. That geometry was tuned
// on M4-class hosts. `NemotronFC1Tune` rebuilds the same body under other
// launch geometries and keeps every output's arithmetic: one output is one
// SIMD group's accumulator, lane l holds values [8l, 8l + 8) of each 256-value
// block, blocks run in K order with the same qdot term order and affine close
// (`scale * accum + sum * bias`) per block, then one closing simd_sum over the
// 32 lanes and the same typed ReLU2 store. A variant moves only
// - RPS: how many consecutive outputs one SIMD group owns (each keeps its own
//   accumulator and order);
// - NSG: how many SIMD groups share a threadgroup (output tiles per group);
// - slot axis: slot on grid x (all 18 slots of an output tile dispatched
//   together) instead of z (all tiles of a slot together);
// - W32: the HALF decode takes its n2/n3 halfwords from the 32-bit word it
//   already loads (the same bits, so the same half and FP32 operands);
// - unroll: `#pragma clang loop unroll_count` on the three main K loops (1:
//   no pragma, the record loops; 4: four blocks per unrolled step).
// None of these touches the lane split, the K order, a term, a rounding point
// or the cross-lane reduction tree. Every tune kernel carries an unread `dep`
// input (FP32): production binds the launch's own prepared activation there.
// `tune` (load time, untimed) still checks every variant bit for bit against
// the launch production makes today on seeded inputs and routes over the real
// routed FC1 banks (every case on every MoE layer), drops any mismatch or Metal
// error (an unavailable incumbent or a timing error keeps the incumbent), and
// checks the incumbent's own geometry rebuilt through the same pipeline (V0's
// timing twin) the same way. It times V0's twin and the survivors round robin
// (`NemotronTuneAdoption.order`, median) as one chain per sample in which each
// launch binds the previous launch's output as `dep`, so the launches run one
// after another as in the round, and adopts the fastest only when
// `NemotronTuneAdoption.decide` admits it (at least 0.5% faster by median, `medianBar`, where the shared rule asks 3%,
// and faster in at least 75% of the cycles). NEMOTRON_FC1_TUNE_BAR sets the median bar (0.97 is the shared one).
// The pair-dictionary form and the plain form are adopted separately: the
// plain form only when some bank without a dictionary took part in the check.
// NEMOTRON_FC1_TUNE=0 keeps the incumbent; NEMOTRON_FC1_FORCE=<n> adopts
// variant n (1-based, `variants`) when it qualified, 0 keeps the incumbent.
enum NemotronFC1Tune {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_FC1_TUNE"] != "0"
    /// An adopted variant's median is at most this fraction of the incumbent's.
    static let medianBar = Double(ProcessInfo.processInfo.environment["NEMOTRON_FC1_TUNE_BAR"] ?? "") ?? 0.995
    static let forced = Int(ProcessInfo.processInfo.environment["NEMOTRON_FC1_FORCE"] ?? "")
    /// Wall time of the timed rounds (graph building included), sized from the warm-up round.
    static let timedMilliseconds = 600.0

    struct Variant: Sendable, Equatable {
        let name: String
        /// Outputs per SIMD group.
        let rps: Int
        /// SIMD groups per threadgroup.
        let nsg: Int
        /// Slot on grid x (slots fastest) instead of z.
        let slotX: Bool
        /// HALF decode reads each lane's weight word once as 32 bits.
        let w32: Bool
        /// unroll_count of the three main K loops; 0: the incumbent's (2 once
        /// `qualifyFC1Unroll` admitted it, else none).
        let unroll: Int
    }

    /// The incumbent is r2n1 (two outputs, one SIMD group, slot on z).
    static let variants: [Variant] = [
        Variant(name: "r2n2", rps: 2, nsg: 2, slotX: false, w32: false, unroll: 0),
        Variant(name: "r2n4", rps: 2, nsg: 4, slotX: false, w32: false, unroll: 0),
        Variant(name: "r1n4", rps: 1, nsg: 4, slotX: false, w32: false, unroll: 0),
        Variant(name: "r4n1", rps: 4, nsg: 1, slotX: false, w32: false, unroll: 0),
        Variant(name: "r4n2", rps: 4, nsg: 2, slotX: false, w32: false, unroll: 0),
        Variant(name: "r2n1x", rps: 2, nsg: 1, slotX: true, w32: false, unroll: 0),
        Variant(name: "r2n4x", rps: 2, nsg: 4, slotX: true, w32: false, unroll: 0),
        Variant(name: "r2n1w", rps: 2, nsg: 1, slotX: false, w32: true, unroll: 0),
        Variant(name: "r2n8", rps: 2, nsg: 8, slotX: false, w32: false, unroll: 0),
        Variant(name: "r1n8", rps: 1, nsg: 8, slotX: false, w32: false, unroll: 0),
        Variant(name: "r4n4", rps: 4, nsg: 4, slotX: false, w32: false, unroll: 0),
        Variant(name: "r8n1", rps: 8, nsg: 1, slotX: false, w32: false, unroll: 0),
        Variant(name: "r8n2", rps: 8, nsg: 2, slotX: false, w32: false, unroll: 0),
        Variant(name: "r2n2x", rps: 2, nsg: 2, slotX: true, w32: false, unroll: 0),
        Variant(name: "r4n1x", rps: 4, nsg: 1, slotX: true, w32: false, unroll: 0),
        Variant(name: "r2n1u1", rps: 2, nsg: 1, slotX: false, w32: false, unroll: 1),
        Variant(name: "r2n1u4", rps: 2, nsg: 1, slotX: false, w32: false, unroll: 4),
        Variant(name: "r4n2u1", rps: 4, nsg: 2, slotX: false, w32: false, unroll: 1),
    ]

    /// The same launch geometry and text (name aside).
    static func sameGeometry(_ a: Variant, _ b: Variant) -> Bool {
        a.rps == b.rps && a.nsg == b.nsg && a.slotX == b.slotX && a.w32 == b.w32 && unrollCount(a) == unrollCount(b)
    }

    /// The incumbent's geometry through the tune's pipeline (the timing twin of the production launch).
    static var incumbentTwin: Variant {
        let rps = NemotronMoELeader.fc1Results ?? NemotronMoELeader.leaderResultsPerSimdgroup(rows: 3)
        let nsg = (1856 / rps) % NemotronMoELeader.fc1SimdGroups == 0 ? NemotronMoELeader.fc1SimdGroups : 2
        return Variant(name: "r\(rps)n\(nsg)", rps: rps, nsg: nsg, slotX: false, w32: false, unroll: 0)
    }

    /// The adopted geometry (nil: the incumbent). Set once at load, before any
    /// verify graph is traced; `plainQualified` admits it for banks without a
    /// pair dictionary too.
    nonisolated(unsafe) private(set) static var choice: Variant? = nil
    nonisolated(unsafe) private(set) static var plainQualified = false

    private static let lock = NSLock()
    nonisolated(unsafe) private static var kernels: [String: MLXFast.MLXFastKernel?] = [:]

    private static func unrollCount(_ v: Variant) -> Int {
        v.unroll > 0 ? v.unroll
            : (NemotronMoELeader.fc1Unroll && NemotronMoELeader.fc1UnrollQualified ? 2 : 1)
    }

    /// The incumbent's source pipeline (`NemotronMoELeaderPreparedKernels`:
    /// unroll, then tail-8, prepared pointers and pair dictionary) with the
    /// variant's slot axis; nil when the record text moved.
    static func source(_ v: Variant, tail8: Bool, dict: Bool) -> String? {
        var s = nemotronMoETripleGateUpSource.replacingOccurrences(
            of: "x2 += block_size;", with: "x2 += PSTRIDE;")
        if v.slotX {
            let from = "  const int slot = tid.z;\n"
            guard s.components(separatedBy: from).count == 2 else { return nil }
            s = s.replacingOccurrences(of: from, with: "  const int slot = tid.x;\n")
        }
        let count = unrollCount(v)
        if count > 1 {
            let loop = "    for (; k < K - block_size; k += block_size) {\n"
            guard s.components(separatedBy: loop).count == 4 else { return nil }
            s = s.replacingOccurrences(of: loop, with: "    #pragma clang loop unroll_count(\(count))\n" + loop)
        }
        if tail8 {
            guard let t = nmlTail8Source(s) else { return nil }
            s = t
        }
        guard var p = nmlPreparedSource(s, pointers: [
            ("const device T* x0 = x + (XSLOT ? slot : row) * K + simd_lid * values_per_thread;",
             "const device float* x0 = prepared + row * (PTILES * PSTRIDE) + simd_lid;"),
            ("const device T* x1 = x + (XSLOT ? partner : partner / TOPK) * K +\n        simd_lid * values_per_thread;",
             "const device float* x1 = prepared + (partner / TOPK) * (PTILES * PSTRIDE) + simd_lid;"),
            ("const device T* x2 = x + (XSLOT ? third : third / TOPK) * K +\n        simd_lid * values_per_thread;",
             "const device float* x2 = prepared + (third / TOPK) * (PTILES * PSTRIDE) + simd_lid;"),
        ]) else { return nil }
        if dict {
            guard let d = nemotronSBDictSource(p) else { return nil }
            p = d
        }
        return p
    }

    /// The incumbent's header; with W32 the HALF helpers' n2/n3 halfword loads
    /// `ws[i]` (i = 0, 1 of a lane's 32-bit word) read as `w2 >> 16 * (i % 2)`
    /// from the word each helper already holds (little endian: the same bits).
    static func header(w32: Bool) -> String? {
        let base = nemotronMoELeaderHeader + nemotronMoELeaderStore + nemotronMoELeaderPreparedHeader
            + nemotronMoETripleDecodeHeader
        guard w32 else { return base }
        var s = base
        for mask in ["0x0f00", "0xf000"] {
            let from = "U(half(ushort(ws[i] & \(mask))))"
            guard s.components(separatedBy: from).count == 5 else { return nil }
            s = s.replacingOccurrences(of: from, with: "U(half(ushort((w2 >> (16 * (i % 2))) & \(mask))))")
        }
        guard !s.contains("ushort(ws[i]") else { return nil }
        return s
    }

    static func kernel(_ v: Variant, tail8: Bool, dict: Bool) -> MLXFast.MLXFastKernel? {
        let key = "\(v.name)_u\(unrollCount(v))_\(tail8 ? "t8" : "tr")_\(dict ? "sbd" : "bf")"
        return lock.withLock { () -> MLXFast.MLXFastKernel? in
            if let made = kernels[key] { return made }
            var made: MLXFast.MLXFastKernel? = nil
            if let body = source(v, tail8: tail8, dict: dict), let header = header(w32: v.w32) {
                made = MLXFast.metalKernel(
                    name: "nemotron_fc1_tune_" + key,
                    inputNames: ["w", "scales", "biases", "prepared", "inds", "dep"],
                    outputNames: ["y"], source: body, header: header)
            }
            kernels[key] = made
            return made
        }
    }

    /// One variant launch with the incumbent's operands and template; nil when
    /// the geometry does not tile N, the kernel is missing, or (`explicit`
    /// false, production) the variant is not admitted for this form. `dep`
    /// (FP32, unread; nil: `packed`) only orders the launch after its producer.
    static func launch(_ v: Variant, packed: MLXArray, inds: MLXArray, bank: NemotronMoELeader.Bank,
                       dict: NemotronSBDict.Dict?, relu2: Bool, tail8: Bool, explicit: Bool,
                       dep: MLXArray? = nil) -> MLXArray? {
        let n = bank.n
        guard explicit || dict != nil || plainQualified,
              n % v.rps == 0, (n / v.rps) % v.nsg == 0,
              let kernel = kernel(v, tail8: tail8, dict: dict != nil)
        else { return nil }
        let operands: [MLXArray] = dict.map {
            [NemotronBankBinding.bind(bank.weight), NemotronBankBinding.bind($0.index), $0.table, packed, inds]
        } ?? [NemotronBankBinding.bind(bank.weight), NemotronBankBinding.bind(bank.scales),
              NemotronBankBinding.bind(bank.biases), packed, inds]
        return kernel(
            operands + [dep ?? packed],
            template: [("T", DType.bfloat16), ("R", 3), ("TOPK", 6),
                       ("K", bank.k), ("N", n), ("XSLOT", false),
                       ("RELU2", relu2), ("RPS", v.rps)],
            grid: v.slotX ? (32 * 18, n / v.rps, 1) : (32, n / v.rps, 18),
            threadGroup: (32, v.nsg, 1),
            outputShapes: [[18, n]], outputDTypes: [.bfloat16])[0]
    }

    /// Deterministic route generator (SplitMix64): three rows of six distinct
    /// experts; rows 1 and 2 reuse `overlap` experts of the rows above.
    private struct Routes {
        var state: UInt64
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        mutating func pick(_ n: Int) -> Int { Int(next() % UInt64(n)) }
        mutating func route(experts: Int, overlap: Int) -> [UInt32] {
            var rows: [[Int]] = []
            for r in 0 ..< 3 {
                var row: [Int] = []
                let seen = Array(Set(rows.flatMap { $0 })).sorted()
                if r > 0 {
                    var pool = seen
                    while row.count < min(overlap, pool.count) {
                        row.append(pool.remove(at: pick(pool.count)))
                    }
                }
                while row.count < 6 {
                    let e = pick(experts)
                    if !row.contains(e) && !seen.contains(e) { row.append(e) }
                }
                for i in stride(from: row.count - 1, to: 0, by: -1) { row.swapAt(i, pick(i + 1)) }
                rows.append(row)
            }
            return rows.flatMap { $0 }.map { UInt32($0) }
        }
    }

    private struct Case {
        let packed: MLXArray
        let inds: MLXArray
        let x: MLXArray
    }

    /// Load time, after `qualifyFC1Unroll` and the pair dictionaries and before
    /// any verify graph is traced or warmed.
    static func tune(_ model: Module) {
        choice = nil
        plainQualified = false
        let t0 = DispatchTime.now().uptimeNanoseconds
        func report(_ message: String) {
            let s = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
            FileHandle.standardError.write(Data(
                "nemotron fc1 tune: \(message) (\(String(format: "%.2f", s)) s)\n".utf8))
        }
        guard enabled else { return report("off (setting)") }
        guard NemotronMoELeader.enabled, NemotronMoELeader.fuseRelu2 else {
            return report("off (triple FC1 not in use)")
        }
        var layers: [(bank: NemotronMoELeader.Bank, dict: NemotronSBDict.Dict?)] = []
        for module in model.modules() {
            guard let moe = module as? NemotronHMoE,
                  type(of: moe.switchMLP.fc1) == QuantizedSwitchLinear.self,
                  let up = NemotronMoELeader.bank(moe.switchMLP.fc1, k: 2688, dtype: .bfloat16),
                  up.weight.shape == [128, 1856, 336] else { continue }
            layers.append((up, NemotronSBDict.enabled ? NemotronSBDict.dict(moe.switchMLP.fc1) : nil))
        }
        guard !layers.isEmpty else { return report("off (no routed FC1 bank)") }

        // Seeded activations and routes: two fixed routes with a three-row and
        // a two-row expert, decode-like random routes (about 15 experts), all
        // 18 distinct, one six-expert triple set; plain, wide, ReLU2-sparse and
        // large inputs.
        var gen = Routes(state: 0xFC1_7E5E)
        let routes: [[UInt32]] = [
            [3, 17, 42, 77, 100, 121, 17, 5, 64, 90, 3, 11, 3, 8, 50, 66, 101, 127],
            [96, 30, 1, 65, 127, 33, 65, 2, 96, 40, 12, 7, 33, 96, 9, 19, 64, 65],
            gen.route(experts: 128, overlap: 2),
            gen.route(experts: 128, overlap: 0),
            [9, 44, 71, 2, 120, 33, 33, 120, 2, 9, 71, 44, 44, 9, 33, 71, 120, 2],
            gen.route(experts: 128, overlap: 1),
            gen.route(experts: 128, overlap: 3),
        ]
        var cases: [Case] = []
        do {
            cases = try withError { () -> [Case] in
                routes.enumerated().map { i, route -> Case in
                    let key = MLXRandom.key(UInt64(0xFC1_0000 + i))
                    var x = MLXRandom.normal([3, 2688], key: key) * ([0.5, 2, 1, 8, 1, 0.5, 1] as [Float])[i]
                    if i == 2 {
                        x = MLX.maximum(x - 1, MLXArray(Float(0)))
                        x = x * x
                    }
                    let xb = x.asType(.bfloat16)
                    let c = Case(packed: NemotronHeadPreparedActivation.prepareContiguousRows(xb, pv: 8),
                                 inds: MLXArray(route), x: xb)
                    eval([c.packed, c.inds, c.x])
                    return c
                }
            }
        } catch {
            return report("off (inputs: \(error))")
        }
        func incumbent(_ l: Int, _ c: Case) -> MLXArray? {
            NemotronMoELeader.applyTripleWithoutPlan(c.x, c.inds, bank: layers[l].bank, relu2: true,
                prepared: c.packed, dict: layers[l].dict)
        }
        func variant(_ v: Variant, _ l: Int, _ c: Case, dep: MLXArray? = nil) -> MLXArray? {
            NemotronMoELeader.applyTripleWithoutPlan(c.x, c.inds, bank: layers[l].bank, relu2: true,
                prepared: c.packed, dict: layers[l].dict, tuned: v, dep: dep)
        }

        // Qualify: bit for bit against the incumbent; any mismatch, missing
        // kernel or error drops a variant. V0's twin (the incumbent geometry
        // through the tune's pipeline, timed in the incumbent's place) is
        // checked the same way; without it the incumbent is kept. A variant's
        // comparisons over every layer of a case are evaluated together, one
        // evaluation per variant and case (its own: a Metal error drops only it).
        let twin = incumbentTwin
        var alive = [twin] + variants.filter { !sameGeometry($0, twin) }
        var dropped: [String] = []
        var plainChecked = false
        var checks = 0
        qualify: for (ci, c) in cases.enumerated() {
            let references: [MLXArray]
            do {
                references = try withError { () -> [MLXArray] in
                    let outs = layers.indices.compactMap { incumbent($0, c) }
                    guard outs.count == layers.count else { return [] }
                    eval(outs)
                    return outs
                }
            } catch {
                references = []
            }
            guard references.count == layers.count else {
                return report("off (incumbent unavailable in case \(ci))")
            }
            if layers.contains(where: { $0.dict == nil }) { plainChecked = true }
            alive = alive.filter { v in
                do {
                    let same = try withError { () -> Bool in
                        var ok = MLXArray(true)
                        for (l, reference) in references.enumerated() {
                            guard let out = variant(v, l, c) else { return false }
                            ok = ok .&& (out.view(dtype: .uint16) .== reference.view(dtype: .uint16)).all()
                        }
                        return ok.item(Bool.self)
                    }
                    if !same { dropped.append("\(v.name) differs@\(ci)") }
                    return same
                } catch {
                    dropped.append("\(v.name) error")
                    return false
                }
            }
            checks += layers.count
            if alive.first != twin || alive.count < 2 { break qualify }
        }
        guard alive.first == twin else {
            return report("off (V0 twin \(twin.name) unavailable or not bitwise equal; "
                + dropped.prefix(4).joined(separator: ", ") + ")")
        }
        alive.removeFirst()
        let dictLayers = layers.filter { $0.dict != nil }.count

        // Time V0's twin and the survivors round robin (`NemotronTuneAdoption.order`): one
        // chain per sample over every MoE layer's FC1 on two decode-like
        // routes, each launch binding the previous launch's output (viewed as
        // FP32) as `dep`, so the launches run one after another as in the
        // round; the chains are built before the clock starts; median ms per
        // launch.
        let timed = [cases[2], cases[5]]
        let candidates: [Variant] = [twin] + alive
        let chain = layers.count * timed.count
        var medians: [Double] = []
        var verdict: NemotronTuneAdoption.Verdict? = nil
        var reps = 0
        if alive.count > 0 {
            var samples = Array(repeating: [Double](), count: candidates.count)
            do {
                try withError {
                    // One candidate's chain (its last output), built lazily.
                    func build(_ i: Int) -> MLXArray? {
                        var prev: MLXArray? = nil
                        for j in 0 ..< chain {
                            guard let out = variant(candidates[i], j % layers.count, timed[j / layers.count],
                                                    dep: prev?.view(dtype: .float32)) else { return nil }
                            prev = out
                        }
                        return prev
                    }
                    // A round: every candidate's chain built first, then evaluated back to back in `order`.
                    func round(_ order: [Int]) -> [Double?] {
                        let chains = order.map { build($0) }
                        return chains.map { last -> Double? in
                            guard let last else { return nil }
                            let start = DispatchTime.now().uptimeNanoseconds
                            eval(last)
                            Stream.gpu.synchronize()
                            return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6 / Double(chain)
                        }
                    }
                    // One warm-up round (every pipeline ran in the check) sizes the timed rounds to
                    // ~`timedMilliseconds` (at least `NemotronTuneAdoption.minCycles`).
                    let warm = DispatchTime.now().uptimeNanoseconds
                    _ = round(Array(candidates.indices))
                    let wall = Double(DispatchTime.now().uptimeNanoseconds - warm) / 1e6
                    reps = NemotronTuneAdoption.cycles(target: timedMilliseconds, wall: wall, cap: 41)
                    for rep in 0 ..< reps {
                        let order = NemotronTuneAdoption.order(candidates.count, cycle: rep)
                        for (i, ms) in zip(order, round(order)) {
                            if let ms { samples[i].append(ms) }
                        }
                    }
                }
            } catch {
                dropped.append("timing error")
                alive = []
            }
            if !alive.isEmpty, samples.allSatisfy({ $0.count == reps }) {
                verdict = NemotronTuneAdoption.decide(samples, medianBar: medianBar)
                medians = verdict?.medians ?? []
            } else {
                if !alive.isEmpty { dropped.append("timing incomplete") }
                alive = []
            }
        }

        var picked: Variant? = nil
        var why = "kept"
        if !medians.isEmpty, !alive.isEmpty {
            if let f = forced {
                if f >= 1, f <= variants.count, let i = alive.firstIndex(of: variants[f - 1]) {
                    picked = alive[i]
                    why = "forced"
                } else {
                    why = f == 0 ? "forced V0" : "forced \(f) not qualified, kept"
                }
            } else if let best = verdict?.adopted {
                picked = candidates[best]
                why = "adopted"
            }
        }
        plainQualified = picked != nil && plainChecked
        choice = picked

        func ms(_ v: Variant?) -> String {
            guard let i = candidates.firstIndex(of: v ?? twin), i < medians.count else { return "n/a" }
            return String(format: "%.4f", medians[i])
        }
        let times = candidates.indices.dropFirst().map {
            "\(candidates[$0].name) \(ms(candidates[$0])) \(verdict?.won($0) ?? "n/a")"
        }
        let rule = verdict?.rule({ candidates[$0].name }) ?? "no candidate timed"
        report("\(why) \(picked?.name ?? "V0") \(ms(picked)) ms vs V0 (\(twin.name)) \(ms(nil)) ms per launch"
            + " (\(rule); median of \(reps), \(chain) chained launches/sample on \(layers.count) layers x 2 routes;"
            + " \(alive.count)/\(variants.count) bitwise over \(checks) layer cases, V0 twin bitwise,"
            + " dict \(dictLayers)/\(layers.count), plain \(plainQualified ? "on" : "off")"
            + (times.isEmpty ? "" : "; [" + times.joined(separator: ", ") + "]")
            + (dropped.isEmpty ? "" : "; dropped: " + dropped.prefix(8).joined(separator: ", ")) + ")")
    }
}

// MARK: - Load-time geometry tune of the verify's dictionary QMVs (NEMOTRON_QMV_TUNE)
//
// The verify's R = 3 pair-dictionary QMVs (`nemotronSBDictSource` of the prepared body, with the vector-store
// closing loops) read most of the round's dense bytes: the Mamba in_proj with its conv/dt epilogue (K 2688,
// N 10304, 23 layers), the target head (K 2688, N 131072), the Mamba out_proj and attention o_proj (K 4096,
// N 2688, 29 layers) and the shared-expert up (K 2688, N 3712, ReLU2, 23 layers). Their geometry (two outputs
// per SIMD group, two SIMD groups per threadgroup, the head's eight; the main-loop load forms; unroll by two)
// was picked on M1/M4-class hosts. `NemotronQMVTune` rebuilds each family's launch through the incumbent's own
// source pipeline (record body, counted load-form and NSG swaps, `nemotronSBDictSource`, `nemotronU2Source`)
// under other geometries and keeps every output's arithmetic: one output is one SIMD group's accumulator, lane l
// holds the same K values of each block, blocks run in K order with the same decode, products, term order and
// affine close, then one closing simd_sum over the 32 lanes and the same typed store (and, for in_proj, the
// same epilogue expressions). A variant moves only
// - RPS: how many consecutive outputs one SIMD group owns (each keeps its own accumulator and order);
// - NSG: how many SIMD groups share a threadgroup (the row -> SIMD-group map);
// - load: the main-loop load form (`nemotronPreparedVariantSource`: 32-bit word loads, hoisted loads, next-block
//   prefetch; the same bits reach the same decode lines);
// - u2: `#pragma clang loop unroll_count(2)` on the main K loop (`NemotronU2`).
// The decode form, dictionary, epilogue and every template argument but RPS / NSG stay the incumbent's. The
// head, out_proj and shared-up tune kernels carry an unread `dep` input (FP32): production binds the launch's own
// prepared activation there.
// `tune` (load time, untimed, after `NemotronU2.qualify` and `qualifyRegs`, before any verify graph is traced)
// checks every variant bit for bit against the production launch on every layer's real weights with seeded rows
// (all four in_proj outputs), drops any mismatch, missing kernel or Metal error, times the incumbent and the
// survivors round robin (`NemotronTuneAdoption.order`, median) and adopts the fastest only when
// `NemotronTuneAdoption.decide` admits it (at least 3% faster by median and faster in at least 75% of the cycles). The
// timing runs one chain per sample in which each launch depends on the one before it, so the launches run one
// after another as in the round: an in_proj launch takes the previous launch's convolution output as its conv
// state (the production kernels, incumbent included); a head, out_proj or shared-up launch binds the previous
// output (viewed as FP32) as `dep`, the incumbent timed as its own geometry through the tune's pipeline (V0's
// twin, checked bit for bit against the production launch like a variant; without it the family keeps its
// incumbent). Only
// the verify's dictionary launches read the choice (`preparedLaunch`, `inProjLaunch`), under the production
// context it was measured in (ReLU2 and unroll state); plain launches, the draft head, the prompt prefill and
// explicit record calls keep their kernels. NEMOTRON_QMV_TUNE=0 keeps every incumbent;
// NEMOTRON_QMV_FORCE=<family>:<n>[,<family>:<n>] (families inproj, head, outproj, sharedup) adopts variant n
// (1-based, `variants`) of that family when it qualified, 0 keeps its incumbent. The shared up runs beside the
// router in one barrier stage, where a geometry timed alone does not carry over, so its family keeps the incumbent
// unless NEMOTRON_QMV_TUNE_SHAREDUP=1 (or a sharedup force) asks for it.
enum NemotronQMVTune {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_QMV_TUNE"] != "0"
    static let sharedUpTuned = ProcessInfo.processInfo.environment["NEMOTRON_QMV_TUNE_SHAREDUP"] == "1"
    static let forced: [Family: Int] = {
        var out: [Family: Int] = [:]
        for item in (ProcessInfo.processInfo.environment["NEMOTRON_QMV_FORCE"] ?? "").split(separator: ",") {
            let parts = item.split(separator: ":")
            guard parts.count == 2, let f = Family(rawValue: String(parts[0])), let n = Int(parts[1]) else { continue }
            out[f] = n
        }
        return out
    }()
    /// Soft cap on the tune's own load time: no family, variant (its first launch JITs the pipeline) or timing
    /// phase starts past it; what has not started keeps its incumbent.
    static let budgetSeconds = 3.0
    /// Wall time of one family's timed rounds (graph building included), sized from the last warm-up round.
    static let timedMilliseconds = 300.0

    enum Family: String, CaseIterable, Sendable {
        case inProj = "inproj", head = "head", outProj = "outproj", sharedUp = "sharedup"
        var k: Int { self == .outProj ? 4096 : 2688 }
        var n: Int {
            switch self {
            case .inProj: return 10304
            case .head: return 131072
            case .outProj: return 2688
            case .sharedUp: return 3712
            }
        }
        var relu2: Bool { self == .sharedUp }
        init?(k: Int, n: Int) {
            guard let f = Family.allCases.first(where: { $0.k == k && $0.n == n }) else { return nil }
            self = f
        }
    }

    /// One launch geometry, resolved against the incumbent.
    struct Config: Sendable, Equatable {
        let rps: Int
        let nsg: Int
        /// Main-loop load form (0 crown, 1 W32, 2 W32 + hoist, 3 + next-block prefetch).
        let load: Int
        /// Main K loop unrolled by two.
        let u2: Bool
        /// The prepared body's HF decode form (nil: crown); the incumbent's.
        let hf: Int?
        /// In_proj register epilogue; the incumbent's.
        let regs: Bool
        /// In_proj 32-bit epilogue (`NemotronMambaInProjConvDt.epilogue32Verified`); the incumbent's.
        var e32 = false
    }

    struct Variant: Sendable, Equatable {
        let name: String
        let rps: Int
        let nsg: Int
        /// nil: the incumbent's load form / unroll.
        let load: Int?
        let u2: Bool?
        func resolved(_ inc: Config) -> Config {
            Config(rps: rps, nsg: nsg, load: load ?? inc.load, u2: u2 ?? inc.u2, hf: inc.hf, regs: inc.regs,
                   e32: inc.e32)
        }
    }

    /// Incumbents: in_proj r2n2 crown load 0 (u2 once admitted); head r2n8 load 2 (u2 once admitted);
    /// out_proj / o_proj r2n2 HF2 load 2; shared up r2n2 crown load 0. A variant that resolves to the
    /// incumbent is skipped.
    static func variants(_ family: Family) -> [Variant] {
        switch family {
        case .inProj:
            return [
                Variant(name: "r2n4", rps: 2, nsg: 4, load: nil, u2: nil),
                Variant(name: "r2n1", rps: 2, nsg: 1, load: nil, u2: nil),
                Variant(name: "r4n2", rps: 4, nsg: 2, load: nil, u2: nil),
                Variant(name: "r2n2l1", rps: 2, nsg: 2, load: 1, u2: nil),
                Variant(name: "r2n2u0", rps: 2, nsg: 2, load: nil, u2: false),
                Variant(name: "r1n4", rps: 1, nsg: 4, load: nil, u2: nil),
                Variant(name: "r2n8", rps: 2, nsg: 8, load: nil, u2: nil),
                Variant(name: "r2n2l2", rps: 2, nsg: 2, load: 2, u2: nil),
            ]
        case .head:
            return [
                Variant(name: "r2n4", rps: 2, nsg: 4, load: nil, u2: nil),
                Variant(name: "r2n16", rps: 2, nsg: 16, load: nil, u2: nil),
                Variant(name: "r4n4", rps: 4, nsg: 4, load: nil, u2: nil),
                Variant(name: "r2n8u0", rps: 2, nsg: 8, load: nil, u2: false),
                Variant(name: "r1n16", rps: 1, nsg: 16, load: nil, u2: nil),
                Variant(name: "r2n32", rps: 2, nsg: 32, load: nil, u2: nil),
                Variant(name: "r2n8l3", rps: 2, nsg: 8, load: 3, u2: nil),
            ]
        case .outProj:
            return [
                Variant(name: "r2n4", rps: 2, nsg: 4, load: nil, u2: nil),
                Variant(name: "r2n1", rps: 2, nsg: 1, load: nil, u2: nil),
                Variant(name: "r4n2", rps: 4, nsg: 2, load: nil, u2: nil),
                Variant(name: "r2n2l0", rps: 2, nsg: 2, load: 0, u2: nil),
                Variant(name: "r2n2u1", rps: 2, nsg: 2, load: nil, u2: true),
                Variant(name: "r1n4", rps: 1, nsg: 4, load: nil, u2: nil),
                Variant(name: "r2n8", rps: 2, nsg: 8, load: nil, u2: nil),
                Variant(name: "r2n2l3", rps: 2, nsg: 2, load: 3, u2: nil),
            ]
        case .sharedUp:
            return [
                Variant(name: "r2n4", rps: 2, nsg: 4, load: nil, u2: nil),
                Variant(name: "r2n1", rps: 2, nsg: 1, load: nil, u2: nil),
                Variant(name: "r4n2", rps: 4, nsg: 2, load: nil, u2: nil),
                Variant(name: "r2n2l2", rps: 2, nsg: 2, load: 2, u2: nil),
                Variant(name: "r2n2u1", rps: 2, nsg: 2, load: nil, u2: true),
                Variant(name: "r1n4", rps: 1, nsg: 4, load: nil, u2: nil),
                Variant(name: "r2n8", rps: 2, nsg: 8, load: nil, u2: nil),
                Variant(name: "r2n2l1", rps: 2, nsg: 2, load: 1, u2: nil),
            ]
        }
    }

    /// The variant body: the incumbent's record body (`base`) through the counted load-form and NSG swaps of
    /// `nemotronPreparedVariantSource` (nil instead of a trap when a swap's text moved), the pair dictionary
    /// (`nemotronSBDictSource`, vector-store closing loops) and, with `u2`, the unroll pragma on the main K loop.
    static func source(base: String, load: Int, u2: Bool) -> String? {
        let loads: [Int: [(String, String, Int)]] = [
            1: nemotronPreparedW32Swaps, 2: nemotronPreparedHoistSwaps, 3: nemotronPreparedPrefetchSwaps]
        guard load == 0 || loads[load] != nil else { return nil }
        var s = base
        for (old, new, count) in (loads[load] ?? []) + nemotronPreparedNSGSwaps {
            guard s.components(separatedBy: old).count - 1 == count else { return nil }
            s = s.replacingOccurrences(of: old, with: new)
        }
        guard let d = nemotronSBDictSource(s) else { return nil }
        return u2 ? nemotronU2Source(d, loop: NemotronU2.qmvLoop, count: 1) : d
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var kernels: [String: MLXFast.MLXFastKernel?] = [:]
    /// Adopted configs by family (set once at load, before any verify graph is traced), with the production
    /// unroll state they replace.
    nonisolated(unsafe) private static var adopted: [Family: (config: Config, u2: Bool)] = [:]

    private static func cached(_ key: String, _ make: () -> MLXFast.MLXFastKernel?) -> MLXFast.MLXFastKernel? {
        lock.withLock {
            if let made = kernels[key] { return made }
            let made = make()
            kernels[key] = made
            return made
        }
    }

    static func preparedKernel(_ c: Config) -> MLXFast.MLXFastKernel? {
        let key = "nemotron_qmv_tune_prep_hf\(c.hf ?? 0)_l\(c.load)_u\(c.u2 ? 2 : 1)"
        return cached(key) {
            let base = c.hf != nil
                ? NemotronHeadPreparedActivation.preparedHalfSource : NemotronHeadPreparedActivation.crownSource
            return source(base: base, load: c.load, u2: c.u2).map {
                MLXFast.metalKernel(
                    name: key, inputNames: ["w", "scales", "biases", "prepared", "K", "N", "dep"], outputNames: ["y"],
                    source: $0, header: NemotronHeadPreparedActivation.variantHeader(half: c.hf != nil, load: c.load))
            }
        }
    }

    static func inProjKernel(_ c: Config) -> MLXFast.MLXFastKernel? {
        let key = "nemotron_qmv_tune_inproj_rg\(c.regs ? 1 : 0)_l\(c.load)_u\(c.u2 ? 2 : 1)" + (c.e32 ? "_e32" : "")
        return cached(key) {
            NemotronMambaInProjConvDt.tuneBase(regs: c.regs, e32: c.e32)
                .flatMap { source(base: $0, load: c.load, u2: c.u2) }.map {
                MLXFast.metalKernel(
                    name: key,
                    inputNames: ["w", "scales", "biases", "prepared", "K", "N",
                                 "oldState", "cw", "cb", "dtBias", "dtLo", "dtHi"],
                    outputNames: ["y", "conv", "padded", "dt"],
                    source: $0, header: NemotronHeadPreparedActivation.variantHeader(half: false, load: c.load),
                    ensureRowContiguous: c.e32)
            }
        }
    }

    /// The R = 3 dictionary launch of [K, N] (kernel, extra template arguments, RPS, NSG): `config` (a load-time
    /// check's), else the adopted config when this is its family's production launch (same ReLU2, same unroll
    /// state), else nil (the caller's `dictLaunch`). The kernel's last input is the unread FP32 `dep`.
    static func preparedLaunch(k: Int, n: Int, relu2: Bool, u2: Bool, config: Config? = nil)
        -> (MLXFast.MLXFastKernel, [(String, any KernelTemplateArg)], Int, Int)? {
        let c: Config
        if let config {
            c = config
        } else {
            guard let f = Family(k: k, n: n), f != .inProj, relu2 == f.relu2,
                  let pick = adopted[f], pick.u2 == u2 else { return nil }
            c = pick.config
        }
        guard let kernel = preparedKernel(c) else { return nil }
        let hf: [(String, any KernelTemplateArg)] = c.hf.map { [("HF", $0)] } ?? []
        return (kernel, hf + [("NSG", c.nsg)], c.rps, c.nsg)
    }

    /// The in_proj dictionary launch: `config` (a load-time check's), else the adopted config on the production
    /// call (`production`: no explicit unroll or epilogue choice), else nil (the record selection).
    static func inProjLaunch(_ config: Config?, production: Bool)
        -> (MLXFast.MLXFastKernel, [(String, any KernelTemplateArg)], Int, Int)? {
        guard let c = config ?? (production ? adopted[.inProj]?.config : nil),
              let kernel = inProjKernel(c) else { return nil }
        return (kernel, [("NSG", c.nsg)], c.rps, c.nsg)
    }

    /// One family's check cases: each runs the production launch (nil) or a config's.
    private typealias Run = (Config?) -> [MLXArray]?

    /// Load time, after `NemotronU2.qualify` and `NemotronMambaInProjConvDt.qualifyRegs`, before any verify graph
    /// is traced or warmed.
    static func tune(_ model: Module) {
        adopted = [:]
        let t0 = DispatchTime.now().uptimeNanoseconds
        func elapsed() -> Double { Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9 }
        func report(_ family: String, _ message: String) {
            FileHandle.standardError.write(Data(
                "nemotron qmv tune \(family): \(message) (\(String(format: "%.2f", elapsed())) s)\n".utf8))
        }
        guard enabled else { return report("all", "off (setting)") }
        guard NemotronSBDict.enabled, NemotronMultiRowQMV.enabled, NemotronMultiRowQMV.narrowEnabled,
              Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu
        else { return report("all", "off (dictionary QMV not in use)") }
        for family in Family.allCases {
            if family == .sharedUp, !sharedUpTuned, forced[.sharedUp] == nil {
                report(family.rawValue, "incumbent kept (runs beside the router; NEMOTRON_QMV_TUNE_SHAREDUP=1 tunes it)")
                continue
            }
            guard elapsed() < budgetSeconds else {
                report(family.rawValue, "skipped (load budget), incumbent kept")
                continue
            }
            report(family.rawValue, tuneFamily(family, model, elapsed: elapsed))
        }
    }

    /// One link of a family's timing chain: the production launch (nil) or a config's, run after `previous` (the
    /// value the link before it returned, nil for the first link); returns what the next link depends on.
    private typealias Step = (Config?, MLXArray?) -> MLXArray?

    private static func tuneFamily(_ family: Family, _ model: Module, elapsed: () -> Double) -> String {
        let k = family.k, n = family.n
        let inc: Config
        var checks: [Run] = []
        var steps: [Step] = []
        var layers = 0
        do {
            try withError {
                if family == .inProj {
                    guard NemotronMambaInProjConvDt.enabled, NemotronMambaInProjConvDt.dictAvailable else { return }
                    for module in model.modules() {
                        guard let mixer = module as? NemotronHMamba2Mixer, let q = mixer.inProj as? QuantizedLinear,
                              type(of: q) == QuantizedLinear.self, q.weight.shape == [n, k / 8],
                              NemotronSBDict.dict(q) != nil, let convBias = mixer.conv1d.bias else { continue }
                        for seed in 0 ..< 2 {
                            let keys = MLXRandom.split(key: MLXRandom.key(UInt64(0x9a10 + 2 * layers + seed)), into: 2)
                            let x = (MLXRandom.normal([3, k], key: keys[0]) * Float(seed == 0 ? 0.5 : 4))
                                .asType(.bfloat16)
                            let state = (MLXRandom.normal([1, 3, mixer.convDim], key: keys[1]) * 0.5).asType(.bfloat16)
                            let packed = NemotronHeadPreparedActivation.pack(x)
                            eval(x, state, packed)
                            func launch(_ c: Config?, _ old: MLXArray)
                                -> (projected: MLXArray, convolution: MLXArray, padded: MLXArray, dt: MLXArray)? {
                                NemotronMambaInProjConvDt.apply(
                                    x, q, prepared: packed, oldState: old, convWeight: mixer.conv1d.weight,
                                    convBias: convBias, dtBias: mixer.dtBias, timeStepLimit: mixer.timeStepLimit,
                                    convStart: mixer.intermediateSize, convDim: mixer.convDim,
                                    heads: mixer.numHeads, qmvTune: c)
                            }
                            checks.append { c in launch(c, state).map { [$0.projected, $0.convolution, $0.padded, $0.dt] } }
                            // Timing link: the previous link's convolution output is this launch's conv state.
                            if seed == 0 { steps.append { c, previous in launch(c, previous ?? state)?.convolution } }
                        }
                        layers += 1
                    }
                } else {
                    guard NemotronHeadPreparedActivation.dictLaunchAvailable(k: k, n: n) else { return }
                    let callerRPS = NemotronMultiRowQMV.headRPS(n: n)
                    let scales: [Float] = family == .head ? [0.5, 2, 4] : [0.5, 4]
                    for module in model.modules() {
                        guard let q = module as? QuantizedLinear, type(of: q) == QuantizedLinear.self,
                              q.weight.shape == [n, k / 8], q.bias == nil, NemotronSBDict.dict(q) != nil
                        else { continue }
                        for (seed, scale) in scales.enumerated() {
                            let key = MLXRandom.key(UInt64(0x9a40 + 8 * layers + seed) ^ UInt64(n))
                            let x = (MLXRandom.normal([3, k], key: key) * scale).asType(.bfloat16)
                            let packed = NemotronHeadPreparedActivation.pack(x)
                            eval(x, packed)
                            func launch(_ c: Config?, dep: MLXArray?) -> MLXArray {
                                NemotronHeadPreparedActivation.apply(x, q, relu2: family.relu2, rps: callerRPS,
                                                                     prepared: packed, qmvTune: c, qmvDep: dep)
                            }
                            checks.append { c in [launch(c, dep: nil)] }
                            // Timing link: `dep` is the previous link's output viewed as FP32. The head is one tensor:
                            // four links per sample.
                            if seed == 0 {
                                let step: Step = { c, previous in launch(c, dep: previous?.view(dtype: .float32)) }
                                steps += Array(repeating: step, count: family == .head ? 4 : 1)
                            }
                        }
                        layers += 1
                    }
                }
            }
        } catch {
            return "off (inputs: \(error))"
        }
        guard !checks.isEmpty else { return "off (no dictionary launch of this family)" }
        if family == .inProj {
            inc = Config(rps: 2, nsg: 2, load: 0, u2: NemotronU2.inProj, hf: nil,
                         regs: NemotronMambaInProjConvDt.regsVerified,
                         e32: NemotronMambaInProjConvDt.epilogue32Verified && !NemotronMambaInProjConvDt.regsVerified)
        } else {
            let callerRPS = NemotronMultiRowQMV.headRPS(n: n)
            let geometry = NemotronHeadPreparedActivation.preparedGeometry[[k, n]] ?? (rps: callerRPS, nsg: 2)
            inc = Config(rps: geometry.rps, nsg: geometry.nsg,
                         load: NemotronHeadPreparedActivation.loadForm(k: k, n: n),
                         u2: NemotronU2.prepared(rows: 3, n: n),
                         hf: NemotronHeadPreparedActivation.decodeForm(rows: 3, fast: k % 512 == 0), regs: false)
        }

        // Applicable variants: the geometry tiles N, fits the vector store and a threadgroup, differs from the
        // incumbent and its kernel builds.
        let all = variants(family)
        var dropped: [String] = []
        var alive: [(variant: Variant, config: Config)] = []
        for v in all {
            let c = v.resolved(inc)
            guard c != inc else { continue }
            guard n % c.rps == 0, (n / c.rps) % c.nsg == 0, 3 * c.rps <= 32, 32 * c.nsg <= 1024,
                  (family == .inProj ? inProjKernel(c) : preparedKernel(c)) != nil else {
                dropped.append("\(v.name) n/a")
                continue
            }
            alive.append((v, c))
        }
        // V0's twin (head, out_proj, shared up): the incumbent's config through the tune's pipeline, timed in the
        // incumbent's place; checked first, like a variant. In_proj times its production launch.
        let twin: (variant: Variant, config: Config)? = family == .inProj ? nil
            : (Variant(name: "V0twin", rps: inc.rps, nsg: inc.nsg, load: nil, u2: nil), inc)
        if let twin {
            guard preparedKernel(twin.config) != nil else { return "off (V0 twin unavailable)" }
            alive.insert(twin, at: 0)
        }

        // Qualify: bit for bit against the production launch on every case.
        var references: [[MLXArray]] = []
        do {
            references = try withError { () -> [[MLXArray]] in
                let outs = checks.compactMap { $0(nil) }
                guard outs.count == checks.count else { return [] }
                eval(outs.flatMap { $0 })
                return outs
            }
        } catch {
            return "off (incumbent: \(error))"
        }
        guard references.count == checks.count else { return "off (incumbent unavailable)" }
        alive = alive.filter { entry in
            // A variant's first launch JITs its pipeline: none starts past the load budget.
            guard elapsed() < budgetSeconds else {
                dropped.append("\(entry.variant.name) budget")
                return false
            }
            do {
                let same = try withError { () -> Bool in
                    var ok = MLXArray(true)
                    for (i, run) in checks.enumerated() {
                        guard let out = run(entry.config), out.count == references[i].count else { return false }
                        for (a, b) in zip(out, references[i]) { ok = logicalAnd(ok, NemotronU2.equal(a, b)) }
                    }
                    return ok.item(Bool.self)
                }
                if !same { dropped.append("\(entry.variant.name) differs") }
                return same
            } catch {
                dropped.append("\(entry.variant.name) error")
                return false
            }
        }
        if let twin {
            guard alive.first?.variant == twin.variant else {
                return "off (V0 twin not bitwise equal to the production launch"
                    + (dropped.isEmpty ? "" : "; dropped: " + dropped.prefix(4).joined(separator: ", ")) + ")"
            }
            alive.removeFirst()
        }

        // Time V0 and the survivors round robin (`NemotronTuneAdoption.order`): one chain of every timing link per
        // sample, each link depending on the one before it, built before the clock starts; median ms per launch.
        let candidates: [Config?] = [twin?.config] + alive.map { Optional($0.config) }
        let chain = steps.count
        var medians: [Double] = []
        var verdict: NemotronTuneAdoption.Verdict? = nil
        var reps = 0
        if !alive.isEmpty, elapsed() >= budgetSeconds {
            dropped.append("timing skipped (load budget)")
            alive = []
        }
        if !alive.isEmpty {
            var samples = Array(repeating: [Double](), count: candidates.count)
            do {
                try withError {
                    // One candidate's chain (its last link's value), built lazily.
                    func build(_ i: Int) -> MLXArray? {
                        var previous: MLXArray? = nil
                        for step in steps {
                            guard let out = step(candidates[i], previous) else { return nil }
                            previous = out
                        }
                        return previous
                    }
                    // A round: every candidate's chain built first, then evaluated back to back in `order`.
                    func round(_ order: [Int]) -> [Double?] {
                        let chains = order.map { build($0) }
                        return chains.map { last -> Double? in
                            guard let last else { return nil }
                            let start = DispatchTime.now().uptimeNanoseconds
                            eval(last)
                            Stream.gpu.synchronize()
                            return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6 / Double(chain)
                        }
                    }
                    var wall = 0.0
                    for _ in 0 ..< 2 {
                        let start = DispatchTime.now().uptimeNanoseconds
                        _ = round(Array(candidates.indices))
                        wall = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
                    }
                    reps = NemotronTuneAdoption.cycles(target: timedMilliseconds, wall: wall, cap: 31)
                    for rep in 0 ..< reps {
                        let order = NemotronTuneAdoption.order(candidates.count, cycle: rep)
                        for (i, ms) in zip(order, round(order)) {
                            if let ms { samples[i].append(ms) }
                        }
                    }
                }
            } catch {
                dropped.append("timing error")
                alive = []
            }
            if !alive.isEmpty, samples.allSatisfy({ $0.count == reps }) {
                verdict = NemotronTuneAdoption.decide(samples)
                medians = verdict?.medians ?? []
            } else {
                if !alive.isEmpty { dropped.append("timing incomplete") }
                alive = []
            }
        }

        var picked: (variant: Variant, config: Config)? = nil
        var why = "kept"
        if !medians.isEmpty, !alive.isEmpty {
            if let f = forced[family] {
                if f >= 1, f <= all.count, let i = alive.firstIndex(where: { $0.variant == all[f - 1] }) {
                    picked = alive[i]
                    why = "forced"
                } else {
                    why = f == 0 ? "forced V0" : "forced \(f) not qualified, kept"
                }
            } else if let best = verdict?.adopted {
                picked = alive[best - 1]
                why = "adopted"
            }
        }
        if let picked { adopted[family] = (picked.config, inc.u2) }

        func ms(_ i: Int) -> String {
            guard i < medians.count else { return "n/a" }
            return String(format: "%.4f", medians[i])
        }
        let pickedIndex = picked.flatMap { p in alive.firstIndex(where: { $0.variant == p.variant }).map { $0 + 1 } }
        let times = alive.indices.map { "\(alive[$0].variant.name) \(ms($0 + 1)) \(verdict?.won($0 + 1) ?? "n/a")" }
        let incName = "r\(inc.rps)n\(inc.nsg)l\(inc.load)u\(inc.u2 ? 1 : 0)\(inc.hf.map { "hf\($0)" } ?? "")"
            + (inc.regs ? "rg" : "") + (inc.e32 ? "e32" : "")
        let rule = verdict?.rule({ alive[$0 - 1].variant.name }) ?? "no candidate timed"
        return "\(why) \(picked?.variant.name ?? "V0") \(ms(pickedIndex ?? 0)) ms vs V0 (\(incName)) \(ms(0)) ms per launch"
            + " (\(rule); median of \(reps), \(chain) chained launches/sample; \(alive.count)/\(all.count)"
            + " bitwise over"
            + " \(checks.count) cases on \(layers) tensors" + (twin != nil ? ", V0 twin bitwise" : "")
            + (times.isEmpty ? "" : "; [" + times.joined(separator: ", ") + "]")
            + (dropped.isEmpty ? "" : "; dropped: " + dropped.prefix(8).joined(separator: ", ")) + ")"
    }
}

// MARK: - Load-time geometry tune of the routed FC2 zero-skip slot kernels (NEMOTRON_FC2_TUNE, NEMOTRON_DFC2_TUNE)
//
// The verify's routed FC2 (`NemotronZSkip.products` at R = 3: 18 slots x 21 tiles of 128 outputs, 23 launches a
// round) and the draft tail's (R = 1: 6 slots, the `dep` twin) run the fast tiled pair-dictionary slot kernel with the
// vector store and unmasked loads that are on, as one 512-thread threadgroup (16 SIMD groups) per (tile, slot) with
// the slot on grid y. That geometry was chosen on M4-class hosts. `NemotronFC2Tune` rebuilds the same kernel text
// (the selection `products` makes) under other geometries. In every variant, output n of a tile is formed as before:
// lane partial l (0 ... 31) runs its chunks l, l + 32, ... in increasing order with the same statements (fast paths,
// general path, decode, products, affine close) into the same accumulators, partial l of output n waits in row l of
// the tile's `part`, and the closing simd_sum reads partial l in lane l; the rounding and the store are unchanged.
// A variant moves only
// - SGS: the SIMD groups per threadgroup (8 or 32; the template argument): SIMD group s runs lane partials s,
//   s + SGS, ... and the closing sums of outputs s * 128 / SGS ...;
// - slot axis: the slot on grid x and the tile on grid y (the slots of one tile are dispatched together);
// - tiles: one threadgroup stages its slot's row and chunk masks once, then runs 3 or 7 consecutive tiles in turn,
//   each with the incumbent's per-tile text and a barrier after it (the tiles reuse `part`);
// - paired walk (with SGS 8): a SIMD group walks its four lane partials as two pairs in one chunk loop; where both
//   partials' chunks hold one non-zero value, the incumbent's one-value statements of the two run interleaved line by
//   line (each partial's own copy of the statements and accumulators), else each partial's chunk runs the incumbent's
//   chunk statements in turn (`pairedSource`).
// Each variant kernel carries the dep twins' unread `dep` input; the verify launch binds its own rows there.
// `tune` (load time, after the copies and the bank-binding views, before any verify graph is traced) checks every
// variant bit for bit against the launch production makes today, over every MoE layer's real dictionary copy, on
// seeded ReLU2-like rows (8.75% with -0, an all-zero, a one-hot and a dense-tail slot, an infinity, a NaN and a tiny
// value; 45%; full) with experts shared between rows and slot scores with 0, -0, NaN and both infinities. It drops any
// mismatch or Metal error, times the incumbent and the survivors round robin (`NemotronTuneAdoption.order`, median)
// as one chain per sample in which each launch binds the previous one's output as `dep` (so the launches run one
// after another, as in the round), and adopts the fastest only when `NemotronTuneAdoption.decide` admits it (at
// least 3% faster by median and faster in at least 75% of the cycles). `tuneDraft` does the same for the
// draft head's copy at R = 1 (rows at 45% with the special values, 8.75%, 20%, 70% and full; timed at 45%, the
// draft tail's density). Plain BF16 copies keep the incumbent.
// NEMOTRON_FC2_TUNE=0 / NEMOTRON_DFC2_TUNE=0 keep the incumbents; NEMOTRON_FC2_FORCE=<n> / NEMOTRON_DFC2_FORCE=<n>
// adopt variant n (1-based, `variants`) when it qualified, 0 keeps the incumbent.
enum NemotronFC2Tune {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_FC2_TUNE"] != "0"
    static let forced = Int(ProcessInfo.processInfo.environment["NEMOTRON_FC2_FORCE"] ?? "")
    static let draftEnabled = ProcessInfo.processInfo.environment["NEMOTRON_DFC2_TUNE"] != "0"
    static let draftForced = Int(ProcessInfo.processInfo.environment["NEMOTRON_DFC2_FORCE"] ?? "")
    /// Wall time of one tune's timed rounds (graph building included), sized from the last warm-up round.
    static let timedMilliseconds = 150.0

    struct Variant: Sendable, Equatable {
        let name: String
        /// SIMD groups per threadgroup.
        let sgs: Int
        /// Slot on grid x, tile on grid y.
        let slotX: Bool
        /// Consecutive tiles per threadgroup.
        let tiles: Int
        /// Lane partials walked in pairs (`pairedSource`).
        var paired = false
    }

    /// The incumbent is s16 (16 SIMD groups, slot on y, one tile per threadgroup).
    static let variants: [Variant] = [
        Variant(name: "s8", sgs: 8, slotX: false, tiles: 1),
        Variant(name: "s32", sgs: 32, slotX: false, tiles: 1),
        Variant(name: "x16", sgs: 16, slotX: true, tiles: 1),
        Variant(name: "t3", sgs: 16, slotX: false, tiles: 3),
        Variant(name: "t7", sgs: 16, slotX: false, tiles: 7),
        Variant(name: "s8p", sgs: 8, slotX: false, tiles: 1, paired: true),
    ]

    /// The adopted geometries (nil: the incumbent), set once at load before any verify or draft graph is traced.
    nonisolated(unsafe) private(set) static var choice: Variant? = nil
    nonisolated(unsafe) private(set) static var draftChoice: Variant? = nil

    /// The adopted geometry of a `products` launch: the verify's (R = 3, no dep) or the draft tail's (R = 1 with dep),
    /// on a dictionary copy only.
    static func adopted(rows: Int, dep: Bool, dict: Bool) -> Variant? {
        guard dict else { return nil }
        if rows == 3, !dep { return choice }
        if rows == 1, dep { return draftChoice }
        return nil
    }

    /// The incumbent's text: the selection `NemotronZSkip` makes for a dictionary copy (the fast tiled source
    /// through the dictionary, vector-store and unmasked-load transforms that are on); nil for any other form.
    static func incumbentSource() -> String? {
        guard NemotronZSkip.tiled, NemotronZSkip.fast,
              let base = nemotronZSkipDictSource(nemotronZSkipTiledFastSource) else { return nil }
        guard NemotronZSkip.vstore else { return base }
        let vs = nemotronZSkipVectorStoreSource(base)
        return (NemotronZSkip.unmasked ? vs.flatMap(nemotronZSkipUnmaskedLoadsSource) : nil) ?? vs
    }

    /// The variant's text from the incumbent's by counted replacements; nil when a statement moved.
    static func source(_ v: Variant) -> String? {
        guard var s = incumbentSource() else { return nil }
        let slotY = "  const int slot = int(threadgroup_position_in_grid.y);\n"
        let tileX = "int(threadgroup_position_in_grid.x)"
        guard s.components(separatedBy: slotY).count == 2, s.components(separatedBy: tileX).count == 3,
              !s.contains("return") else { return nil }
        if v.slotX {
            s = s.replacingOccurrences(of: tileX, with: "int(threadgroup_position_in_grid.y)")
                .replacingOccurrences(of: slotY, with: "  const int slot = int(threadgroup_position_in_grid.x);\n")
        }
        if v.tiles > 1 {
            let n0 = "  const int n0 = int(threadgroup_position_in_grid.x) * OT;\n"
            let tb = "  const ulong tb = (ulong)expert * (N / OT) + int(threadgroup_position_in_grid.x);\n"
            guard !v.slotX, s.components(separatedBy: n0).count == 2, s.components(separatedBy: tb).count == 2
            else { return nil }
            s = s.replacingOccurrences(of: n0, with: "")
                .replacingOccurrences(of: tb, with: "  for (int nmt = 0; nmt < \(v.tiles); nmt++) {\n"
                    + "  const int nmt_tile = int(threadgroup_position_in_grid.x) * \(v.tiles) + nmt;\n"
                    + "  const int n0 = nmt_tile * OT;\n"
                    + "  const ulong tb = (ulong)expert * (N / OT) + nmt_tile;\n")
                + "\n  threadgroup_barrier(mem_flags::mem_threadgroup);\n  }\n"
        }
        if v.paired {
            guard let paired = pairedSource(s) else { return nil }
            s = paired
        }
        return s
    }

    /// The paired walk: SIMD group s walks its lane partials as pairs (l_a, l_b) = (s + 2k SGS, s + (2k + 1) SGS) in
    /// one chunk loop over j, reading both chunk masks. When both chunks hold one non-zero value, the incumbent's
    /// one-value statements run for both partials, interleaved line by line, each partial's copy renamed with its
    /// suffix (its own value, code word, index word, table entries, products and accumulators): both code and index
    /// loads, then both table loads, then each partial's products. Otherwise partial a's chunk, then partial b's, runs
    /// the incumbent's chunk statements (every path; `continue` becomes `break` out of a one-pass loop), skipping a
    /// chunk without a non-zero value as before. Each partial adds its chunks in increasing j with the same statements
    /// into its own accumulators and stores them to its `part` row. Nil when the loop text moved.
    static func pairedSource(_ s: String) -> String? {
        let head = "  for (int ll = 0; ll < LPS; ll++) {\n    const int l = sg + SGS * ll;\n    thread U p[4];\n"
            + "    for (int o = 0; o < 4; o++) {\n      p[o] = 0;\n    }\n    const int count = l < tail ? 8 : 7;\n"
            + "    for (int j = 0; j < count; j++) {\n      const int c = j < 7 ? l + 32 * j : 224 + l;\n"
            + "      const uint m = cmask[c];\n      if (m == 0) {\n        continue;\n      }\n"
        let end = "    }\n    for (int o = 0; o < 4; o++) {\n      part[l * PS + 4 * lid + o] = p[o];\n    }\n  }\n"
        let oneHead = "      if (popcount(m) == 1) {\n"
        let loopHead = "        for (int o = 0; o < 4; o++) {\n"
        let oneEnd = "        }\n        continue;\n      }\n"
        guard s.components(separatedBy: head).count == 2, let h = s.range(of: head),
              let e = s.range(of: end, range: h.upperBound ..< s.endIndex) else { return nil }
        let body = String(s[h.upperBound ..< e.lowerBound])
        guard body.hasPrefix(oneHead), body.components(separatedBy: "continue;").count == 4, !body.contains("break"),
              let oe = body.range(of: oneEnd) else { return nil }
        let one = body[body.index(body.startIndex, offsetBy: oneHead.count) ..< oe.lowerBound]
            .components(separatedBy: loopHead)
        guard one.count == 2, one[0].hasSuffix("\n"), one[1].hasSuffix("\n") else { return nil }
        let decls = one[0].dropLast().components(separatedBy: "\n")
        let products = one[1].dropLast().components(separatedBy: "\n")
        // Every name the one-value statements declare or read per partial takes the partial's suffix.
        let names = ["j1", "xvj", "xtj", "sum1", "w", "gi", "ir", "e0", "e1", "e2", "e3", "sv", "bv", "code", "accum",
                     "m", "c", "p"]
        guard let rename = try? NSRegularExpression(pattern: "\\b(" + names.joined(separator: "|") + ")\\b"),
              let declared = try? NSRegularExpression(pattern: "^ *(?:const )?\\w+ (\\w+)(?:\\[4\\])? = ")
        else { return nil }
        for line in decls + products {
            let range = NSRange(line.startIndex..., in: line)
            if let match = declared.firstMatch(in: line, range: range), let r = Range(match.range(at: 1), in: line) {
                guard names.contains(String(line[r])) else { return nil }
            } else {
                guard line.contains("+=") else { return nil }
            }
        }
        func renamed(_ line: String, _ x: String) -> String {
            rename.stringByReplacingMatches(in: line, range: NSRange(line.startIndex..., in: line), withTemplate: "$1_" + x)
        }
        var pair = "      if (popcount(m_a) == 1 && popcount(m_b) == 1) {\n"
        for line in decls { pair += renamed(line, "a") + "\n" + renamed(line, "b") + "\n" }
        pair += loopHead
        for line in products { pair += renamed(line, "a") + "\n" + renamed(line, "b") + "\n" }
        pair += oneEnd
        var each = ""
        for x in ["a", "b"] {
            each += "      if (m_\(x) != 0) {\n        const uint m = m_\(x);\n        const int c = c_\(x);\n"
                + "        thread U* p = p_\(x);\n        do {\n"
                + body.replacingOccurrences(of: "continue;", with: "break;")
                + "        } while (false);\n      }\n"
        }
        let loop = "  static_assert(LPS % 2 == 0, \"paired lane partials\");\n"
            + "  for (int nmp = 0; nmp < LPS / 2; nmp++) {\n"
            + "    const int l_a = sg + SGS * (2 * nmp);\n    const int l_b = l_a + SGS;\n"
            + "    thread U p_a[4];\n    thread U p_b[4];\n"
            + "    for (int o = 0; o < 4; o++) {\n      p_a[o] = 0;\n      p_b[o] = 0;\n    }\n"
            + "    const int count_a = l_a < tail ? 8 : 7;\n    const int count_b = l_b < tail ? 8 : 7;\n"
            + "    for (int j = 0; j < count_a || j < count_b; j++) {\n"
            + "      const int c_a = j < 7 ? l_a + 32 * j : 224 + l_a;\n"
            + "      const int c_b = j < 7 ? l_b + 32 * j : 224 + l_b;\n"
            + "      const uint m_a = j < count_a ? uint(cmask[c_a]) : 0u;\n"
            + "      const uint m_b = j < count_b ? uint(cmask[c_b]) : 0u;\n"
            + pair + each
            + "    }\n    for (int o = 0; o < 4; o++) {\n"
            + "      part[l_a * PS + 4 * lid + o] = p_a[o];\n      part[l_b * PS + 4 * lid + o] = p_b[o];\n    }\n  }\n"
        return s.replacingCharacters(in: h.lowerBound ..< e.upperBound, with: loop)
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var kernels: [String: MLXFast.MLXFastKernel?] = [:]

    /// The variant's kernel (with the `dep` input): the production dep twin when only SGS moves, else its own text.
    static func kernel(_ v: Variant, _ copy: NemotronZSkip.Copy) -> MLXFast.MLXFastKernel? {
        guard v.slotX || v.tiles > 1 || v.paired else { return NemotronZSkip.depKernel(for: copy) }
        let key = (v.slotX ? "x" : v.paired ? "p" : "t\(v.tiles)")
            + (NemotronZSkip.vstore ? (NemotronZSkip.unmasked ? "_ul" : "_vs") : "")
        return lock.withLock { () -> MLXFast.MLXFastKernel? in
            if let made = kernels[key] { return made }
            let made = source(v).map {
                MLXFast.metalKernel(
                    name: "nemotron_fc2_tune_sbd_" + key,
                    inputNames: ["wk", "sk", "bk", "x", "inds", "scores", "dep"], outputNames: ["prod"],
                    source: $0, header: nemotronMoELeaderHeader + nemotronMoELeaderStore + nemotronZSkipHeader,
                    ensureRowContiguous: true)
            }
            kernels[key] = made
            return made
        }
    }

    /// One variant launch of the slot products of `x` ([slots, 1856]) on a dictionary copy: [slots * 2688] FP32,
    /// the incumbent's operands and template but SGS; nil when the kernel is missing.
    static func launch(_ v: Variant, _ x: MLXArray, _ inds: MLXArray, _ scores: MLXArray,
                       copy: NemotronZSkip.Copy, dep: MLXArray) -> MLXArray? {
        guard copy.dict, x.ndim == 2, let kernel = kernel(v, copy) else { return nil }
        let slots = x.dim(0), tiles = 2688 / 128, threads = 32 * v.sgs
        return kernel(
            [NemotronBankBinding.bind(copy.codes), NemotronBankBinding.bind(copy.scales),
             NemotronBankBinding.bind(copy.biases), x, inds, scores, dep],
            template: [("T", x.dtype), ("K", 1856), ("N", 2688), ("SGS", v.sgs)],
            grid: v.slotX ? (threads * slots, tiles, 1) : (threads * (tiles / v.tiles), slots, 1),
            threadGroup: (threads, 1, 1),
            outputShapes: [[slots * 2688]], outputDTypes: [.float32])[0]
    }

    /// Seeded ReLU2-like rows [slots, 1856] at `density`; `special`: -0 for some zeros, an all-zero slot (1), a
    /// one-hot slot (2), dense tail chunks in slot 0, an infinity, a NaN and a tiny value (the general path's
    /// sentinel) in slots 3, 4 and 5.
    private static func rows(_ slots: Int, density: Float, seed: UInt64, special: Bool) -> MLXArray {
        let k = 1856
        let r = MLXRandom.split(key: MLXRandom.key(seed), into: 3)
        let zero = MLXArray(Float(0))
        let h = abs(MLXRandom.normal([slots, k], key: r[0])) * 0.5 + 1e-3
        let u = MLXRandom.uniform(Float(0) ..< 1, [slots, k], key: r[1])
        let slot = MLXArray((0 ..< slots).map(Int32.init)).reshaped(slots, 1)
        let column = MLXArray((0 ..< k).map(Int32.init)).reshaped(1, k)
        var picked = which(u .< MLXArray(density), h, zero)
        if special {
            picked = which(slot .== MLXArray(Int32(1)), zero, picked)
            picked = which(slot .== MLXArray(Int32(2)),
                           which(column .== MLXArray(Int32(k - 56)), MLXArray(Float(1.5)), zero), picked)
            picked = which(logicalAnd(slot .== MLXArray(Int32(0)), column .>= MLXArray(Int32(k - 64))), h, picked)
        }
        let b = picked.asType(.bfloat16)
        let squared = (b * b).asType(.bfloat16)
        guard special else { return squared }
        let bf16 = { (v: Float) in MLXArray(v).asType(.bfloat16) }
        let negative = MLXRandom.uniform(Float(0) ..< 1, [slots, k], key: r[2]) .< MLXArray(Float(0.05))
        let signed = which(logicalAnd(squared .== bf16(0), negative), bf16(-0.0), squared)
        func at(_ s: Int, _ c: Int) -> MLXArray {
            logicalAnd(slot .== MLXArray(Int32(s % slots)), column .== MLXArray(Int32(c)))
        }
        return which(at(3, 77), bf16(.infinity), which(at(4, 901), bf16(.nan), which(at(5, 1300), bf16(1e-35), signed)))
    }

    /// Seeded slot scores; `special`: 0, -0, NaN, both infinities and a tie first.
    private static func scores(_ slots: Int, seed: UInt64, special: Bool) -> MLXArray {
        let s = MLXRandom.uniform(Float(0.01) ..< 0.5, [slots], key: MLXRandom.key(seed))
        guard special else { return s }
        let head: [Float] = [0, -0.0, .nan, .infinity, -.infinity, 0.25, 0.25]
        let fixed = MLXArray((0 ..< slots).map { $0 < head.count ? head[$0] : 0 })
        return which(MLXArray((0 ..< slots).map(Int32.init)) .< MLXArray(Int32(min(slots, head.count))), fixed, s)
    }

    /// Seeded route of slots / 6 rows of six distinct experts; each row after the first reuses `overlap` experts of
    /// the row before it.
    private static func route(_ seed: UInt64, slots: Int, overlap: Int) -> MLXArray {
        var state = seed
        func next() -> Int {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Int(state >> 33)
        }
        var rows: [[Int]] = []
        for r in 0 ..< slots / 6 {
            var row = r > 0 ? Array(rows[r - 1].prefix(overlap)) : []
            while row.count < 6 {
                let e = next() % 128
                if !row.contains(e) { row.append(e) }
            }
            rows.append(row)
        }
        return MLXArray(rows.flatMap { $0 }.map { UInt32($0) })
    }

    /// Load time, after `NemotronZSkip.prepare` and the bank-binding views, before any verify graph is traced.
    static func tune(_ model: Module) {
        choice = nil
        let t0 = DispatchTime.now().uptimeNanoseconds
        func elapsed() -> Double { Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9 }
        func report(_ message: String) {
            FileHandle.standardError.write(Data(
                "nemotron fc2 tune: \(message) (\(String(format: "%.2f", elapsed())) s)\n".utf8))
        }
        guard enabled else { return report("off (setting)") }
        guard incumbentSource() != nil else { return report("off (slot kernel is not the fast tiled form)") }
        let owners = NemotronZSkip.decodeCopyOwners(model)
        let all = NemotronZSkip.lock.withLock { owners.compactMap { NemotronZSkip.copies[$0] } }
        let copies = all.filter(\.dict)
        guard !copies.isEmpty else { return report("off (no dictionary copy)") }
        let (picked, message) = run(copies: copies, slots: 18, densities: [0.0875, 0.45, 1], timedDensity: 0.0875,
                                    chain: 2 * copies.count, forced: forced, budget: 1.0, elapsed: elapsed)
        choice = picked
        report(message + ", dictionary copies \(copies.count)/\(all.count))")
    }

    /// Load time, after `NemotronZSkip.prepareDraft` and before the draft colaunch check and any draft graph trace.
    static func tuneDraft(_ head: Module) {
        draftChoice = nil
        let t0 = DispatchTime.now().uptimeNanoseconds
        func elapsed() -> Double { Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9 }
        func report(_ message: String) {
            FileHandle.standardError.write(Data(
                "nemotron draft fc2 tune: \(message) (\(String(format: "%.2f", elapsed())) s)\n".utf8))
        }
        guard draftEnabled else { return report("off (setting)") }
        guard incumbentSource() != nil else { return report("off (slot kernel is not the fast tiled form)") }
        var copies: [NemotronZSkip.Copy] = []
        for module in head.modules() {
            guard let moe = module as? NemotronHMoE,
                  let copy = NemotronZSkip.lock.withLock({ NemotronZSkip.draftCopies[ObjectIdentifier(moe.switchMLP.fc2)] })
            else { continue }
            copies.append(copy)
        }
        let dicts = copies.filter(\.dict)
        guard !dicts.isEmpty, NemotronZSkip.depKernel(for: dicts[0]) != nil else {
            return report("off (no draft dictionary copy with a dep twin)")
        }
        let (picked, message) = run(copies: dicts, slots: 6, densities: [0.45, 0.0875, 1, 0.2, 0.7], timedDensity: 0.45,
                                    chain: 40, forced: draftForced, budget: 0.8, elapsed: elapsed)
        draftChoice = picked
        report(message + ")")
    }

    /// The check, timing and pick of one tune over `copies` at `slots` slots (18: the verify's launch without dep;
    /// 6: the draft tail's dep launch). Returns the pick and the report (without its closing parenthesis).
    private static func run(copies: [NemotronZSkip.Copy], slots: Int, densities: [Float], timedDensity: Float,
                            chain: Int, forced: Int?, budget: Double, elapsed: () -> Double)
        -> (Variant?, String) {
        let rows = slots / 6
        let draft = rows == 1
        // Check cases: every density on every copy, each copy with its own routes (experts shared between rows).
        typealias Case = (x: MLXArray, scores: MLXArray, inds: [MLXArray])
        var cases: [Case] = []
        var timed: [Case] = []
        do {
            try withError {
                for (i, density) in densities.enumerated() {
                    let x = Self.rows(slots, density: density, seed: UInt64(0xF2C0 + 16 * slots + i), special: i == 0)
                    let s = scores(slots, seed: UInt64(0xF2D0 + 16 * slots + i), special: i == 0)
                    let inds = copies.indices.map { route(UInt64(0xF2E0 + 64 * $0 + 8 * i + slots), slots: slots,
                                                          overlap: i % 3) }
                    eval([x, s] + inds)
                    cases.append((x, s, inds))
                }
                for i in 0 ..< 2 {
                    let x = Self.rows(slots, density: timedDensity, seed: UInt64(0xF2F0 + 16 * slots + i), special: false)
                    let s = scores(slots, seed: UInt64(0xF300 + 16 * slots + i), special: false)
                    let inds = copies.indices.map { route(UInt64(0xF310 + 64 * $0 + 8 * i + slots), slots: slots,
                                                          overlap: 2 - i) }
                    eval([x, s] + inds)
                    timed.append((x, s, inds))
                }
            }
        } catch {
            return (nil, "off (inputs: \(error))")
        }
        func incumbent(_ k: Case, _ l: Int, dep: MLXArray?) -> MLXArray {
            NemotronZSkip.products(k.x, k.inds[l], k.scores, copy: copies[l], rows: rows, dep: dep)
        }
        func variant(_ v: Variant, _ k: Case, _ l: Int, dep: MLXArray) -> MLXArray? {
            launch(v, k.x, k.inds[l], k.scores, copy: copies[l], dep: dep)
        }

        // Qualify: bit for bit against the production launch (the verify's has no dep, the draft's binds one). Every
        // `dep` binding is BF16 rows, as in production (MLX builds one pipeline per input dtype).
        var references: [[MLXArray]] = []
        do {
            references = try withError { () -> [[MLXArray]] in
                let outs = copies.indices.map { l in cases.map { incumbent($0, l, dep: draft ? $0.x : nil) } }
                eval(outs.flatMap { $0 })
                return outs
            }
        } catch {
            return (nil, "off (incumbent: \(error))")
        }
        var alive: [Variant] = []
        var dropped: [String] = []
        for v in variants {
            // A variant's first launch JITs its pipeline: none starts past the load budget.
            guard elapsed() < budget else {
                dropped.append("\(v.name) budget")
                continue
            }
            guard kernel(v, copies[0]) != nil else {
                dropped.append("\(v.name) n/a")
                continue
            }
            do {
                let same = try withError { () -> Bool in
                    var ok = MLXArray(true)
                    for l in copies.indices {
                        for (i, k) in cases.enumerated() {
                            guard let out = variant(v, k, l, dep: k.x) else { return false }
                            ok = ok .&& (out.view(dtype: .uint32) .== references[l][i].view(dtype: .uint32)).all()
                        }
                    }
                    return ok.item(Bool.self)
                }
                if same { alive.append(v) } else { dropped.append("\(v.name) differs") }
            } catch {
                dropped.append("\(v.name) error")
            }
        }
        let checks = copies.count * cases.count
        let checked = elapsed()

        // Time the incumbent and the survivors round robin (`NemotronTuneAdoption.order`): one chain of `chain`
        // launches per sample, each launch binding the previous output (viewed as BF16) as `dep`, the graph built
        // before the clock starts; median ms per launch.
        let candidates: [Variant?] = [nil] + alive.map { Optional($0) }
        var medians: [Double] = []
        var verdict: NemotronTuneAdoption.Verdict? = nil
        var reps = 0
        if !alive.isEmpty, elapsed() >= budget {
            dropped.append("timing skipped (load budget)")
            alive = []
        }
        if !alive.isEmpty {
            var samples = Array(repeating: [Double](), count: candidates.count)
            do {
                try withError {
                    // One candidate's chain (its last output), built lazily.
                    func build(_ i: Int) -> MLXArray? {
                        var prev = timed[0].x
                        for j in 0 ..< chain {
                            let l = j % copies.count, k = timed[(j / copies.count) % timed.count]
                            guard let out = candidates[i].map({ variant($0, k, l, dep: prev) })
                                ?? incumbent(k, l, dep: prev) else { return nil }
                            prev = out.view(dtype: .bfloat16)
                        }
                        return prev
                    }
                    // A round: every candidate's chain built first, then evaluated back to back in `order`.
                    func round(_ order: [Int]) -> [Double?] {
                        let chains = order.map { build($0) }
                        return chains.map { last -> Double? in
                            guard let last else { return nil }
                            let start = DispatchTime.now().uptimeNanoseconds
                            eval(last)
                            Stream.gpu.synchronize()
                            return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6 / Double(chain)
                        }
                    }
                    var wall = 0.0
                    for _ in 0 ..< 2 {
                        let start = DispatchTime.now().uptimeNanoseconds
                        _ = round(Array(candidates.indices))
                        wall = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
                    }
                    reps = NemotronTuneAdoption.cycles(target: timedMilliseconds, wall: wall, cap: 41)
                    for rep in 0 ..< reps {
                        let order = NemotronTuneAdoption.order(candidates.count, cycle: rep)
                        for (i, ms) in zip(order, round(order)) {
                            if let ms { samples[i].append(ms) }
                        }
                    }
                }
            } catch {
                dropped.append("timing error")
                alive = []
            }
            if !alive.isEmpty, samples.allSatisfy({ $0.count == reps }) {
                verdict = NemotronTuneAdoption.decide(samples)
                medians = verdict?.medians ?? []
            } else {
                if !alive.isEmpty { dropped.append("timing incomplete") }
                alive = []
            }
        }

        var picked: Variant? = nil
        var why = "kept"
        if !medians.isEmpty, !alive.isEmpty {
            if let f = forced {
                if f >= 1, f <= variants.count, let i = alive.firstIndex(of: variants[f - 1]) {
                    picked = alive[i]
                    why = "forced"
                } else {
                    why = f == 0 ? "forced V0" : "forced \(f) not qualified, kept"
                }
            } else if let best = verdict?.adopted {
                picked = candidates[best]
                why = "adopted"
            }
        }
        func ms(_ v: Variant?) -> String {
            guard let i = candidates.firstIndex(of: v), i < medians.count else { return "n/a" }
            return String(format: "%.4f", medians[i])
        }
        let times = alive.indices.map { "\(alive[$0].name) \(ms(alive[$0])) \(verdict?.won($0 + 1) ?? "n/a")" }
        let rule = verdict?.rule({ candidates[$0]?.name ?? "V0" }) ?? "no candidate timed"
        return (picked, "\(why) \(picked?.name ?? "V0") \(ms(picked)) ms vs V0 (s16) \(ms(nil)) ms per launch"
            + " (\(rule); median of \(reps), \(chain) chained launches/sample; \(alive.count)/\(variants.count)"
            + " bitwise over"
            + " \(checks) cases at R = \(rows) by \(String(format: "%.2f", checked)) s"
            + (times.isEmpty ? "" : "; [" + times.joined(separator: ", ") + "]")
            + (dropped.isEmpty ? "" : "; dropped: " + dropped.prefix(8).joined(separator: ", ")))
    }
}

extension NemotronMoELeader {
    /// `NemotronDraftTune`'s one-row routed FC1 launch of geometry `g` (fc1): the production pipeline of the draft
    /// leader (tail-8 when `tail8`, pair dictionary, `g.unroll` on both main K loops) with the SIMD groups per
    /// threadgroup from the template in `out_row` and, with `g.slotX`, the slot on grid x; unread `dep` input last.
    /// nil when it cannot run.
    static func draftTuneLaunch(_ g: NemotronDraftTune.Geometry, _ x: MLXArray, _ inds: MLXArray, bank: Bank,
                                dict: NemotronSBDict.Dict, relu2: Bool, tail8: Bool, dep: MLXArray) -> MLXArray? {
        let n = bank.n
        let key = "nemotron_draft_tune_fc1_\(tail8 ? "t8" : "tr")_u\(g.unroll)" + (g.slotX ? "_x" : "")
        guard inds.size == 6, x.size == bank.k, n % g.rps == 0, (n / g.rps) % g.nsg == 0, g.rps <= 32, g.nsg <= 32,
              let kernel = NemotronDraftTune.cached(key, {
                  var swaps = [("  const int out_row = tid.y * (2 * results_per_simdgroup) + simd_gid * results_per_simdgroup;\n",
                                "  const int out_row = tid.y * (NSG * results_per_simdgroup) + simd_gid * results_per_simdgroup;\n", 1)]
                  if g.slotX { swaps.append(("  const int slot = tid.z;\n", "  const int slot = tid.x;\n", 1)) }
                  var s = nemotronCountedSwaps(nemotronMoELeaderSource, swaps)
                  if tail8 { s = s.flatMap(nmlTail8Source) }
                  return s.flatMap(nemotronSBDictSource)
                      .flatMap { nemotronUnrollSource($0, loop: NemotronU2.leaderLoop, count: 2, unroll: g.unroll) }
                      .map {
                          MLXFast.metalKernel(
                              name: key, inputNames: ["w", "scales", "biases", "x", "inds", "dep"],
                              outputNames: ["y"], source: $0, header: nemotronMoELeaderHeader + nemotronMoELeaderStore)
                      }
              })
        else { return nil }
        return kernel(
            [NemotronBankBinding.bind(bank.weight), NemotronBankBinding.bind(dict.index), dict.table, x, inds, dep],
            template: [("T", x.dtype), ("R", 1), ("TOPK", 6), ("K", bank.k), ("N", n), ("XSLOT", false),
                       ("RELU2", relu2), ("RPS", g.rps), ("NSG", g.nsg)],
            grid: g.slotX ? (32 * 6, n / g.rps, 1) : (32, n / g.rps, 6), threadGroup: (32, g.nsg, 1),
            outputShapes: [[6, n]], outputDTypes: [x.dtype])[0]
    }
}

// MARK: - Shared adoption rule of the load-time tunes
//
// Every load-time tune (FC1, dictionary QMV, verify and draft FC2, SSM anchor, verify SDPA, the seed retiles and
// the seed SSM tune) times its incumbent (candidate 0) and the candidates that passed its bit check in cycles: each
// cycle takes one sample of every candidate, in `order`. `decide` picks the candidate with the lowest median and
// adopts it only when both hold:
// - its median is at least 3% below the incumbent's;
// - it was faster than the incumbent in at least 75% of the cycles (each cycle's two samples compared).
// Otherwise the incumbent stays. Every tune times at least `minCycles` cycles. A tune's FORCE switch is applied by
// the tune before the rule and bypasses it.
enum NemotronTuneAdoption {
    /// Fewest timed cycles of a tune.
    static let minCycles = 16
    /// An adopted candidate's median is at most this fraction of the incumbent's.
    static let medianBar = 0.97
    /// An adopted candidate beats the incumbent in at least this fraction of the cycles.
    static let winBar = 0.75

    /// The candidates' order in cycle `cycle`: rotated by cycle / 2, reversed on odd cycles. Over each pair of
    /// cycles every candidate takes mirrored positions and every two candidates run in both orders, so a drift
    /// within or across cycles weighs on the incumbent and each candidate alike.
    static func order(_ count: Int, cycle: Int) -> [Int] {
        let rotated = (0 ..< count).map { ($0 + cycle / 2) % count }
        return cycle % 2 == 0 ? rotated : Array(rotated.reversed())
    }

    /// Timed cycles for rounds of `wall` ms: `target` ms worth, at least `minCycles`, at most `cap`, even.
    static func cycles(target: Double, wall: Double, cap: Int) -> Int {
        let n = max(minCycles, min(cap, Int(target / max(wall, 0.01))))
        return n - n % 2
    }

    struct Verdict {
        /// Every candidate's median (0: the incumbent).
        let medians: [Double]
        /// Cycles in which each candidate was faster than the incumbent (0 for the incumbent).
        let wins: [Int]
        let cycles: Int
        /// The candidate with the lowest median besides the incumbent.
        let best: Int?
        /// `best` when it passes both bars.
        let adopted: Int?

        /// "w/n": candidate i's wins over the cycles.
        func won(_ i: Int) -> String { i < wins.count ? "\(wins[i])/\(cycles)" : "n/a" }

        /// The rule's inputs: "best <name> <change of the median vs the incumbent's> wins <w>/<n>".
        func rule(_ name: (Int) -> String) -> String {
            guard let b = best, medians[0] > 0 else { return "no candidate timed" }
            return "best \(name(b)) " + String(format: "%+.1f%%", 100 * (medians[b] / medians[0] - 1))
                + " wins \(won(b))"
        }
    }

    /// `samples[i][c]`: candidate i's time in cycle c (i = 0: the incumbent); every candidate has the same cycles.
    /// `eligible(i)`: whether candidate i may be judged (a tune that drops a candidate after its timing leaves it
    /// out; its median and wins are still reported).
    static func decide(_ samples: [[Double]], eligible: (Int) -> Bool = { _ in true },
                       medianBar bar: Double = medianBar) -> Verdict {
        let cycles = samples.first?.count ?? 0
        let medians = samples.map { $0.isEmpty ? Double.nan : $0.sorted()[$0.count / 2] }
        let wins = samples.map { s in zip(s, samples[0]).filter { $0.0 < $0.1 }.count }
        let best = medians.indices.dropFirst().filter(eligible).min(by: { medians[$0] < medians[$1] })
        var adopted: Int? = nil
        if let b = best, cycles > 0, samples.allSatisfy({ $0.count == cycles }), medians[b] <= bar * medians[0],
           Double(wins[b]) >= winBar * Double(cycles) {
            adopted = b
        }
        return Verdict(medians: medians, wins: wins, cycles: cycles, best: best, adopted: adopted)
    }
}
