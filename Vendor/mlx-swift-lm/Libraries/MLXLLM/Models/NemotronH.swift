//
//  NemotronH.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/nemotron_h.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Block Type

private enum NemotronHBlockType {
    case mamba  // "M"
    case attention  // "*"
    case mlp  // "-"
    case moe  // "E"

    init(from char: Character) {
        switch char {
        case "M": self = .mamba
        case "*": self = .attention
        case "-": self = .mlp
        case "E": self = .moe
        default: fatalError("Unknown NemotronH block type: \(char)")
        }
    }
}

// MARK: - Mixer Protocol

/// Protocol for all mixer types in NemotronH blocks
private protocol NemotronHMixer: Module {
    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray
}

// MARK: - Activations

/// Layers between early asyncEval calls in the CBv2 forward (0 disables).
let nemotronEarlyEvalInterval = 4

/// Captured verify only: layers between its submissions after the first one
/// (evaluation order only; the fused verify graph does not depend on it).
/// NEMOTRON_MTP_VERIFY_SUBMIT_INTERVAL=4 restores the shared spacing.
let nemotronVerifySubmitInterval =
    Int(ProcessInfo.processInfo.environment["NEMOTRON_MTP_VERIFY_SUBMIT_INTERVAL"] ?? "") ?? 13

/// Captured verify only: also submit after the first layer.
/// `NEMOTRON_MTP_VERIFY_FIRST_SUBMIT=0` keeps the plain interval.
let nemotronVerifyFirstSubmit =
    ProcessInfo.processInfo.environment["NEMOTRON_MTP_VERIFY_FIRST_SUBMIT"] != "0"

/// Captured verify only: the layer counts after which it submits (evaluation
/// order only). Short chunks first, so the GPU has the verify's first layers
/// while the host builds the next chunk, then one submit every seven layers to
/// the end, so the GPU never waits on the host for a long tail chunk.
/// `NEMOTRON_MTP_VERIFY_SUBMITS` takes a comma list ("1,4,13,26" is the
/// previous schedule); setting `NEMOTRON_MTP_VERIFY_SUBMIT_INTERVAL` uses the
/// interval instead.
let nemotronVerifySubmitLayers: Set<Int>? = {
    let env = ProcessInfo.processInfo.environment
    guard env["NEMOTRON_MTP_VERIFY_SUBMIT_INTERVAL"] == nil else { return nil }
    return Set((env["NEMOTRON_MTP_VERIFY_SUBMITS"] ?? "1,4,8,13,20,27,34,41,48")
        .split(separator: ",").compactMap { Int($0) })
}()

/// Layers per submitted chunk in a multi-token (prompt prefill) CBv2 forward,
/// where each chunk also waits for the previous one so its temporaries return
/// to MLX's buffer cache before the next chunk allocates. Every prompt prefill
/// (engine request prefill and teacher-forced stepper alike) runs this one
/// path. `NEMOTRON_PREFILL_REEVAL=0` restores the default spacing.
let nemotronPrefillEvalInterval =
    ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_REEVAL"] == "0"
    ? nemotronEarlyEvalInterval : 2

/// Multi-token (prompt prefill) CBv2 forward: also submit after the first
/// layer, as `nemotronVerifyFirstSubmit` does for the verify, so the GPU
/// starts the first chunk while the host builds the next layer. Outside the
/// recycling chain (the next chunk waits on nothing new). Evaluation order
/// only. `NEMOTRON_PREFILL_FIRST_SUBMIT=0` keeps the plain interval.
let nemotronPrefillFirstSubmit =
    ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_FIRST_SUBMIT"] != "0"

/// Squared ReLU activation: relu(x)^2
private func relu2(_ x: MLXArray) -> MLXArray {
    let y = MLX.maximum(x, MLXArray(0))
    return y * y
}

// MARK: - Target residual and next block normalization

enum NemotronHResidualRMSNorm {
    private static let source = """
            constexpr uint n_reads = 4;
            const uint gid = threadgroup_position_in_grid.x;
            const uint lid = thread_position_in_threadgroup.x;
            const uint simd_lane_id = thread_index_in_simdgroup;
            const uint simd_group_id = simdgroup_index_in_threadgroup;
            threadgroup float local_inv_mean[1];
            threadgroup float local_sums[32];
            float acc = 0;
            float thread_x[n_reads];
            const size_t offset = size_t(gid) * axis_size + lid * n_reads;
            for (uint i = 0; i < n_reads; i++) {
                T sum = lhs[offset + i] + rhs[offset + i];
                residual[offset + i] = sum;
                thread_x[i] = sum;
                acc += thread_x[i] * thread_x[i];
            }
            acc = simd_sum(acc);
            if (simd_group_id == 0) {
                local_sums[simd_lane_id] = 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_lane_id == 0) {
                local_sums[simd_group_id] = acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group_id == 0) {
                acc = simd_sum(local_sums[simd_lane_id]);
                if (simd_lane_id == 0) {
                    local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = 0; i < n_reads; i++) {
                normalized[offset + i] = weight[lid * n_reads + i]
                    * static_cast<T>(thread_x[i] * local_inv_mean[0]);
            }
            """

    /// NEMOTRON_MTP_JOIN_ONE_BARRIER=0 keeps the record's three-barrier RMS reduction in these joins.
    /// Otherwise every SIMD group stores its sum of squares, and after one barrier every SIMD group forms the
    /// record's 32-lane sum itself (the threadgroup's SIMD-group sums in their lanes, zeros elsewhere: the
    /// vector SIMD group 0 summed after zeroing the slots) and the same `precise::rsqrt`, so each thread holds
    /// the record's inverse mean without the zeroing and broadcast barriers.
    static let oneBarrier = ProcessInfo.processInfo.environment["NEMOTRON_MTP_JOIN_ONE_BARRIER"] != "0"
    static func oneBarrierSource(_ s: String) -> String {
        guard oneBarrier else { return s }
        let block = """
            acc = simd_sum(acc);
            if (simd_group_id == 0) {
                local_sums[simd_lane_id] = 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_lane_id == 0) {
                local_sums[simd_group_id] = acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group_id == 0) {
                acc = simd_sum(local_sums[simd_lane_id]);
                if (simd_lane_id == 0) {
                    local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            """
        precondition(s.components(separatedBy: block).count == 2, "join rms block")
        return s.replacingOccurrences(of: block, with: """
            acc = simd_sum(acc);
            if (simd_lane_id == 0) {
                local_sums[simd_group_id] = acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint nrm_groups = (threads_per_threadgroup.x + 31) / 32;
            acc = simd_sum(simd_lane_id < nrm_groups ? local_sums[simd_lane_id] : 0.0f);
            const float nrm_inv_mean = metal::precise::rsqrt(acc / axis_size + eps);
            """).replacingOccurrences(of: "local_inv_mean[0]", with: "nrm_inv_mean")
    }
    private static func name(_ base: String) -> String { oneBarrier ? base + "_1b" : base }

    private static let kernel = MLXFast.metalKernel(
        name: name("nemotron_h_residual_rms_bf16"),
        inputNames: ["lhs", "rhs", "weight", "axis_size", "eps"],
        outputNames: ["residual", "normalized"],
        source: oneBarrierSource(source), ensureRowContiguous: true)

    static let preparedMoEEnabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_RMS_PREPARED_MOE"] != "0"

    private static func preparedSource() -> String {
        let scratch = "threadgroup float local_inv_mean[1];"
        let store = "normalized[offset + i] = weight[lid * n_reads + i]"
        let product = "* static_cast<T>(thread_x[i] * local_inv_mean[0]);"
        precondition(source.components(separatedBy: scratch).count == 2)
        precondition(source.components(separatedBy: store).count == 2)
        precondition(source.components(separatedBy: product).count == 2)
        guard joinShuffle else {
            let s = source.replacingOccurrences(of: scratch,
                with: scratch + "\nthreadgroup T rounded[2688];")
                .replacingOccurrences(of: store, with: "T value = weight[lid * n_reads + i]")
                .replacingOccurrences(of: product, with: product
                    + "\nnormalized[offset + i] = value;\nrounded[lid * n_reads + i] = value;")
            return s + """

                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (lid < 336) {
                    const int chunk = int(lid);
                    float values[8];
                    float total = 0;
                    for (int i = 0; i < 8; i += 4) {
                        total += float(rounded[chunk * 8 + i]) + float(rounded[chunk * 8 + i + 1])
                            + float(rounded[chunk * 8 + i + 2]) + float(rounded[chunk * 8 + i + 3]);
                        values[i] = rounded[chunk * 8 + i];
                        values[i + 1] = rounded[chunk * 8 + i + 1] / 16.0f;
                        values[i + 2] = rounded[chunk * 8 + i + 2] / 256.0f;
                        values[i + 3] = rounded[chunk * 8 + i + 3] / 4096.0f;
                    }
                    const int base = (int(gid) * 11 + chunk / 32) * 9 * 32 + chunk % 32;
                    for (int i = 0; i < 8; i++) prepared[base + i * 32] = values[i];
                    prepared[base + 8 * 32] = total;
                }
                """
        }
        // Chunk c (values 8c ..< 8c + 8) is lanes 2c and 2c + 1 of one SIMD group: each lane keeps its four
        // rounded values, reads its partner's through simd_shuffle_xor, and the pair writes the chunk's nine
        // words (the even lane its four, the total; the odd lane its four) with the same expressions and order.
        let s = source.replacingOccurrences(of: store, with: "T value = weight[lid * n_reads + i]")
            .replacingOccurrences(of: product, with: product
                + "\nnormalized[offset + i] = value;\nnmj_r[i] = value;")
            .replacingOccurrences(of: "float thread_x[n_reads];", with: "float thread_x[n_reads];\nT nmj_r[n_reads];")
        precondition(s.contains("T nmj_r[n_reads];"))
        return s + """

            {
                const bool odd = (lid & 1u) != 0;
                T other[4];
                for (int i = 0; i < 4; i++) other[i] = as_type<T>(simd_shuffle_xor(as_type<ushort>(nmj_r[i]), ushort(1)));
                const int chunk = int(lid >> 1);
                const int base = (int(gid) * 11 + chunk / 32) * 9 * 32 + chunk % 32;
                if (!odd) {
                    float total = 0;
                    total += float(nmj_r[0]) + float(nmj_r[1]) + float(nmj_r[2]) + float(nmj_r[3]);
                    total += float(other[0]) + float(other[1]) + float(other[2]) + float(other[3]);
                    prepared[base + 0 * 32] = float(nmj_r[0]);
                    prepared[base + 1 * 32] = float(nmj_r[1]) / 16.0f;
                    prepared[base + 2 * 32] = float(nmj_r[2]) / 256.0f;
                    prepared[base + 3 * 32] = float(nmj_r[3]) / 4096.0f;
                    prepared[base + 8 * 32] = total;
                } else {
                    prepared[base + 4 * 32] = float(nmj_r[0]);
                    prepared[base + 5 * 32] = float(nmj_r[1]) / 16.0f;
                    prepared[base + 6 * 32] = float(nmj_r[2]) / 256.0f;
                    prepared[base + 7 * 32] = float(nmj_r[3]) / 4096.0f;
                }
            }
            """
    }
    /// NEMOTRON_MTP_JOIN_SHUFFLE=0 keeps the threadgroup copy and barrier ahead of the prepared epilogue.
    static let joinShuffle = ProcessInfo.processInfo.environment["NEMOTRON_MTP_JOIN_SHUFFLE"] != "0"
    private static let preparedKernel = MLXFast.metalKernel(
        name: name(joinShuffle ? "nemotron_h_residual_rms_prepared_moe_bf16_sh" : "nemotron_h_residual_rms_prepared_moe_bf16"),
        inputNames: ["lhs", "rhs", "weight", "axis_size", "eps"],
        outputNames: ["residual", "normalized", "prepared"],
        source: oneBarrierSource(preparedSource()), ensureRowContiguous: true)

    static func applyPrepared(_ lhs: MLXArray, _ rhs: MLXArray, norm: RMSNorm)
        -> (MLXArray, MLXArray, MLXArray)?
    {
        guard preparedMoEEnabled, NemotronMTPExecution.batchedM1,
            NemotronMTPExecution.batchedNorm, NemotronMoELeader.enabled,
            NemotronMoELeader.preparedGateUp,
            Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
            lhs.shape == [1, 3, 2688], rhs.shape == lhs.shape,
            lhs.dtype == .bfloat16, rhs.dtype == .bfloat16,
            type(of: norm) == RMSNorm.self,
            norm.weight.shape == [2688], norm.weight.dtype == .bfloat16
        else { return nil }
        let outputs = preparedKernel([lhs, rhs, norm.weight, UInt32(2688), norm.eps],
            template: [("T", DType.bfloat16)],
            grid: (672 * 3, 1, 1), threadGroup: (672, 1, 1),
            outputShapes: [lhs.shape, lhs.shape, [3, 11, 9 * 32]],
            outputDTypes: [.bfloat16, .bfloat16, .float32])
        if NemotronMoELeader.selfTest, let expected = apply(lhs, rhs, norm: norm) {
            return (nemotronBitwiseOrNaN(outputs[0], expected.0),
                    nemotronBitwiseOrNaN(outputs[1], expected.1), outputs[2])
        }
        return (outputs[0], outputs[1], outputs[2])
    }

    // MARK: Captured-verify plumbing (one launch per layer join)

    /// NEMOTRON_MTP_FUSED_PLUMBING=0 restores the original captured verify
    /// joins (stock add / stock rms at early-eval boundaries, separate
    /// activation prepares, MoE shared add). Captured verify only.
    static let fusedPlumbing =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_FUSED_PLUMBING"] != "0"

    /// `x + (y + s)`: the MoE's stock `y + shared` Add moves in front of the
    /// residual Add; both stay separate BF16 roundings, in the same order.
    private static func sumPreparedSource() -> String {
        let add = "T sum = lhs[offset + i] + rhs[offset + i];"
        let s = preparedSource()
        precondition(s.components(separatedBy: add).count == 2)
        return s.replacingOccurrences(of: add, with: "T part = rhs[offset + i] + shared[offset + i];\n"
            + "                T sum = lhs[offset + i] + part;")
    }

    /// Stock rms_single_row input (thread_x = x) with the prepared epilogue.
    private static func normPreparedSource() -> String {
        let add = "T sum = lhs[offset + i] + rhs[offset + i];"
        let store = "residual[offset + i] = sum;"
        let s = preparedSource()
        precondition(s.components(separatedBy: add).count == 2)
        precondition(s.components(separatedBy: store).count == 2)
        return s.replacingOccurrences(of: add, with: "T sum = lhs[offset + i];")
            .replacingOccurrences(of: store, with: "")
    }
    private static let sumPreparedKernel = MLXFast.metalKernel(
        name: name(joinShuffle ? "nemotron_h_residual_sum_rms_prepared_bf16_sh" : "nemotron_h_residual_sum_rms_prepared_bf16"),
        inputNames: ["lhs", "rhs", "shared", "weight", "axis_size", "eps"],
        outputNames: ["residual", "normalized", "prepared"],
        source: oneBarrierSource(sumPreparedSource()), ensureRowContiguous: true)

    /// The MoE join reading the routed half as the six weighted slot products
    /// of its row ([18 * 2688] FP32, slot-major): the zero-skip top-six reduce
    /// (first slot, then `slot + total` in slot order, BF16 store) runs here
    /// instead of in its own launch. Every later expression is unchanged.
    private static func sumProductsPreparedSource() -> String {
        withProductsReduce(sumPreparedSource())
    }

    private static func withProductsReduce(_ s: String) -> String {
        let add = "T part = rhs[offset + i] + shared[offset + i];"
        precondition(s.components(separatedBy: add).count == 2)
        return s.replacingOccurrences(of: add, with: """
            const size_t column = lid * n_reads + i;
                            float total = rhs[size_t(gid * 6) * axis_size + column];
                            for (int j = 1; j < 6; j++) {
                                total = rhs[size_t(gid * 6 + j) * axis_size + column] + total;
                            }
                            T part = static_cast<T>(total) + shared[offset + i];
            """)
    }
    private static let sumProductsPreparedKernel = MLXFast.metalKernel(
        name: name(joinShuffle ? "nemotron_h_residual_products_rms_prepared_bf16_sh" : "nemotron_h_residual_products_rms_prepared_bf16"),
        inputNames: ["lhs", "rhs", "shared", "weight", "axis_size", "eps"],
        outputNames: ["residual", "normalized", "prepared"],
        source: oneBarrierSource(sumProductsPreparedSource()), ensureRowContiguous: true)
    private static let normPreparedKernel = MLXFast.metalKernel(
        name: name(joinShuffle ? "nemotron_h_rms_prepared_bf16_sh" : "nemotron_h_rms_prepared_bf16"),
        inputNames: ["lhs", "weight", "axis_size", "eps"],
        outputNames: ["normalized", "prepared"],
        source: oneBarrierSource(normPreparedSource()), ensureRowContiguous: true)

    /// Input-independent scalar kernel arguments, built once instead of per
    /// call (same values and the same scalar kernel instantiation).
    private static let scalarLock = NSLock()
    nonisolated(unsafe) private static var axisScalar: MLXArray?
    nonisolated(unsafe) private static var epsScalars: [UInt32: MLXArray] = [:]
    private static func scalars(_ eps: Float) -> (MLXArray, MLXArray) {
        scalarLock.lock(); defer { scalarLock.unlock() }
        let axis = axisScalar ?? MLXArray(UInt32(2688))
        axisScalar = axis
        if let value = epsScalars[eps.bitPattern] { return (axis, value) }
        let value = MLXArray(eps)
        epsScalars[eps.bitPattern] = value
        return (axis, value)
    }

    /// Configuration gate of the fused captured-verify joins (the consumers
    /// read a prepared activation only on the narrow multi-row route).
    static var plumbingAvailable: Bool {
        fusedPlumbing && NemotronMTPExecution.batchedM1 && NemotronMTPExecution.batchedNorm
            && NemotronMultiRowQMV.enabled && NemotronMultiRowQMV.narrowEnabled
            && Device.defaultDevice().deviceType == .gpu && StreamOrDevice.default == .gpu
    }

    private static func plumbingRows(_ x: MLXArray, norm: RMSNorm) -> Bool {
        plumbingAvailable
            && x.shape == [1, 3, 2688] && x.dtype == .bfloat16
            && type(of: norm) == RMSNorm.self
            && norm.weight.shape == [2688] && norm.weight.dtype == .bfloat16
    }

    /// Captured verify join: (residual, normalized, prepared K=2688 rows).
    /// `shared` non-nil computes residual = lhs + (rhs + shared).
    static func applyPreparedRows(
        _ lhs: MLXArray, _ rhs: MLXArray, shared: MLXArray? = nil, norm: RMSNorm
    ) -> (MLXArray, MLXArray, MLXArray)? {
        if rhs.shape == [18 * 2688] {
            return applyProductsRows(lhs, rhs, shared: shared, norm: norm)
        }
        guard plumbingRows(lhs, norm: norm),
              rhs.shape == lhs.shape, rhs.dtype == .bfloat16,
              shared == nil || (shared!.shape == lhs.shape && shared!.dtype == .bfloat16)
        else { return nil }
        let (axis, eps) = scalars(norm.eps)
        let shapes = [lhs.shape, lhs.shape, [3, 11, 9 * 32]]
        let outputs: [MLXArray]
        if let shared {
            outputs = sumPreparedKernel([lhs, rhs, shared, norm.weight, axis, eps],
                template: [("T", DType.bfloat16)],
                grid: (672 * 3, 1, 1), threadGroup: (672, 1, 1),
                outputShapes: shapes, outputDTypes: [.bfloat16, .bfloat16, .float32])
        } else {
            outputs = preparedKernel([lhs, rhs, norm.weight, axis, eps],
                template: [("T", DType.bfloat16)],
                grid: (672 * 3, 1, 1), threadGroup: (672, 1, 1),
                outputShapes: shapes, outputDTypes: [.bfloat16, .bfloat16, .float32])
        }
        guard NemotronMoELeader.selfTest else { return (outputs[0], outputs[1], outputs[2]) }
        // Stock reference: MLX Add(s) then stock RMSNorm. Any bit difference
        // poisons every output, the prepared activation included (consumers
        // also re-check prepared against a self-prepared reference).
        let summed = lhs + (shared.map { rhs + $0 } ?? rhs)
        let ok = MLX.logicalAnd(
            (outputs[0].view(dtype: .uint16) .== summed.view(dtype: .uint16)).all(),
            (outputs[1].view(dtype: .uint16) .== norm(summed).view(dtype: .uint16)).all())
        return (which(ok, outputs[0], MLXArray(Float.nan).asType(.bfloat16)),
                which(ok, outputs[1], MLXArray(Float.nan).asType(.bfloat16)),
                which(ok, outputs[2], MLXArray(Float.nan)))
    }

    /// Set at load by `checkProductsJoin`; the verify asks the MoE for its
    /// slot products only after the products join matched there.
    nonisolated(unsafe) static var productsJoinVerified = false

    /// Load time: the products join against the separate top-six reduce plus
    /// the two-operand join, bit for bit, on seeded rows and a real norm.
    static func checkProductsJoin(_ model: Module) {
        productsJoinVerified = false
        guard plumbingAvailable, let norm = model.modules().compactMap({ $0 as? RMSNorm }).first(where: {
            type(of: $0) == RMSNorm.self && $0.weight.shape == [2688] && $0.weight.dtype == .bfloat16
        }) else { return }
        var ok = MLXArray(true)
        for seed in 0 ..< 3 {
            let keys = MLXRandom.split(key: MLXRandom.key(UInt64(0x6a01 + seed)), into: 3)
            let lhs = MLXRandom.normal([1, 3, 2688], key: keys[0]).asType(.bfloat16)
            var products = MLXRandom.normal([18 * 2688], scale: 0.05, key: keys[1])
            if seed == 1 { products = which(abs(products) .< MLXArray(Float(0.03)), MLXArray(Float(0)), products) }
            let shared = MLXRandom.normal([1, 3, 2688], scale: 0.1, key: keys[2]).asType(.bfloat16)
            let fused = applyProductsRows(lhs, products, shared: shared, norm: norm)
            guard let expected = applyPreparedRows(
                lhs, NemotronZSkip.reduceProducts(products, rows: 3), shared: shared, norm: norm)
            else { return }
            ok = MLX.logicalAnd(ok, MLX.logicalAnd(
                (fused.0.view(dtype: .uint16) .== expected.0.view(dtype: .uint16)).all(),
                (fused.1.view(dtype: .uint16) .== expected.1.view(dtype: .uint16)).all()))
        }
        productsJoinVerified = ok.item(Bool.self)
        if !productsJoinVerified {
            FileHandle.standardError.write(
                Data("nemotron zskip join off: the products join differs from reduce + join\n".utf8))
        }
    }

    /// The MoE join over the routed slot products (`NemotronZSkip`): rhs is
    /// [18 * 2688] FP32, slot-major (row r owns slots 6r ..< 6r + 6).
    private static func applyProductsRows(
        _ lhs: MLXArray, _ products: MLXArray, shared: MLXArray?, norm: RMSNorm
    ) -> (MLXArray, MLXArray, MLXArray) {
        guard plumbingRows(lhs, norm: norm), products.dtype == .float32, let shared,
              shared.shape == lhs.shape, shared.dtype == .bfloat16 else {
            // Not reachable on the plumbing route that requested products.
            let routed = NemotronZSkip.reduceProducts(products, rows: 3)
            let summed = lhs + (shared.map { routed + $0 } ?? routed)
            let normalized = norm(summed)
            return (summed, normalized, NemotronHeadPreparedActivation.pack(normalized.reshaped(3, 2688)))
        }
        let (axis, eps) = scalars(norm.eps)
        let outputs = sumProductsPreparedKernel([lhs, products, shared, norm.weight, axis, eps],
            template: [("T", DType.bfloat16)],
            grid: (672 * 3, 1, 1), threadGroup: (672, 1, 1),
            outputShapes: [lhs.shape, lhs.shape, [3, 11, 9 * 32]],
            outputDTypes: [.bfloat16, .bfloat16, .float32])
        guard NemotronMoELeader.selfTest else { return (outputs[0], outputs[1], outputs[2]) }
        // Reference: the separate top-six reduce, then the two-operand join.
        let routed = NemotronZSkip.reduceProducts(products, rows: 3)
        let expected = applyPreparedRows(lhs, routed, shared: shared, norm: norm)!
        // The prepared output's padding slots are never written; compare the
        // residual and the normalized rows it is formed from, as the base does.
        let ok = MLX.logicalAnd(
            (outputs[0].view(dtype: .uint16) .== expected.0.view(dtype: .uint16)).all(),
            (outputs[1].view(dtype: .uint16) .== expected.1.view(dtype: .uint16)).all())
        return (which(ok, outputs[0], MLXArray(Float.nan).asType(.bfloat16)),
                which(ok, outputs[1], MLXArray(Float.nan).asType(.bfloat16)),
                which(ok, outputs[2], MLXArray(Float.nan)))
    }

    /// Captured verify entry norm: (normalized, prepared K=2688 rows).
    static func normPreparedRows(_ x: MLXArray, norm: RMSNorm) -> (MLXArray, MLXArray)? {
        guard plumbingRows(x, norm: norm) else { return nil }
        let (axis, eps) = scalars(norm.eps)
        let outputs = normPreparedKernel([x, norm.weight, axis, eps],
            template: [("T", DType.bfloat16)],
            grid: (672 * 3, 1, 1), threadGroup: (672, 1, 1),
            outputShapes: [x.shape, [3, 11, 9 * 32]], outputDTypes: [.bfloat16, .float32])
        guard NemotronMoELeader.selfTest else { return (outputs[0], outputs[1]) }
        let ok = (outputs[0].view(dtype: .uint16) .== norm(x).view(dtype: .uint16)).all()
        return (which(ok, outputs[0], MLXArray(Float.nan).asType(.bfloat16)),
                which(ok, outputs[1], MLXArray(Float.nan)))
    }

    /// `kernel`'s text with the residual read moved to row a of `lhs` [1, W, 2688], where a is the device
    /// accept walk's count over the acceptance packet (the count of leading packet[targets + i] ==
    /// packet[drafts + i], i < count; the row pick's loop) formed in the launch. Nil if the read moved.
    private static let walkSource: String? = {
        let base = oneBarrierSource(source)
        let read = "T sum = lhs[offset + i] + rhs[offset + i];"
        guard base.components(separatedBy: read).count == 2 else { return nil }
        return """
            size_t nmw_row = 0;
            {
                int nmw_a = 0;
                if (thread_index_in_simdgroup == 0) {
                    const int64_t nmw_step = packet_strides[0];
                    while (nmw_a < walk[0] && packet[(walk[2] + nmw_a) * nmw_step] == packet[(walk[1] + nmw_a) * nmw_step]) {
                        ++nmw_a;
                    }
                }
                nmw_a = simd_broadcast(nmw_a, ushort(0));
                nmw_row = size_t(nmw_a) * axis_size;
            }

            """ + base.replacingOccurrences(of: read, with: "T sum = lhs[nmw_row + offset + i] + rhs[offset + i];")
    }()
    private static let walkKernel = walkSource.map {
        MLXFast.metalKernel(
            name: name("nemotron_h_residual_rms_walk_bf16"),
            inputNames: ["lhs", "rhs", "weight", "axis_size", "eps", "packet", "walk"],
            outputNames: ["residual", "normalized"], source: $0, ensureRowContiguous: true)
    }
    static var walkKernelAvailable: Bool { walkKernel != nil }

    /// Whether `applyWalk` serves these operands (checked before the caller builds anything).
    static func walkServes(_ rows: MLXArray, packet: MLXArray, count: Int, drafts: Int, targets: Int,
                           norm: RMSNorm) -> Bool {
        walkKernel != nil && Device.defaultDevice().deviceType == .gpu && StreamOrDevice.default == .gpu
            && rows.ndim == 3 && rows.dim(0) == 1 && rows.dim(2) == 2688 && rows.dtype == .bfloat16
            && rows.strides == [rows.dim(1) * 2688, 2688, 1]
            && count >= 0 && count < rows.dim(1)
            && packet.ndim == 1 && packet.dtype == .int32 && drafts >= 0 && targets >= 0
            && drafts + count <= packet.dim(0) && targets + count <= packet.dim(0)
            && type(of: norm) == RMSNorm.self && norm.weight.shape == [2688] && norm.weight.dtype == .bfloat16
    }

    /// `apply(rows[a], rhs, norm)` for the row a the device accept walk selects (see `walkSource`), without the
    /// row being gathered first: rows [1, W, 2688] row-contiguous, rhs [1, 1, 2688], both BF16. Nil outside these
    /// shapes.
    static func applyWalk(_ rows: MLXArray, _ rhs: MLXArray, packet: MLXArray, count: Int, drafts: Int,
                          targets: Int, norm: RMSNorm) -> (MLXArray, MLXArray)? {
        guard walkServes(rows, packet: packet, count: count, drafts: drafts, targets: targets, norm: norm),
              rhs.shape == [1, 1, 2688], rhs.dtype == .bfloat16, let walkKernel
        else { return nil }
        let result = walkKernel(
            [rows, rhs, norm.weight, UInt32(2688), norm.eps, packet,
             MLXArray([Int32(count), Int32(drafts), Int32(targets)])],
            template: [("T", DType.bfloat16)],
            grid: (672, 1, 1), threadGroup: (672, 1, 1),
            outputShapes: [rhs.shape, rhs.shape], outputDTypes: [.bfloat16, .bfloat16])
        return (result[0], result[1])
    }

    static func apply(_ lhs: MLXArray, _ rhs: MLXArray, norm: RMSNorm)
        -> (MLXArray, MLXArray)?
    {
        guard Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
            lhs.ndim == 3, lhs.dim(0) == 1, lhs.dim(1) > 0, lhs.dim(2) == 2688,
            rhs.shape == lhs.shape, lhs.dtype == .bfloat16, rhs.dtype == .bfloat16,
            type(of: norm) == RMSNorm.self,
            norm.weight.shape == [2688], norm.weight.dtype == .bfloat16
        else { return nil }
        let result = kernel([lhs, rhs, norm.weight, UInt32(lhs.dim(2)), norm.eps],
            template: [("T", DType.bfloat16)],
            grid: (672 * lhs.dim(1), 1, 1), threadGroup: (672, 1, 1),
            outputShapes: [lhs.shape, lhs.shape], outputDTypes: [.bfloat16, .bfloat16])
        return (result[0], result[1])
    }


    private static func sumSource() -> String {
        let add = "T sum = lhs[offset + i] + rhs[offset + i];"
        precondition(source.components(separatedBy: add).count == 2)
        return source.replacingOccurrences(of: add, with: "T part = rhs[offset + i] + shared[offset + i];\n"
            + "                T sum = lhs[offset + i] + part;")
    }
    private static let sumKernel = MLXFast.metalKernel(
        name: name("nemotron_h_residual_sum_rms_bf16"),
        inputNames: ["lhs", "rhs", "shared", "weight", "axis_size", "eps"],
        outputNames: ["residual", "normalized"],
        source: oneBarrierSource(sumSource()), ensureRowContiguous: true)
    /// The draft tail's join over its six routed slot products ([6 * 2688]
    /// FP32): the top-six reduce of `sumProductsPreparedSource`, one row.
    private static let sumProductsKernel = MLXFast.metalKernel(
        name: name("nemotron_h_residual_products_rms_bf16"),
        inputNames: ["lhs", "rhs", "shared", "weight", "axis_size", "eps"],
        outputNames: ["residual", "normalized"],
        source: oneBarrierSource(withProductsReduce(sumSource())), ensureRowContiguous: true)
    /// `sumProductsKernel` with `shared` bound before `rhs` (same source text; the body names its arguments).
    private static let sumProductsSharedFirstKernel = MLXFast.metalKernel(
        name: name("nemotron_h_residual_products_rms_sf_bf16"),
        inputNames: ["lhs", "shared", "rhs", "weight", "axis_size", "eps"],
        outputNames: ["residual", "normalized"],
        source: oneBarrierSource(withProductsReduce(sumSource())), ensureRowContiguous: true)

    /// Load time: the one-row products join against the separate top-six
    /// reduce plus `applySum`, bit for bit, on seeded rows and the head's norms.
    static func checkDraftProductsJoin(_ head: Module) -> Bool {
        let norms = head.modules().compactMap { $0 as? RMSNorm }.filter {
            type(of: $0) == RMSNorm.self && $0.weight.shape == [2688] && $0.weight.dtype == .bfloat16
        }
        guard !norms.isEmpty else { return false }
        var ok = MLXArray(true)
        for (seed, norm) in norms.enumerated() {
            let keys = MLXRandom.split(key: MLXRandom.key(UInt64(0x6a11 + seed)), into: 3)
            let lhs = MLXRandom.normal([1, 1, 2688], key: keys[0]).asType(.bfloat16)
            var products = MLXRandom.normal([6 * 2688], scale: 0.05, key: keys[1])
            if seed % 2 == 1 { products = which(abs(products) .< MLXArray(Float(0.03)), MLXArray(Float(0)), products) }
            let shared = MLXRandom.normal([1, 1, 2688], scale: 0.1, key: keys[2]).asType(.bfloat16)
            guard let fused = applySum(lhs, products, shared: shared, norm: norm),
                  let expected = applySum(lhs, NemotronZSkip.reduceProducts(products, rows: 1),
                                          shared: shared, norm: norm)
            else { return false }
            ok = MLX.logicalAnd(ok, MLX.logicalAnd(
                (fused.0.view(dtype: .uint16) .== expected.0.view(dtype: .uint16)).all(),
                (fused.1.view(dtype: .uint16) .== expected.1.view(dtype: .uint16)).all()))
        }
        return ok.item(Bool.self)
    }

    /// `rhs` [6 * 2688] FP32 is the draft tail's routed slot products
    /// (`NemotronZSkip.draftProducts`); the join sums them itself.
    /// `sharedFirst` (products only; `NemotronDraftColaunch`): the same join kernel with `shared` bound before
    /// the products (part of the draft tail's encode order; the body names its arguments).
    static func applySum(_ lhs: MLXArray, _ rhs: MLXArray, shared: MLXArray, norm: RMSNorm,
                         sharedFirst: Bool = false)
        -> (MLXArray, MLXArray)?
    {
        if rhs.shape == [6 * 2688] {
            return applyProductsSum(lhs, rhs, shared: shared, norm: norm, sharedFirst: sharedFirst)
        }
        guard Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
            lhs.ndim == 3, lhs.dim(0) == 1, lhs.dim(1) > 0, lhs.dim(2) == 2688,
            rhs.shape == lhs.shape, shared.shape == lhs.shape, lhs.dtype == .bfloat16,
            rhs.dtype == .bfloat16, shared.dtype == .bfloat16,
            type(of: norm) == RMSNorm.self,
            norm.weight.shape == [2688], norm.weight.dtype == .bfloat16
        else { return nil }
        let (axis, eps) = scalars(norm.eps)
        let result = sumKernel([lhs, rhs, shared, norm.weight, axis, eps],
            template: [("T", DType.bfloat16)],
            grid: (672 * lhs.dim(1), 1, 1), threadGroup: (672, 1, 1),
            outputShapes: [lhs.shape, lhs.shape], outputDTypes: [.bfloat16, .bfloat16])
        guard NemotronMoELeader.selfTest else { return (result[0], result[1]) }
        let summed = lhs + (rhs + shared)
        let ok = MLX.logicalAnd(
            (result[0].view(dtype: .uint16) .== summed.view(dtype: .uint16)).all(),
            (result[1].view(dtype: .uint16) .== norm(summed).view(dtype: .uint16)).all())
        return (which(ok, result[0], MLXArray(Float.nan).asType(.bfloat16)),
                which(ok, result[1], MLXArray(Float.nan).asType(.bfloat16)))
    }

    private static func applyProductsSum(
        _ lhs: MLXArray, _ products: MLXArray, shared: MLXArray, norm: RMSNorm, sharedFirst: Bool = false
    ) -> (MLXArray, MLXArray) {
        guard Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
            lhs.shape == [1, 1, 2688], shared.shape == lhs.shape, lhs.dtype == .bfloat16,
            shared.dtype == .bfloat16, products.dtype == .float32,
            type(of: norm) == RMSNorm.self,
            norm.weight.shape == [2688], norm.weight.dtype == .bfloat16
        else {
            // Not reachable on the draft tail that requested products.
            let summed = lhs + (NemotronZSkip.reduceProducts(products, rows: 1) + shared)
            return (summed, norm(summed))
        }
        let (axis, eps) = scalars(norm.eps)
        let result = sharedFirst
            ? sumProductsSharedFirstKernel([lhs, shared, products, norm.weight, axis, eps],
                template: [("T", DType.bfloat16)],
                grid: (672, 1, 1), threadGroup: (672, 1, 1),
                outputShapes: [lhs.shape, lhs.shape], outputDTypes: [.bfloat16, .bfloat16])
            : sumProductsKernel([lhs, products, shared, norm.weight, axis, eps],
                template: [("T", DType.bfloat16)],
                grid: (672, 1, 1), threadGroup: (672, 1, 1),
                outputShapes: [lhs.shape, lhs.shape], outputDTypes: [.bfloat16, .bfloat16])
        guard NemotronMoELeader.selfTest else { return (result[0], result[1]) }
        let summed = lhs + (NemotronZSkip.reduceProducts(products, rows: 1) + shared)
        let ok = MLX.logicalAnd(
            (result[0].view(dtype: .uint16) .== summed.view(dtype: .uint16)).all(),
            (result[1].view(dtype: .uint16) .== norm(summed).view(dtype: .uint16)).all())
        return (which(ok, result[0], MLXArray(Float.nan).asType(.bfloat16)),
                which(ok, result[1], MLXArray(Float.nan).asType(.bfloat16)))
    }
}


enum NemotronHEmbeddingRows {
    static let enabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_DRAFT_EMBED"] != "0"
        && ProcessInfo.processInfo.environment["NEMOTRON_MTP_DRAFT_FUSED"] != "0"
    private static let kernel = MLXFast.metalKernel(
        name: "nemotron_h_embedding_rows_affine4_gs64",
        inputNames: ["ids", "w", "scales", "biases"], outputNames: ["out"],
        source: """
            const uint b = thread_position_in_grid.x;
            const uint t = thread_position_in_grid.y;
            int64_t id = int64_t(ids[t]);
            if (id < 0) {
                id += int64_t(V);
            }
            const device uint8_t* row = (const device uint8_t*)w + id * int64_t(BYTES);
            const size_t oindex = size_t(b) * 2;
            const size_t gindex = oindex / 64;
            T scale = scales[id * int64_t(G) + int64_t(gindex)];
            T bias = biases[id * int64_t(G) + int64_t(gindex)];
            device T* o = out + size_t(t) * size_t(2 * BYTES) + oindex;
            uint val = row[b];
            #pragma clang loop unroll(full)
            for (int i = 0; i < 2; i++) {
                uint8_t d;
                d = (val >> (4 * i)) & 0x0f;
                o[i] = scale * d + bias;
            }
            """, ensureRowContiguous: true)

    static func apply(_ tokens: MLXArray, _ embedding: Embedding) -> MLXArray? {
        guard enabled, Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              tokens.ndim == 2, tokens.dim(0) == 1, (1...512).contains(tokens.dim(1)),
              tokens.dtype == .int32 || tokens.dtype == .uint32,
              type(of: embedding) == QuantizedEmbedding.self,
              let q = embedding as? QuantizedEmbedding,
              q.mode == .affine, q.bits == 4, q.groupSize == 64,
              let biases = q.biases, q.weight.dtype == .uint32, q.weight.ndim == 2,
              q.scales.dtype == .bfloat16, biases.dtype == .bfloat16
        else { return nil }
        let vocabulary = q.weight.dim(0), words = q.weight.dim(1)
        let dims = words * 8
        guard dims % 64 == 0, q.scales.shape == [vocabulary, dims / 64],
              biases.shape == q.scales.shape, (dims / 2) % 64 == 0
        else { return nil }
        let count = tokens.dim(1)
        return kernel([tokens.reshaped(count), q.weight, q.scales, biases],
            template: [("T", DType.bfloat16), ("V", vocabulary), ("BYTES", dims / 2),
                       ("G", dims / 64)],
            grid: (dims / 2, count, 1), threadGroup: (64, 1, 1),
            outputShapes: [[1, count, dims]], outputDTypes: [.bfloat16])[0]
    }
}

enum NemotronHVerifyEntry {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_VERIFY_ENTRY_FUSED"] != "0"
    private static let lock = NSLock()
    nonisolated(unsafe) private static var admittedEmbedding: QuantizedEmbedding?
    nonisolated(unsafe) private static var admittedNorm: RMSNorm?

    static func matches(_ embedding: QuantizedEmbedding, _ norm: RMSNorm) -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard let e = admittedEmbedding, let n = admittedNorm else { return false }
        return e === embedding && n === norm
    }

    static func publish(_ embedding: QuantizedEmbedding, _ norm: RMSNorm, accepted: Bool) {
        lock.lock(); defer { lock.unlock() }
        admittedEmbedding = accepted ? embedding : nil
        admittedNorm = accepted ? norm : nil
    }
}

extension NemotronHResidualRMSNorm {
    private static func verifyEntrySource() -> String {
        let add = "T sum = lhs[offset + i] + rhs[offset + i];"
        let s = preparedSource()
        precondition(s.components(separatedBy: add).count == 2)
        return s.replacingOccurrences(of: add, with: """
            const uint column = lid * n_reads + i;
                            int64_t id = int64_t(ids[gid]);
                            if (id < 0) {
                                id += int64_t(V);
                            }
                            const device uint8_t* row = (const device uint8_t*)w + id * int64_t(BYTES);
                            T scale = scales[id * int64_t(G) + int64_t(column / 64)];
                            T bias = biases[id * int64_t(G) + int64_t(column / 64)];
                            uint val = row[column / 2];
                            uint8_t d;
                            d = (val >> (4 * (column % 2))) & 0x0f;
                            T sum = scale * d + bias;
            """)
    }
    private static let verifyEntryKernel = MLXFast.metalKernel(
        name: oneBarrier ? "nemotron_h_verify_entry_rms_prepared_1b" : "nemotron_h_verify_entry_rms_prepared",
        inputNames: ["ids", "w", "scales", "biases", "weight", "axis_size", "eps"],
        outputNames: ["residual", "normalized", "prepared"],
        source: oneBarrierSource(verifyEntrySource()), ensureRowContiguous: true)

    private static func verifyEntryRows(_ ids: MLXArray, _ q: QuantizedEmbedding, norm: RMSNorm)
        -> [MLXArray]
    {
        let count: Int = ids.size
        let dims: Int = 2688
        let vocabulary: Int = q.weight.dim(0)
        let biases: MLXArray = q.biases!
        let (axis, eps) = scalars(norm.eps)
        let inputs: [MLXArray] = [ids.reshaped(count), q.weight, q.scales, biases, norm.weight, axis, eps]
        let template: [(String, any KernelTemplateArg)] = [
            ("T", DType.bfloat16), ("V", vocabulary), ("BYTES", dims / 2), ("G", dims / 64),
        ]
        let residualShape: [Int] = [1, count, dims]
        let preparedShape: [Int] = [count, 11, 288]
        return verifyEntryKernel(
            inputs, template: template,
            grid: (672 * count, 1, 1), threadGroup: (672, 1, 1),
            outputShapes: [residualShape, residualShape, preparedShape],
            outputDTypes: [.bfloat16, .bfloat16, .float32])
    }

    private static func verifyEntryShape(_ embedding: Embedding, norm: RMSNorm) -> QuantizedEmbedding? {
        guard NemotronHVerifyEntry.enabled, NemotronHEmbeddingRows.enabled else { return nil }
        guard type(of: embedding) == QuantizedEmbedding.self,
              let q = embedding as? QuantizedEmbedding else { return nil }
        guard q.mode == .affine, q.bits == 4, q.groupSize == 64 else { return nil }
        guard let biases = q.biases else { return nil }
        let w: MLXArray = q.weight
        guard w.dtype == .uint32, w.ndim == 2 else { return nil }
        let vocabulary: Int = w.dim(0)
        guard w.dim(1) == 336 else { return nil }
        let groupShape: [Int] = [vocabulary, 42]
        guard q.scales.dtype == .bfloat16, q.scales.shape == groupShape else { return nil }
        guard biases.dtype == .bfloat16, biases.shape == groupShape else { return nil }
        guard type(of: norm) == RMSNorm.self else { return nil }
        let normShape: [Int] = [2688]
        guard norm.weight.shape == normShape, norm.weight.dtype == .bfloat16 else { return nil }
        return q
    }

    nonisolated(unsafe) private static let preparedWritten: MLXArray = {
        var flags: [Bool] = []
        flags.reserveCapacity(11 * 288)
        for block in 0 ..< 11 {
            for word in 0 ..< 288 {
                let lane: Int = word % 32
                let chunk: Int = block * 32 + lane
                flags.append(chunk < 336)
            }
        }
        return MLXArray(flags).reshaped(11, 288)
    }()

    private static func preparedEqual(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
        let equal: MLXArray = a.view(dtype: .uint32) .== b.view(dtype: .uint32)
        return which(preparedWritten, equal, MLXArray(true)).all()
    }

    private static func halfEqual(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
        let equal: MLXArray = a.view(dtype: .uint16) .== b.view(dtype: .uint16)
        return equal.all()
    }

    static func verifyEntry(_ tokens: MLXArray, _ embedding: Embedding, norm: RMSNorm)
        -> (MLXArray, MLXArray, MLXArray)?
    {
        guard plumbingAvailable, tokens.shape == [1, 3] else { return nil }
        guard tokens.dtype == .int32 || tokens.dtype == .uint32 else { return nil }
        guard let q = verifyEntryShape(embedding, norm: norm) else { return nil }
        guard NemotronHVerifyEntry.matches(q, norm) else { return nil }
        let outputs = verifyEntryRows(tokens, q, norm: norm)
        guard NemotronMoELeader.selfTest else { return (outputs[0], outputs[1], outputs[2]) }
        let hidden = embedding(tokens)
        guard let expected = normPreparedRows(hidden, norm: norm) else { return nil }
        let ok: MLXArray = preparedEqual(outputs[2], expected.1)
        return (nemotronBitwiseOrNaN(outputs[0], hidden), nemotronBitwiseOrNaN(outputs[1], expected.0),
                which(ok, outputs[2], MLXArray(Float.nan)))
    }

    static func qualifyVerifyEntry(_ embedding: Embedding, norm: RMSNorm) {
        guard plumbingAvailable, let q = verifyEntryShape(embedding, norm: norm) else {
            FileHandle.standardError.write(Data("nemotron verify entry: off\n".utf8))
            return
        }
        let (axis, eps) = scalars(norm.eps)
        func rows(_ ids: MLXArray) -> Bool {
            let count: Int = ids.size
            let got = verifyEntryRows(ids, q, norm: norm)
            let hidden = embedding(ids)
            let preparedShape: [Int] = [count, 11, 288]
            let stock = normPreparedKernel([hidden, norm.weight, axis, eps],
                template: [("T", DType.bfloat16)],
                grid: (672 * count, 1, 1), threadGroup: (672, 1, 1),
                outputShapes: [hidden.shape, preparedShape],
                outputDTypes: [.bfloat16, .float32])
            let first: MLXArray = halfEqual(got[0], hidden)
            let second: MLXArray = halfEqual(got[1], stock[0])
            let third: MLXArray = preparedEqual(got[2], stock[1])
            let all: MLXArray = MLX.logicalAnd(first, MLX.logicalAnd(second, third))
            return all.item(Bool.self)
        }
        let vocabulary: Int = q.weight.dim(0)
        NemotronHVerifyEntry.publish(q, norm, accepted: false)
        var admitted = false
        var note = "mismatch, stock kept"
        do {
            admitted = try withError { () -> Bool in
                var start = 0
                while start < vocabulary {
                    let count: Int = min(4096, vocabulary - start)
                    let range: Range<Int32> = Int32(start) ..< Int32(start + count)
                    if !rows(MLXArray(range).reshaped(1, count)) { return false }
                    start += count
                }
                var state: UInt64 = 0x3C6E_F372_FE94_F82B
                func draw() -> Int {
                    state = state &* 6364136223846793005 &+ 1442695040888963407
                    return Int(state >> 33) % vocabulary
                }
                for _ in 0 ..< 8 {
                    let signed: [Int32] = [Int32(draw()), Int32(draw()), Int32(draw())]
                    if !rows(MLXArray(signed).reshaped(1, 3)) { return false }
                    let unsigned: [UInt32] = [UInt32(draw()), UInt32(draw()), UInt32(draw())]
                    if !rows(MLXArray(unsigned).reshaped(1, 3)) { return false }
                }
                let v32: Int32 = Int32(vocabulary)
                let edgeA: [Int32] = [-v32, -1, 0]
                let edgeB: [Int32] = [v32 - 1, -(v32 / 2), 1]
                if !rows(MLXArray(edgeA).reshaped(1, 3)) { return false }
                if !rows(MLXArray(edgeB).reshaped(1, 3)) { return false }
                return true
            }
        } catch {
            admitted = false
            note = "\(error)"
        }
        NemotronHVerifyEntry.publish(q, norm, accepted: admitted)
        let line = admitted ? "on (all \(vocabulary) rows bitwise equal)" : "off (\(note))"
        FileHandle.standardError.write(Data("nemotron verify entry: \(line)\n".utf8))
    }
}

// MARK: - MambaRMSNormGated

enum NemotronHGatedRMSKernel {
    private static let source = """
            uint gid = threadgroup_position_in_grid.x;
            uint lid = thread_position_in_threadgroup.x;
            uint simd_lane_id = thread_index_in_simdgroup;
            uint simd_group_id = simdgroup_index_in_threadgroup;
            int64_t groups = int64_t(x_shape[2]) / 512;
            int64_t row = int64_t(gid) / groups;
            int64_t feature = (int64_t(gid) % groups) * 512 + int64_t(lid) * 4;
            threadgroup float local_inv_mean[1];
            threadgroup float local_sums[32];
            float acc = 0;
            float thread_x[4];
            for (int i = 0; i < 4; i++) {
                int64_t f = feature + int64_t(i);
                T g = gate[row * gate_strides[1] + f * gate_strides[2]];
                auto y = 1 / (1 + metal::exp(metal::abs(g)));
                T sig = g < 0 ? y : 1 - y;
                T activated = g * sig;
                T gated = x[row * x_strides[1] + f * x_strides[2]] * activated;
                thread_x[i] = gated;
                acc += thread_x[i] * thread_x[i];
            }
            acc = simd_sum(acc);
            if (simd_group_id == 0) {
                local_sums[simd_lane_id] = 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_lane_id == 0) {
                local_sums[simd_group_id] = acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group_id == 0) {
                acc = simd_sum(local_sums[simd_lane_id]);
                if (simd_lane_id == 0) {
                    local_inv_mean[0] = metal::precise::rsqrt(acc / 512 + eps);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int i = 0; i < 4; i++) {
                int64_t f = feature + int64_t(i);
                T normalized = static_cast<T>(thread_x[i] * local_inv_mean[0]);
                T unweighted = identity * normalized;
                T weighted = weight[f * weight_strides[0]] * unweighted;
                out[row * int64_t(x_shape[2]) + f] = weighted;
            }
            """

    /// The kernel above with each thread's four features read and written as
    /// one 4-element vector (contiguous rows only) and G groups of 512 per
    /// threadgroup (one 128-thread slice per group). Every per-element
    /// statement, the per-thread order, and both `simd_sum` levels with their
    /// lane layout are the kernel's.
    private static let wideSource = """
            uint slice = thread_position_in_threadgroup.x / 128;
            uint gid = threadgroup_position_in_grid.x * G + slice;
            uint lid = thread_position_in_threadgroup.x % 128;
            uint simd_lane_id = thread_index_in_simdgroup;
            uint simd_group_id = (thread_position_in_threadgroup.x % 128) / 32;
            int64_t groups = int64_t(x_shape[2]) / 512;
            int64_t row = int64_t(gid) / groups;
            int64_t feature = (int64_t(gid) % groups) * 512 + int64_t(lid) * 4;
            threadgroup float local_inv_mean[G];
            threadgroup float local_sums[G][32];
            const bool live = int64_t(gid) < int64_t(x_shape[1]) * groups;
            float acc = 0;
            float thread_x[4];
            vec<T, 4> xs4 = vec<T, 4>(0);
            vec<T, 4> gs4 = vec<T, 4>(0);
            if (live) {
                xs4 = *((const device vec<T, 4>*)(x + row * x_strides[1] + feature));
                gs4 = *((const device vec<T, 4>*)(gate + row * gate_strides[1] + feature));
            }
            for (int i = 0; i < 4; i++) {
                T g = gs4[i];
                auto y = 1 / (1 + metal::exp(metal::abs(g)));
                T sig = g < 0 ? y : 1 - y;
                T activated = g * sig;
                T gated = xs4[i] * activated;
                thread_x[i] = gated;
                acc += thread_x[i] * thread_x[i];
            }
            acc = simd_sum(acc);
            if (simd_group_id == 0) {
                local_sums[slice][simd_lane_id] = 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_lane_id == 0) {
                local_sums[slice][simd_group_id] = acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group_id == 0) {
                acc = simd_sum(local_sums[slice][simd_lane_id]);
                if (simd_lane_id == 0) {
                    local_inv_mean[slice] = metal::precise::rsqrt(acc / 512 + eps);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (live) {
                const vec<T, 4> ws4 = *((const device vec<T, 4>*)(weight + feature));
                vec<T, 4> o4;
                for (int i = 0; i < 4; i++) {
                    T normalized = static_cast<T>(thread_x[i] * local_inv_mean[slice]);
                    T unweighted = identity * normalized;
                    T weighted = ws4[i] * unweighted;
                    o4[i] = weighted;
                }
                *((device vec<T, 4>*)(out + row * int64_t(x_shape[2]) + feature)) = o4;
            }
            """
    static let wideKernel = MLXFast.metalKernel(
        name: "nemotron_h_gated_rms_wide",
        inputNames: ["x", "gate", "weight", "identity", "eps"], outputNames: ["out"],
        source: wideSource, ensureRowContiguous: false)

    nonisolated(unsafe) static let identity = MLXArray(1).asType(.bfloat16)
    static let kernel = MLXFast.metalKernel(
        name: "nemotron_h_gated_rms",
        inputNames: ["x", "gate", "weight", "identity", "eps"], outputNames: ["out"],
        source: source, ensureRowContiguous: false)
    private static func preparedSource() -> String {
        let scratch = "threadgroup float local_inv_mean[1];"
        let store = "out[row * int64_t(x_shape[2]) + f] = weighted;"
        precondition(source.components(separatedBy: scratch).count == 2)
        precondition(source.components(separatedBy: store).count == 2)
        return source.replacingOccurrences(of: scratch,
            with: scratch + "\nthreadgroup T rounded[512];")
            .replacingOccurrences(of: store, with: store + "\nrounded[lid * 4 + i] = weighted;")
            + """

            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lid < 32) {
                float values[16];
                float total = 0;
                for (int i = 0; i < 16; i += 4) {
                    total += float(rounded[lid * 16 + i]) + float(rounded[lid * 16 + i + 1])
                        + float(rounded[lid * 16 + i + 2]) + float(rounded[lid * 16 + i + 3]);
                    values[i] = rounded[lid * 16 + i];
                    values[i + 1] = rounded[lid * 16 + i + 1] / 16.0f;
                    values[i + 2] = rounded[lid * 16 + i + 2] / 256.0f;
                    values[i + 3] = rounded[lid * 16 + i + 3] / 4096.0f;
                }
                const int base = int(gid) * 17 * 32 + int(lid);
                for (int i = 0; i < 16; i++) prepared[base + i * 32] = values[i];
                prepared[base + 16 * 32] = total;
            }
            """
    }
    static let preparedKernel = MLXFast.metalKernel(
        name: "nemotron_h_gated_rms_prepared_output",
        inputNames: ["x", "gate", "weight", "identity", "eps"],
        outputNames: ["out", "prepared"], source: preparedSource(), ensureRowContiguous: false)

}

class NemotronHRMSNormGated: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float
    let groupSize: Int

    init(dimensions: Int, eps: Float, groupSize: Int) {
        self.eps = eps
        self.groupSize = groupSize
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    private static let preparedOutputEnabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_GATED_PREPARED_OUT"] != "0"

    /// Native compact verify only. The original BF16 gate/norm/identity/weight
    /// boundaries remain; preparation reads the final rounded BF16 values.
    func preparedRows(_ x: MLXArray, gate: MLXArray) -> (MLXArray, MLXArray)? {
        guard Self.preparedOutputEnabled, NemotronMTPExecution.batchedNorm,
              NemotronMultiRowQMV.enabled, NemotronMultiRowQMV.narrowEnabled,
              Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              type(of: self) == NemotronHRMSNormGated.self,
              groupSize == 512, x.shape == [1, 3, 4096], gate.shape == x.shape,
              weight.shape == [4096], x.dtype == .bfloat16,
              gate.dtype == .bfloat16, weight.dtype == .bfloat16
        else { return nil }
        let outputs = NemotronHGatedRMSKernel.preparedKernel(
            [x, gate, weight, NemotronHGatedRMSKernel.identity, eps],
            template: [("T", DType.bfloat16)],
            grid: (128 * 3 * 8, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [x.shape, [3, 8, 17 * 32]], outputDTypes: [.bfloat16, .float32])
        let normalized = NemotronMoELeader.selfTest
            ? nemotronBitwiseOrNaN(outputs[0], callAsFunction(x, gate: gate)) : outputs[0]
        return (normalized, outputs[1])
    }

    func callAsFunction(_ x: MLXArray, gate: MLXArray?) -> MLXArray {
        if let gate, groupSize == 512, x.ndim == 3, x.dim(0) == 1,
            x.dim(1) > 0, x.dim(2) > 0, x.dim(2) % 512 == 0,
            gate.shape == x.shape, weight.shape == [x.dim(2)],
            x.dtype == .bfloat16, gate.dtype == .bfloat16, weight.dtype == .bfloat16
        {
            if NemotronPrefillGatedRMSWide.verified, NemotronHPrefillX5.active, x.dim(1) > 1,
               let wide = NemotronPrefillGatedRMSWide.apply(x, gate: gate, weight: weight, eps: eps) {
                return wide
            }
            return NemotronHGatedRMSKernel.kernel(
                [x, gate, weight, NemotronHGatedRMSKernel.identity, eps],
                template: [("T", DType.bfloat16)],
                grid: (128 * x.dim(1) * (x.dim(2) / 512), 1, 1),
                threadGroup: (128, 1, 1), outputShapes: [x.shape], outputDTypes: [.bfloat16])[0]
        }
        var states = x
        if let gate {
            states = states * silu(gate)
        }

        // Python: x = mx.unflatten(x, axis=-1, shape=(-1, self.group_size))
        // Reshape [..., hidden] -> [..., nGroups, groupSize] for per-group normalization
        let shape = states.shape
        var newShape = Array(shape.dropLast())
        newShape.append(-1)
        newShape.append(groupSize)
        let unflattened = states.reshaped(newShape)

        // Python: x = mx.fast.rms_norm(x, weight=None, eps=self.eps)
        // Apply RMS norm per group WITHOUT scaling (pass ones as weight)
        // Swift rmsNorm doesn't accept nil, so we use identity weight
        // The optional-weight Python RMSNorm preserves x.dtype. A default
        // FP32 identity silently widens BF16 activations for the rest of the
        // trunk (and disables native BF16 verification kernels).
        let identityWeight = MLXArray.ones([groupSize], dtype: unflattened.dtype)
        let normed = MLXFast.rmsNorm(unflattened, weight: identityWeight, eps: eps)

        // Python: return self.weight * x.flatten(-2)
        // Flatten back to [..., hidden] and apply learned weight
        let flattened = normed.reshaped(shape)
        return weight * flattened
    }
}

// Compact verify SSM + gated RMS norm (+ out_proj preparation) in one launch.
// The body up to the arrival is nemotron_mtp_ssm_compact's text (WO, L = 3)
// and still stores y. Each 512-feature norm group is 8 heads = 64 of its
// threadgroups; the last of those to finish (device counter per group, reset
// by that group) runs nemotron_h_gated_rms_prepared_output's per-thread text
// for the group's rows in order, on threads 0-127 (simdgroups 0-3, as in that
// launch), reading the stored y and the gate in place from in_proj's
// contiguous [1, 3, 10304] output (the gate is its first 4096 columns). Same y, same per-row code: y, the final
// state, the normalized rows and the prepared activation are bitwise those of
// the two launches, and the norm leaves the SSM -> out_proj dependency chain.
enum NemotronHCompactSSMGated {
    static let enabled =
        ProcessInfo.processInfo.environment["NEMOTRON_SSM_GATED_FUSED"] != "0"
    static let source = """
            auto n = thread_position_in_grid.z;
            auto h_idx = n % H;
            auto g_idx = n / G;
            constexpr int n_per_t = (Ds + 31) / 32;
            auto ds_idx = thread_position_in_threadgroup.x;
            auto d_idx = thread_position_in_grid.y;
            float prior[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * ds_idx + i;
                if constexpr (Ds % 32 != 0) { if (s_idx >= Ds) continue; }
                prior[i] = state_in[n * Dh * Ds + d_idx * Ds + s_idx];
            }
            for (int t = 0; t < L; ++t) {
                auto x_ = static_cast<float>(conv[t * CD + n * Dh + d_idx]);
                auto C_ = conv + t * CD + CO + g_idx * Ds;
                auto B_ = conv + t * CD + BO + g_idx * Ds;
                auto dt_ = static_cast<float>(dt[t * H + n]);
                auto A = -fast::exp(static_cast<float>(A_log[h_idx]));
                auto dA = fast::exp(A * dt_);
                float acc = 0.0;
                for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * ds_idx + i;
                    if constexpr (Ds % 32 != 0) { if (s_idx >= Ds) continue; }
                    auto dB_by_x = x_ * dt_ * static_cast<float>(B_[s_idx]);
                    auto state = dA * prior[i] + dB_by_x;
                    if (t == L - 1) {
                        state_out[(n * Dh + d_idx) * Ds + s_idx] = static_cast<U>(state);
                    }
                    prior[i] = static_cast<U>(state);
                    acc += state * C_[s_idx];
                }
                acc = simd_sum(acc);
                if (thread_index_in_simdgroup == 0) {
                    out[(t * H + n) * Dh + d_idx] = static_cast<T>(acc + x_ * D[h_idx]);
                }
            }

            threadgroup uint gr_last;
            threadgroup float local_inv_mean[1];
            threadgroup float local_sums[32];
            threadgroup T rounded[512];
            const uint flat = thread_position_in_threadgroup.x + 32 * thread_position_in_threadgroup.y;
            const uint grp = n / (512 / Dh);
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (flat == 0) {
                const uint prev = atomic_fetch_add_explicit(
                    (device atomic_uint*)(counter + grp), 1u, memory_order_relaxed);
                gr_last = prev == uint((512 / Dh) * (Dh / 8)) - 1u ? 1u : 0u;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (gr_last == 0u) { return; }
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
            if (flat == 0) {
                atomic_store_explicit((device atomic_uint*)(counter + grp), 0u, memory_order_relaxed);
            }
            uint simd_lane_id = thread_index_in_simdgroup;
            uint simd_group_id = simdgroup_index_in_threadgroup;
            const uint lid = flat;
            const bool on = lid < 128;
            for (int row = 0; row < L; ++row) {
                const uint gid = uint(row) * uint(H * Dh / 512) + grp;
                int64_t feature = int64_t(grp) * 512 + int64_t(lid) * 4;
                float acc = 0;
                float thread_x[4];
                if (on) {
                    for (int i = 0; i < 4; i++) {
                        int64_t f = feature + int64_t(i);
                        T g = proj[int64_t(row) * PS + f];
                        auto y = 1 / (1 + metal::exp(metal::abs(g)));
                        T sig = g < 0 ? y : 1 - y;
                        T activated = g * sig;
                        T gated = out[int64_t(row) * (H * Dh) + f] * activated;
                        thread_x[i] = gated;
                        acc += thread_x[i] * thread_x[i];
                    }
                }
                acc = simd_sum(acc);
                if (simd_group_id == 0) {
                    local_sums[simd_lane_id] = 0;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (on && simd_lane_id == 0) {
                    local_sums[simd_group_id] = acc;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (simd_group_id == 0) {
                    acc = simd_sum(local_sums[simd_lane_id]);
                    if (simd_lane_id == 0) {
                        local_inv_mean[0] = metal::precise::rsqrt(acc / 512 + eps);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (on) {
                    for (int i = 0; i < 4; i++) {
                        int64_t f = feature + int64_t(i);
                        T normalized = static_cast<T>(thread_x[i] * local_inv_mean[0]);
                        T unweighted = identity * normalized;
                        T weighted = weight[f] * unweighted;
                        normed[int64_t(row) * (H * Dh) + f] = weighted;
                        rounded[lid * 4 + i] = weighted;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (lid < 32) {
                    float values[16];
                    float total = 0;
                    for (int i = 0; i < 16; i += 4) {
                        total += float(rounded[lid * 16 + i]) + float(rounded[lid * 16 + i + 1])
                            + float(rounded[lid * 16 + i + 2]) + float(rounded[lid * 16 + i + 3]);
                        values[i] = rounded[lid * 16 + i];
                        values[i + 1] = rounded[lid * 16 + i + 1] / 16.0f;
                        values[i + 2] = rounded[lid * 16 + i + 2] / 256.0f;
                        values[i + 3] = rounded[lid * 16 + i + 3] / 4096.0f;
                    }
                    const int base = int(gid) * 17 * 32 + int(lid);
                    for (int i = 0; i < 16; i++) prepared[base + i * 32] = values[i];
                    prepared[base + 16 * 32] = total;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            """
    /// NEMOTRON_SSM_TAIL_SHUFFLE=0 keeps the threadgroup copy of the rounded rows. Otherwise the group's last
    /// threadgroup runs each row with one barrier: simdgroups 0-3 store their sums in that row's slots, every
    /// one of them then forms the record's 32-lane sum (the four sums, zeros elsewhere) and inverse mean itself,
    /// and each lane writes its own four prepared words; the chunk's total adds the four lanes' partial sums
    /// (each the record's four-value expression) in lane order through simd_shuffle_down.
    /// The launch uses it only after `qualifyShuffle` found its outputs bitwise equal to the record kernel's.
    static let tailShuffle = ProcessInfo.processInfo.environment["NEMOTRON_SSM_TAIL_SHUFFLE"] != "0"
    static func shuffledTail(_ source: String) -> String? {
        let head = "threadgroup T rounded[512];\n"
        let loop = "for (int row = 0; row < L; ++row) {\n"
        guard tailShuffle, source.components(separatedBy: head).count == 2,
              let start = source.range(of: loop) else { return nil }
        return String(source[..<start.lowerBound]).replacingOccurrences(of: head, with:
            "            threadgroup float nst_sums[L][4];\n") + """
                        for (int row = 0; row < L; ++row) {
                            const uint gid = uint(row) * uint(H * Dh / 512) + grp;
                            int64_t feature = int64_t(grp) * 512 + int64_t(lid) * 4;
                            float acc = 0;
                            float thread_x[4];
                            if (on) {
                                for (int i = 0; i < 4; i++) {
                                    int64_t f = feature + int64_t(i);
                                    T g = proj[int64_t(row) * PS + f];
                                    auto y = 1 / (1 + metal::exp(metal::abs(g)));
                                    T sig = g < 0 ? y : 1 - y;
                                    T activated = g * sig;
                                    T gated = out[int64_t(row) * (H * Dh) + f] * activated;
                                    thread_x[i] = gated;
                                    acc += thread_x[i] * thread_x[i];
                                }
                            }
                            acc = simd_sum(acc);
                            if (on && simd_lane_id == 0) {
                                nst_sums[row][simd_group_id] = acc;
                            }
                            threadgroup_barrier(mem_flags::mem_threadgroup);
                            if (on) {
                                const float total_sq = simd_sum(simd_lane_id < 4 ? nst_sums[row][simd_lane_id] : 0.0f);
                                const float inv_mean = metal::precise::rsqrt(total_sq / 512 + eps);
                                T w[4];
                                for (int i = 0; i < 4; i++) {
                                    int64_t f = feature + int64_t(i);
                                    T normalized = static_cast<T>(thread_x[i] * inv_mean);
                                    T unweighted = identity * normalized;
                                    T weighted = weight[f] * unweighted;
                                    normed[int64_t(row) * (H * Dh) + f] = weighted;
                                    w[i] = weighted;
                                }
                                const float part = float(w[0]) + float(w[1]) + float(w[2]) + float(w[3]);
                                const float p1 = simd_shuffle_down(part, ushort(1));
                                const float p2 = simd_shuffle_down(part, ushort(2));
                                const float p3 = simd_shuffle_down(part, ushort(3));
                                const int k = int(lid & 3u);
                                const int base = int(gid) * 17 * 32 + int(lid >> 2);
                                prepared[base + (4 * k + 0) * 32] = float(w[0]);
                                prepared[base + (4 * k + 1) * 32] = float(w[1]) / 16.0f;
                                prepared[base + (4 * k + 2) * 32] = float(w[2]) / 256.0f;
                                prepared[base + (4 * k + 3) * 32] = float(w[3]) / 4096.0f;
                                if (k == 0) {
                                    float total = 0;
                                    total += part;
                                    total += p1;
                                    total += p2;
                                    total += p3;
                                    prepared[base + 16 * 32] = total;
                                }
                            }
                        }

            """
    }
    /// NEMOTRON_SSM_TAIL_ONE_BARRIER=0 keeps one barrier per row in the shuffle tail. Otherwise the group's
    /// last threadgroup first forms every row's gated values and SIMD-group sums of squares (the shuffle tail's
    /// statements, row by row), stores the sums of all rows, passes one barrier, and then finishes the rows in
    /// order with the shuffle tail's statements: the same values, sums, inverse means and prepared words, with
    /// one barrier instead of L. The launch uses it only after `qualifyRowsTail` found it bitwise equal to the
    /// shuffle-tail launch.
    static let tailOneBarrier = ProcessInfo.processInfo.environment["NEMOTRON_SSM_TAIL_ONE_BARRIER"] != "0"
    static func rowsTail(_ source: String) -> String? {
        let head = "threadgroup T rounded[512];\n"
        let loop = "for (int row = 0; row < L; ++row) {\n"
        guard tailShuffle, tailOneBarrier, source.components(separatedBy: head).count == 2,
              let start = source.range(of: loop) else { return nil }
        return String(source[..<start.lowerBound]).replacingOccurrences(of: head, with:
            "            threadgroup float nst_sums[L][4];\n") + """
                        float nst_x[L][4];
                        for (int row = 0; row < L; ++row) {
                            int64_t feature = int64_t(grp) * 512 + int64_t(lid) * 4;
                            float acc = 0;
                            if (on) {
                                for (int i = 0; i < 4; i++) {
                                    int64_t f = feature + int64_t(i);
                                    T g = proj[int64_t(row) * PS + f];
                                    auto y = 1 / (1 + metal::exp(metal::abs(g)));
                                    T sig = g < 0 ? y : 1 - y;
                                    T activated = g * sig;
                                    T gated = out[int64_t(row) * (H * Dh) + f] * activated;
                                    nst_x[row][i] = gated;
                                    acc += nst_x[row][i] * nst_x[row][i];
                                }
                            }
                            acc = simd_sum(acc);
                            if (on && simd_lane_id == 0) {
                                nst_sums[row][simd_group_id] = acc;
                            }
                        }
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                        if (on) {
                            for (int row = 0; row < L; ++row) {
                                const uint gid = uint(row) * uint(H * Dh / 512) + grp;
                                int64_t feature = int64_t(grp) * 512 + int64_t(lid) * 4;
                                const float total_sq = simd_sum(simd_lane_id < 4 ? nst_sums[row][simd_lane_id] : 0.0f);
                                const float inv_mean = metal::precise::rsqrt(total_sq / 512 + eps);
                                T w[4];
                                for (int i = 0; i < 4; i++) {
                                    int64_t f = feature + int64_t(i);
                                    T normalized = static_cast<T>(nst_x[row][i] * inv_mean);
                                    T unweighted = identity * normalized;
                                    T weighted = weight[f] * unweighted;
                                    normed[int64_t(row) * (H * Dh) + f] = weighted;
                                    w[i] = weighted;
                                }
                                const float part = float(w[0]) + float(w[1]) + float(w[2]) + float(w[3]);
                                const float p1 = simd_shuffle_down(part, ushort(1));
                                const float p2 = simd_shuffle_down(part, ushort(2));
                                const float p3 = simd_shuffle_down(part, ushort(3));
                                const int k = int(lid & 3u);
                                const int base = int(gid) * 17 * 32 + int(lid >> 2);
                                prepared[base + (4 * k + 0) * 32] = float(w[0]);
                                prepared[base + (4 * k + 1) * 32] = float(w[1]) / 16.0f;
                                prepared[base + (4 * k + 2) * 32] = float(w[2]) / 256.0f;
                                prepared[base + (4 * k + 3) * 32] = float(w[3]) / 4096.0f;
                                if (k == 0) {
                                    float total = 0;
                                    total += part;
                                    total += p1;
                                    total += p2;
                                    total += p3;
                                    prepared[base + 16 * 32] = total;
                                }
                            }
                        }

            """
    }
    static let kernel = MLXFast.metalKernel(
        name: "nemotron_mtp_ssm_compact_gated_prepared",
        inputNames: ["conv", "A_log", "D", "dt", "state_in", "proj", "weight", "identity",
                     "eps", "counter"],
        outputNames: ["out", "state_out", "normed", "prepared"],
        source: source, mutableInputs: ["counter"])
    static let rowsKernel = rowsTail(source).map {
        MLXFast.metalKernel(
            name: "nemotron_mtp_ssm_compact_gated_prepared_sh1b",
            inputNames: ["conv", "A_log", "D", "dt", "state_in", "proj", "weight", "identity",
                         "eps", "counter"],
            outputNames: ["out", "state_out", "normed", "prepared"],
            source: $0, mutableInputs: ["counter"])
    }
    static let shuffleKernel = shuffledTail(source).map {
        MLXFast.metalKernel(
            name: "nemotron_mtp_ssm_compact_gated_prepared_sh",
            inputNames: ["conv", "A_log", "D", "dt", "state_in", "proj", "weight", "identity",
                         "eps", "counter"],
            outputNames: ["out", "state_out", "normed", "prepared"],
            source: $0, mutableInputs: ["counter"])
    }
    /// NEMOTRON_MTP_SSM_PENDING_FOLD (see NemotronMTPPendingSSM): the same
    /// launch with `P` pending strict-prefix steps of earlier windows (`pconv`
    /// [1, P, CD], `pdt` [1, P, H]) run first from `state_in`, with the
    /// replay kernel's per-step statements (no y, no store), then the window
    /// from the resulting state exactly as above.
    static let pendingSource: String = {
        let loop = "\nfor (int t = 0; t < L; ++t) {\n"
        precondition(source.components(separatedBy: loop).count == 2, "pending ssm source")
        let prelude = """
                    for (int t = 0; t < P; ++t) {
                        auto x_ = static_cast<float>(pconv[t * CD + n * Dh + d_idx]);
                        auto B_ = pconv + t * CD + BO + g_idx * Ds;
                        auto dt_ = static_cast<float>(pdt[t * H + n]);
                        auto A = -fast::exp(static_cast<float>(A_log[h_idx]));
                        auto dA = fast::exp(A * dt_);
                        for (int i = 0; i < n_per_t; ++i) {
                            auto s_idx = n_per_t * ds_idx + i;
                            if constexpr (Ds % 32 != 0) { if (s_idx >= Ds) continue; }
                            auto dB_by_x = x_ * dt_ * static_cast<float>(B_[s_idx]);
                            auto state = dA * prior[i] + dB_by_x;
                            prior[i] = static_cast<U>(state);
                        }
                    }

        """
        return source.replacingOccurrences(of: loop, with: "\n" + prelude + loop)
    }()
    static let pendingKernel = MLXFast.metalKernel(
        name: "nemotron_mtp_ssm_compact_gated_prepared_pending",
        inputNames: ["conv", "A_log", "D", "dt", "state_in", "proj", "weight", "identity",
                     "eps", "counter", "pconv", "pdt"],
        outputNames: ["out", "state_out", "normed", "prepared"],
        source: pendingSource, mutableInputs: ["counter"])
    static let shufflePendingKernel = shuffledTail(pendingSource).map {
        MLXFast.metalKernel(
            name: "nemotron_mtp_ssm_compact_gated_prepared_pending_sh",
            inputNames: ["conv", "A_log", "D", "dt", "state_in", "proj", "weight", "identity",
                         "eps", "counter", "pconv", "pdt"],
            outputNames: ["out", "state_out", "normed", "prepared"],
            source: $0, mutableInputs: ["counter"])
    }
    static let rowsPendingKernel = rowsTail(pendingSource).map {
        MLXFast.metalKernel(
            name: "nemotron_mtp_ssm_compact_gated_prepared_pending_sh1b",
            inputNames: ["conv", "A_log", "D", "dt", "state_in", "proj", "weight", "identity",
                         "eps", "counter", "pconv", "pdt"],
            outputNames: ["out", "state_out", "normed", "prepared"],
            source: $0, mutableInputs: ["counter"])
    }
    /// Set at load by `qualifyShuffle`.
    nonisolated(unsafe) private(set) static var shuffleQualified = false
    /// Set at load by `qualifyRowsTail`.
    nonisolated(unsafe) private(set) static var rowsTailQualified = false

    /// Load time, after `qualifyShuffle` admitted the shuffle tail: the one-barrier tail launches against the
    /// shuffle-tail launches on synthetic operands with `mixer`'s parameters (0 ... maxSteps pending steps,
    /// three seeds), adopted only if the normalized rows, the prepared activation and the final state are
    /// bitwise equal in every case.
    static func qualifyRowsTail(_ mixer: NemotronHMamba2Mixer) {
        rowsTailQualified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron ssm tail one barrier: \(message)\n".utf8))
        }
        guard tailOneBarrier else { return report("off (setting)") }
        guard shuffleQualified, rowsKernel != nil, rowsPendingKernel != nil else {
            return report("off (shuffle tail off)")
        }
        let (heads, dim, groups, stateDim) = (mixer.numHeads, mixer.headDim, mixer.numGroups, mixer.ssmStateSize)
        func bits(_ x: MLXArray) -> MLXArray {
            x.dtype == .bfloat16 ? x.view(dtype: .uint16) : x.view(dtype: .uint32)
        }
        var same = MLXArray(true)
        var cases = 0
        do {
            try withError {
                for seed in 0 ..< 3 {
                    for steps in 0 ... NemotronMTPPendingSSM.maxSteps {
                        let keys = MLXRandom.split(key: MLXRandom.key(UInt64(0x55B1 + 16 * seed + steps)), into: 6)
                        let scale = Float(1 + 3 * seed)
                        let conv = (MLXRandom.normal([1, 3, mixer.convDim], key: keys[0]) * scale).asType(.bfloat16)
                        let dt = MLXRandom.uniform(Float(0.001) ..< 0.2, [1, 3, heads], key: keys[1])
                        let state = MLXRandom.normal([1, heads, dim, stateDim], key: keys[2]) * 0.1
                        let projected = (MLXRandom.normal(
                            [1, 3, mixer.intermediateSize + mixer.convDim + heads], key: keys[3]) * scale)
                            .asType(.bfloat16)
                        let pending: (conv: MLXArray, dt: MLXArray, steps: Int)? = steps == 0 ? nil : (
                            MLXRandom.normal([1, steps, mixer.convDim], key: keys[4]).asType(.bfloat16),
                            MLXRandom.uniform(Float(0.001) ..< 0.2, [1, steps, heads], key: keys[5]), steps)
                        func run(_ rows: Bool) -> (MLXArray, MLXArray, MLXArray)? {
                            mixer.norm.compactSSMPreparedRows(
                                convolution: conv, ALog: mixer.aLog, D: mixer.D, dt: dt, state: state,
                                projected: projected, heads: heads, headDim: dim, groups: groups,
                                stateDim: stateDim, pending: pending, shuffle: true, rowsTail: rows)
                        }
                        guard let a = run(false), let b = run(true) else {
                            same = MLXArray(false); return
                        }
                        for (x, y) in [(a.0, b.0), (a.1, b.1), (a.2, b.2)] {
                            same = logicalAnd(same, (bits(x) .== bits(y)).all())
                        }
                        eval(same)
                        cases += 1
                    }
                }
            }
        } catch {
            return report("off (\(error))")
        }
        rowsTailQualified = same.item(Bool.self)
        report(rowsTailQualified
            ? "on (\(cases) cases: normalized rows, prepared activation and state bitwise equal to the shuffle tail)"
            : "off (differs from the shuffle-tail launch)")
    }

    /// Load time, after `prepareCounter`: runs the record and the one-barrier tail launches (no pending steps
    /// and two pending steps) on synthetic operands with `mixer`'s parameters and adopts the tail only if the
    /// normalized rows, the prepared activation and the final state are bitwise equal.
    static func qualifyShuffle(_ mixer: NemotronHMamba2Mixer) {
        shuffleQualified = false
        guard tailShuffle, enabled, counter != nil, shuffleKernel != nil, shufflePendingKernel != nil else { return }
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron ssm tail shuffle: \(message)\n".utf8))
        }
        let (heads, dim, groups, stateDim) = (mixer.numHeads, mixer.headDim, mixer.numGroups, mixer.ssmStateSize)
        let keys = MLXRandom.split(key: MLXRandom.key(0x55A1), into: 6)
        let conv = MLXRandom.normal([1, 3, mixer.convDim], key: keys[0]).asType(.bfloat16)
        let dt = MLXRandom.uniform(Float(0.001) ..< 0.2, [1, 3, heads], key: keys[1])
        let state = MLXRandom.normal([1, heads, dim, stateDim], key: keys[2]) * 0.1
        let projected = MLXRandom.normal(
            [1, 3, mixer.intermediateSize + mixer.convDim + heads], key: keys[3]).asType(.bfloat16)
        let pconv = MLXRandom.normal([1, 2, mixer.convDim], key: keys[4]).asType(.bfloat16)
        let pdt = MLXRandom.uniform(Float(0.001) ..< 0.2, [1, 2, heads], key: keys[5])
        func bits(_ x: MLXArray) -> MLXArray {
            x.dtype == .bfloat16 ? x.view(dtype: .uint16) : x.view(dtype: .uint32)
        }
        var same = MLXArray(true)
        for pending in [nil, (conv: pconv, dt: pdt, steps: 2)] as [(conv: MLXArray, dt: MLXArray, steps: Int)?] {
            func run(_ shuffle: Bool) -> (MLXArray, MLXArray, MLXArray)? {
                mixer.norm.compactSSMPreparedRows(
                    convolution: conv, ALog: mixer.aLog, D: mixer.D, dt: dt, state: state,
                    projected: projected, heads: heads, headDim: dim, groups: groups, stateDim: stateDim,
                    pending: pending, shuffle: shuffle)
            }
            guard let a = run(false), let b = run(true) else { return report("off (launch declined)") }
            for (x, y) in [(a.0, b.0), (a.1, b.1), (a.2, b.2)] {
                same = logicalAnd(same, (bits(x) .== bits(y)).all())
            }
            eval(same)
        }
        shuffleQualified = same.item(Bool.self)
        report(shuffleQualified
            ? "on (normalized rows, prepared activation and state bitwise equal, 0 and 2 pending steps)"
            : "off (differs from the record launch)")
    }
    /// Per-norm-group arrival counters; every launch leaves them at zero.
    /// `counter` is a mutable input, so MLX's encoder orders any two launches
    /// that share it (a barrier), even without a data edge between them.
    /// Launches are ordered (each layer's conv depends on the previous layer).
    nonisolated(unsafe) static var counter: MLXArray?

    /// Load time, outside any compile trace: the zeroed counters.
    static func prepareCounter() {
        guard enabled, counter == nil else { return }
        let c = MLXArray.zeros([64], dtype: .uint32)
        eval(c)
        counter = c
    }
}

extension NemotronHRMSNormGated {
    /// nemotronMTPSSMCompact(steps = writeOutput = 3) + preparedRows in one
    /// launch, under preparedRows' guards; returns (normalized, prepared, final
    /// SSM state) or nil to keep the two launches.
    func compactSSMPreparedRows(
        convolution: MLXArray, ALog: MLXArray, D: MLXArray, dt: MLXArray,
        state: MLXArray, projected: MLXArray, heads: Int, headDim: Int, groups: Int,
        stateDim: Int, pending: (conv: MLXArray, dt: MLXArray, steps: Int)? = nil, shuffle: Bool? = nil,
        rowsTail: Bool? = nil
    ) -> (MLXArray, MLXArray, MLXArray)? {
        if let pending {
            guard (1 ... NemotronMTPPendingSSM.maxSteps).contains(pending.steps),
                  pending.conv.shape == [1, pending.steps, 6144], pending.conv.dtype == .bfloat16,
                  pending.dt.shape == [1, pending.steps, 64], pending.dt.dtype == .float32
            else { return nil }
        }
        guard NemotronHCompactSSMGated.enabled, NemotronMTPExecution.batchedNorm,
              NemotronMultiRowQMV.enabled, NemotronMultiRowQMV.narrowEnabled,
              Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              type(of: self) == NemotronHRMSNormGated.self,
              groupSize == 512, heads == 64, headDim == 64, groups == 8, stateDim == 128,
              convolution.shape == [1, 3, 6144], convolution.dtype == .bfloat16,
              dt.dtype == .float32, dt.shape == [1, 3, 64],
              state.dtype == .float32, state.shape == [1, 64, 64, 128],
              projected.shape == [1, 3, 10304], projected.dtype == .bfloat16,
              weight.shape == [4096], weight.dtype == .bfloat16
        else { return nil }
        // Created and evaluated at load (`prepareCounter`), never inside a
        // compile trace; without it the two launches stay.
        guard NemotronHCompactSSMGated.counter != nil else { return nil }
        var inputs: [any ScalarOrArray] = [
            convolution, ALog, D, dt, state, projected, weight, NemotronHGatedRMSKernel.identity,
            eps, NemotronHCompactSSMGated.counter!]
        var template: [(String, any KernelTemplateArg)] = [
            ("T", DType.bfloat16), ("U", DType.float32),
            ("Dh", 64), ("Ds", 128), ("H", 64), ("G", 8), ("L", 3),
            ("CD", 6144), ("BO", 4096), ("CO", 5120), ("PS", 10304)]
        let tail = shuffle ?? NemotronHCompactSSMGated.shuffleQualified
        // The one-barrier tail (NEMOTRON_SSM_TAIL_ONE_BARRIER) once qualified; `shuffle` set by a load-time
        // check keeps the kernel it names unless `rowsTail` asks for the one-barrier one.
        let rows = rowsTail ?? (shuffle == nil && NemotronHCompactSSMGated.rowsTailQualified)
        var kernel = (rows ? NemotronHCompactSSMGated.rowsKernel : nil)
            ?? (tail ? NemotronHCompactSSMGated.shuffleKernel : nil) ?? NemotronHCompactSSMGated.kernel
        if let pending {
            inputs += [pending.conv, pending.dt]
            template.append(("P", pending.steps))
            kernel = (rows ? NemotronHCompactSSMGated.rowsPendingKernel : nil)
                ?? (tail ? NemotronHCompactSSMGated.shufflePendingKernel : nil)
                ?? NemotronHCompactSSMGated.pendingKernel
        }
        let outputs = kernel(
            inputs, template: template,
            grid: (32, 64, 64), threadGroup: (32, 8, 1),
            outputShapes: [[1, 3, 64, 64], [1, 64, 64, 128], [1, 3, 4096], [3, 8, 17 * 32]],
            outputDTypes: [.bfloat16, .float32, .bfloat16, .float32])
        return (outputs[2], outputs[3], outputs[1])
    }
}


// MARK: - Prefill x5 switches

/// Prefill x5: stable counting sort with expert-run tiles, expert-run NAX
/// gather, the fused segsum scan and the fused scan-window output.
/// NEMOTRON_PREFILL_X5=0 restores the argsort / stock gather / masked-decay
/// graph.
enum NemotronHPrefillX5 {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_X5"] != "0"
    /// The run gather clones the NAX kernel; without NAX stock runs another one.
    static let runGather = enabled && GPU.gemma4ExpertQMMDiagnostics().naxAvailable
    /// The state update `product + exp(cumsum)[-1] * previous` is one kernel.
    /// `NEMOTRON_PREFILL_SSM_STATE_UPDATE=0` restores the stock pair.
    static let ssmStateUpdate = enabled
        && ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_SSM_STATE_UPDATE"] != "0"
    /// `dtx * decay` for the state product is written in the GEMM's layout
    /// by one kernel. `NEMOTRON_PREFILL_SSM_DTX_DECAY=0` restores the stock pair.
    static let ssmDtxDecay = enabled
        && ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_SSM_DTX_DECAY"] != "0"
    /// The SSM window-output kernel also applies `+ x * D` and writes the
    /// concatenated output. `NEMOTRON_PREFILL_SSM_EPILOGUE=0` restores the
    /// separate concatenate, multiply and add.
    static let ssmEpilogue = enabled
        && ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_SSM_EPILOGUE"] != "0"
    /// The four kernels above and the prefill router select run only inside the
    /// engine's request prefill (the seed of the scored decode); the stepper's
    /// prefill keeps the base graph. `NEMOTRON_PREFILL_SEED_ONLY=0` uses them on
    /// every multi-token prefill.
    static let seedOnly = ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_SEED_ONLY"] != "0"
    nonisolated(unsafe) static var requestPrefill = false
    static var active: Bool { !seedOnly || requestPrefill }
}

/// Seed request prefill only: the first layer's input projection (with the
/// embedding gather and norm ahead of it) is committed as its own command
/// buffer, before the rest of that layer is encoded. The layer's remaining
/// command buffer then starts behind the projection instead of on an idle
/// GPU. Evaluation order only; values are unchanged. The teacher-forced
/// stepper never sets `requestPrefill`, so its prefill is untouched.
/// NEMOTRON_SEED_EARLY_COMMIT=0 keeps one command buffer for the layer.
enum NemotronSeedEarlyCommit {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_SEED_EARLY_COMMIT"] != "0"

    static func applies(modelLayerIndex: Int, rows: Int) -> Bool {
        enabled && modelLayerIndex == 0 && rows > 8 && NemotronHPrefillX5.requestPrefill
    }
}

/// Seed-only: a fresh request's first scan window computes
/// `next + lastDecay * (+0.0)` (a [b, h, 1, 1] multiply, then a broadcast
/// add) through `stateUpdate` with the zero-stride +0.0 state, i.e. the same
/// `e * 0` and add per element in one launch. `qualify` checks it bitwise
/// against the stock pair (signed zeros, infinities and NaN included).
/// NEMOTRON_FRESH_STATE_UPDATE=0 keeps the stock pair.
enum NemotronFreshStateUpdate {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_FRESH_STATE_UPDATE"] != "0"
    nonisolated(unsafe) static var verified = false

    static func qualify() {
        verified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron fresh state update: \(message)\n".utf8))
        }
        guard enabled, NemotronHPrefillX5.ssmStateUpdate else { return report("off (setting)") }
        let (h, dh, d) = (64, 64, 128)
        var checked = 0
        do {
            let matches = try withError { () -> Bool in
                for (index, n) in [256, 232, 9, 1].enumerated() {
                    let seed = UInt64(0x5f7e_0000 + index)
                    var prod = MLXRandom.normal([1, h, dh, d], dtype: .float32, scale: 3,
                                                key: MLXRandom.key(seed))
                    // Special values in the product and the decay.
                    let specials = MLXArray([Float(-0.0), 0, .infinity, -.infinity, .nan, Float(bitPattern: 0x0001_0000), Float(bitPattern: 0x8001_0000), 3e38])
                    prod = concatenated([specials, prod.reshaped(-1)[8...]]).reshaped(1, h, dh, d)
                    var ec = MLXRandom.uniform(0 ..< 1, [1, n, h], key: MLXRandom.key(seed + 1))
                    ec = concatenated([ec.reshaped(-1)[..<(n * h - 4)],
                                       MLXArray([Float(0), .infinity, .nan, 1e-30])]).reshaped(1, n, h)
                    let zero = NemotronMambaZeroState.view([1, h, dh, d], dtype: .float32)
                    let fused = NemotronHScanKernels.shared.stateUpdate(
                        [prod, zero, ec, n],
                        template: [("H", h), ("DH", dh), ("DS", d)],
                        grid: (dh * d / 4, h, 1), threadGroup: (256, 1, 1),
                        outputShapes: [[1, h, dh, d]], outputDTypes: [.float32])[0]
                    let lastDecay = ec[0..., -1, 0..., .newAxis, .newAxis]
                    let stock = prod + lastDecay * NemotronMambaZeroState.scalar(.float32)
                    let same = (fused.view(dtype: .uint32) .== stock.view(dtype: .uint32)).all()
                    if !same.item(Bool.self) { return false }
                    checked += 1
                }
                return true
            }
            verified = matches
        } catch {
            return report("off (\(error))")
        }
        report(verified ? "on (\(checked) cases bitwise equal)"
                        : "off (outputs differ from the stock pair after \(checked) cases)")
    }
}

/// Seed-only (request prefill) fold of the scan windows' cumsum / exp
/// launches. NEMOTRON_PREFILL_SCAN_EXP=0 keeps the stock launches.
enum NemotronPrefillScanExp {
    static let scanExpEnabled = ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_SCAN_EXP"] != "0"
    nonisolated(unsafe) static var scanExpVerified = false

    /// `exp(cumsum(dtA[window], axis: -2))` for every 256-row scan window in
    /// one launch: per (window, head) column, MLX strided_scan's statements
    /// (32-row blocks from the window's first row, simd_prefix_inclusive_sum,
    /// `+ prefix` with prefix starting at +0.0, carried by simd_shuffle of
    /// lane 31; rows past the window read 0), then Exp's metal::precise::exp.
    /// Row r of the output holds its own window's value.
    static let scanExp = MLXFast.metalKernel(
        name: "nemotron_prefill_window_cumsum_exp",
        inputNames: ["dtA", "L"], outputNames: ["out"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const int head = int(threadgroup_position_in_grid.y);
            const int start = int(threadgroup_position_in_grid.z) * STEP;
            const int end = min(start + STEP, L);
            float prefix = 0.0f;
            for (int r0 = start; r0 < end; r0 += 32) {
                const int i = r0 + int(lane);
                float v = i < end
                    ? dtA[i * dtA_strides[1] + head * dtA_strides[2]] : 0.0f;
                v = simd_prefix_inclusive_sum(v);
                v = v + prefix;
                prefix = simd_shuffle(v, 31);
                if (i < end) {
                    out[size_t(i) * H + head] = metal::precise::exp(v);
                }
            }
            """, ensureRowContiguous: false)

    static func applyScanExp(_ dtA: MLXArray, step: Int) -> MLXArray? {
        guard scanExpVerified, dtA.ndim == 3, dtA.dim(0) == 1, dtA.dtype == .float32, step == 256
        else { return nil }
        let l = dtA.dim(1), h = dtA.dim(2)
        return scanExp(
            [dtA, l], template: [("H", h), ("STEP", step)],
            grid: (32, h, (l + step - 1) / step), threadGroup: (32, 1, 1),
            outputShapes: [[1, l, h]], outputDTypes: [.float32])[0]
    }

    static func qualify() {
        scanExpVerified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron prefill scan exp: \(message)\n".utf8))
        }
        guard NemotronHPrefillX5.enabled else { return report("off (setting)") }
        var scanChecked = 0
        if scanExpEnabled {
            do {
                scanExpVerified = try withError { () -> Bool in
                    for (index, l) in [488, 300, 256, 257, 9, 512].enumerated() {
                        let seed = UInt64(0x5ca0_0000 + index)
                        // dt (softplus-range, some large) times A (negative).
                        let dt = MLXRandom.uniform(0 ..< 2, [1, l, 64], key: MLXRandom.key(seed))
                        let a = -MLX.exp(MLXRandom.normal([64], key: MLXRandom.key(seed + 1)))
                        let dtA = (dt * a.reshaped(1, 1, -1)).asType(.float32)
                        guard let fused = applyScanExpUnchecked(dtA) else { return false }
                        var same = MLXArray(true)
                        for start in stride(from: 0, to: l, by: 256) {
                            let end = min(start + 256, l)
                            let stock = MLX.exp(MLX.cumsum(dtA[0..., start ..< end, 0...], axis: -2))
                            same = same .&& (fused[0..., start ..< end, 0...].view(dtype: .uint32)
                                .== stock.view(dtype: .uint32)).all()
                        }
                        if !same.item(Bool.self) { return false }
                        scanChecked += 1
                    }
                    return true
                }
            } catch {
                scanExpVerified = false
            }
        }
        report("\(scanExpVerified ? "on" : "off") (\(scanChecked) cases bitwise equal)")
    }

    private static func applyScanExpUnchecked(_ dtA: MLXArray) -> MLXArray? {
        let saved = scanExpVerified
        scanExpVerified = true
        defer { scanExpVerified = saved }
        return applyScanExp(dtA, step: 256)
    }
}

/// Seed-only: at the early-eval layer boundaries the block's `x + output`
/// and the next block's RMSNorm run as `NemotronHResidualRMSNorm.apply`
/// (the join every other layer already uses) instead of the stock add and
/// rms launches. `qualify` checks the join against the stock pair with the
/// target's own norm weights. NEMOTRON_PREFILL_BOUNDARY_JOIN=0 keeps the pair.
enum NemotronPrefillBoundaryJoin {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_BOUNDARY_JOIN"] != "0"
    nonisolated(unsafe) static var verified = false

    static func qualify(_ target: Module) {
        verified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron prefill boundary join: \(message)\n".utf8))
        }
        guard enabled, NemotronHPrefillX5.enabled else { return report("off (setting)") }
        let norms = target.modules().compactMap { $0 as? RMSNorm }
            .filter { type(of: $0) == RMSNorm.self && $0.weight.shape == [2688] }
        guard !norms.isEmpty else { return report("off (no 2688 RMSNorm)") }
        var checked = 0
        do {
            let matches = try withError { () -> Bool in
                for (index, norm) in norms.enumerated() where index % 4 == 0 || index == norms.count - 1 {
                    for rows in [488, 9] {
                        let seed = UInt64(0xb7a0_0000 + index * 8 + rows)
                        let x = MLXRandom.normal([1, rows, 2688], dtype: .bfloat16, scale: 4,
                                                 key: MLXRandom.key(seed))
                        let y = MLXRandom.normal([1, rows, 2688], dtype: .bfloat16, scale: 2,
                                                 key: MLXRandom.key(seed + 1))
                        guard let joined = NemotronHResidualRMSNorm.apply(x, y, norm: norm) else { return false }
                        let sum = x + y
                        let same = MLX.logicalAnd(
                            (joined.0.view(dtype: .uint16) .== sum.view(dtype: .uint16)).all(),
                            (joined.1.view(dtype: .uint16) .== norm(sum).view(dtype: .uint16)).all())
                        if !same.item(Bool.self) { return false }
                        checked += 1
                    }
                }
                return true
            }
            verified = matches
        } catch {
            return report("off (\(error))")
        }
        report(verified ? "on (\(checked) cases bitwise equal)"
                        : "off (join differs from add + rms after \(checked) cases)")
    }
}

// MARK: - Multi-token SSM scan (copy of SSM.swift ssmAttn, fewer copies)

/// tril(CB_g * exp(S), 0) with CB broadcast over the R heads of its group:
/// exactly the stock where(i >= j, astype(repeat(CB), f32) * Exp(S), 0)
/// (Exp is metal::precise::exp) without the [B, H, L, L] repeat, promotion,
/// exp and product intermediates.
private final class NemotronHScanKernels: Sendable {
    static let shared = NemotronHScanKernels()
    let maskedDecay: MLXFast.MLXFastKernel
    let scanDecay: MLXFast.MLXFastKernel
    let windowOut: MLXFast.MLXFastKernel
    let stateUpdate: MLXFast.MLXFastKernel
    let dtxDecayT: MLXFast.MLXFastKernel
    let dtxDecayT4: MLXFast.MLXFastKernel
    let windowOutFinal: MLXFast.MLXFastKernel
    let windowOutFresh: MLXFast.MLXFastKernel
    let windowOutFinalSG: MLXFast.MLXFastKernel
    /// `windowOutFinalSG`'s source (also the base of `NemotronPrefillFreshStateFold.windowOut`).
    static let windowOutFinalSGSource = """
                constexpr int RB = SRB;
                constexpr int TB = 32 / RB;
                
                const uint lane = thread_index_in_simdgroup;
                const uint sg = simdgroup_index_in_threadgroup;
                const uint bh = threadgroup_position_in_grid.z;
                const uint bb = bh / H;
                const uint hh = bh % H;
                const int nb = (n + TC - 1) / TC;
                if (int(threadgroup_position_in_grid.y) >= nb) {
                    if (COPY) {
                        const int cb = int(threadgroup_position_in_grid.y) - nb;
                        constexpr int RT = RB * SGS;
                        for (int q = int(sg) * 32 + int(lane); q < TC * RT; q += 32 * SGS) {
                            const int t = cb * TC + q / RT;
                            const int i = int(threadgroup_position_in_grid.x) * RT + q % RT;
                            if (t < s0) {
                                out[((size_t(bb) * LT + t) * H + hh) * DH + i] =
                                    P[bb * P_strides[0] + t * P_strides[1] + hh * P_strides[2]
                                      + i * P_strides[3]];
                            }
                        }
                    }
                    return;
                }
                const int i0 = (int(threadgroup_position_in_grid.x) * SGS + int(sg)) * RB;
                const int k0 = int(lane) * 4;
                const int t0 = int(threadgroup_position_in_grid.y) * TC;
                const int t1 = min(t0 + TC, n);
                const device float* sp = state + bb * state_strides[0] + hh * state_strides[1];
                float st[RB][4];
                for (int r = 0; r < RB; ++r) {
                    for (int tn = 0; tn < 4; ++tn) {
                        st[r][tn] = sp[(i0 + r) * state_strides[2] + (k0 + tn) * state_strides[3]];
                    }
                }
                const device CT* cp = C + bb * C_strides[0] + (hh / R) * C_strides[2];
                const int orow = int(lane) / TB;
                const int ot = int(lane) % TB;
                const float dv = static_cast<float>(Dw[hh * Dw_strides[0]]);
                for (int tb = t0; tb < t1; tb += TB) {
                    // v[r * TB + j]: this lane's partial of output (row r, token j).
                    float v[32];
                    for (int j = 0; j < TB; ++j) {
                        const int t = min(tb + j, n - 1);
                        float cv[4];
                        if (C_strides[3] == 1) {
                            const vec<CT, 4> c4 = *((const device vec<CT, 4>*)(cp + t * C_strides[1] + k0));
                            cv[0] = static_cast<float>(c4.x);
                            cv[1] = static_cast<float>(c4.y);
                            cv[2] = static_cast<float>(c4.z);
                            cv[3] = static_cast<float>(c4.w);
                        } else {
                            for (int tn = 0; tn < 4; ++tn) {
                                cv[tn] = static_cast<float>(cp[t * C_strides[1] + (k0 + tn) * C_strides[3]]);
                            }
                        }
                        for (int r = 0; r < RB; ++r) {
                            float acc = 0.0f;
                            for (int tn = 0; tn < 4; ++tn) {
                                acc += st[r][tn] * cv[tn];
                            }
                            v[r * TB + j] = acc;
                        }
                    }
                    // windowOutFinal's tree (lane offsets 16, 8, 4, 2, 1) as a
                    // butterfly: at offset m a lane keeps the outputs whose bit m
                    // matches its own and adds its partner's partial of each, so
                    // every add pairs the same lanes; lane L ends with output L.
                    float w16[16];
                    const bool b16 = (lane & 16) != 0;
                    for (int q = 0; q < 16; ++q) {
                        const float keep = b16 ? v[q + 16] : v[q];
                        const float send = b16 ? v[q] : v[q + 16];
                        w16[q] = keep + simd_shuffle_xor(send, 16);
                    }
                    float w8[8];
                    const bool b8 = (lane & 8) != 0;
                    for (int q = 0; q < 8; ++q) {
                        const float keep = b8 ? w16[q + 8] : w16[q];
                        const float send = b8 ? w16[q] : w16[q + 8];
                        w8[q] = keep + simd_shuffle_xor(send, 8);
                    }
                    float w4[4];
                    const bool b4 = (lane & 4) != 0;
                    for (int q = 0; q < 4; ++q) {
                        const float keep = b4 ? w8[q + 4] : w8[q];
                        const float send = b4 ? w8[q] : w8[q + 4];
                        w4[q] = keep + simd_shuffle_xor(send, 4);
                    }
                    float w2[2];
                    const bool b2 = (lane & 2) != 0;
                    for (int q = 0; q < 2; ++q) {
                        const float keep = b2 ? w4[q + 2] : w4[q];
                        const float send = b2 ? w4[q] : w4[q + 2];
                        w2[q] = keep + simd_shuffle_xor(send, 2);
                    }
                    const bool b1 = (lane & 1) != 0;
                    const float keep1 = b1 ? w2[1] : w2[0];
                    const float send1 = b1 ? w2[0] : w2[1];
                    const float yp = keep1 + simd_shuffle_xor(send1, 1);
                    const int t = tb + ot;
                    if (t < t1) {
                        const float e = ec[bb * ec_strides[0] + t * ec_strides[1] + hh * ec_strides[2]];
                        const float p = e * yp;
                        const float yv = y[bb * y_strides[0] + hh * y_strides[1] + t * y_strides[2]
                            + (i0 + orow) * y_strides[3]];
                        const float u = yv + p;
                        const OT w = static_cast<OT>(u);
                        const float xv = static_cast<float>(X[bb * X_strides[0] + (xs + t) * X_strides[1]
                            + hh * X_strides[2] + (i0 + orow) * X_strides[3]]);
                        const OT xd = static_cast<OT>(xv * dv);
                        const OT o = static_cast<OT>(static_cast<float>(w) + static_cast<float>(xd));
                        out[((size_t(bb) * LT + s0 + t) * H + hh) * DH + i0 + orow] = o;
                    }
                }
            """
    let scanDecaySkip: MLXFast.MLXFastKernel
    private init() {
        // (dtx * decay) written as the contiguous [B, H, DH, n] array that
        // `.swappedAxes(1, 2).swappedAxes(2, 3).reshaped(...)` copies it into:
        // the same single fp32 product per element, one pass instead of the
        // product and the reshape copy. A 32 x 32 (token, row) tile per
        // threadgroup keeps both the read and the transposed write coalesced.
        // dtxDecayT over a 32-token x 64-row tile per 256-thread group, the
        // dtx rows read and the output rows written as float4 (contiguous
        // inner strides only): the same single fp32 product per element.
        dtxDecayT4 = MLXFast.metalKernel(
            name: "nemotron_ssm_dtx_decay_t4",
            inputNames: ["dtx", "last", "n"], outputNames: ["out"],
            source: """
                threadgroup float tile[32][65];
                const uint tid = thread_position_in_threadgroup.x;
                const int tb = int(threadgroup_position_in_grid.x) * 32;
                const uint bh = threadgroup_position_in_grid.z;
                const uint bb = bh / H;
                const uint hh = bh % H;
                // Read: 16 threads cover a token's 64 rows as float4; 16 tokens per pass.
                for (int pass = 0; pass < 2; ++pass) {
                    const int tl = int(tid / 16) + 16 * pass;
                    const int i4 = int(tid % 16) * 4;
                    const int t = tb + tl;
                    float4 v = float4(0.0f);
                    if (t < n) {
                        const float4 a = *((const device float4*)(dtx + bb * dtx_strides[0] + t * dtx_strides[1]
                            + hh * dtx_strides[2] + i4));
                        const float e = last[bb * last_strides[0] + hh * last_strides[1]
                            + t * last_strides[3]];
                        v = float4(a.x * e, a.y * e, a.z * e, a.w * e);
                    }
                    tile[tl][i4 + 0] = v.x;
                    tile[tl][i4 + 1] = v.y;
                    tile[tl][i4 + 2] = v.z;
                    tile[tl][i4 + 3] = v.w;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                // Write: 8 threads cover a row's 32 tokens as float4; 32 rows per pass.
                for (int pass = 0; pass < 2; ++pass) {
                    const int i = int(tid / 8) + 32 * pass;
                    const int t4 = int(tid % 8) * 4;
                    const int t = tb + t4;
                    device float* o = out + (size_t(bh) * DH + i) * n + t;
                    if (t + 3 < n) {
                        *((device float4*)o) = float4(tile[t4][i], tile[t4 + 1][i], tile[t4 + 2][i], tile[t4 + 3][i]);
                    } else {
                        for (int u = 0; u < 4; ++u) {
                            if (t + u < n) {
                                o[u] = tile[t4 + u][i];
                            }
                        }
                    }
                }
            """, ensureRowContiguous: false)
        dtxDecayT = MLXFast.metalKernel(
            name: "nemotron_ssm_dtx_decay_t",
            inputNames: ["dtx", "last", "n"], outputNames: ["out"],
            source: """
                threadgroup float tile[32][33];
                const uint tx = thread_position_in_threadgroup.x;
                const uint ty = thread_position_in_threadgroup.y;
                const int tb = int(threadgroup_position_in_grid.x) * 32;
                const int ib = int(threadgroup_position_in_grid.y) * 32;
                const uint bh = threadgroup_position_in_grid.z;
                const uint bb = bh / H;
                const uint hh = bh % H;
                for (int k = 0; k < 4; ++k) {
                    const int t = tb + int(ty) + 8 * k;
                    const int i = ib + int(tx);
                    float v = 0.0f;
                    if (t < n) {
                        const float a = dtx[bb * dtx_strides[0] + t * dtx_strides[1]
                            + hh * dtx_strides[2] + i * dtx_strides[3]];
                        const float e = last[bb * last_strides[0] + hh * last_strides[1]
                            + t * last_strides[3]];
                        v = a * e;
                    }
                    tile[ty + 8 * k][tx] = v;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int k = 0; k < 4; ++k) {
                    const int i = ib + int(ty) + 8 * k;
                    const int t = tb + int(tx);
                    if (t < n) {
                        out[(size_t(bh) * DH + i) * n + t] = tile[tx][ty + 8 * k];
                    }
                }
            """, ensureRowContiguous: false)
        // nextState = product + ec[..., -1, :, None, None] * previous: the
        // stock broadcast multiply and add (each one fp32 rounding, in that
        // order) in one pass over the [B, H, DH, DS] state.
        stateUpdate = MLXFast.metalKernel(
            name: "nemotron_ssm_state_update",
            inputNames: ["prod", "prev", "ec", "n"], outputNames: ["out"],
            source: """
                const uint idx = thread_position_in_grid.x * 4;
                const uint bh = thread_position_in_grid.y;
                const uint bb = bh / H;
                const uint hh = bh % H;
                const uint i = idx / DS;
                const uint k = idx % DS;
                const float e = ec[bb * ec_strides[0] + (n - 1) * ec_strides[1] + hh * ec_strides[2]];
                const device float* ps = prev + bb * prev_strides[0] + hh * prev_strides[1]
                    + i * prev_strides[2];
                const device float* qs = prod + bb * prod_strides[0] + hh * prod_strides[1]
                    + i * prod_strides[2];
                device float* o = out + (size_t(bh) * DH + i) * DS;
                for (uint j = 0; j < 4; ++j) {
                    const float p = e * ps[(k + j) * prev_strides[3]];
                    o[k + j] = qs[(k + j) * prod_strides[3]] + p;
                }
            """, ensureRowContiguous: false)
        // One scan window's output, astype(y + ec[..., None] * yPrev, OT) with
        // yPrev = state @ C as the stock broadcast matmul computes it: MLX
        // gemv_float32_bm4_bn1_sm1_sn32_tm4_tn4 for K = 128, i.e. per output
        // lane l sums k = 4l..4l+3 from 0 with `acc += s * c`, then
        // simd_shuffle_down 16, 8, 4, 2, 1 adds lane l + sn into lane l. Here
        // lane l's partial goes to threadgroup memory and one lane adds the
        // 32 partials in that tree order, so a simdgroup keeps 8 state rows
        // in registers for TC tokens instead of one 128-thread group per
        // (token, head, 16 rows). Product, sum and narrowing stay in
        // registers with one rounding each, as in the stock kernels.
        windowOut = MLXFast.metalKernel(
            name: "nemotron_ssm_window_out",
            inputNames: ["state", "C", "y", "ec", "n"], outputNames: ["out"],
            source: """
                constexpr int RB = 8;
                constexpr int TB = 32 / RB;
                threadgroup float part[4][32][33];
                const uint lane = thread_index_in_simdgroup;
                const uint sg = simdgroup_index_in_threadgroup;
                const uint bh = threadgroup_position_in_grid.z;
                const uint bb = bh / H;
                const uint hh = bh % H;
                const int i0 = (int(threadgroup_position_in_grid.x) * 4 + int(sg)) * RB;
                const int k0 = int(lane) * 4;
                const int t0 = int(threadgroup_position_in_grid.y) * TC;
                const int t1 = min(t0 + TC, n);
                const device float* sp = state + bb * state_strides[0] + hh * state_strides[1];
                float st[RB][4];
                for (int r = 0; r < RB; ++r) {
                    for (int tn = 0; tn < 4; ++tn) {
                        st[r][tn] = sp[(i0 + r) * state_strides[2] + (k0 + tn) * state_strides[3]];
                    }
                }
                const device CT* cp = C + bb * C_strides[0] + (hh / R) * C_strides[2];
                threadgroup float (*pp)[33] = part[sg];
                const int orow = int(lane) / TB;
                const int ot = int(lane) % TB;
                for (int tb = t0; tb < t1; tb += TB) {
                    for (int j = 0; j < TB; ++j) {
                        const int t = min(tb + j, n - 1);
                        float cv[4];
                        for (int tn = 0; tn < 4; ++tn) {
                            cv[tn] = static_cast<float>(cp[t * C_strides[1] + (k0 + tn) * C_strides[3]]);
                        }
                        for (int r = 0; r < RB; ++r) {
                            float acc = 0.0f;
                            for (int tn = 0; tn < 4; ++tn) {
                                acc += st[r][tn] * cv[tn];
                            }
                            pp[r * TB + j][lane] = acc;
                        }
                    }
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    const threadgroup float* v = pp[orow * TB + ot];
                    float a[16];
                    for (int q = 0; q < 16; ++q) {
                        a[q] = v[q] + v[q + 16];
                    }
                    float b[8];
                    for (int q = 0; q < 8; ++q) {
                        b[q] = a[q] + a[q + 8];
                    }
                    float c[4];
                    for (int q = 0; q < 4; ++q) {
                        c[q] = b[q] + b[q + 4];
                    }
                    const float d0 = c[0] + c[2];
                    const float d1 = c[1] + c[3];
                    const float yp = d0 + d1;
                    const int t = tb + ot;
                    if (t < t1) {
                        const float e = ec[bb * ec_strides[0] + t * ec_strides[1] + hh * ec_strides[2]];
                        const float p = e * yp;
                        const float yv = y[bb * y_strides[0] + hh * y_strides[1] + t * y_strides[2]
                            + (i0 + orow) * y_strides[3]];
                        const float u = yv + p;
                        out[((size_t(bb) * n + t) * H + hh) * DH + i0 + orow] = static_cast<OT>(u);
                    }
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                }
            """, ensureRowContiguous: false)
        // windowOut with the prefill epilogue `concatenated(outputs) + x * D`
        // folded in: each token's OT(u) is followed by OT(x * D) and OT(sum),
        // the two stock bf16 element-wise ops (each a float op rounded once;
        // a bf16 product is exact in float and float add then rounding is a
        // correctly rounded bf16 add, so this is bitwise the stock pair).
        // Token t of the window reads X row xs + t and writes output row
        // s0 + t. Rows t < s0 come from the earlier windows' epilogued outputs P
        // (grid.y blocks past this window's), copied into the full [B, LT, H,
        // DH] output in place of the concatenation.
        // windowOutFinal with SGS simdgroups (of SRB rows each) per threadgroup
        // instead of four of eight. Each lane's partials are windowOutFinal's
        // four `acc += st * cv` statements (C read as one 4-element vector when
        // its last stride is 1), and the 32 lane partials of an output are
        // added in windowOutFinal's tree (lane offsets 16, 8, 4, 2, 1) as a
        // simd_shuffle_xor butterfly instead of a threadgroup exchange, so
        // every add pairs the same lanes and lane L ends with output L. The
        // copied prefix is spread over the group's rows.
        windowOutFinalSG = MLXFast.metalKernel(
            name: "nemotron_ssm_window_out_final_sg",
            inputNames: ["state", "C", "y", "ec", "n", "X", "Dw", "P", "s0", "xs"], outputNames: ["out"],
            source: Self.windowOutFinalSGSource, ensureRowContiguous: false)
        windowOutFinal = MLXFast.metalKernel(
            name: "nemotron_ssm_window_out_final",
            inputNames: ["state", "C", "y", "ec", "n", "X", "Dw", "P", "s0", "xs"], outputNames: ["out"],
            source: """
                constexpr int RB = 8;
                constexpr int TB = 32 / RB;
                threadgroup float part[4][32][33];
                const uint lane = thread_index_in_simdgroup;
                const uint sg = simdgroup_index_in_threadgroup;
                const uint bh = threadgroup_position_in_grid.z;
                const uint bb = bh / H;
                const uint hh = bh % H;
                const int nb = (n + TC - 1) / TC;
                if (int(threadgroup_position_in_grid.y) >= nb) {
                    if (COPY) {
                        const int cb = int(threadgroup_position_in_grid.y) - nb;
                        const int i = int(threadgroup_position_in_grid.x) * 32 + int(lane);
                        for (int q = int(sg); q < TC; q += 4) {
                            const int t = cb * TC + q;
                            if (t < s0) {
                                out[((size_t(bb) * LT + t) * H + hh) * DH + i] =
                                    P[bb * P_strides[0] + t * P_strides[1] + hh * P_strides[2]
                                      + i * P_strides[3]];
                            }
                        }
                    }
                    return;
                }
                const int i0 = (int(threadgroup_position_in_grid.x) * 4 + int(sg)) * RB;
                const int k0 = int(lane) * 4;
                const int t0 = int(threadgroup_position_in_grid.y) * TC;
                const int t1 = min(t0 + TC, n);
                const device float* sp = state + bb * state_strides[0] + hh * state_strides[1];
                float st[RB][4];
                for (int r = 0; r < RB; ++r) {
                    for (int tn = 0; tn < 4; ++tn) {
                        st[r][tn] = sp[(i0 + r) * state_strides[2] + (k0 + tn) * state_strides[3]];
                    }
                }
                const device CT* cp = C + bb * C_strides[0] + (hh / R) * C_strides[2];
                threadgroup float (*pp)[33] = part[sg];
                const int orow = int(lane) / TB;
                const int ot = int(lane) % TB;
                const float dv = static_cast<float>(Dw[hh * Dw_strides[0]]);
                for (int tb = t0; tb < t1; tb += TB) {
                    for (int j = 0; j < TB; ++j) {
                        const int t = min(tb + j, n - 1);
                        float cv[4];
                        for (int tn = 0; tn < 4; ++tn) {
                            cv[tn] = static_cast<float>(cp[t * C_strides[1] + (k0 + tn) * C_strides[3]]);
                        }
                        for (int r = 0; r < RB; ++r) {
                            float acc = 0.0f;
                            for (int tn = 0; tn < 4; ++tn) {
                                acc += st[r][tn] * cv[tn];
                            }
                            pp[r * TB + j][lane] = acc;
                        }
                    }
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    const threadgroup float* v = pp[orow * TB + ot];
                    float a[16];
                    for (int q = 0; q < 16; ++q) {
                        a[q] = v[q] + v[q + 16];
                    }
                    float b[8];
                    for (int q = 0; q < 8; ++q) {
                        b[q] = a[q] + a[q + 8];
                    }
                    float c[4];
                    for (int q = 0; q < 4; ++q) {
                        c[q] = b[q] + b[q + 4];
                    }
                    const float d0 = c[0] + c[2];
                    const float d1 = c[1] + c[3];
                    const float yp = d0 + d1;
                    const int t = tb + ot;
                    if (t < t1) {
                        const float e = ec[bb * ec_strides[0] + t * ec_strides[1] + hh * ec_strides[2]];
                        const float p = e * yp;
                        const float yv = y[bb * y_strides[0] + hh * y_strides[1] + t * y_strides[2]
                            + (i0 + orow) * y_strides[3]];
                        const float u = yv + p;
                        const OT w = static_cast<OT>(u);
                        const float xv = static_cast<float>(X[bb * X_strides[0] + (xs + t) * X_strides[1]
                            + hh * X_strides[2] + (i0 + orow) * X_strides[3]]);
                        const OT xd = static_cast<OT>(xv * dv);
                        const OT o = static_cast<OT>(static_cast<float>(w) + static_cast<float>(xd));
                        out[((size_t(bb) * LT + s0 + t) * H + hh) * DH + i0 + orow] = o;
                    }
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                }
            """, ensureRowContiguous: false)
        // windowOutFinal for a fresh request's first window (`state` all
        // +0.0, nothing copied): every lane partial of state . C is +0.0, so
        // yp is +0.0 and u = yv + e * yp; the epilogue statements are
        // windowOutFinal's. One thread per (token, row), no state or C reads.
        windowOutFresh = MLXFast.metalKernel(
            name: "nemotron_ssm_window_out_fresh",
            inputNames: ["y", "ec", "n", "X", "Dw"], outputNames: ["out"],
            source: """
                // Four consecutive rows per thread (vector reads of y and X and
                // a vector store when their row stride is 1).
                const uint i0 = thread_position_in_grid.x * 4;
                const int t = int(thread_position_in_grid.y);
                const uint bh = thread_position_in_grid.z;
                if (int(i0) >= DH || t >= n) {
                    return;
                }
                const uint bb = bh / H;
                const uint hh = bh % H;
                const float dv = static_cast<float>(Dw[hh * Dw_strides[0]]);
                const float e = ec[bb * ec_strides[0] + t * ec_strides[1] + hh * ec_strides[2]];
                const bool vy = y_strides[3] == 1;
                const bool vx = X_strides[3] == 1;
                const device float* yr = y + bb * y_strides[0] + hh * y_strides[1] + t * y_strides[2];
                const device OT* xr = X + bb * X_strides[0] + t * X_strides[1] + hh * X_strides[2];
                float4 y4;
                vec<OT, 4> x4;
                if (vy) {
                    y4 = *((const device float4*)(yr + i0));
                } else {
                    for (int u = 0; u < 4; ++u) y4[u] = yr[(i0 + u) * y_strides[3]];
                }
                if (vx) {
                    x4 = *((const device vec<OT, 4>*)(xr + i0));
                } else {
                    for (int u = 0; u < 4; ++u) x4[u] = xr[(i0 + u) * X_strides[3]];
                }
                vec<OT, 4> o4;
                for (int u = 0; u < 4; ++u) {
                    const float yp = 0.0f;
                    const float p = e * yp;
                    const float yv = y4[u];
                    const float uu = yv + p;
                    const OT w = static_cast<OT>(uu);
                    const float xv = static_cast<float>(x4[u]);
                    const OT xd = static_cast<OT>(xv * dv);
                    o4[u] = static_cast<OT>(static_cast<float>(w) + static_cast<float>(xd));
                }
                *((device vec<OT, 4>*)(out + ((size_t(bb) * LT + t) * H + hh) * DH + i0)) = o4;
            """, ensureRowContiguous: false)
        // Fused segsum (tril(x, -1) -> cumsum(axis -2) in strided_scan order)
        // + exp + CB product + tril, plus the decay's last row.
        scanDecay = MLXFast.metalKernel(
            name: "nemotron_ssm_scan_decay",
            inputNames: ["dtA", "CB", "L"], outputNames: ["out", "last"],
            source: """
                // One threadgroup = 8 simdgroups x 32 lanes: column block
                // j0..j0+7 of one (batch, head). Simdgroup s owns column
                // j0+s, lane r is row rb*32+r: the same lane/row mapping and
                // per-32-row simd_prefix_inclusive_sum + carried prefix as
                // MLX strided_scan (cumsum axis -2) over tril(x, -1).
                threadgroup float tile[32][9];
                const uint lane = thread_index_in_simdgroup;
                const uint s = simdgroup_index_in_threadgroup;
                const uint j0 = threadgroup_position_in_grid.x * 8;
                const uint bh = threadgroup_position_in_grid.z;
                const uint bb = bh / H;
                const uint hh = bh % H;
                const uint j = j0 + s;
                const device CBT* cb = CB + size_t(bb * (H / R) + hh / R) * L * L;
                device float* o = out + size_t(bh) * L * L;
                float prefix = 0.0f;
                for (int rb = 0; rb < L; rb += 32) {
                    const int i = rb + int(lane);
                    float v = 0.0f;
                    if (i < L && int(j) < L && i > int(j)) {
                        v = dtA[(size_t(bb) * L + i) * H + hh];
                    }
                    v = simd_prefix_inclusive_sum(v);
                    v = v + prefix;
                    prefix = simd_shuffle(v, 31);
                    float r = 0.0f;
                    if (i < L && int(j) < L) {
                        float d = metal::precise::exp(v);
                        if (i == L - 1) {
                            last[size_t(bh) * L + j] = d;
                        }
                        if (int(j) <= i) {
                            float c = static_cast<float>(cb[size_t(i) * L + j]);
                            r = c * d;
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    tile[lane][s] = r;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    const uint writer = s * 32 + lane;
                    const uint wr = writer / 8;
                    const uint wc = writer % 8;
                    const int wi = rb + int(wr);
                    const uint wj = j0 + wc;
                    if (wi < L && int(wj) < L) {
                        o[size_t(wi) * L + wj] = tile[wr][wc];
                    }
                }
            """)
        // scanDecay with the work its zeros do not need skipped: a 32-row
        // block that lies wholly above the group's eight columns writes only
        // its zeros (its masked scan adds +0.0 and leaves the carried prefix
        // +0.0), and exp runs only where its value is stored (j <= i, or the
        // last row). Every stored value comes from scanDecay's statements.
        scanDecaySkip = MLXFast.metalKernel(
            name: "nemotron_ssm_scan_decay_skip",
            inputNames: ["dtA", "CB", "L"], outputNames: ["out", "last"],
            source: """
                threadgroup float tile[32][9];
                const uint lane = thread_index_in_simdgroup;
                const uint s = simdgroup_index_in_threadgroup;
                const uint j0 = threadgroup_position_in_grid.x * 8;
                const uint bh = threadgroup_position_in_grid.z;
                const uint bb = bh / H;
                const uint hh = bh % H;
                const uint j = j0 + s;
                const device CBT* cb = CB + size_t(bb * (H / R) + hh / R) * L * L;
                device float* o = out + size_t(bh) * L * L;
                float prefix = 0.0f;
                const uint writer = s * 32 + lane;
                const uint wr = writer / 8;
                const uint wc = writer % 8;
                for (int rb = 0; rb < L; rb += 32) {
                    const int wi = rb + int(wr);
                    const uint wj = j0 + wc;
                    if (rb + 31 < int(j0)) {
                        if (wi < L && int(wj) < L) {
                            o[size_t(wi) * L + wj] = 0.0f;
                        }
                        continue;
                    }
                    const int i = rb + int(lane);
                    float v = 0.0f;
                    if (i < L && int(j) < L && i > int(j)) {
                        v = dtA[(size_t(bb) * L + i) * H + hh];
                    }
                    v = simd_prefix_inclusive_sum(v);
                    v = v + prefix;
                    prefix = simd_shuffle(v, 31);
                    float r = 0.0f;
                    if (i < L && int(j) < L && (int(j) <= i || i == L - 1)) {
                        float d = metal::precise::exp(v);
                        if (i == L - 1) {
                            last[size_t(bh) * L + j] = d;
                        }
                        if (int(j) <= i) {
                            float c = static_cast<float>(cb[size_t(i) * L + j]);
                            r = c * d;
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    tile[lane][s] = r;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (wi < L && int(wj) < L) {
                        o[size_t(wi) * L + wj] = tile[wr][wc];
                    }
                }
            """)
        maskedDecay = MLXFast.metalKernel(
            name: "nemotron_ssm_masked_decay",
            inputNames: ["CB", "S", "L", "R"], outputNames: ["out"],
            source: """
                uint j = thread_position_in_grid.x;
                uint i = thread_position_in_grid.y;
                uint bh = thread_position_in_grid.z;
                size_t idx = (size_t(bh) * L + i) * L + j;
                float v;
                if (j <= i) {
                    float c = static_cast<float>(CB[(size_t(bh / R) * L + i) * L + j]);
                    float d = metal::precise::exp(S[idx]);
                    v = c * d;
                } else {
                    v = 0.0f;
                }
                out[idx] = v;
            """)
    }
}

private let nemotronCompiledPrefillDt = compile(shapeless: true) { (inputs: [MLXArray]) -> [MLXArray] in
    let dt = softplus(inputs[0].asType(.float32) + inputs[1])
    return [MLX.clip(dt, min: inputs[2], max: inputs[3])]
}

func nemotronPrefillComputeDt(
    _ dt: MLXArray, _ dtBias: MLXArray, _ timeStepLimit: (Float, Float)
) -> MLXArray {
    guard Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
          [DType.bfloat16, .float16, .float32].contains(dt.dtype),
          [DType.bfloat16, .float16, .float32].contains(dtBias.dtype)
    else { return computeDt(dt, dtBias, timeStepLimit) }
    // Keep the native Float32 cast, bias promotion, softplus and clip order.
    // All values are dynamic inputs; no model or request state is captured.
    return nemotronCompiledPrefillDt([
        dt, dtBias, MLXArray(timeStepLimit.0), MLXArray(timeStepLimit.1)])[0]
}

/// Bitwise copy of `ssmAttn` (SSM.swift, not editable) for L > 1 with a
/// prior state. Only materialized copies feeding element-wise ops change:
/// the segsum input is a broadcast view instead of `repeated` (the tril
/// `where` writes the same contiguous array either way, so the cumsum sees
/// identical input), and the CB repeat/promotion/exp/product/tril chain is
/// one kernel with the same per-element arithmetic. Every matmul, cumsum
/// and the state/output algebra is unchanged.
/// `state == nil` is a fresh request's FP32 zero state (see
/// `NemotronMambaZeroState`): the first window adds `decay * 0` per head
/// instead of `decay * zeros` per element, and the window output reads a
/// zero-stride +0.0 view, so every element sees the same IEEE operations
/// on the same operands without a [b, h, dh, d] zero buffer.
func nemotronSSMAttn(
    x: MLXArray, ALog: MLXArray, B: MLXArray, C: MLXArray, D: MLXArray,
    dt: MLXArray, dtBias: MLXArray, state: MLXArray?,
    timeStepLimit: (Float, Float),
    prepared: (dtA: MLXArray, dtx: MLXArray, bF32: MLXArray?)? = nil
) -> (MLXArray, MLXArray) {
    let (b, l, h, dh) = x.shape4
    let (_, _, g, d) = B.shape4

    let repeats = h / g
    let dtA: MLXArray
    let dtx: MLXArray
    if let prepared {
        // `NemotronPrefillConvDt`: the same dtA and dtx, bit for bit.
        (dtA, dtx) = (prepared.dtA, prepared.dtx)
    } else {
        let dt = nemotronPrefillComputeDt(dt, dtBias, timeStepLimit)
        let A = -MLX.exp(ALog).asType(dt.dtype)
        dtA = dt * A.reshaped(1, 1, -1)
        dtx = dt.reshaped(b, l, h, 1) * x
    }

    let step = 256
    // Seed only: every window's exp(cumsum(dtA)) in one launch.
    let windowCumExp: MLXArray? = NemotronHPrefillX5.active && b == 1
        ? NemotronPrefillScanExp.applyScanExp(dtA, step: step) : nil
    // The fused epilogue needs every window on the windowOut path below,
    // whose dtype conditions hold for an fp32 state and fp32 dtA.
    let fuseEpilogue = NemotronHPrefillX5.ssmEpilogue && NemotronHPrefillX5.active && d == 128 && dh % 32 == 0
        && h % g == 0 && (state?.dtype ?? .float32) == .float32 && dtA.dtype == .float32
        && D.dtype == x.dtype && [DType.bfloat16, .float16, .float32].contains(x.dtype)
    var currentState = state
    var outputs = [MLXArray]()
    outputs.reserveCapacity((l + step - 1) / step)
    // Seed only: a fresh first window's state left to the second window
    // (`NemotronPrefillFreshStateFold`).
    let foldFresh = NemotronPrefillFreshStateFold.usable(l: l, step: step, fuseEpilogue: fuseEpilogue)
    var freshFold: (prod: MLXArray, ec: MLXArray, n: Int)? = nil
    // Seed only: the fold's two launches with 32-bit addressing (`NemotronPrefillSeedI32`).
    let seedI32 = NemotronPrefillSeedI32.applies(l)
    for start in stride(from: 0, to: l, by: step) {
        let end = min(start + step, l)
        let n = end - start
        let stepDtx = dtx[0..., start ..< end, 0..., 0...]
        let stepDtA = dtA[0..., start ..< end, 0...]
        var stepB = B[0..., start ..< end, 0..., 0...]
        let stepC = C[0..., start ..< end, 0..., 0...]

        stepB = MLX.transposed(stepB, axes: [0, 2, 3, 1])
        let CB = MLX.swappedAxes(stepC, 1, 2).matmul(stepB)

        // Seed only: the decay matrix and its matmul in one launch
        // (`NemotronPrefillScanDecayGemm`).
        // A fresh request's first window also takes `windowOutFresh`'s
        // epilogue in that launch.
        let freshGemm = start == 0 && currentState == nil && fuseEpilogue && b == 1
            && NemotronPrefillFreshWindowOut.verified
            ? windowCumExp.flatMap { cumExp in
                NemotronPrefillScanDecayGemm.applyFresh(
                    dtA: stepDtA, CB: CB, dtx: stepDtx, n: n, h: h, g: g, dh: dh,
                    ec: cumExp[0..., start ..< end, 0...], x: x, D: D, lt: end == l ? l : n)
            } : nil
        let scanGemm = freshGemm == nil ? NemotronPrefillScanDecayGemm.apply(
            dtA: stepDtA, CB: CB, dtx: stepDtx, n: n, b: b, h: h, g: g, dh: dh) : nil
        let surrogateAttentionMatrix: MLXArray?
        let decayLast: MLXArray
        if let freshGemm {
            surrogateAttentionMatrix = nil
            decayLast = freshGemm.last
        } else if let scanGemm {
            surrogateAttentionMatrix = nil
            decayLast = scanGemm.last
        } else if NemotronHPrefillX5.enabled, stepDtA.dtype == .float32, h % g == 0 {
            // Seed only: the launch that skips work its zeros do not need.
            let fused = (NemotronPrefillScanDecaySkip.verified && NemotronHPrefillX5.active
                ? NemotronHScanKernels.shared.scanDecaySkip : NemotronHScanKernels.shared.scanDecay)(
                [stepDtA, CB, n],
                template: [("CBT", CB.dtype), ("H", h), ("R", repeats)],
                grid: (((n + 7) / 8) * 256, 1, b * h), threadGroup: (256, 1, 1),
                outputShapes: [[b, h, n, n], [b, h, 1, n]], outputDTypes: [.float32, .float32])
            surrogateAttentionMatrix = fused[0]
            decayLast = fused[1]
        } else {
            // segsum(stepDtA.swappedAxes(1, 2)) with a broadcast instead of repeat.
            let segIn = stepDtA.swappedAxes(1, 2)[.ellipsis, .newAxis]
            let segTri = MLX.tril(MLX.broadcast(segIn, to: [b, h, n, n]), k: -1)
            let segsum = MLX.cumsum(segTri, axis: -2)
            decayLast = MLX.exp(segsum[0..., 0..., (-1)..., 0...])
            if segsum.dtype == .float32 {
                surrogateAttentionMatrix = NemotronHScanKernels.shared.maskedDecay(
                    [CB, segsum, n, repeats],
                    template: [("T", CB.dtype)],
                    grid: (n, n, b * h), threadGroup: (32, 8, 1),
                    outputShapes: [[b, h, n, n]], outputDTypes: [.float32])[0]
            } else {
                let decayFull = MLX.exp(segsum)
                surrogateAttentionMatrix = MLX.tril(
                    MLX.repeated(CB, count: repeats, axis: 1) * decayFull, k: 0)
            }
        }
        // With `freshGemm` the window output is already formed and nothing reads `yWindow`.
        let yWindow = freshGemm != nil ? NemotronMambaZeroState.scalar(.float32)
            : scanGemm?.y ?? surrogateAttentionMatrix!.matmul(stepDtx.swappedAxes(1, 2))

        // decay[..., -1:, :] == exp(segsum[..., -1:, :]) element-wise.
        let decay = decayLast.transposed(0, 3, 1, 2)
        let nextStateProduct: MLXArray
        if b == 1, h == 64, g == 8, dh == 64, d == 128, n >= 64,
           Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu {
            // Seed: B already written as the f32 block the promotion would make.
            let groupedBF32 = prepared?.bF32.map {
                nemotronPrefillBF32Window($0, start: start, end: end, groups: g, state: d)
            }
            // Seed only: the left operand and the matmul in one launch
            // (`NemotronPrefillStateGemm`).
            if let groupedBF32, let fusedProduct = NemotronPrefillStateGemm.apply(
                dtx: stepDtx, last: decayLast, bWindow: groupedBF32, n: n, h: h, g: g, dh: dh, d: d) {
                nextStateProduct = fusedProduct
            } else {
            // Broadcast each B group over its heads, preserving the reduction
            // axis and input order without materializing eight copies.
            let groupedB = stepB.swappedAxes(2, 3).reshaped(b, g, 1, n, d)
            let groupedDtxDecay: MLXArray
            if NemotronHPrefillX5.ssmDtxDecay, NemotronHPrefillX5.active, stepDtx.dtype == .float32,
               decayLast.dtype == .float32 {
                groupedDtxDecay = NemotronPrefillDtxDecay4.usable(stepDtx, n: n, dh: dh)
                    ? NemotronHScanKernels.shared.dtxDecayT4(
                        [stepDtx, decayLast, n],
                        template: [("H", h), ("DH", dh)],
                        grid: (((n + 31) / 32) * 256, 1, b * h), threadGroup: (256, 1, 1),
                        outputShapes: [[b, g, repeats, dh, n]], outputDTypes: [.float32])[0]
                    : NemotronHScanKernels.shared.dtxDecayT(
                    [stepDtx, decayLast, n],
                    template: [("H", h), ("DH", dh)],
                    grid: (((n + 31) / 32) * 32, dh / 32 * 8, b * h), threadGroup: (32, 8, 1),
                    outputShapes: [[b, g, repeats, dh, n]], outputDTypes: [.float32])[0]
            } else {
                groupedDtxDecay = (stepDtx * decay).swappedAxes(1, 2).swappedAxes(2, 3)
                    .reshaped(b, g, repeats, dh, n)
            }
            nextStateProduct = groupedDtxDecay.matmul(groupedBF32 ?? groupedB).reshaped(b, h, dh, d)
            }
        } else {
            let dtxdecay = (stepDtx * decay).swappedAxes(1, 2).swappedAxes(2, 3)
            stepB = MLX.repeated(stepB, count: repeats, axis: 1).swappedAxes(2, 3)
            nextStateProduct = dtxdecay.matmul(stepB)
        }
        var nextState = nextStateProduct

        let expDtACumsum = windowCumExp.map { $0[0..., start ..< end, 0...] }
            ?? MLX.exp(MLX.cumsum(stepDtA, axis: -2))
        let lastDecay = expDtACumsum[0..., -1, 0..., .newAxis, .newAxis]
        let previousState: MLXArray
        var freshWindow = false
        var foldedFrom: (prod: MLXArray, ec: MLXArray, n: Int)? = nil
        if var currentState {
            if let fold = freshFold {
                freshFold = nil
                if nextState.dtype == .float32, nextState.shape == fold.prod.shape,
                   expDtACumsum.dtype == .float32 {
                    foldedFrom = fold
                } else {
                    currentState = NemotronHScanKernels.shared.stateUpdate(
                        [fold.prod, NemotronMambaZeroState.view([b, h, dh, d], dtype: .float32), fold.ec, fold.n],
                        template: [("H", h), ("DH", dh), ("DS", d)],
                        grid: (dh * d / 4, b * h, 1), threadGroup: (256, 1, 1),
                        outputShapes: [[b, h, dh, d]], outputDTypes: [.float32])[0]
                }
            }
            previousState = currentState
            if let fold = foldedFrom {
                // `previousState` is the first window's product here.
                nextState = (seedI32 ? NemotronPrefillSeedI32.update : NemotronPrefillFreshStateFold.update)(
                    nextState, fold.prod, expDtACumsum, fold.ec, n, fold.n, b, h, dh, d)
            } else if NemotronHPrefillX5.ssmStateUpdate, NemotronHPrefillX5.active, nextState.dtype == .float32,
               previousState.dtype == .float32, expDtACumsum.dtype == .float32,
               nextState.shape == previousState.shape, d % 4 == 0 {
                precondition(nextState.dim(1) == h && nextState.dim(2) == dh && nextState.dim(3) == d,
                             "NemotronH SSM state shape")
                nextState = NemotronHScanKernels.shared.stateUpdate(
                    [nextState, previousState, expDtACumsum, n],
                    template: [("H", h), ("DH", dh), ("DS", d)],
                    grid: (dh * d / 4, b * h, 1), threadGroup: (256, 1, 1),
                    outputShapes: [[b, h, dh, d]], outputDTypes: [.float32])[0]
            } else {
                nextState = nextState + lastDecay * previousState
            }
        } else {
            // zeros(FP32) is uniform +0.0, so (decay * zeros)[h, i, k] is
            // decay[h] * +0.0 for every (i, k): the same Multiply (same
            // promotion and operand order) on [b, h, 1, 1], then the same Add.
            previousState = NemotronMambaZeroState.view([b, h, dh, d], dtype: .float32)
            freshWindow = true
            if NemotronFreshStateUpdate.verified, NemotronHPrefillX5.ssmStateUpdate,
               NemotronHPrefillX5.active, nextState.dtype == .float32,
               expDtACumsum.dtype == .float32, nextState.shape == [b, h, dh, d], d % 4 == 0 {
                // The same e * (+0.0) then add per element, in one launch
                // (`NemotronFreshStateUpdate`), or in the second window's launches.
                if foldFresh, start == 0, end < l {
                    freshFold = (nextState, expDtACumsum, n)
                } else {
                    nextState = NemotronHScanKernels.shared.stateUpdate(
                        [nextState, previousState, expDtACumsum, n],
                        template: [("H", h), ("DH", dh), ("DS", d)],
                        grid: (dh * d / 4, b * h, 1), threadGroup: (256, 1, 1),
                        outputShapes: [[b, h, dh, d]], outputDTypes: [.float32])[0]
                }
            } else {
                nextState = nextState + lastDecay * NemotronMambaZeroState.scalar(.float32)
            }
        }
        if fuseEpilogue {
            precondition(previousState.dtype == .float32 && yWindow.dtype == .float32
                && expDtACumsum.dtype == .float32, "NemotronH SSM epilogue dtypes")
            let last = end == l
            let copied = last ? start : 0
            let prior: MLXArray = copied == 0 ? stepC
                : outputs.count == 1 ? outputs[0] : MLX.concatenated(outputs, axis: 1)
            let copyBlocks = (copied + 31) / 32
            // Seed only: a fresh request's first window reads no state.
            let fused: MLXArray
            if let freshGemm, copied == 0 {
                fused = freshGemm.o
            } else if let fold = foldedFrom, seedI32 {
                fused = NemotronPrefillSeedI32.windowOutFold(
                    [previousState, stepC, yWindow, expDtACumsum, n, x, D, prior, copied, start,
                     NemotronMambaZeroState.view([b, h, dh, d], dtype: .float32), fold.ec, fold.n],
                    C: stepC, prior: prior, sgs: NemotronPrefillWindowOutSG.sgs, ot: x.dtype, h: h, r: repeats,
                    dh: dh, lt: last ? l : n, copied: copied, n: n, b: b)
            } else if let fold = foldedFrom, let foldOut = NemotronPrefillFreshStateFold.windowOut {
                fused = NemotronPrefillWindowOutSG.run(
                    foldOut, sgs: NemotronPrefillWindowOutSG.sgs,
                    [previousState, stepC, yWindow, expDtACumsum, n, x, D, prior, copied, start,
                     NemotronMambaZeroState.view([b, h, dh, d], dtype: .float32), fold.ec, fold.n],
                    ct: stepC.dtype, ot: x.dtype, h: h, r: repeats, dh: dh, lt: last ? l : n, copy: copied > 0,
                    n: n, copyBlocks: copyBlocks, b: b, outRows: last ? l : n)
            } else {
                fused = freshWindow && copied == 0 && start == 0 && NemotronPrefillFreshWindowOut.verified
                    ? NemotronPrefillFreshWindowOut.apply(
                        y: yWindow, ec: expDtACumsum, n: n, x: x, D: D, lt: last ? l : n)
                    : NemotronPrefillWindowOutSG.run(
                        NemotronPrefillWindowOutSG.verified ? NemotronHScanKernels.shared.windowOutFinalSG
                            : NemotronHScanKernels.shared.windowOutFinal,
                        sgs: NemotronPrefillWindowOutSG.verified ? NemotronPrefillWindowOutSG.sgs : 0,
                        [previousState, stepC, yWindow, expDtACumsum, n, x, D, prior, copied, start],
                        ct: stepC.dtype, ot: x.dtype, h: h, r: repeats, dh: dh, lt: last ? l : n, copy: copied > 0,
                        n: n, copyBlocks: copyBlocks, b: b, outRows: last ? l : n)
            }
            if last {
                return (fused, nextState)
            }
            outputs.append(fused)
        } else if NemotronHPrefillX5.enabled, d == 128, dh % 32 == 0, h % g == 0,
           previousState.dtype == .float32, yWindow.dtype == .float32,
           expDtACumsum.dtype == .float32 {
            outputs.append(NemotronHScanKernels.shared.windowOut(
                [previousState, stepC, yWindow, expDtACumsum, n],
                template: [("CT", stepC.dtype), ("OT", x.dtype), ("H", h), ("R", repeats),
                           ("DH", dh), ("TC", 32)],
                grid: (dh / 32 * 128, (n + 31) / 32, b * h), threadGroup: (128, 1, 1),
                outputShapes: [[b, n, h, dh]], outputDTypes: [x.dtype])[0])
        } else {
            let reshapedState = previousState.reshaped(b, 1, g, repeats, dh, d)
            let reshapedC = stepC.reshaped(b, n, g, 1, d, 1)
            let yPrev = (reshapedState.matmul(reshapedC))
                .squeezed(axis: -1)
                .flattened(start: 2, end: 3)
            let y = MLX.swappedAxes(yWindow, 1, 2) + expDtACumsum[.ellipsis, .newAxis] * yPrev
            outputs.append(y.asType(x.dtype))
        }
        currentState = nextState
    }

    let y = MLX.concatenated(outputs, axis: 1) + x * D.reshaped(1, 1, h, 1)
    return (y, currentState ?? MLXArray.zeros([b, h, dh, d], dtype: .float32))
}

/// Seed-only (request prefill, `NemotronHPrefillX5.active`): a fresh
/// request's first scan window state, `prod1 + e1 * (+0.0)` (the
/// `NemotronFreshStateUpdate` launch), is not written; the second window's
/// two readers evaluate it per element with the same statements instead.
/// `stateUpdate` computes `prod + e * (prod1 + e1 * zero)` and `windowOut`
/// is `windowOutFinalSG` whose state rows are loaded as `prod1 + e1 * zero`.
/// `qualify` checks both bit for bit against the fresh state update followed
/// by the stock second-window launches, with signed zeros, infinities and NaN
/// in the products and decays. NEMOTRON_PREFILL_FRESH_STATE_FOLD=0 keeps the
/// first window's launch.
enum NemotronPrefillFreshStateFold {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_FRESH_STATE_FOLD"] != "0"
    nonisolated(unsafe) static var verified = false

    static let stateUpdate = MLXFast.metalKernel(
        name: "nemotron_ssm_state_update_fresh_fold",
        inputNames: ["prod", "prod1", "zero", "ec", "ec1", "n", "n1"], outputNames: ["out"],
        source: """
            const uint idx = thread_position_in_grid.x * 4;
            const uint bh = thread_position_in_grid.y;
            const uint bb = bh / H;
            const uint hh = bh % H;
            const uint i = idx / DS;
            const uint k = idx % DS;
            const float e = ec[bb * ec_strides[0] + (n - 1) * ec_strides[1] + hh * ec_strides[2]];
            const float e1 = ec1[bb * ec1_strides[0] + (n1 - 1) * ec1_strides[1] + hh * ec1_strides[2]];
            const device float* zs = zero + bb * zero_strides[0] + hh * zero_strides[1]
                + i * zero_strides[2];
            const device float* ps = prod1 + bb * prod1_strides[0] + hh * prod1_strides[1]
                + i * prod1_strides[2];
            const device float* qs = prod + bb * prod_strides[0] + hh * prod_strides[1]
                + i * prod_strides[2];
            device float* o = out + (size_t(bh) * DH + i) * DS;
            for (uint j = 0; j < 4; ++j) {
                const float p1 = e1 * zs[(k + j) * zero_strides[3]];
                const float s1 = ps[(k + j) * prod1_strides[3]] + p1;
                const float p = e * s1;
                o[k + j] = qs[(k + j) * prod_strides[3]] + p;
            }
        """, ensureRowContiguous: false)

    private static let stateBase =
        "const device float* sp = state + bb * state_strides[0] + hh * state_strides[1];"
    private static let foldedBase =
        "const device float* sp = state + bb * state_strides[0] + hh * state_strides[1];"
        + " const float e1 = ec1[bb * ec1_strides[0] + (n1 - 1) * ec1_strides[1] + hh * ec1_strides[2]];"
        + " const device float* zp = zero + bb * zero_strides[0] + hh * zero_strides[1];"
    private static let stateLoad =
        "st[r][tn] = sp[(i0 + r) * state_strides[2] + (k0 + tn) * state_strides[3]];"
    private static let foldedLoad =
        "{ const float p1 = e1 * zp[(i0 + r) * zero_strides[2] + (k0 + tn) * zero_strides[3]];"
        + " st[r][tn] = sp[(i0 + r) * state_strides[2] + (k0 + tn) * state_strides[3]] + p1; }"

    /// `windowOutFinalSG` with `state` = the first window's product; nil if
    /// its state load is not found exactly once.
    static let windowOut: MLXFast.MLXFastKernel? = {
        var source = NemotronHScanKernels.windowOutFinalSGSource
        for (old, new) in [(NemotronPrefillFreshStateFold.stateBase, NemotronPrefillFreshStateFold.foldedBase),
                           (NemotronPrefillFreshStateFold.stateLoad, NemotronPrefillFreshStateFold.foldedLoad)] {
            guard source.components(separatedBy: old).count == 2 else { return nil }
            source = source.replacingOccurrences(of: old, with: new)
        }
        return MLXFast.metalKernel(
            name: "nemotron_ssm_window_out_final_sg_fresh_fold",
            inputNames: ["state", "C", "y", "ec", "n", "X", "Dw", "P", "s0", "xs", "zero", "ec1", "n1"],
            outputNames: ["out"], source: source, ensureRowContiguous: false)
    }()

    /// Whether a fresh first window of `l` rows may leave its state to the
    /// second window (the second window then takes these two launches).
    static func usable(l: Int, step: Int, fuseEpilogue: Bool) -> Bool {
        verified && NemotronHPrefillX5.active && fuseEpilogue && l > step && windowOut != nil
            && NemotronPrefillWindowOutSG.verified && NemotronHPrefillX5.ssmStateUpdate
    }

    static func update(prod: MLXArray, prod1: MLXArray, ec: MLXArray, ec1: MLXArray, n: Int, n1: Int,
                       b: Int, h: Int, dh: Int, d: Int) -> MLXArray {
        let zero = NemotronMambaZeroState.view([b, h, dh, d], dtype: .float32)
        return stateUpdate(
            [prod, prod1, zero, ec, ec1, n, n1],
            template: [("H", h), ("DH", dh), ("DS", d)],
            grid: (dh * d / 4, b * h, 1), threadGroup: (256, 1, 1),
            outputShapes: [[b, h, dh, d]], outputDTypes: [.float32])[0]
    }

    static func qualify() {
        verified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron prefill fresh state fold: \(message)\n".utf8))
        }
        guard enabled, NemotronFreshStateUpdate.verified, NemotronPrefillWindowOutSG.verified,
              NemotronHPrefillX5.ssmStateUpdate, NemotronHPrefillX5.ssmEpilogue
        else { return report("off (setting)") }
        guard let windowOut else { return report("off (window output source not as expected)") }
        let (h, g, dh, d) = (64, 8, 64, 128)
        let sgs = NemotronPrefillWindowOutSG.sgs
        let kernels = NemotronHScanKernels.shared
        let specials = MLXArray([Float(-0.0), 0, .infinity, -.infinity, .nan, Float(bitPattern: 0x0001_0000),
                                 Float(bitPattern: 0x8001_0000), 3e38])
        var cases = 0
        do {
            let matches = try withError { () -> Bool in
                for (seed, (n1, n, l)) in [(256, 232, 488), (256, 256, 512), (256, 17, 273), (256, 100, 600)]
                    .enumerated() {
                    let key = MLXRandom.split(key: MLXRandom.key(UInt64(0xf01d + seed)), into: 10)
                    var prod1 = MLXRandom.normal([1, h, dh, d], key: key[0]) * Float(2 + seed)
                    prod1 = concatenated([specials, prod1.reshaped(-1)[8...]]).reshaped(1, h, dh, d)
                    let prod = MLXRandom.normal([1, h, dh, d], key: key[1]) * Float(1 + seed)
                    var ec1 = MLX.exp(-MLX.abs(MLXRandom.normal([1, n1, h], key: key[2])))
                    // The decay the state update reads (row n1 - 1) of four heads.
                    ec1 = concatenated([ec1.reshaped(-1)[..<((n1 - 1) * h)],
                                        MLXArray([Float(0), .infinity, .nan, 1e-30]),
                                        ec1.reshaped(-1)[((n1 - 1) * h + 4)...]]).reshaped(1, n1, h)
                    let ec = MLX.exp(-MLX.abs(MLXRandom.normal([1, n, h], key: key[3])))
                    let zero = NemotronMambaZeroState.view([1, h, dh, d], dtype: .float32)
                    let state1 = kernels.stateUpdate(
                        [prod1, zero, ec1, n1],
                        template: [("H", h), ("DH", dh), ("DS", d)],
                        grid: (dh * d / 4, h, 1), threadGroup: (256, 1, 1),
                        outputShapes: [[1, h, dh, d]], outputDTypes: [.float32])[0]
                    let stock = kernels.stateUpdate(
                        [prod, state1, ec, n],
                        template: [("H", h), ("DH", dh), ("DS", d)],
                        grid: (dh * d / 4, h, 1), threadGroup: (256, 1, 1),
                        outputShapes: [[1, h, dh, d]], outputDTypes: [.float32])[0]
                    let folded = update(prod: prod, prod1: prod1, ec: ec, ec1: ec1, n: n, n1: n1,
                                        b: 1, h: h, dh: dh, d: d)
                    var same = (folded.view(dtype: .uint32) .== stock.view(dtype: .uint32)).all()
                    // The second window's output, final (copying the first window's rows) when it
                    // ends the prefill.
                    let C = (MLXRandom.normal([1, n, g, d], key: key[4]) * Float(1 + seed)).asType(.bfloat16)
                    let y = MLXRandom.normal([1, h, n, dh], key: key[5])
                    let x = MLXRandom.normal([1, l, h, dh], key: key[6]).asType(.bfloat16)
                    let D = MLXRandom.normal([h], key: key[7]).asType(.bfloat16)
                    let start = n1
                    let last = start + n == l
                    let copied = last ? start : 0
                    let prior = copied == 0 ? C
                        : MLXRandom.normal([1, copied, h, dh], key: key[8]).asType(.bfloat16)
                    let copyBlocks = (copied + 31) / 32
                    func go(_ kernel: MLXFast.MLXFastKernel, _ inputs: [any ScalarOrArray]) -> MLXArray {
                        NemotronPrefillWindowOutSG.run(
                            kernel, sgs: sgs, inputs, ct: C.dtype, ot: x.dtype, h: h, r: h / g, dh: dh,
                            lt: last ? l : n, copy: copied > 0, n: n, copyBlocks: copyBlocks, b: 1,
                            outRows: last ? l : n)
                    }
                    let a = go(kernels.windowOutFinalSG, [state1, C, y, ec, n, x, D, prior, copied, start])
                    let bOut = go(windowOut, [prod1, C, y, ec, n, x, D, prior, copied, start, zero, ec1, n1])
                    same = same .&& (a.view(dtype: .uint16) .== bOut.view(dtype: .uint16)).all()
                    if !same.item(Bool.self) { return false }
                    cases += 1
                }
                return true
            }
            verified = matches
        } catch {
            return report("off (\(error))")
        }
        report(verified ? "on (\(cases) cases bitwise equal)"
                        : "off (differs from the fresh state update and second window after \(cases) cases)")
    }
}

/// Seed request prefill only (`NemotronHPrefillX5.active`): the second scan
/// window's two launches of a fresh request (`NemotronPrefillFreshStateFold`'s
/// state update and folded window output) with 32-bit index arithmetic.
/// Every statement that produces a value and its order per element is the
/// existing kernels' (the folded state load `sp + e1 * zero`, the `acc += s *
/// c` partials, the butterfly tree, the epilogue casts; the state update's
/// `e1 * zero`, `+`, `e *`, `+`). What changes: strides are read once as
/// `int` (MLX passes them as int64 and the stock kernels multiply them per
/// access), the state update moves its four floats as one vector when the
/// rows are contiguous, and the window output copies the first window's rows
/// 16 bytes at a time when they are the contiguous [b, s0, h, dh] output.
/// The thread grid and each output's thread are the stock kernels'.
/// `qualify` runs the whole scan (`nemotronSSMAttn`) with and without them on
/// the real first and last Mamba layers and compares output and state bit
/// for bit, then the two kernels directly on special values.
/// NEMOTRON_PREFILL_SEED_I32=0 keeps the existing kernels.
enum NemotronPrefillSeedI32 {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_SEED_I32"] != "0"
    nonisolated(unsafe) static var verified = false

    /// Used for a request prefill of `l` rows (32-bit offsets stay in range).
    static func applies(_ l: Int) -> Bool {
        verified && NemotronHPrefillX5.active && l <= 65536
    }

    static let stateUpdate = MLXFast.metalKernel(
        name: "nemotron_ssm_state_update_fresh_fold_i32",
        inputNames: ["prod", "prod1", "zero", "ec", "ec1", "n", "n1"], outputNames: ["out"],
        source: """
            const int idx = int(thread_position_in_grid.x) * 4;
            const int bh = int(thread_position_in_grid.y);
            const int bb = bh / H;
            const int hh = bh % H;
            const int i = idx / DS;
            const int k = idx % DS;
            const float e = ec[bb * int(ec_strides[0]) + (n - 1) * int(ec_strides[1]) + hh * int(ec_strides[2])];
            const float e1 = ec1[bb * int(ec1_strides[0]) + (n1 - 1) * int(ec1_strides[1])
                + hh * int(ec1_strides[2])];
            const int z3 = int(zero_strides[3]), p13 = int(prod1_strides[3]), q3 = int(prod_strides[3]);
            const device float* zs = zero + bb * int(zero_strides[0]) + hh * int(zero_strides[1])
                + i * int(zero_strides[2]);
            const device float* ps = prod1 + bb * int(prod1_strides[0]) + hh * int(prod1_strides[1])
                + i * int(prod1_strides[2]);
            const device float* qs = prod + bb * int(prod_strides[0]) + hh * int(prod_strides[1])
                + i * int(prod_strides[2]);
            device float* o = out + (bh * DH + i) * DS;
            if (VEC) {
                const float4 s4 = *((const device float4*)(ps + k));
                const float4 q4 = *((const device float4*)(qs + k));
                float4 o4;
                for (int j = 0; j < 4; ++j) {
                    const float p1 = e1 * zs[(k + j) * z3];
                    const float s1 = s4[j] + p1;
                    const float p = e * s1;
                    o4[j] = q4[j] + p;
                }
                *((device float4*)(o + k)) = o4;
            } else {
                for (int j = 0; j < 4; ++j) {
                    const float p1 = e1 * zs[(k + j) * z3];
                    const float s1 = ps[(k + j) * p13] + p1;
                    const float p = e * s1;
                    o[k + j] = qs[(k + j) * q3] + p;
                }
            }
        """, ensureRowContiguous: false)

    static let windowOut = MLXFast.metalKernel(
        name: "nemotron_ssm_window_out_final_sg_fresh_fold_i32",
        inputNames: ["state", "C", "y", "ec", "n", "X", "Dw", "P", "s0", "xs", "zero", "ec1", "n1"],
        outputNames: ["out"],
        source: """
            constexpr int RB = SRB;
            constexpr int TB = 32 / RB;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const int bh = int(threadgroup_position_in_grid.z);
            const int bb = bh / H;
            const int hh = bh % H;
            const int nb = (n + TC - 1) / TC;
            if (int(threadgroup_position_in_grid.y) >= nb) {
                if (COPY) {
                    const int cb = int(threadgroup_position_in_grid.y) - nb;
                    constexpr int RT = RB * SGS;
                    if (PV) {
                        // Rows i .. i + 7 of token t as one 16-byte move.
                        constexpr int V = 8;
                        constexpr int RV = RT / V;
                        const device OT* pb = P + (bb * s0 * H + hh) * DH;
                        device OT* ob = out + (bb * LT * H + hh) * DH;
                        for (int q = int(sg) * 32 + int(lane); q < TC * RV; q += 32 * SGS) {
                            const int t = cb * TC + q / RV;
                            const int i = int(threadgroup_position_in_grid.x) * RT + (q % RV) * V;
                            if (t < s0) {
                                *((device uint4*)(ob + t * (H * DH) + i)) =
                                    *((const device uint4*)(pb + t * (H * DH) + i));
                            }
                        }
                    } else {
                        const int ps0 = int(P_strides[0]), ps1 = int(P_strides[1]);
                        const int ps2 = int(P_strides[2]), ps3 = int(P_strides[3]);
                        for (int q = int(sg) * 32 + int(lane); q < TC * RT; q += 32 * SGS) {
                            const int t = cb * TC + q / RT;
                            const int i = int(threadgroup_position_in_grid.x) * RT + q % RT;
                            if (t < s0) {
                                out[((bb * LT + t) * H + hh) * DH + i] = P[bb * ps0 + t * ps1 + hh * ps2 + i * ps3];
                            }
                        }
                    }
                }
                return;
            }
            const int i0 = (int(threadgroup_position_in_grid.x) * SGS + int(sg)) * RB;
            const int k0 = int(lane) * 4;
            const int t0 = int(threadgroup_position_in_grid.y) * TC;
            const int t1 = min(t0 + TC, n);
            const int ss2 = int(state_strides[2]), ss3 = int(state_strides[3]);
            const int zs2 = int(zero_strides[2]), zs3 = int(zero_strides[3]);
            const device float* sp = state + bb * int(state_strides[0]) + hh * int(state_strides[1]);
            const float e1 = ec1[bb * int(ec1_strides[0]) + (n1 - 1) * int(ec1_strides[1]) + hh * int(ec1_strides[2])];
            const device float* zp = zero + bb * int(zero_strides[0]) + hh * int(zero_strides[1]);
            float st[RB][4];
            for (int r = 0; r < RB; ++r) {
                for (int tn = 0; tn < 4; ++tn) {
                    const float p1 = e1 * zp[(i0 + r) * zs2 + (k0 + tn) * zs3];
                    st[r][tn] = sp[(i0 + r) * ss2 + (k0 + tn) * ss3] + p1;
                }
            }
            const int cs1 = int(C_strides[1]), cs3 = int(C_strides[3]);
            const device CT* cp = C + bb * int(C_strides[0]) + (hh / R) * int(C_strides[2]);
            const int orow = int(lane) / TB;
            const int ot = int(lane) % TB;
            const float dv = static_cast<float>(Dw[hh * int(Dw_strides[0])]);
            const int es1 = int(ec_strides[1]);
            const device float* ep = ec + bb * int(ec_strides[0]) + hh * int(ec_strides[2]);
            const int ys2 = int(y_strides[2]);
            const device float* ypp = y + bb * int(y_strides[0]) + hh * int(y_strides[1])
                + (i0 + orow) * int(y_strides[3]);
            const int xs1 = int(X_strides[1]);
            const device OT* xp = X + bb * int(X_strides[0]) + hh * int(X_strides[2]) + (i0 + orow) * int(X_strides[3]);
            device OT* op = out + ((bb * LT + s0) * H + hh) * DH + i0 + orow;
            for (int tb = t0; tb < t1; tb += TB) {
                float v[32];
                for (int j = 0; j < TB; ++j) {
                    const int t = min(tb + j, n - 1);
                    float cv[4];
                    if (CV) {
                        const vec<CT, 4> c4 = *((const device vec<CT, 4>*)(cp + t * cs1 + k0));
                        cv[0] = static_cast<float>(c4.x);
                        cv[1] = static_cast<float>(c4.y);
                        cv[2] = static_cast<float>(c4.z);
                        cv[3] = static_cast<float>(c4.w);
                    } else {
                        for (int tn = 0; tn < 4; ++tn) {
                            cv[tn] = static_cast<float>(cp[t * cs1 + (k0 + tn) * cs3]);
                        }
                    }
                    for (int r = 0; r < RB; ++r) {
                        float acc = 0.0f;
                        for (int tn = 0; tn < 4; ++tn) {
                            acc += st[r][tn] * cv[tn];
                        }
                        v[r * TB + j] = acc;
                    }
                }
                float w16[16];
                const bool b16 = (lane & 16) != 0;
                for (int q = 0; q < 16; ++q) {
                    const float keep = b16 ? v[q + 16] : v[q];
                    const float send = b16 ? v[q] : v[q + 16];
                    w16[q] = keep + simd_shuffle_xor(send, 16);
                }
                float w8[8];
                const bool b8 = (lane & 8) != 0;
                for (int q = 0; q < 8; ++q) {
                    const float keep = b8 ? w16[q + 8] : w16[q];
                    const float send = b8 ? w16[q] : w16[q + 8];
                    w8[q] = keep + simd_shuffle_xor(send, 8);
                }
                float w4[4];
                const bool b4 = (lane & 4) != 0;
                for (int q = 0; q < 4; ++q) {
                    const float keep = b4 ? w8[q + 4] : w8[q];
                    const float send = b4 ? w8[q] : w8[q + 4];
                    w4[q] = keep + simd_shuffle_xor(send, 4);
                }
                float w2[2];
                const bool b2 = (lane & 2) != 0;
                for (int q = 0; q < 2; ++q) {
                    const float keep = b2 ? w4[q + 2] : w4[q];
                    const float send = b2 ? w4[q] : w4[q + 2];
                    w2[q] = keep + simd_shuffle_xor(send, 2);
                }
                const bool b1 = (lane & 1) != 0;
                const float keep1 = b1 ? w2[1] : w2[0];
                const float send1 = b1 ? w2[0] : w2[1];
                const float yp = keep1 + simd_shuffle_xor(send1, 1);
                const int t = tb + ot;
                if (t < t1) {
                    const float e = ep[t * es1];
                    const float p = e * yp;
                    const float yv = ypp[t * ys2];
                    const float u = yv + p;
                    const OT w = static_cast<OT>(u);
                    const float xv = static_cast<float>(xp[(xs + t) * xs1]);
                    const OT xd = static_cast<OT>(xv * dv);
                    const OT o = static_cast<OT>(static_cast<float>(w) + static_cast<float>(xd));
                    op[t * (H * DH)] = o;
                }
            }
        """, ensureRowContiguous: false)

    static func contiguous(_ a: MLXArray) -> Bool {
        var expected = 1
        for (dim, stride) in zip(a.shape, a.strides).reversed() {
            if dim != 1 && stride != expected { return false }
            expected *= dim
        }
        return true
    }

    /// `NemotronPrefillFreshStateFold.update`'s values.
    static func update(prod: MLXArray, prod1: MLXArray, ec: MLXArray, ec1: MLXArray, n: Int, n1: Int,
                       b: Int, h: Int, dh: Int, d: Int) -> MLXArray {
        let zero = NemotronMambaZeroState.view([b, h, dh, d], dtype: .float32)
        let vec = contiguous(prod) && contiguous(prod1)
        return stateUpdate(
            [prod, prod1, zero, ec, ec1, n, n1],
            template: [("H", h), ("DH", dh), ("DS", d), ("VEC", vec)],
            grid: (dh * d / 4, b * h, 1), threadGroup: (256, 1, 1),
            outputShapes: [[b, h, dh, d]], outputDTypes: [.float32])[0]
    }

    /// `NemotronPrefillWindowOutSG.run(NemotronPrefillFreshStateFold.windowOut, ...)`'s values;
    /// `inputs` as that call's.
    static func windowOutFold(_ inputs: [any ScalarOrArray], C: MLXArray, prior: MLXArray, sgs: Int,
                              ot: DType, h: Int, r: Int, dh: Int, lt: Int, copied: Int, n: Int, b: Int)
        -> MLXArray
    {
        let rb = NemotronPrefillWindowOutSG.rows
        let pv = copied > 0 && ot.size == 2 && prior.dtype == ot && prior.shape == [b, copied, h, dh]
            && contiguous(prior) && (rb * sgs) % 8 == 0
        let cv = C.strides.last == 1
        return windowOut(
            inputs,
            template: [("CT", C.dtype), ("OT", ot), ("H", h), ("R", r), ("DH", dh), ("TC", 32), ("LT", lt),
                       ("COPY", copied > 0), ("SGS", sgs), ("SRB", rb), ("PV", pv), ("CV", cv)],
            grid: (dh / (rb * sgs) * 32 * sgs, (n + 31) / 32 + (copied + 31) / 32, b * h),
            threadGroup: (32 * sgs, 1, 1),
            outputShapes: [[b, lt, h, dh]], outputDTypes: [ot])[0]
    }

    static func qualify(_ target: Module) {
        verified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron prefill seed i32: \(message)\n".utf8))
        }
        guard enabled, NemotronHPrefillX5.enabled else { return report("off (setting)") }
        guard NemotronPrefillFreshStateFold.verified, NemotronPrefillFreshStateFold.windowOut != nil,
              NemotronPrefillWindowOutSG.verified, NemotronPrefillConvDt.verified
        else { return report("off (fresh state fold off)") }
        let mixers = target.modules().compactMap { $0 as? NemotronHMamba2Mixer }
        guard let firstMixer = mixers.first, let lastMixer = mixers.last else {
            return report("off (no Mamba layer)")
        }
        let savedRequest = NemotronHPrefillX5.requestPrefill
        NemotronHPrefillX5.requestPrefill = true
        defer { NemotronHPrefillX5.requestPrefill = savedRequest }
        // (length, fresh state)
        let cases: [(Int, Bool)] = [(488, true), (488, false), (300, true), (257, true), (512, true),
                                    (600, true), (9, true)]
        var checked = 0
        do {
            let matches = try withError { () -> Bool in
                for (mi, mixer) in [firstMixer, lastMixer].enumerated() {
                    for (ci, (length, fresh)) in cases.enumerated() {
                        let seed = UInt64(0x5e32_0000 + mi * 64 + ci)
                        let width = mixer.intermediateSize + mixer.convDim + mixer.numHeads
                        let projected = MLXRandom.normal(
                            [1, length, width], dtype: .bfloat16, scale: 2, key: MLXRandom.key(seed))
                        let input = projected[
                            0..., 0..., mixer.intermediateSize ..< (mixer.intermediateSize + mixer.convDim)]
                        let dt = projected[0..., 0..., (mixer.intermediateSize + mixer.convDim) ..< width]
                        let oldState = NemotronMambaZeroState.view([1, 3, mixer.convDim], dtype: .bfloat16)
                        guard let front = mixer.prefillConvDt(
                            oldState, input, dt, writeB: NemotronPrefillConvDt.bf32Verified)
                        else { return false }
                        let (g, d, h, dh) = (mixer.numGroups, mixer.ssmStateSize, mixer.numHeads, mixer.headDim)
                        let parts = split(front.0, indices: [mixer.intermediateSize, mixer.intermediateSize + g * d],
                                          axis: -1)
                        let x = parts[0].reshaped(1, length, h, dh)
                        let B = parts[1].reshaped(1, length, g, d)
                        let C = parts[2].reshaped(1, length, g, d)
                        let state: MLXArray? = fresh ? nil
                            : MLXRandom.normal([1, h, dh, d], scale: 3, key: MLXRandom.key(seed + 1))
                        func run(_ on: Bool) -> (MLXArray, MLXArray) {
                            verified = on
                            defer { verified = false }
                            let r = nemotronSSMAttn(
                                x: x, ALog: mixer.aLog, B: B, C: C, D: mixer.D,
                                dt: dt.reshaped(1, length, h), dtBias: mixer.dtBias, state: state,
                                timeStepLimit: mixer.timeStepLimit,
                                prepared: (dtA: front.2, dtx: front.3, bF32: front.4))
                            eval(r.0, r.1)
                            return r
                        }
                        let stock = run(false)
                        let mine = run(true)
                        let same = MLX.logicalAnd(
                            (mine.0.view(dtype: .uint16) .== stock.0.view(dtype: .uint16)).all(),
                            (mine.1.view(dtype: .uint32) .== stock.1.view(dtype: .uint32)).all())
                        if !same.item(Bool.self) { return false }
                        checked += 1
                    }
                }
                // The two kernels directly, with signed zeros, infinities, NaN and
                // subnormals in the first window's product and decays.
                guard let stockOut = NemotronPrefillFreshStateFold.windowOut else { return false }
                let (h, g, dh, d) = (64, 8, 64, 128)
                let sgs = NemotronPrefillWindowOutSG.sgs
                let specials = MLXArray([Float(-0.0), 0, .infinity, -.infinity, .nan, Float(bitPattern: 0x0001_0000),
                                         Float(bitPattern: 0x8001_0000), 3e38])
                for (si, (n1, n, l)) in [(256, 232, 488), (256, 256, 512), (256, 17, 273), (256, 100, 600)]
                    .enumerated() {
                    let key = MLXRandom.split(key: MLXRandom.key(UInt64(0x5e33 + si)), into: 10)
                    var prod1 = MLXRandom.normal([1, h, dh, d], key: key[0]) * Float(2 + si)
                    prod1 = concatenated([specials, prod1.reshaped(-1)[8...]]).reshaped(1, h, dh, d)
                    let prod = MLXRandom.normal([1, h, dh, d], key: key[1]) * Float(1 + si)
                    var ec1 = MLX.exp(-MLX.abs(MLXRandom.normal([1, n1, h], key: key[2])))
                    ec1 = concatenated([ec1.reshaped(-1)[..<((n1 - 1) * h)],
                                        MLXArray([Float(0), .infinity, .nan, 1e-30]),
                                        ec1.reshaped(-1)[((n1 - 1) * h + 4)...]]).reshaped(1, n1, h)
                    let ec = MLX.exp(-MLX.abs(MLXRandom.normal([1, n, h], key: key[3])))
                    let a0 = NemotronPrefillFreshStateFold.update(
                        prod: prod, prod1: prod1, ec: ec, ec1: ec1, n: n, n1: n1, b: 1, h: h, dh: dh, d: d)
                    let a1 = update(prod: prod, prod1: prod1, ec: ec, ec1: ec1, n: n, n1: n1, b: 1, h: h, dh: dh, d: d)
                    var same = (a0.view(dtype: .uint32) .== a1.view(dtype: .uint32)).all()
                    // C, x as column slices of a conv-output-wide row block.
                    let wide = (MLXRandom.normal([1, l, h * dh + 2 * g * d], key: key[4]) * Float(1 + si))
                        .asType(.bfloat16)
                    let x = wide[0..., 0..., 0 ..< (h * dh)].reshaped(1, l, h, dh)
                    let C = wide[0..., n1 ..< (n1 + n), (h * dh + g * d) ..< (h * dh + 2 * g * d)]
                        .reshaped(1, n, g, d)
                    let y = MLXRandom.normal([1, h, n, dh], key: key[5])
                    let D = MLXRandom.normal([h], key: key[7]).asType(.bfloat16)
                    let last = n1 + n == l
                    let copied = last ? n1 : 0
                    let prior = copied == 0 ? C
                        : MLXRandom.normal([1, copied, h, dh], key: key[8]).asType(.bfloat16)
                    let zero = NemotronMambaZeroState.view([1, h, dh, d], dtype: .float32)
                    let inputs: [any ScalarOrArray] = [prod1, C, y, ec, n, x, D, prior, copied, n1, zero, ec1, n1]
                    let o0 = NemotronPrefillWindowOutSG.run(
                        stockOut, sgs: sgs, inputs, ct: C.dtype, ot: x.dtype, h: h, r: h / g, dh: dh,
                        lt: last ? l : n, copy: copied > 0, n: n, copyBlocks: (copied + 31) / 32, b: 1,
                        outRows: last ? l : n)
                    let o1 = windowOutFold(inputs, C: C, prior: prior, sgs: sgs, ot: x.dtype, h: h, r: h / g,
                                           dh: dh, lt: last ? l : n, copied: copied, n: n, b: 1)
                    same = same .&& (o0.view(dtype: .uint16) .== o1.view(dtype: .uint16)).all()
                    if !same.item(Bool.self) { return false }
                    checked += 1
                }
                return true
            }
            verified = matches
        } catch {
            verified = false
            return report("off (\(error))")
        }
        report(verified ? "on (\(checked) cases bitwise equal)"
                        : "off (outputs differ from the existing kernels after \(checked) cases)")
    }
}

/// Seed-only (request prefill, `NemotronHPrefillX5.active`): a scan window's
/// `scanDecay(Skip)` launch and the FP32 matmul of its [b, h, n, n] masked
/// decay matrix with `dtx` in one launch that never writes the matrix. Each
/// threadgroup rebuilds its 64-row strip of the matrix, 32 columns at a time,
/// with `scanDecaySkip`'s statements (the column's 32-row
/// `simd_prefix_inclusive_sum` blocks from the block holding the column, the
/// carried prefix, `precise::exp`, the CB product and the `last` row), and
/// multiplies it with the MLX NAX GEMM loop's fragments: 32 x 32 simdgroup
/// tiles, K in 32-column steps from 0 with the last step zero-padded, and
/// `tile_matmad_nax` per step. `qualify` compares `y` and `last` with the two
/// launches bit for bit. NEMOTRON_PREFILL_SCAN_GEMM=0 keeps them.
enum NemotronPrefillScanDecayGemm {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_SCAN_GEMM"] != "0"
    nonisolated(unsafe) static var verified = false

    /// The kernel body; with `freshOut` the accumulators go through
    /// `windowOutFresh`'s epilogue (a fresh request's first window) instead
    /// of being stored as `y`.
    private static func source(freshOut: Bool) -> String {
        let body = """
            constexpr short SM = 32;
            constexpr short SN = 32;
            constexpr short SK = 32;
            constexpr short TM = SM / 16;
            constexpr short TN = SN / 16;
            constexpr short TK = SK / 16;
            constexpr int AS = SK + 1;
            threadgroup float As[64 * AS];
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const int m0 = int(threadgroup_position_in_grid.x) * 64;
            const uint bh = threadgroup_position_in_grid.z;
            const uint bb = bh / H;
            const uint hh = bh % H;
            const device CBT* cb = CB + size_t(bb * (H / R) + hh / R) * L * L;
            const device float* da = dtA + size_t(bb) * L * H + hh;
            const int ldb = H * DH;
            const device float* xb = dtx + size_t(bb) * L * ldb + size_t(hh) * DH;
            const short tm = SM * short(sg / 2);
            const short tn = SN * short(sg % 2);
            NAXTile<float, TM, TN> Dtile;
            Dtile.clear();
            for (int k0 = 0; k0 < L; k0 += SK) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int c = int(sg); c < SK; c += 4) {
                    const int j = k0 + c;
                    const int jb = (j / 32) * 32;
                    float prefix = 0.0f;
                    for (int rb = jb; rb < m0; rb += 32) {
                        const int i = rb + int(lane);
                        float v = 0.0f;
                        if (i < L && j < L && i > j) {
                            v = da[size_t(i) * H];
                        }
                        v = simd_prefix_inclusive_sum(v);
                        v = v + prefix;
                        prefix = simd_shuffle(v, 31);
                    }
                    for (int q = 0; q < 2; ++q) {
                        const int rb = m0 + 32 * q;
                        const int i = rb + int(lane);
                        float r = 0.0f;
                        if (rb + 31 >= jb) {
                            float v = 0.0f;
                            if (i < L && j < L && i > j) {
                                v = da[size_t(i) * H];
                            }
                            v = simd_prefix_inclusive_sum(v);
                            v = v + prefix;
                            prefix = simd_shuffle(v, 31);
                            if (i < L && j < L && (j <= i || i == L - 1)) {
                                float d = metal::precise::exp(v);
                                if (i == L - 1) {
                                    last[size_t(bh) * L + j] = d;
                                }
                                if (j <= i) {
                                    float cv = static_cast<float>(cb[size_t(i) * L + j]);
                                    r = cv * d;
                                }
                            }
                        }
                        As[(32 * q + int(lane)) * AS + c] = r;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                NAXTile<float, TM, TK> Atile;
                NAXTile<float, TK, TN> Btile;
                Atile.template load<float, AS, 1>(As + tm * AS);
                const device float* bp = xb + size_t(k0) * ldb + tn;
                if (k0 + SK <= L) {
                    Btile.load(bp, ldb);
                } else {
                    Btile.load_safe(bp, ldb, short2(SN, short(L - k0)));
                }
                tile_matmad_nax(
                    Dtile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<false>{});
            }
            STORE_SECTION
        """
        let store = freshOut ? """
            // windowOutFresh's statements on each accumulator element.
            const short2 sc = BaseNAXFrag::get_coord();
            const float dv = static_cast<float>(Dw[hh * Dw_strides[0]]);
            STEEL_PRAGMA_UNROLL
            for (short fm = 0; fm < TM; ++fm) {
                STEEL_PRAGMA_UNROLL
                for (short fn = 0; fn < TN; ++fn) {
                    const thread typename NAXTile<float, TM, TN>::frag_type& fr = Dtile.frag_at(fm, fn);
                    STEEL_PRAGMA_UNROLL
                    for (short ii = 0; ii < 2; ++ii) {
                        const int t = m0 + int(tm) + fm * 16 + int(sc.y) + ii * 8;
                        if (t < L) {
                            const int i0 = int(tn) + fn * 16 + int(sc.x);
                            const float e = ec[(size_t(bb) * L + t) * H + hh];
                            const device OT* xr = X + bb * X_strides[0] + t * X_strides[1] + hh * X_strides[2];
                            vec<OT, 4> o4;
                            for (int u = 0; u < 4; ++u) {
                                const float yp = 0.0f;
                                const float p = e * yp;
                                const float yv = fr[ii * 4 + u];
                                const float uu = yv + p;
                                const OT w = static_cast<OT>(uu);
                                const float xv = static_cast<float>(xr[(i0 + u) * X_strides[3]]);
                                const OT xd = static_cast<OT>(xv * dv);
                                o4[u] = static_cast<OT>(static_cast<float>(w) + static_cast<float>(xd));
                            }
                            *((device vec<OT, 4>*)(o + ((size_t(bb) * LT + t) * H + hh) * DH + i0)) = o4;
                        }
                    }
                }
            }
            """ : """
            const int rows = min(int(SM), L - (m0 + int(tm)));
            if (rows > 0) {
                device float* yp = y + (size_t(bh) * L + m0 + tm) * DH + tn;
                if (rows == int(SM)) {
                    Dtile.store(yp, DH);
                } else {
                    Dtile.store_slice(yp, DH, short2(0, 0), short2(SN, short(rows)));
                }
            }
            """
        return body.replacingOccurrences(of: "STORE_SECTION", with: store)
    }

    static let kernel = MLXFast.metalKernel(
        name: "nemotron_ssm_scan_decay_gemm",
        inputNames: ["dtA", "CB", "dtx", "L"], outputNames: ["y", "last"],
        source: NemotronPrefillScanDecayGemm.source(freshOut: false),
        header: nemotronRunGatherHeader)

    /// The fresh first window's output ([b, LT, h, dh], `windowOutFresh`'s
    /// values) and `last`.
    static let freshKernel = MLXFast.metalKernel(
        name: "nemotron_ssm_scan_decay_gemm_fresh_out",
        inputNames: ["dtA", "CB", "dtx", "L", "ec", "X", "Dw"], outputNames: ["o", "last"],
        source: NemotronPrefillScanDecayGemm.source(freshOut: true),
        header: nemotronRunGatherHeader, ensureRowContiguous: false)

    /// The window's `(yWindow, decayLast)`; nil outside the request prefill or
    /// the kernel's shapes.
    static func apply(dtA: MLXArray, CB: MLXArray, dtx: MLXArray, n: Int, b: Int, h: Int, g: Int, dh: Int)
        -> (y: MLXArray, last: MLXArray)?
    {
        guard verified, NemotronHPrefillX5.active else { return nil }
        return launch(dtA: dtA, CB: CB, dtx: dtx, n: n, b: b, h: h, g: g, dh: dh)
    }

    private static func launch(dtA: MLXArray, CB: MLXArray, dtx: MLXArray, n: Int, b: Int, h: Int, g: Int,
                               dh: Int) -> (y: MLXArray, last: MLXArray)? {
        guard NemotronHPrefillX5.runGather, b == 1, dh == 64, g > 0, h % g == 0,
              dtA.dtype == .float32, dtx.dtype == .float32,
              CB.dtype == .bfloat16 || CB.dtype == .float16 || CB.dtype == .float32,
              dtA.shape == [b, n, h], dtx.shape == [b, n, h, dh], CB.shape == [b, g, n, n]
        else { return nil }
        let out = kernel(
            [dtA, CB, dtx, n],
            template: [("CBT", CB.dtype), ("H", h), ("R", h / g), ("DH", dh)],
            grid: (((n + 63) / 64) * 128, 1, b * h), threadGroup: (128, 1, 1),
            outputShapes: [[b, h, n, dh], [b, h, 1, n]], outputDTypes: [.float32, .float32])
        return (out[0], out[1])
    }

    nonisolated(unsafe) static var freshVerified = false

    /// A fresh request's first window: `(windowOutFresh output, decayLast)`
    /// from one launch; nil outside the request prefill or the shapes.
    static func applyFresh(dtA: MLXArray, CB: MLXArray, dtx: MLXArray, n: Int, h: Int, g: Int, dh: Int,
                           ec: MLXArray, x: MLXArray, D: MLXArray, lt: Int) -> (o: MLXArray, last: MLXArray)?
    {
        guard verified, freshVerified, NemotronHPrefillX5.active else { return nil }
        return launchFresh(dtA: dtA, CB: CB, dtx: dtx, n: n, h: h, g: g, dh: dh, ec: ec, x: x, D: D, lt: lt)
    }

    private static func launchFresh(dtA: MLXArray, CB: MLXArray, dtx: MLXArray, n: Int, h: Int, g: Int, dh: Int,
                                    ec: MLXArray, x: MLXArray, D: MLXArray, lt: Int)
        -> (o: MLXArray, last: MLXArray)?
    {
        guard NemotronHPrefillX5.runGather, dh == 64, g > 0, h % g == 0,
              dtA.dtype == .float32, dtx.dtype == .float32,
              CB.dtype == .bfloat16 || CB.dtype == .float16 || CB.dtype == .float32,
              dtA.shape == [1, n, h], dtx.shape == [1, n, h, dh], CB.shape == [1, g, n, n],
              dtA.strides.last == 1, dtx.strides.last == 1, CB.strides.last == 1,
              dtA.strides[1] == h, dtx.strides[1] == h * dh, dtx.strides[2] == dh, CB.strides[2] == n,
              CB.strides[1] == n * n,
              ec.dtype == .float32, ec.shape == [1, n, h], ec.strides[1] == h, ec.strides[2] == 1,
              x.dtype == .bfloat16 || x.dtype == .float16, x.ndim == 4, x.dim(0) == 1, x.dim(1) >= n,
              x.dim(2) == h, x.dim(3) == dh, D.dtype == x.dtype, D.shape == [h], lt >= n
        else { return nil }
        let out = freshKernel(
            [dtA, CB, dtx, n, ec, x, D],
            template: [("CBT", CB.dtype), ("H", h), ("R", h / g), ("DH", dh), ("OT", x.dtype), ("LT", lt)],
            grid: (((n + 63) / 64) * 128, 1, h), threadGroup: (128, 1, 1),
            outputShapes: [[1, lt, h, dh], [1, h, 1, n]], outputDTypes: [x.dtype, .float32])
        return (out[0], out[1])
    }

    /// `freshKernel` against `scanDecay(Skip)`, the matmul and
    /// `windowOutFresh`, bit for bit.
    static func qualifyFresh() {
        freshVerified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron prefill scan gemm fresh out: \(message)\n".utf8))
        }
        guard verified, NemotronPrefillFreshWindowOut.verified else { return report("off (setting)") }
        let (h, g, dh) = (64, 8, 64)
        let specials = MLXArray([Float(-0.0), 0, .infinity, -.infinity, .nan, Float(bitPattern: 0x0001_0000),
                                 Float(bitPattern: 0x8001_0000), 3e38])
        var cases = 0
        do {
            let matches = try withError { () -> Bool in
                for (seed, (n, l)) in [(256, 488), (256, 300), (100, 100), (17, 17)].enumerated() {
                    let key = MLXRandom.split(key: MLXRandom.key(UInt64(0xf5e0 + seed)), into: 6)
                    let dtAFull = -MLX.abs(MLXRandom.normal([1, l, h], key: key[0])) * Float(0.05 * Double(1 + seed))
                    var dtxFull = MLXRandom.normal([1, l, h, dh], key: key[1]) * Float(1 + seed)
                    if seed == 1 {
                        dtxFull = concatenated([specials, dtxFull.reshaped(-1)[8...]]).reshaped(1, l, h, dh)
                    }
                    let CB = (MLXRandom.normal([1, g, n, n], key: key[2]) * Float(2 + seed)).asType(.bfloat16)
                    let ecFull = MLX.exp(-MLX.abs(MLXRandom.normal([1, l, h], key: key[3])))
                    // x as a column slice of a wider row, as the convolution split leaves it.
                    let wide = MLXRandom.normal([1, l, h * dh + 64], key: key[4]).asType(.bfloat16)
                    let x = wide[0..., 0..., 0 ..< (h * dh)].reshaped(1, l, h, dh)
                    let D = MLXRandom.normal([h], key: key[5]).asType(.bfloat16)
                    let dtA = dtAFull[0..., 0 ..< n, 0...]
                    let dtx = dtxFull[0..., 0 ..< n, 0..., 0...]
                    let ec = ecFull[0..., 0 ..< n, 0...]
                    let lt = n == l ? l : n
                    let decay = (NemotronPrefillScanDecaySkip.verified ? NemotronHScanKernels.shared.scanDecaySkip
                        : NemotronHScanKernels.shared.scanDecay)(
                        [dtA, CB, n],
                        template: [("CBT", CB.dtype), ("H", h), ("R", h / g)],
                        grid: (((n + 7) / 8) * 256, 1, h), threadGroup: (256, 1, 1),
                        outputShapes: [[1, h, n, n], [1, h, 1, n]], outputDTypes: [.float32, .float32])
                    let stockY = decay[0].matmul(dtx.swappedAxes(1, 2))
                    let stockO = NemotronPrefillFreshWindowOut.apply(y: stockY, ec: ec, n: n, x: x, D: D, lt: lt)
                    guard let fused = launchFresh(dtA: dtA, CB: CB, dtx: dtx, n: n, h: h, g: g, dh: dh,
                                                  ec: ec, x: x, D: D, lt: lt)
                    else { return false }
                    let rows = stockO[0..., 0 ..< n, 0..., 0...]
                    let same = (fused.o[0..., 0 ..< n, 0..., 0...].view(dtype: .uint16) .== rows.view(dtype: .uint16)).all()
                        .&& (fused.last.view(dtype: .uint32) .== decay[1].view(dtype: .uint32)).all()
                    if !same.item(Bool.self) { return false }
                    cases += 1
                }
                return true
            }
            freshVerified = matches
        } catch {
            return report("off (\(error))")
        }
        report(freshVerified ? "on (\(cases) cases bitwise equal)"
                             : "off (differs from the two launches and windowOutFresh after \(cases) cases)")
    }

    static func qualify() {
        verified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron prefill scan gemm: \(message)\n".utf8))
        }
        guard enabled, NemotronHPrefillX5.enabled, NemotronHPrefillX5.runGather else {
            return report("off (setting)")
        }
        let (h, g, dh) = (64, 8, 64)
        let specials = MLXArray([Float(-0.0), 0, .infinity, -.infinity, .nan, Float(bitPattern: 0x0001_0000),
                                 Float(bitPattern: 0x8001_0000), 3e38])
        var cases = 0
        do {
            let matches = try withError { () -> Bool in
                for (seed, (n, l, start)) in [(256, 488, 0), (232, 488, 256), (256, 512, 256), (100, 356, 256),
                                              (17, 17, 0)].enumerated() {
                    let key = MLXRandom.split(key: MLXRandom.key(UInt64(0x5ca9 + seed)), into: 4)
                    let dtAFull = -MLX.abs(MLXRandom.normal([1, l, h], key: key[0])) * Float(0.05 * Double(1 + seed))
                    var dtxFull = MLXRandom.normal([1, l, h, dh], key: key[1]) * Float(1 + seed)
                    if seed == 1 {
                        dtxFull = concatenated([dtxFull.reshaped(-1)[..<(start * h * dh + 3 * dh)], specials,
                                                dtxFull.reshaped(-1)[(start * h * dh + 3 * dh + 8)...]])
                            .reshaped(1, l, h, dh)
                    }
                    var CB = MLXRandom.normal([1, g, n, n], key: key[2]) * Float(2 + seed)
                    if seed == 2 {
                        CB = concatenated([specials, CB.reshaped(-1)[8...]]).reshaped(1, g, n, n)
                    }
                    CB = CB.asType(.bfloat16)
                    let dtA = dtAFull[0..., start ..< (start + n), 0...]
                    let dtx = dtxFull[0..., start ..< (start + n), 0..., 0...]
                    let decay = (NemotronPrefillScanDecaySkip.verified ? NemotronHScanKernels.shared.scanDecaySkip
                        : NemotronHScanKernels.shared.scanDecay)(
                        [dtA, CB, n],
                        template: [("CBT", CB.dtype), ("H", h), ("R", h / g)],
                        grid: (((n + 7) / 8) * 256, 1, h), threadGroup: (256, 1, 1),
                        outputShapes: [[1, h, n, n], [1, h, 1, n]], outputDTypes: [.float32, .float32])
                    let stockY = decay[0].matmul(dtx.swappedAxes(1, 2))
                    guard let fused = launch(dtA: dtA, CB: CB, dtx: dtx, n: n, b: 1, h: h, g: g, dh: dh) else {
                        return false
                    }
                    let same = (fused.y.view(dtype: .uint32) .== stockY.view(dtype: .uint32)).all()
                        .&& (fused.last.view(dtype: .uint32) .== decay[1].view(dtype: .uint32)).all()
                    if !same.item(Bool.self) { return false }
                    cases += 1
                }
                return true
            }
            verified = matches
        } catch {
            return report("off (\(error))")
        }
        report(verified ? "on (\(cases) cases bitwise equal)"
                        : "off (differs from the scan decay launch and matmul after \(cases) cases)")
        qualifyFresh()
    }
}

/// Seed-only (request prefill, `NemotronHPrefillX5.active`): a scan window's
/// state product, `dtxDecayT` (`dtx[t, h, i] * last[h, t]` written as the
/// [b, g, R, dh, n] left operand) and its FP32 matmul with the window's FP32 B
/// block, in one launch that never writes the left operand. Each threadgroup
/// (one head) forms the operand 32 tokens at a time with `dtxDecayT`'s
/// product and multiplies it with the MLX NAX GEMM loop's fragments (32 x 32
/// simdgroup tiles, K in 32-token steps from 0 with the last step
/// zero-padded, `tile_matmad_nax` per step). `qualify` compares the product
/// with the two launches bit for bit. NEMOTRON_PREFILL_STATE_GEMM=0 keeps them.
enum NemotronPrefillStateGemm {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_STATE_GEMM"] != "0"
    nonisolated(unsafe) static var verified = false

    static let kernel = MLXFast.metalKernel(
        name: "nemotron_ssm_state_gemm",
        inputNames: ["dtx", "last", "B", "n"], outputNames: ["out"],
        source: """
            constexpr short SM = 32;
            constexpr short SN = 32;
            constexpr short SK = 32;
            constexpr short TM = SM / 16;
            constexpr short TN = SN / 16;
            constexpr short TK = SK / 16;
            constexpr int AS = SK + 1;
            threadgroup float As[DH * AS];
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const int tid = int(sg) * 32 + int(lane);
            const uint bh = threadgroup_position_in_grid.z;
            const uint bb = bh / H;
            const uint hh = bh % H;
            const uint gg = hh / R;
            const short tm = SM * short(sg / 4);
            const short tn = SN * short(sg % 4);
            const device float* xr = dtx + size_t(bb) * n * H * DH + size_t(hh) * DH;
            const device float* er = last + size_t(bh) * n;
            const device float* bp0 = B + (size_t(bb) * (H / R) + gg) * size_t(n) * DS;
            NAXTile<float, TM, TN> Dtile;
            Dtile.clear();
            for (int k0 = 0; k0 < n; k0 += SK) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int e = tid; e < DH * SK; e += 256) {
                    const int i = e % DH;
                    const int t = e / DH;
                    const int tt = k0 + t;
                    float v = 0.0f;
                    if (tt < n) {
                        const float a = xr[size_t(tt) * H * DH + i];
                        const float ee = er[tt];
                        v = a * ee;
                    }
                    As[i * AS + t] = v;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                NAXTile<float, TM, TK> Atile;
                NAXTile<float, TK, TN> Btile;
                Atile.template load<float, AS, 1>(As + tm * AS);
                const device float* bp = bp0 + size_t(k0) * DS + tn;
                if (k0 + SK <= n) {
                    Btile.load(bp, DS);
                } else {
                    Btile.load_safe(bp, DS, short2(SN, short(n - k0)));
                }
                tile_matmad_nax(
                    Dtile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<false>{});
            }
            Dtile.store(out + (size_t(bh) * DH + tm) * DS + tn, DS);
        """,
        header: nemotronRunGatherHeader)

    /// `dtxDecayT(dtx, last).matmul(bWindow).reshaped(b, h, dh, d)`; nil
    /// outside the request prefill or the kernel's shapes.
    static func apply(dtx: MLXArray, last: MLXArray, bWindow: MLXArray, n: Int, h: Int, g: Int, dh: Int, d: Int)
        -> MLXArray?
    {
        guard verified, NemotronHPrefillX5.active else { return nil }
        return launch(dtx: dtx, last: last, bWindow: bWindow, n: n, h: h, g: g, dh: dh, d: d)
    }

    private static func launch(dtx: MLXArray, last: MLXArray, bWindow: MLXArray, n: Int, h: Int, g: Int,
                               dh: Int, d: Int) -> MLXArray? {
        guard NemotronHPrefillX5.runGather, dh == 64, d == 128, g > 0, h % g == 0,
              dtx.dtype == .float32, last.dtype == .float32, bWindow.dtype == .float32,
              dtx.shape == [1, n, h, dh], last.shape == [1, h, 1, n], bWindow.shape == [1, g, 1, n, d]
        else { return nil }
        return kernel(
            [dtx, last, bWindow, n],
            template: [("H", h), ("R", h / g), ("DH", dh), ("DS", d)],
            grid: (256, 1, h), threadGroup: (256, 1, 1),
            outputShapes: [[1, h, dh, d]], outputDTypes: [.float32])[0]
    }

    static func qualify() {
        verified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron prefill state gemm: \(message)\n".utf8))
        }
        guard enabled, NemotronHPrefillX5.enabled, NemotronHPrefillX5.runGather else {
            return report("off (setting)")
        }
        let (h, g, dh, d) = (64, 8, 64, 128)
        let specials = MLXArray([Float(-0.0), 0, .infinity, -.infinity, .nan, Float(bitPattern: 0x0001_0000),
                                 Float(bitPattern: 0x8001_0000), 3e38])
        var cases = 0
        do {
            let matches = try withError { () -> Bool in
                for (seed, (n, l, start)) in [(256, 488, 0), (232, 488, 256), (256, 512, 256), (100, 356, 256),
                                              (64, 64, 0)].enumerated() {
                    let key = MLXRandom.split(key: MLXRandom.key(UInt64(0x57a7 + seed)), into: 3)
                    var dtxFull = MLXRandom.normal([1, l, h, dh], key: key[0]) * Float(1 + seed)
                    if seed == 1 {
                        dtxFull = concatenated([specials, dtxFull.reshaped(-1)[8...]]).reshaped(1, l, h, dh)
                    }
                    var last = MLX.exp(-MLX.abs(MLXRandom.normal([1, h, 1, n], key: key[1])))
                    if seed == 2 {
                        last = concatenated([MLXArray([Float(0), .infinity, .nan, 1e-30]),
                                             last.reshaped(-1)[4...]]).reshaped(1, h, 1, n)
                    }
                    let flat = MLXRandom.normal([l * g * d], key: key[2]) * Float(2 + seed)
                    let bWindow = nemotronPrefillBF32Window(flat, start: start, end: start + n, groups: g, state: d)
                    let dtx = dtxFull[0..., start ..< (start + n), 0..., 0...]
                    let operand = NemotronHScanKernels.shared.dtxDecayT(
                        [dtx, last, n],
                        template: [("H", h), ("DH", dh)],
                        grid: (((n + 31) / 32) * 32, dh / 32 * 8, h), threadGroup: (32, 8, 1),
                        outputShapes: [[1, g, h / g, dh, n]], outputDTypes: [.float32])[0]
                    let stock = operand.matmul(bWindow).reshaped(1, h, dh, d)
                    guard let fused = launch(dtx: dtx, last: last, bWindow: bWindow, n: n, h: h, g: g, dh: dh, d: d)
                    else { return false }
                    let same = (fused.view(dtype: .uint32) .== stock.view(dtype: .uint32)).all()
                    if !same.item(Bool.self) { return false }
                    cases += 1
                }
                return true
            }
            verified = matches
        } catch {
            return report("off (\(error))")
        }
        report(verified ? "on (\(cases) cases bitwise equal)"
                        : "off (differs from the transpose launch and matmul after \(cases) cases)")
    }
}

/// Seed-only: `windowOutFinalSG` (two simdgroups of eight rows per
/// threadgroup) in place of `windowOutFinal`; `qualify` checks it bit for bit,
/// with and without the copied prefix. NEMOTRON_PREFILL_WINDOW_OUT_SG=0 keeps
/// `windowOutFinal`.
enum NemotronPrefillWindowOutSG {
    static let sgs = Int(ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_WINDOW_OUT_SG"] ?? "") ?? 2
    nonisolated(unsafe) static var verified = false

    static let rows = 8
    static func run(_ kernel: MLXFast.MLXFastKernel, sgs: Int, rb: Int = rows, _ inputs: [any ScalarOrArray],
                    ct: DType, ot: DType, h: Int, r: Int, dh: Int, lt: Int, copy: Bool, n: Int, copyBlocks: Int,
                    b: Int, outRows: Int) -> MLXArray {
        let original = sgs == 0
        var template: [(String, any KernelTemplateArg)] = [("CT", ct), ("OT", ot), ("H", h), ("R", r),
                       ("DH", dh), ("TC", 32), ("LT", lt), ("COPY", copy)]
        if !original { template.append(("SGS", sgs)); template.append(("SRB", rb)) }
        return kernel(
            inputs, template: template,
            grid: (original ? dh / 32 * 128 : dh / (rb * sgs) * 32 * sgs, (n + 31) / 32 + copyBlocks, b * h),
            threadGroup: (original ? 128 : 32 * sgs, 1, 1),
            outputShapes: [[b, outRows, h, dh]], outputDTypes: [ot])[0]
    }

    static func qualify() {
        verified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron prefill window out sg: \(message)\n".utf8))
        }
        guard sgs > 0, NemotronHPrefillX5.enabled, NemotronHPrefillX5.ssmEpilogue else { return report("off (setting)") }
        let (h, g, dh, d) = (64, 8, 64, 128)
        var ok = MLXArray(true)
        var cases = 0
        for (seed, (n, l, start)) in [(256, 488, 0), (232, 488, 256), (100, 100, 0), (17, 273, 256)].enumerated() {
            let key = MLXRandom.split(key: MLXRandom.key(UInt64(0x5a5a + seed)), into: 7)
            let state = MLXRandom.normal([1, h, dh, d], key: key[0]) * Float(1 + seed)
            let C = (MLXRandom.normal([1, n, g, d], key: key[1]) * Float(1 + seed)).asType(.bfloat16)
            let y = MLXRandom.normal([1, h, n, dh], key: key[2])
            let ec = MLX.exp(-MLX.abs(MLXRandom.normal([1, n, h], key: key[3])))
            let x = MLXRandom.normal([1, l, h, dh], key: key[4]).asType(.bfloat16)
            let D = MLXRandom.normal([h], key: key[5]).asType(.bfloat16)
            let last = start + n == l
            let copied = last ? start : 0
            let prior = copied == 0 ? C : MLXRandom.normal([1, copied, h, dh], key: key[6]).asType(.bfloat16)
            let copyBlocks = (copied + 31) / 32
            let inputs: [any ScalarOrArray] = [state, C, y, ec, n, x, D, prior, copied, start]
            func go(_ kernel: MLXFast.MLXFastKernel, _ sg: Int, rb: Int = rows) -> MLXArray {
                run(kernel, sgs: sg, rb: rb, inputs, ct: C.dtype, ot: x.dtype, h: h, r: h / g, dh: dh,
                    lt: last ? l : n, copy: copied > 0, n: n, copyBlocks: copyBlocks, b: 1, outRows: last ? l : n)
            }
            let a = go(NemotronHScanKernels.shared.windowOutFinal, 0)
            let bOut = go(NemotronHScanKernels.shared.windowOutFinalSG, sgs)
            ok = MLX.logicalAnd(ok, (a.view(dtype: .uint16) .== bOut.view(dtype: .uint16)).all())
            eval(ok)
            cases += 1
        }
        verified = ok.item(Bool.self)
        report(verified ? "on (sgs \(sgs), \(cases) cases bitwise equal)" : "off (differs from windowOutFinal)")
    }
}

/// Seed-only: the request prefill's gated RMS norm through
/// `NemotronHGatedRMSKernel.wideKernel` (vector loads and stores, G groups per
/// threadgroup) when the rows are contiguous. `qualify` checks it against the
/// kernel bit for bit. NEMOTRON_PREFILL_GATED_RMS_WIDE=0 keeps the kernel.
enum NemotronPrefillGatedRMSWide {
    static let groupsPerThreadgroup =
        Int(ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_GATED_RMS_WIDE"] ?? "") ?? 2
    nonisolated(unsafe) static var verified = false

    static func apply(_ x: MLXArray, gate: MLXArray, weight: MLXArray, eps: Float, g: Int = groupsPerThreadgroup)
        -> MLXArray?
    {
        guard g > 0, x.ndim == 3, x.strides.last == 1, gate.strides.last == 1, weight.strides.last == 1,
              x.strides[1] % 4 == 0, gate.strides[1] % 4 == 0 else { return nil }
        let total = x.dim(1) * (x.dim(2) / 512)
        return NemotronHGatedRMSKernel.wideKernel(
            [x, gate, weight, NemotronHGatedRMSKernel.identity, eps],
            template: [("T", DType.bfloat16), ("G", g)],
            grid: (128 * g * ((total + g - 1) / g), 1, 1),
            threadGroup: (128 * g, 1, 1), outputShapes: [x.shape], outputDTypes: [.bfloat16])[0]
    }

    static func qualify() {
        verified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron prefill gated rms wide: \(message)\n".utf8))
        }
        guard groupsPerThreadgroup > 0, NemotronHPrefillX5.enabled else { return report("off (setting)") }
        var ok = MLXArray(true)
        var cases = 0
        for (seed, rows) in [488, 256, 17, 3].enumerated() {
            let key = MLXRandom.split(key: MLXRandom.key(UInt64(0x96a7 + seed)), into: 3)
            let x = (MLXRandom.normal([1, rows, 4096], key: key[0]) * Float(1 + seed)).asType(.bfloat16)
            let gate = (MLXRandom.normal([1, rows, 4096], key: key[1]) * 2).asType(.bfloat16)
            let weight = (MLXRandom.normal([4096], key: key[2])).asType(.bfloat16)
            let eps = Float(1e-5)
            let a = NemotronHGatedRMSKernel.kernel(
                [x, gate, weight, NemotronHGatedRMSKernel.identity, eps],
                template: [("T", DType.bfloat16)],
                grid: (128 * rows * 8, 1, 1),
                threadGroup: (128, 1, 1), outputShapes: [x.shape], outputDTypes: [.bfloat16])[0]
            guard let b = apply(x, gate: gate, weight: weight, eps: eps) else { return report("off (layout)") }
            ok = MLX.logicalAnd(ok, (a.view(dtype: .uint16) .== b.view(dtype: .uint16)).all())
            eval(ok)
            cases += 1
        }
        verified = ok.item(Bool.self)
        report(verified ? "on (G \(groupsPerThreadgroup), \(cases) cases bitwise equal)" : "off (differs from the kernel)")
    }
}

/// Seed-only: `scanDecaySkip` in place of `scanDecay` (same grid and outputs);
/// `qualify` checks both outputs bit for bit. NEMOTRON_PREFILL_SCAN_DECAY_SKIP=0
/// keeps `scanDecay`.
enum NemotronPrefillScanDecaySkip {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_SCAN_DECAY_SKIP"] != "0"
    nonisolated(unsafe) static var verified = false

    static func qualify() {
        verified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron prefill scan decay skip: \(message)\n".utf8))
        }
        guard enabled, NemotronHPrefillX5.enabled else { return report("off (setting)") }
        let (h, g) = (64, 8)
        var ok = MLXArray(true)
        var cases = 0
        for (seed, n) in [256, 232, 100, 17, 1].enumerated() {
            let key = MLXRandom.split(key: MLXRandom.key(UInt64(0x5c1d + seed)), into: 2)
            let dtA = -MLX.abs(MLXRandom.normal([1, n, h], key: key[0])) * Float(0.1 * Double(1 + seed))
            let CB = (MLXRandom.normal([1, g, n, n], key: key[1]) * Float(1 + seed)).asType(.bfloat16)
            func go(_ kernel: MLXFast.MLXFastKernel) -> [MLXArray] {
                kernel([dtA, CB, n], template: [("CBT", CB.dtype), ("H", h), ("R", h / g)],
                       grid: (((n + 7) / 8) * 256, 1, h), threadGroup: (256, 1, 1),
                       outputShapes: [[1, h, n, n], [1, h, 1, n]], outputDTypes: [.float32, .float32])
            }
            let a = go(NemotronHScanKernels.shared.scanDecay)
            let b = go(NemotronHScanKernels.shared.scanDecaySkip)
            for (u, w) in zip(a, b) {
                ok = MLX.logicalAnd(ok, (u.view(dtype: .uint32) .== w.view(dtype: .uint32)).all())
            }
            eval(ok)
            cases += 1
        }
        verified = ok.item(Bool.self)
        report(verified ? "on (\(cases) cases bitwise equal)" : "off (differs from scanDecay)")
    }
}

extension NemotronHResidualRMSNorm {
    /// `source` with the request prefill's MoE output as `rhs`: the
    /// `combineAdd` statements (the K weighted sorted expert rows `y[inv]` of
    /// the row summed from +0.0 in slot order, astype(T), then the shared add
    /// of `addend`) feed the unchanged residual add and norm, so the result
    /// is `lhs + (y + shared)` as the stock MoE block forms it.
    private static func moeJoinSource() -> String {
        let offset = "const size_t offset = size_t(gid) * axis_size + lid * n_reads;"
        let add = "T sum = lhs[offset + i] + rhs[offset + i];"
        precondition(source.components(separatedBy: offset).count == 2)
        precondition(source.components(separatedBy: add).count == 2)
        return source.replacingOccurrences(of: offset, with: offset + """

                        float total[n_reads];
                        for (int k = 0; k < K; ++k) {
                            uint src = inv[gid * K + k];
                            float s = scores[gid * K + k];
                            const device T* row = y + size_t(src) * axis_size + lid * n_reads;
                            for (uint j = 0; j < n_reads; ++j) {
                                float v = static_cast<float>(row[j]);
                                float p = v * s;
                                float a = p + 0.0f;
                                if (k == 0) {
                                    total[j] = a;
                                } else {
                                    total[j] = a + total[j];
                                }
                            }
                        }
            """)
            .replacingOccurrences(of: add, with: """
            const T r = static_cast<T>(total[i]);
                            T part = static_cast<T>(static_cast<float>(r) + static_cast<float>(addend[offset + i]));
                            T sum = lhs[offset + i] + part;
            """)
    }
    private static let moeJoinKernel = MLXFast.metalKernel(
        name: "nemotron_h_moe_join_rms_bf16",
        inputNames: ["lhs", "y", "inv", "scores", "addend", "weight", "axis_size", "eps"],
        outputNames: ["residual", "normalized"],
        source: moeJoinSource(), ensureRowContiguous: true)

    /// Request prefill MoE block join (`NemotronPrefillMoEJoin`): `y` is the
    /// [rows * k, 2688] sorted expert rows, `inverse` / `scores` the k slots
    /// of each row; (residual, normalized) as `combineAdd` then `apply`.
    static func applyMoEJoin(
        _ lhs: MLXArray, y: MLXArray, inverse: MLXArray, scores: MLXArray, addend: MLXArray,
        k: Int, norm: RMSNorm
    ) -> (MLXArray, MLXArray)? {
        guard Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
            lhs.ndim == 3, lhs.dim(0) == 1, lhs.dim(1) > 0, lhs.dim(2) == 2688,
            lhs.dtype == .bfloat16, addend.shape == lhs.shape, addend.dtype == .bfloat16,
            (1...8).contains(k), y.dtype == .bfloat16, y.size == lhs.dim(1) * k * 2688,
            inverse.dtype == .uint32, inverse.size == lhs.dim(1) * k,
            scores.dtype == .float32, scores.size == inverse.size,
            type(of: norm) == RMSNorm.self,
            norm.weight.shape == [2688], norm.weight.dtype == .bfloat16
        else { return nil }
        let result = moeJoinKernel(
            [lhs, y, inverse, scores, addend, norm.weight, UInt32(lhs.dim(2)), norm.eps],
            template: [("T", DType.bfloat16), ("K", k)],
            grid: (672 * lhs.dim(1), 1, 1), threadGroup: (672, 1, 1),
            outputShapes: [lhs.shape, lhs.shape], outputDTypes: [.bfloat16, .bfloat16])
        return (result[0], result[1])
    }
}

/// Seed-only (request prefill, `NemotronHPrefillX5.active`): each MoE
/// block's join with the next block's norm (norm_f after the last block)
/// rides the combine-add launch (`NemotronHResidualRMSNorm.applyMoEJoin`), one
/// launch per row instead of the combine add and the residual + norm launch,
/// and the MoE output is no longer stored and reloaded (23 launches per seed;
/// only where the combine add runs, L > 10 rows). `qualify` checks the
/// (residual, normalized) pair bitwise against the stock graph (unsort
/// gather, score multiply, K-way sum, astype, shared add, residual add,
/// rms_norm) and against the two launches it replaces, for every norm these
/// joins use, on seeded rows at L = 488, 300 and 9 with special values in the
/// expert rows, the shared output and the residual, and zero and unit scores.
/// NEMOTRON_PREFILL_MOE_JOIN=0 keeps the two launches.
enum NemotronPrefillMoEJoin {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_MOE_JOIN"] != "0"
    nonisolated(unsafe) static var verified = false

    static func qualify(_ target: Module) {
        verified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron prefill moe join: \(message)\n".utf8))
        }
        guard enabled, NemotronHPrefillX5.enabled, NemotronPrefillCombineAdd.verified else {
            return report("off (setting)")
        }
        guard let backbone = target.modules().compactMap({ $0 as? NemotronHBackbone }).first,
              let moe = backbone.layers.lazy.compactMap({ $0.mixer as? NemotronHMoE }).first
        else { return report("off (no MoE block)") }
        let layers = backbone.layers
        // The norm of each MoE block's join: the next block's, norm_f after the last.
        let norms = layers.indices.filter { layers[$0].blockType == .moe }
            .map { $0 + 1 < layers.count ? layers[$0 + 1].norm : backbone.normF }
        let (k, width) = (moe.numExpertsPerTok, backbone.normF.weight.dim(0))
        // +-0, +-inf, +-NaN, +-subnormal, +-max, 1 (BF16 bits).
        let specials: [UInt16] = [
            0x0000, 0x8000, 0x7f80, 0xff80, 0x7fc0, 0xffc0, 0x0001, 0x8001, 0x7f7f, 0xff7f, 0x3f80,
        ]
        func equal(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
            (a.view(dtype: .uint16) .== b.view(dtype: .uint16)).all()
        }
        var checked = 0
        var failure = "outputs differ"
        do {
            let matches = try withError { () -> Bool in
                for (index, rows) in [488, 300, 9].enumerated() {
                    let keys = MLXRandom.split(key: MLXRandom.key(UInt64(0x30e1_0000 + index)), into: 5)
                    // Special j goes to row 3j of the residual, slot j % k of row
                    // 3j + 1 of the expert rows and row 3j + 2 of the shared output
                    // (one per row: a row's norm reads it whole); later rows stay clean.
                    let count = min(specials.count, (rows - 1) / 3)
                    func mark(_ a: MLXArray, column: Int, at row: (Int) -> Int) -> MLXArray {
                        var value = [UInt16](repeating: 0, count: a.size / width)
                        var hit = [Bool](repeating: false, count: value.count)
                        for j in 0 ..< count {
                            value[row(j)] = specials[j]
                            hit[row(j)] = true
                        }
                        let columns = MLXArray(0 ..< width).reshaped(1, width) .== MLXArray(Int32(column))
                        return which(MLXArray(hit).reshaped(-1, 1) .&& columns,
                                     MLXArray(value).view(dtype: .bfloat16).reshaped(-1, 1),
                                     a.reshaped(-1, width)).reshaped(a.shape)
                    }
                    let lhs = mark(MLXRandom.normal([1, rows, width], dtype: .bfloat16, scale: 8, key: keys[0]),
                                   column: 7) { 3 * $0 }
                    let unsorted = mark(MLXRandom.normal([rows * k, width], dtype: .bfloat16, scale: 4,
                                                         key: keys[1]),
                                        column: 1001) { (3 * $0 + 1) * k + $0 % k }
                    let addend = mark(MLXRandom.normal([1, rows, width], dtype: .bfloat16, scale: 4,
                                                       key: keys[2]),
                                      column: 2002) { 3 * $0 + 2 }
                    // The slot holding an expert-row special weighs it 0 (odd j) or 1 (even j).
                    var weight = [Float](repeating: 0, count: rows * k)
                    var weighted = [Bool](repeating: false, count: rows * k)
                    for j in 0 ..< count {
                        weight[(3 * j + 1) * k + j % k] = j % 2 == 0 ? 1 : 0
                        weighted[(3 * j + 1) * k + j % k] = true
                    }
                    let scores = which(MLXArray(weighted).reshaped(1, rows, k), MLXArray(weight).reshaped(1, rows, k),
                                       MLXRandom.uniform(0 ..< 1, [1, rows, k], key: keys[3]))
                    // Sorted row i holds slot order[i]; slot s sits at sorted row inverse[s].
                    let order = argSort(MLXRandom.uniform(0 ..< 1, [rows * k], key: keys[4]))
                    let inverse = argSort(order).asType(.uint32)
                    let y = unsorted[order]
                    // Stock graph: unsort gather, fp32 score multiply, K-way sum,
                    // astype, the MoE's shared add, then the block's residual add.
                    let routed = (y[inverse].reshaped(1, rows, k, width) * scores[.ellipsis, .newAxis])
                        .sum(axis: -2).asType(.bfloat16)
                    let residual = lhs + (routed + addend)
                    // The launches the join replaces: combine add, then residual + norm.
                    let combined = NemotronHPrefillKernels.shared.combineAdd(
                        [y, inverse, scores, width, addend], template: [("T", DType.bfloat16), ("K", k)],
                        grid: (width / 4, rows, 1), threadGroup: (32, 1, 1),
                        outputShapes: [[1, rows, width]], outputDTypes: [.bfloat16])[0]
                    eval(lhs, y, inverse, scores, addend, residual, combined)
                    for norm in norms {
                        guard let fused = NemotronHResidualRMSNorm.applyMoEJoin(
                                  lhs, y: y, inverse: inverse, scores: scores, addend: addend, k: k, norm: norm),
                              let launches = NemotronHResidualRMSNorm.apply(lhs, combined, norm: norm)
                        else {
                            failure = "a launch refused the shapes"
                            return false
                        }
                        if !(equal(fused.0, residual) .&& equal(fused.1, norm(residual))).item(Bool.self) {
                            failure = "outputs differ from the stock graph"
                            return false
                        }
                        if !(equal(fused.0, launches.0) .&& equal(fused.1, launches.1)).item(Bool.self) {
                            failure = "outputs differ from combine add + residual norm"
                            return false
                        }
                        checked += 1
                    }
                }
                return true
            }
            verified = matches
        } catch {
            return report("off (\(error))")
        }
        report(verified ? "on (\(checked) cases bitwise equal)" : "off (\(failure) after \(checked) cases)")
    }
}

/// Seed-only: `dtxDecayT4` (float4 rows, 32 x 64 tiles) in place of
/// `dtxDecayT` when the rows are contiguous and 16-byte aligned; `qualify`
/// checks it bit for bit. NEMOTRON_PREFILL_DTX_DECAY_VEC=0 keeps `dtxDecayT`.
enum NemotronPrefillDtxDecay4 {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_DTX_DECAY_VEC"] != "0"
    nonisolated(unsafe) static var verified = false

    static func usable(_ dtx: MLXArray, n: Int, dh: Int) -> Bool {
        verified && dh == 64 && dtx.ndim == 4 && dtx.strides[3] == 1 && dtx.strides[2] % 4 == 0
            && dtx.strides[1] % 4 == 0 && dtx.strides[0] % 4 == 0
    }

    static func qualify() {
        verified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron prefill dtx decay vec: \(message)\n".utf8))
        }
        guard enabled, NemotronHPrefillX5.enabled else { return report("off (setting)") }
        let (h, g, dh) = (64, 8, 64)
        var same = MLXArray(true)
        for (seed, (n, l, start)) in [(256, 488, 0), (232, 488, 256), (17, 17, 0), (100, 300, 200)].enumerated() {
            let key = MLXRandom.split(key: MLXRandom.key(UInt64(0xd7c4 + seed)), into: 2)
            let full = MLXRandom.normal([1, l, h, dh], key: key[0])
            let dtx = full[0..., start ..< (start + n), 0..., 0...]
            let last = MLX.exp(-MLX.abs(MLXRandom.normal([1, h, 1, n], key: key[1])))
            let a = NemotronHScanKernels.shared.dtxDecayT(
                [dtx, last, n], template: [("H", h), ("DH", dh)],
                grid: (((n + 31) / 32) * 32, dh / 32 * 8, h), threadGroup: (32, 8, 1),
                outputShapes: [[1, g, h / g, dh, n]], outputDTypes: [.float32])[0]
            let b = NemotronHScanKernels.shared.dtxDecayT4(
                [dtx, last, n], template: [("H", h), ("DH", dh)],
                grid: (((n + 31) / 32) * 256, 1, h), threadGroup: (256, 1, 1),
                outputShapes: [[1, g, h / g, dh, n]], outputDTypes: [.float32])[0]
            same = same .&& (a.view(dtype: .uint32) .== b.view(dtype: .uint32)).all()
            eval(same)
        }
        verified = same.item(Bool.self)
        report(verified ? "on (4 cases bitwise equal)" : "off (differs from dtxDecayT)")
    }
}

/// Seed-only: a fresh request's first scan window has an all +0.0 input
/// state, so the window output's `state . C` term is +0.0 for every (token,
/// row) and `windowOutFinal` reduces to its epilogue. `qualify` checks the
/// shortcut against `windowOutFinal` on the zero state view, bit for bit.
/// NEMOTRON_PREFILL_FRESH_WINDOW_OUT=0 keeps `windowOutFinal`.
enum NemotronPrefillFreshWindowOut {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_FRESH_WINDOW_OUT"] != "0"
    nonisolated(unsafe) static var verified = false

    static func apply(y: MLXArray, ec: MLXArray, n: Int, x: MLXArray, D: MLXArray, lt: Int) -> MLXArray {
        let (b, _, h, dh) = x.shape4
        return NemotronHScanKernels.shared.windowOutFresh(
            [y, ec, n, x, D],
            template: [("OT", x.dtype), ("H", h), ("DH", dh), ("LT", lt)],
            grid: (dh / 4, n, b * h), threadGroup: (dh / 4, max(1, 1024 / dh), 1),
            outputShapes: [[b, lt, h, dh]], outputDTypes: [x.dtype])[0]
    }

    static func qualify() {
        verified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron prefill fresh window out: \(message)\n".utf8))
        }
        guard enabled, NemotronHPrefillX5.enabled, NemotronHPrefillX5.ssmEpilogue else {
            return report("off (setting)")
        }
        let (h, g, dh, d) = (64, 8, 64, 128)
        var ok = MLXArray(true)
        var cases = 0
        for (seed, (n, l)) in [(256, 488), (256, 300), (100, 100), (488, 488)].enumerated() {
            let key = MLXRandom.split(key: MLXRandom.key(UInt64(0x5eed + seed)), into: 5)
            let C = (MLXRandom.normal([1, n, g, d], key: key[0]) * Float(1 + seed)).asType(.bfloat16)
            let y = MLXRandom.normal([1, h, n, dh], key: key[1]) * Float(1 + seed)
            let ec = MLX.exp(-MLX.abs(MLXRandom.normal([1, n, h], key: key[2])))
            let x = MLXRandom.normal([1, l, h, dh], key: key[3]).asType(.bfloat16)
            let D = MLXRandom.normal([h], key: key[4]).asType(.bfloat16)
            let last = n == l
            let state = NemotronMambaZeroState.view([1, h, dh, d], dtype: .float32)
            let stock = NemotronHScanKernels.shared.windowOutFinal(
                [state, C, y, ec, n, x, D, C, 0, 0],
                template: [("CT", C.dtype), ("OT", x.dtype), ("H", h), ("R", h / g),
                           ("DH", dh), ("TC", 32), ("LT", last ? l : n), ("COPY", false)],
                grid: (dh / 32 * 128, (n + 31) / 32, h), threadGroup: (128, 1, 1),
                outputShapes: [[1, last ? l : n, h, dh]], outputDTypes: [x.dtype])[0]
            let fresh = apply(y: y, ec: ec, n: n, x: x, D: D, lt: last ? l : n)
            ok = MLX.logicalAnd(ok, (stock.view(dtype: .uint16) .== fresh.view(dtype: .uint16)).all())
            eval(ok)
            cases += 1
        }
        verified = ok.item(Bool.self)
        report(verified ? "on (\(cases) cases bitwise equal)" : "off (differs from the window output kernel)")
    }
}

/// Fresh-request Mamba prefill: a request with no recurrent input state
/// starts from zero conv and SSM state. Instead of materialising
/// [1, 3, convDim] BF16 and [1, H, Dh, Ds] FP32 zeros per layer (about 49 MB
/// per request on the 23 Mamba layers), the conv history and the window
/// output read a zero-stride view of one +0.0 scalar and the first scan
/// window adds `decay * 0` per head. The values every kernel reads are the
/// same +0.0, so outputs are bitwise identical, signed zeros and NaN
/// propagation included. Restored, carried or partial state keeps the
/// existing path. NEMOTRON_MAMBA_ZERO_STATE_SKIP=0 restores the zero buffers.
enum NemotronMambaZeroState {
    static let enabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MAMBA_ZERO_STATE_SKIP"] != "0"
    nonisolated(unsafe) private static let float32Zero = MLXArray(Float(0))
    nonisolated(unsafe) private static let bfloat16Zero = MLXArray(Float(0)).asType(.bfloat16)

    /// +0.0 of `dtype`, the value `MLXArray.zeros(_:dtype:)` fills with.
    static func scalar(_ dtype: DType) -> MLXArray {
        switch dtype {
        case .float32: return float32Zero
        case .bfloat16: return bfloat16Zero
        default: return MLXArray(Float(0)).asType(dtype)
        }
    }

    /// `zeros(shape, dtype)` as a broadcast view: no buffer of `shape`.
    static func view(_ shape: [Int], dtype: DType) -> MLXArray {
        MLX.broadcast(scalar(dtype), to: shape)
    }
}

private let nemotronCompiledDecodeSSM = compile(shapeless: false) { (inputs: [MLXArray]) -> [MLXArray] in
    let (output, state) = ssmUpdate(
        hiddenStates: inputs[0], ALog: inputs[1], B: inputs[2], C: inputs[3], D: inputs[4],
        dt: inputs[5], dtBias: inputs[6], state: inputs[7], timeStepLimit: (0, .infinity), mask: nil)
    return [output, state]
}

func nemotronDecodeSSM(
    hiddenStates: MLXArray, ALog: MLXArray, B: MLXArray, C: MLXArray, D: MLXArray,
    dt: MLXArray, dtBias: MLXArray, state: MLXArray, timeStepLimit: (Float, Float)
) -> (MLXArray, MLXArray) {
    guard Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
          hiddenStates.dim(1) == 1,
          [DType.bfloat16, .float16, .float32].contains(dt.dtype),
          [DType.bfloat16, .float16, .float32].contains(dtBias.dtype),
          timeStepLimit.0.bitPattern == 0, timeStepLimit.1 == .infinity
    else {
        return ssmUpdate(
            hiddenStates: hiddenStates, ALog: ALog, B: B, C: C, D: D, dt: dt,
            dtBias: dtBias, state: state, timeStepLimit: timeStepLimit, mask: nil)
    }
    let outputs = nemotronCompiledDecodeSSM([hiddenStates, ALog, B, C, D, dt, dtBias, state])
    return (outputs[0], outputs[1])
}

// MARK: - Mamba2Mixer

private enum NemotronPackedConvKernel {
    static let kernel = MLXFast.metalKernel(
        name: "nemotron_packed_conv_silu",
        inputNames: ["oldState", "input", "weight", "bias"],
        outputNames: ["out", "padded"],
        source: """
            int64_t channel = int64_t(thread_position_in_grid.x);
            int64_t position = int64_t(thread_position_in_grid.y);
            padded[(position + 3) * C + channel] =
                input[position * input_strides[1] + channel * input_strides[2]];
            if (position == 0) {
                for (int i = 0; i < 3; ++i) {
                    padded[int64_t(i) * C + channel] =
                        oldState[int64_t(i) * oldState_strides[1] + channel * oldState_strides[2]];
                }
            }
            float acc = 0;
            for (int i = 0; i < 4; ++i) {
                int64_t row = position + int64_t(i);
                T value = row < 3
                    ? oldState[row * oldState_strides[1] + channel * oldState_strides[2]]
                    : input[(row - 3) * input_strides[1] + channel * input_strides[2]];
                acc += static_cast<float>(value)
                    * weight[channel * weight_strides[0] + int64_t(i) * weight_strides[1]];
            }
            T convolved = static_cast<T>(acc);
            T x = convolved + bias[channel * bias_strides[0]];
            auto y = 1 / (1 + metal::exp(metal::abs(x)));
            T sigmoid = x < 0 ? y : 1 - y;
            out[position * C + channel] = x * sigmoid;
            """, ensureRowContiguous: false)
}


/// `nemotron_packed_conv_silu` plus the verify window's timestep transform on
/// H extra threads (x = C + head): softplus(float(dt) + bias) as
/// LogAddExp(v, 0), then clip as Maximum then Minimum, i.e. the expressions of
/// `nemotronCompiledPrefillDt`. One launch instead of two; the conv threads
/// run the packed-conv text unchanged.
private enum NemotronPackedConvDtKernel {
    static let kernel = MLXFast.metalKernel(
        name: "nemotron_packed_conv_silu_dt",
        inputNames: ["oldState", "input", "weight", "bias", "dtIn", "dtBias", "dtLo", "dtHi"],
        outputNames: ["out", "padded", "dt"],
        source: """
            int64_t channel = int64_t(thread_position_in_grid.x);
            int64_t position = int64_t(thread_position_in_grid.y);
            if (channel >= C) {
                int64_t head = channel - C;
                if (head >= H) return;
                float v = static_cast<float>(
                    dtIn[position * dtIn_strides[1] + head * dtIn_strides[2]])
                    + static_cast<float>(dtBias[head * dtBias_strides[0]]);
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
                dt[position * H + head] = metal::isnan(m) ? m : (m < hi ? m : hi);
                return;
            }
            padded[(position + 3) * C + channel] =
                input[position * input_strides[1] + channel * input_strides[2]];
            if (position == 0) {
                for (int i = 0; i < 3; ++i) {
                    padded[int64_t(i) * C + channel] =
                        oldState[int64_t(i) * oldState_strides[1] + channel * oldState_strides[2]];
                }
            }
            float acc = 0;
            for (int i = 0; i < 4; ++i) {
                int64_t row = position + int64_t(i);
                T value = row < 3
                    ? oldState[row * oldState_strides[1] + channel * oldState_strides[2]]
                    : input[(row - 3) * input_strides[1] + channel * input_strides[2]];
                acc += static_cast<float>(value)
                    * weight[channel * weight_strides[0] + int64_t(i) * weight_strides[1]];
            }
            T convolved = static_cast<T>(acc);
            T x = convolved + bias[channel * bias_strides[0]];
            auto y = 1 / (1 + metal::exp(metal::abs(x)));
            T sigmoid = x < 0 ? y : 1 - y;
            out[position * C + channel] = x * sigmoid;
            """, ensureRowContiguous: false)
}

private enum NemotronRequestConvHistory {
    static let copy = MLXFast.metalKernel(
        name: "nemotron_request_conv_history_copy",
        inputNames: ["source"], outputNames: ["out"],
        source: """
            uint channel = thread_position_in_grid.x;
            uint row = thread_position_in_grid.y;
            uint batch = thread_position_in_grid.z;
            out[(batch * N + row) * C + channel] = source[
                batch * source_strides[0] + row * source_strides[1] + channel * source_strides[2]];
            """, ensureRowContiguous: false)
}

/// Seed-only request-prefill front of a Mamba layer in one launch: the
/// stock `silu(conv1d(concatenated([oldState, input])))` (the statements of
/// `nemotron_packed_conv_silu`), the conv history rows, and the scan inputs
/// `dtA = dt * A` and `dtx = dt * float(x)` with dt the statements of
/// `nemotron_packed_conv_silu_dt` (= `nemotronCompiledPrefillDt`). Each x
/// thread recomputes its head's dt with those statements; every output
/// element is one product of the same operands as the stock op. Replaces
/// the concatenate, depthwise conv, bias add, SiLU, dt transform, dtA and
/// dtx multiplies, the x promotion and the history copy.
/// Used only inside the request prefill (`NemotronHPrefillX5.active`) after
/// `qualify` found all four outputs bitwise equal to the stock graph.
/// NEMOTRON_PREFILL_CONV_DT=0 keeps the stock graph.
enum NemotronPrefillConvDt {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_CONV_DT"] != "0"
    nonisolated(unsafe) static var verified = false

    static let header = """
        inline float nemotron_prefill_dt(float v, float lo, float hi) {
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
            float m = metal::isnan(sp) ? sp : (sp > lo ? sp : lo);
            return metal::isnan(m) ? m : (m < hi ? m : hi);
        }
        """

    static let source = """
            const int is1 = int(input_strides[1]), is2 = int(input_strides[2]);
            const int os1 = int(oldState_strides[1]), os2 = int(oldState_strides[2]);
            const int ws0 = int(weight_strides[0]), ws1 = int(weight_strides[1]);
            const int ds1 = int(dtIn_strides[1]), ds2 = int(dtIn_strides[2]);
            int channel = int(thread_position_in_grid.x);
            int position = int(thread_position_in_grid.y);
            if (channel >= C) {
                int head = channel - C;
                if (head >= H) return;
                float dtv = nemotron_prefill_dt(
                    static_cast<float>(dtIn[position * ds1 + head * ds2])
                    + static_cast<float>(dtBias[head * dtBias_strides[0]]), dtLo, dtHi);
                dtA[position * H + head] = dtv * A[head * A_strides[0]];
                return;
            }
            if (position >= L - 3) {
                hist[(position - (L - 3)) * C + channel] =
                    input[position * is1 + channel * is2];
            }
            float acc = 0;
            for (int i = 0; i < 4; ++i) {
                int row = position + int(i);
                T value = row < 3
                    ? oldState[row * os1 + channel * os2]
                    : input[(row - 3) * is1 + channel * is2];
                acc += static_cast<float>(value)
                    * weight[channel * ws0 + int(i) * ws1];
            }
            T convolved = static_cast<T>(acc);
            T x = convolved + bias[channel * bias_strides[0]];
            auto y = 1 / (1 + metal::exp(metal::abs(x)));
            T sigmoid = x < 0 ? y : 1 - y;
            T o = x * sigmoid;
            out[position * C + channel] = o;
            // x threadgroups (uniform: C and H * DH are multiples of the
            // 256-thread group): the group's 256 / DH heads' dt once each.
            const int c0 = int(threadgroup_position_in_grid.x) * 256;
            if (c0 < H * DH) {
                threadgroup float tdt[256 / DH];
                const uint lt = thread_position_in_threadgroup.x;
                if (lt < 256 / DH) {
                    const int head = c0 / DH + int(lt);
                    tdt[lt] = nemotron_prefill_dt(
                        static_cast<float>(dtIn[position * ds1 + head * ds2])
                        + static_cast<float>(dtBias[head * dtBias_strides[0]]), dtLo, dtHi);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const float dtv = tdt[(channel - c0) / DH];
                dtx[position * (H * DH) + channel] = dtv * static_cast<float>(o);
            }
            """

    static let kernel = MLXFast.metalKernel(
        name: "nemotron_prefill_conv_silu_dt",
        inputNames: ["oldState", "input", "weight", "bias", "dtIn", "dtBias", "dtLo", "dtHi", "A"],
        outputNames: ["out", "hist", "dtA", "dtx"],
        source: source, header: header, ensureRowContiguous: false)

    /// The same launch also writing B for the scan windows' state product
    /// (`bf32Verified`): per window w the contiguous f32 [G, n_w, DS] block
    /// at offset start_w * G * DS of a flat buffer. NEMOTRON_PREFILL_BF32=0
    /// keeps the stock promotion copies.
    static let bf32Enabled = ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_BF32"] != "0"
    nonisolated(unsafe) static var bf32Verified = false
    static let kernelB = MLXFast.metalKernel(
        name: "nemotron_prefill_conv_silu_dt_b",
        inputNames: ["oldState", "input", "weight", "bias", "dtIn", "dtBias", "dtLo", "dtHi", "A"],
        outputNames: ["out", "hist", "dtA", "dtx", "bf32"],
        source: """
            const int is1 = int(input_strides[1]), is2 = int(input_strides[2]);
            const int os1 = int(oldState_strides[1]), os2 = int(oldState_strides[2]);
            const int ws0 = int(weight_strides[0]), ws1 = int(weight_strides[1]);
            const int ds1 = int(dtIn_strides[1]), ds2 = int(dtIn_strides[2]);
            int channel = int(thread_position_in_grid.x);
            int position = int(thread_position_in_grid.y);
            if (channel >= C) {
                int head = channel - C;
                if (head >= H) return;
                float dtv = nemotron_prefill_dt(
                    static_cast<float>(dtIn[position * ds1 + head * ds2])
                    + static_cast<float>(dtBias[head * dtBias_strides[0]]), dtLo, dtHi);
                dtA[position * H + head] = dtv * A[head * A_strides[0]];
                return;
            }
            if (position >= L - 3) {
                hist[(position - (L - 3)) * C + channel] =
                    input[position * is1 + channel * is2];
            }
            float acc = 0;
            for (int i = 0; i < 4; ++i) {
                int row = position + int(i);
                T value = row < 3
                    ? oldState[row * os1 + channel * os2]
                    : input[(row - 3) * is1 + channel * is2];
                acc += static_cast<float>(value)
                    * weight[channel * ws0 + int(i) * ws1];
            }
            T convolved = static_cast<T>(acc);
            T x = convolved + bias[channel * bias_strides[0]];
            auto y = 1 / (1 + metal::exp(metal::abs(x)));
            T sigmoid = x < 0 ? y : 1 - y;
            T o = x * sigmoid;
            out[position * C + channel] = o;
            if (channel >= H * DH && channel < H * DH + G * DS) {
                // B as the f32 [G, n, DS] block of its scan window, the
                // contiguous array astype(f32) makes of the state product's
                // swapped / reshaped BF16 B (same values, same layout).
                const int cb = channel - H * DH;
                const int w0 = (position / STEP) * STEP;
                const int nw = min(STEP, L - w0);
                bf32[size_t(w0) * (G * DS) + size_t(cb / DS) * nw * DS
                     + size_t(position - w0) * DS + (cb % DS)] = static_cast<float>(o);
            }
            // x threadgroups (uniform: C and H * DH are multiples of the
            // 256-thread group): the group's 256 / DH heads' dt once each.
            const int c0 = int(threadgroup_position_in_grid.x) * 256;
            if (c0 < H * DH) {
                threadgroup float tdt[256 / DH];
                const uint lt = thread_position_in_threadgroup.x;
                if (lt < 256 / DH) {
                    const int head = c0 / DH + int(lt);
                    tdt[lt] = nemotron_prefill_dt(
                        static_cast<float>(dtIn[position * ds1 + head * ds2])
                        + static_cast<float>(dtBias[head * dtBias_strides[0]]), dtLo, dtHi);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const float dtv = tdt[(channel - c0) / DH];
                dtx[position * (H * DH) + channel] = dtv * static_cast<float>(o);
            }
            """, header: header, ensureRowContiguous: false)

    /// kernelB with four consecutive channels per thread: the input rows,
    /// weights and bias read as 4-element vectors (contiguous channels) and
    /// out, dtx and the f32 B block written as 4-element vectors. Every
    /// channel's statements are kernelB's. Seed only (`b4Verified`).
    /// NEMOTRON_PREFILL_CONV_DT_VEC=0 keeps kernelB.
    static let b4Enabled = ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_CONV_DT_VEC"] != "0"
    nonisolated(unsafe) static var b4Verified = false
    nonisolated(unsafe) static var testingB4 = false
    static let kernelB4 = MLXFast.metalKernel(
        name: "nemotron_prefill_conv_silu_dt_b4",
        inputNames: ["oldState", "input", "weight", "bias", "dtIn", "dtBias", "dtLo", "dtHi", "A"],
        outputNames: ["out", "hist", "dtA", "dtx", "bf32"],
        source: """
            const int is1 = int(input_strides[1]), is2 = int(input_strides[2]);
            const int os1 = int(oldState_strides[1]), os2 = int(oldState_strides[2]);
            const int ws0 = int(weight_strides[0]), ws1 = int(weight_strides[1]);
            const int ds1 = int(dtIn_strides[1]), ds2 = int(dtIn_strides[2]);
            const int quad = int(thread_position_in_grid.x);
            const int position = int(thread_position_in_grid.y);
            if (quad >= C / 4) {
                int head = quad - C / 4;
                if (head >= H) return;
                float dtv = nemotron_prefill_dt(
                    static_cast<float>(dtIn[position * ds1 + head * ds2])
                    + static_cast<float>(dtBias[head * dtBias_strides[0]]), dtLo, dtHi);
                dtA[position * H + head] = dtv * A[head * A_strides[0]];
                return;
            }
            const int ch0 = quad * 4;
            T rows[4][4];
            for (int i = 0; i < 4; ++i) {
                int row = position + i;
                if (row < 3) {
                    for (int u = 0; u < 4; ++u) {
                        rows[i][u] = oldState[row * os1 + (ch0 + u) * os2];
                    }
                } else if (is2 == 1) {
                    const vec<T, 4> r4 = *((const device vec<T, 4>*)(input + (row - 3) * is1 + ch0));
                    for (int u = 0; u < 4; ++u) {
                        rows[i][u] = r4[u];
                    }
                } else {
                    for (int u = 0; u < 4; ++u) {
                        rows[i][u] = input[(row - 3) * is1 + (ch0 + u) * is2];
                    }
                }
            }
            if (position >= L - 3) {
                for (int u = 0; u < 4; ++u) {
                    hist[(position - (L - 3)) * C + ch0 + u] = input[position * is1 + (ch0 + u) * is2];
                }
            }
            vec<T, 4> o4;
            for (int u = 0; u < 4; ++u) {
                const int channel = ch0 + u;
                float acc = 0;
                for (int i = 0; i < 4; ++i) {
                    T value = rows[i][u];
                    acc += static_cast<float>(value)
                        * weight[channel * ws0 + int(i) * ws1];
                }
                T convolved = static_cast<T>(acc);
                T x = convolved + bias[channel * bias_strides[0]];
                auto y = 1 / (1 + metal::exp(metal::abs(x)));
                T sigmoid = x < 0 ? y : 1 - y;
                T o = x * sigmoid;
                o4[u] = o;
            }
            *((device vec<T, 4>*)(out + position * C + ch0)) = o4;
            if (ch0 >= H * DH && ch0 < H * DH + G * DS) {
                const int cb = ch0 - H * DH;
                const int w0 = (position / STEP) * STEP;
                const int nw = min(STEP, L - w0);
                *((device float4*)(bf32 + size_t(w0) * (G * DS) + size_t(cb / DS) * nw * DS
                     + size_t(position - w0) * DS + (cb % DS))) =
                    float4(static_cast<float>(o4[0]), static_cast<float>(o4[1]),
                           static_cast<float>(o4[2]), static_cast<float>(o4[3]));
            }
            // x threadgroups of 256 threads cover 1024 channels (uniform: C and
            // H * DH are multiples of 1024): the group's 1024 / DH heads' dt.
            const int c0 = int(threadgroup_position_in_grid.x) * 1024;
            if (c0 < H * DH) {
                threadgroup float tdt[1024 / DH];
                const uint lt = thread_position_in_threadgroup.x;
                if (lt < 1024 / DH) {
                    const int head = c0 / DH + int(lt);
                    tdt[lt] = nemotron_prefill_dt(
                        static_cast<float>(dtIn[position * ds1 + head * ds2])
                        + static_cast<float>(dtBias[head * dtBias_strides[0]]), dtLo, dtHi);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const float dtv = tdt[(ch0 - c0) / DH];
                *((device float4*)(dtx + position * (H * DH) + ch0)) =
                    float4(dtv * static_cast<float>(o4[0]), dtv * static_cast<float>(o4[1]),
                           dtv * static_cast<float>(o4[2]), dtv * static_cast<float>(o4[3]));
            }
            """, header: header, ensureRowContiguous: false)

    /// Load-time bit check against the stock graph with real layers' conv
    /// weights, dt bias, A_log and limits, on seeded random inputs: fresh
    /// zero-stride and carried conv state, strided split views of a
    /// projection row block, prompt lengths 9 to 1024.
    static func qualify(_ target: Module) {
        verified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron prefill conv dt: \(message)\n".utf8))
        }
        guard enabled, NemotronHPrefillX5.enabled else { return report("off (setting)") }
        let mixers = target.modules().compactMap { $0 as? NemotronHMamba2Mixer }
        guard let firstMixer = mixers.first, let lastMixer = mixers.last else {
            return report("off (no Mamba layer)")
        }
        let cases: [(Int, Bool)] = [(488, true), (488, false), (9, true), (300, false), (1024, true)]
        var checked = 0
        do {
            let matches = try withError { () -> Bool in
                for (mi, mixer) in [firstMixer, lastMixer].enumerated() {
                    for (ci, (length, fresh)) in cases.enumerated() {
                        let seed = UInt64(0x7c0d_0000 + mi * 64 + ci)
                        let width = mixer.intermediateSize + mixer.convDim + mixer.numHeads
                        let projected = MLXRandom.normal(
                            [1, length, width + 5], dtype: .bfloat16, scale: 2, key: MLXRandom.key(seed))
                        let input = projected[
                            0..., 0..., mixer.intermediateSize ..< (mixer.intermediateSize + mixer.convDim)]
                        let dt = projected[0..., 0..., (mixer.intermediateSize + mixer.convDim) ..< width]
                        let oldState = fresh
                            ? NemotronMambaZeroState.view([1, 3, mixer.convDim], dtype: .bfloat16)
                            : MLXRandom.normal(
                                [1, 3, mixer.convDim], dtype: .bfloat16, key: MLXRandom.key(seed + 1))
                        guard let fused = mixer.prefillConvDt(oldState, input, dt) else { return false }
                        let stock = mixer.stockPrefillConvDt(oldState, input, dt)
                        var same = MLXArray(true)
                        for (a, b) in [(fused.0, stock.0), (fused.1, stock.1)] {
                            same = same .&& (a.view(dtype: .uint16) .== b.view(dtype: .uint16)).all()
                        }
                        for (a, b) in [(fused.2, stock.2), (fused.3, stock.3)] {
                            same = same .&& (a.view(dtype: .uint32) .== b.view(dtype: .uint32)).all()
                        }
                        if !same.item(Bool.self) { return false }
                        checked += 1
                    }
                }
                return true
            }
            verified = matches
        } catch {
            return report("off (\(error))")
        }
        report(verified ? "on (\(checked) cases bitwise equal)"
                        : "off (outputs differ from the stock graph after \(checked) cases)")
        qualifyB(firstMixer, lastMixer)
    }

    /// B buffer check: the B-writing launch's four outputs equal the plain
    /// launch's, every window block equals astype(f32) of the stock swapped /
    /// reshaped BF16 B, and the state-product matmul on it is bitwise the
    /// matmul on the BF16 view (the stock promotion path).
    private static func qualifyB(_ mixers: NemotronHMamba2Mixer...) {
        bf32Verified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron prefill b f32: \(message)\n".utf8))
        }
        guard verified, bf32Enabled else { return report("off (setting)") }
        var checked = 0
        do {
            let matches = try withError { () -> Bool in
                for (mi, mixer) in mixers.enumerated() {
                    for (ci, (length, fresh)) in [(488, true), (300, false), (9, true), (257, true)].enumerated() {
                        let seed = UInt64(0x7b32_0000 + mi * 64 + ci)
                        let width = mixer.intermediateSize + mixer.convDim + mixer.numHeads
                        let projected = MLXRandom.normal(
                            [1, length, width + 5], dtype: .bfloat16, scale: 2, key: MLXRandom.key(seed))
                        let input = projected[
                            0..., 0..., mixer.intermediateSize ..< (mixer.intermediateSize + mixer.convDim)]
                        let dt = projected[0..., 0..., (mixer.intermediateSize + mixer.convDim) ..< width]
                        let oldState = fresh
                            ? NemotronMambaZeroState.view([1, 3, mixer.convDim], dtype: .bfloat16)
                            : MLXRandom.normal(
                                [1, 3, mixer.convDim], dtype: .bfloat16, key: MLXRandom.key(seed + 1))
                        guard let plain = mixer.prefillConvDt(oldState, input, dt),
                              let withB = mixer.prefillConvDt(oldState, input, dt, writeB: true),
                              let bf32 = withB.4
                        else { return false }
                        var same = MLXArray(true)
                        for (a, b) in [(withB.0, plain.0), (withB.1, plain.1)] {
                            same = same .&& (a.view(dtype: .uint16) .== b.view(dtype: .uint16)).all()
                        }
                        for (a, b) in [(withB.2, plain.2), (withB.3, plain.3)] {
                            same = same .&& (a.view(dtype: .uint32) .== b.view(dtype: .uint32)).all()
                        }
                        let (g, d) = (mixer.numGroups, mixer.ssmStateSize)
                        let B = split(plain.0, indices: [mixer.intermediateSize, mixer.intermediateSize + g * d],
                                      axis: -1)[1].reshaped(1, length, g, d)
                        for start in stride(from: 0, to: length, by: 256) {
                            let end = min(start + 256, length), n = end - start
                            let stepB = MLX.transposed(B[0..., start ..< end, 0..., 0...], axes: [0, 2, 3, 1])
                            let groupedB = stepB.swappedAxes(2, 3).reshaped(1, g, 1, n, d)
                            let mine = nemotronPrefillBF32Window(bf32, start: start, end: end, groups: g, state: d)
                            same = same .&& (mine.view(dtype: .uint32)
                                .== groupedB.asType(.float32).view(dtype: .uint32)).all()
                            let dtxd = MLXRandom.normal([1, g, mixer.numHeads / g, mixer.headDim, n],
                                                        key: MLXRandom.key(seed + UInt64(10 + start)))
                            same = same .&& (dtxd.matmul(mine).view(dtype: .uint32)
                                .== dtxd.matmul(groupedB).view(dtype: .uint32)).all()
                        }
                        if !same.item(Bool.self) { return false }
                        checked += 1
                    }
                }
                return true
            }
            bf32Verified = matches
        } catch {
            return report("off (\(error))")
        }
        report(bf32Verified ? "on (\(checked) cases bitwise equal, matmul included)"
                            : "off (outputs differ after \(checked) cases)")
        qualifyB4(mixers)
    }

    /// kernelB4 against kernelB, all five outputs bit for bit, on the same
    /// seeded cases as `qualifyB`.
    private static func qualifyB4(_ mixers: [NemotronHMamba2Mixer]) {
        b4Verified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron prefill conv dt vec4: \(message)\n".utf8))
        }
        guard bf32Verified, b4Enabled else { return report("off (setting)") }
        var checked = 0
        var same = MLXArray(true)
        for (mi, mixer) in mixers.enumerated() {
            for (ci, (length, fresh)) in [(488, true), (300, false), (9, true), (257, true)].enumerated() {
                let seed = UInt64(0x7b34_0000 + mi * 64 + ci)
                let width = mixer.intermediateSize + mixer.convDim + mixer.numHeads
                let projected = MLXRandom.normal(
                    [1, length, width + 5], dtype: .bfloat16, scale: 2, key: MLXRandom.key(seed))
                let input = projected[0..., 0..., mixer.intermediateSize ..< (mixer.intermediateSize + mixer.convDim)]
                let dt = projected[0..., 0..., (mixer.intermediateSize + mixer.convDim) ..< width]
                let oldState = fresh
                    ? NemotronMambaZeroState.view([1, 3, mixer.convDim], dtype: .bfloat16)
                    : MLXRandom.normal([1, 3, mixer.convDim], dtype: .bfloat16, key: MLXRandom.key(seed + 1))
                guard let a = mixer.prefillConvDt(oldState, input, dt, writeB: true), let aB = a.4 else {
                    return report("off (shapes)")
                }
                testingB4 = true
                let b = mixer.prefillConvDt(oldState, input, dt, writeB: true)
                testingB4 = false
                guard let b, let bB = b.4 else { return report("off (shapes)") }
                for (u, w) in [(a.0, b.0), (a.1, b.1)] {
                    same = same .&& (u.view(dtype: .uint16) .== w.view(dtype: .uint16)).all()
                }
                for (u, w) in [(a.2, b.2), (a.3, b.3), (aB, bB)] {
                    same = same .&& (u.view(dtype: .uint32) .== w.view(dtype: .uint32)).all()
                }
                eval(same)
                checked += 1
            }
        }
        b4Verified = same.item(Bool.self)
        report(b4Verified ? "on (\(checked) cases bitwise equal)" : "off (differs from kernelB)")
    }
}

/// Window [start, end) of the B buffer `NemotronPrefillConvDt.kernelB`
/// writes, as the contiguous [1, G, 1, n, DS] f32 array.
func nemotronPrefillBF32Window(_ flat: MLXArray, start: Int, end: Int, groups: Int, state: Int) -> MLXArray {
    flat[(start * groups * state) ..< (end * groups * state)].reshaped(1, groups, 1, end - start, state)
}

class NemotronHMamba2Mixer: Module, NemotronHMixer {
    let numHeads: Int
    let hiddenSize: Int
    let ssmStateSize: Int
    let convKernelSize: Int
    let intermediateSize: Int
    let numGroups: Int
    let headDim: Int
    let timeStepLimit: (Float, Float)
    let headsPerGroup: Int
    let keepsFloat32State: Bool

    let convDim: Int

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    /// Seed only: `-exp(A_log)` as FP32, evaluated once on the first request
    /// prefill (the same three operations as `prefillConvDt`'s graph, so the
    /// same values) instead of three launches per request.
    /// NEMOTRON_PREFILL_SEED_DECAY_A=0 keeps them per request.
    private final class SeedDecayA: @unchecked Sendable { var value: MLXArray? }
    private let cachedSeedDecayA = SeedDecayA()
    private static let seedDecayAEnabled =
        ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_SEED_DECAY_A"] != "0"
    private var seedDecayA: MLXArray? {
        guard Self.seedDecayAEnabled, NemotronHPrefillX5.active else { return nil }
        if let value = cachedSeedDecayA.value { return value }
        let value = -MLX.exp(aLog).asType(.float32)
        eval(value)
        cachedSeedDecayA.value = value
        return value
    }
    @ParameterInfo(key: "D") var D: MLXArray

    @ModuleInfo(key: "norm") var norm: NemotronHRMSNormGated
    let mtpCompactGraphCache = NemotronH35MTPCompactGraphCache()

    init(_ args: NemotronHConfiguration) {
        self.numHeads = args.mambaNumHeads
        self.hiddenSize = args.hiddenSize
        self.ssmStateSize = args.ssmStateSize
        self.convKernelSize = args.convKernel
        self.intermediateSize = args.mambaNumHeads * args.mambaHeadDim
        self.numGroups = args.nGroups
        self.headDim = args.mambaHeadDim
        self.timeStepLimit = (args.timeStepLimitMin, args.timeStepLimitMax)
        self.headsPerGroup = numHeads / numGroups
        self.keepsFloat32State = args.mambaSSMCacheDType == "float32"
        self.convDim = intermediateSize + 2 * numGroups * ssmStateSize

        self._conv1d.wrappedValue = Conv1d(
            inputChannels: convDim,
            outputChannels: convDim,
            kernelSize: convKernelSize,
            groups: convDim,
            bias: args.useConvBias
        )

        let projectionSize = intermediateSize + convDim + numHeads
        self._inProj.wrappedValue = Linear(hiddenSize, projectionSize, bias: args.mambaProjBias)

        self._dtBias.wrappedValue = MLXArray.ones([numHeads])
        let headsRange = (MLXArray(0 ..< numHeads).asType(.float32) + 1)
        self._aLog.wrappedValue = MLX.log(headsRange)
        self._D.wrappedValue = MLXArray.ones([numHeads])

        let groupSize = intermediateSize / numGroups
        self._norm.wrappedValue = NemotronHRMSNormGated(
            dimensions: intermediateSize,
            eps: args.layerNormEpsilon,
            groupSize: groupSize
        )

        self._outProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: args.mambaProjBias)

        super.init()
    }

    /// `NemotronPrefillConvDt` for a [1, L > 8] request prefill: (SiLU conv
    /// output, conv history [1, 3, C], dtA [1, L, H], dtx [1, L, H, Dh]);
    /// nil when the shapes or dtypes are not the kernel's.
    func prefillConvDt(_ oldState: MLXArray, _ input: MLXArray, _ dt: MLXArray, writeB: Bool = false)
        -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray?)? {
        guard convKernelSize == 4, oldState.shape == [1, 3, convDim],
            input.ndim == 3, input.dim(0) == 1, input.dim(1) > 8,
            input.dim(2) == convDim, oldState.dtype == .bfloat16, input.dtype == .bfloat16,
            conv1d.weight.dtype == .bfloat16, conv1d.weight.shape == [convDim, 4, 1],
            conv1d.groups == convDim, conv1d.stride == 1, conv1d.dilation == 1,
            conv1d.padding == 0, let bias = conv1d.bias,
            bias.dtype == .bfloat16, bias.shape == [convDim],
            dt.shape == [1, input.dim(1), numHeads], dt.dtype == .bfloat16,
            dtBias.shape == [numHeads], [DType.bfloat16, .float32].contains(dtBias.dtype),
            aLog.shape == [numHeads], intermediateSize == numHeads * headDim,
            convDim % 256 == 0, intermediateSize % 256 == 0, 256 % headDim == 0,
            input.dim(1) <= 65536,  // 32-bit indexing: <= 65536 rows of the in_proj split (row width 10304)
            Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu
        else { return nil }
        let length = input.dim(1)
        let A = seedDecayA ?? -MLX.exp(aLog).asType(.float32)
        if writeB {
            guard numGroups * ssmStateSize + intermediateSize <= convDim else { return nil }
            // Seed only: four channels per thread (`kernelB4`).
            let vec4 = (NemotronPrefillConvDt.b4Verified && NemotronHPrefillX5.active
                    || NemotronPrefillConvDt.testingB4)
                && convDim % 1024 == 0 && intermediateSize % 1024 == 0 && headDim <= 1024 && ssmStateSize % 4 == 0
            let arrays = (vec4 ? NemotronPrefillConvDt.kernelB4 : NemotronPrefillConvDt.kernelB)(
                [oldState, input, conv1d.weight, bias, dt, dtBias,
                 MLXArray(timeStepLimit.0), MLXArray(timeStepLimit.1), A],
                template: [("T", DType.bfloat16), ("C", convDim), ("H", numHeads), ("DH", headDim),
                           ("L", length), ("G", numGroups), ("DS", ssmStateSize), ("STEP", 256)],
                grid: (vec4 ? convDim / 4 + numHeads : convDim + numHeads, length, 1), threadGroup: (256, 1, 1),
                outputShapes: [[1, length, convDim], [1, 3, convDim], [1, length, numHeads],
                               [1, length, numHeads, headDim], [length * numGroups * ssmStateSize]],
                outputDTypes: [.bfloat16, .bfloat16, .float32, .float32, .float32])
            return (arrays[0], arrays[1], arrays[2], arrays[3], arrays[4])
        }
        let arrays = NemotronPrefillConvDt.kernel(
            [oldState, input, conv1d.weight, bias, dt, dtBias,
             MLXArray(timeStepLimit.0), MLXArray(timeStepLimit.1), A],
            template: [("T", DType.bfloat16), ("C", convDim), ("H", numHeads), ("DH", headDim),
                       ("L", length)],
            grid: (convDim + numHeads, length, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1, length, convDim], [1, 3, convDim], [1, length, numHeads],
                           [1, length, numHeads, headDim]],
            outputDTypes: [.bfloat16, .bfloat16, .float32, .float32])
        return (arrays[0], arrays[1], arrays[2], arrays[3], nil)
    }

    /// The stock graph `prefillConvDt` replaces (for its load-time check).
    func stockPrefillConvDt(_ oldState: MLXArray, _ input: MLXArray, _ dt: MLXArray)
        -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        let padded = concatenated([oldState, input], axis: 1)
        let convolution = silu(conv1d(padded))
        let end = padded.dim(1)
        let history = padded[0..., (end - 3) ..< end, 0...]
        let length = input.dim(1)
        let t = nemotronPrefillComputeDt(dt.reshaped(1, length, numHeads), dtBias, timeStepLimit)
        let A = -MLX.exp(aLog).asType(t.dtype)
        let x = split(convolution, indices: [intermediateSize], axis: -1)[0]
            .reshaped(1, length, numHeads, headDim)
        return (convolution, history, t * A.reshaped(1, 1, -1), t.reshaped(1, length, numHeads, 1) * x)
    }

    func packedConv(_ oldState: MLXArray, _ input: MLXArray)
        -> (convolution: MLXArray, padded: MLXArray) {
        guard convKernelSize == 4, oldState.shape == [1, 3, convDim],
            input.ndim == 3, input.dim(0) == 1, (1...8).contains(input.dim(1)),
            input.dim(2) == convDim, oldState.dtype == .bfloat16, input.dtype == .bfloat16,
            conv1d.weight.dtype == .bfloat16, conv1d.weight.shape == [convDim, 4, 1],
            conv1d.groups == convDim, conv1d.stride == 1, conv1d.dilation == 1,
            conv1d.padding == 0, let bias = conv1d.bias,
            bias.dtype == .bfloat16, bias.shape == [convDim]
        else {
            let padded = concatenated([oldState, input], axis: 1)
            return (silu(conv1d(padded)), padded)
        }
        let length = input.dim(1)
        let arrays = NemotronPackedConvKernel.kernel(
            [oldState, input, conv1d.weight, bias],
            template: [("T", DType.bfloat16), ("C", convDim)],
            grid: (convDim, length, 1), threadGroup: (min(convDim, 256), 1, 1),
            outputShapes: [[1, length, convDim], [1, length + 3, convDim]],
            outputDTypes: [.bfloat16, .bfloat16])
        return (arrays[0], arrays[1])
    }

    /// Verify-window conv + SiLU and timestep transform in one launch: the
    /// same (convolution, padded) as `packedConv` and the same FP32 dt as
    /// `nemotronPrefillComputeDt`. nil when the shapes are not the window's.
    /// NEMOTRON_MTP_CONV_DT=0 keeps the two launches.
    func packedConvDt(_ oldState: MLXArray, _ input: MLXArray, _ dt: MLXArray)
        -> (convolution: MLXArray, padded: MLXArray, dt: MLXArray)? {
        guard Self.convDtEnabled, convKernelSize == 4, oldState.shape == [1, 3, convDim],
            input.ndim == 3, input.dim(0) == 1, (2...4).contains(input.dim(1)),
            input.dim(2) == convDim, oldState.dtype == .bfloat16, input.dtype == .bfloat16,
            conv1d.weight.dtype == .bfloat16, conv1d.weight.shape == [convDim, 4, 1],
            conv1d.groups == convDim, conv1d.stride == 1, conv1d.dilation == 1,
            conv1d.padding == 0, let bias = conv1d.bias,
            bias.dtype == .bfloat16, bias.shape == [convDim],
            dt.shape == [1, input.dim(1), numHeads], dt.dtype == .bfloat16,
            dtBias.shape == [numHeads], [DType.bfloat16, .float32].contains(dtBias.dtype),
            Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu
        else { return nil }
        let length = input.dim(1)
        let arrays = NemotronPackedConvDtKernel.kernel(
            [oldState, input, conv1d.weight, bias, dt, dtBias,
             MLXArray(timeStepLimit.0), MLXArray(timeStepLimit.1)],
            template: [("T", DType.bfloat16), ("C", convDim), ("H", numHeads)],
            grid: (convDim + numHeads, length, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1, length, convDim], [1, length + 3, convDim], [1, length, numHeads]],
            outputDTypes: [.bfloat16, .bfloat16, .float32])
        return (arrays[0], arrays[1], arrays[2])
    }
    static let convDtEnabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_CONV_DT"] != "0"

    private func applyConv(_ input: MLXArray, mask: MLXArray?, cache: MambaCache?) -> MLXArray {
        var convInput = input

        // Apply mask if present
        if let mask {
            let expandedMask = expandedDimensions(mask, axis: -1)
            convInput = MLX.where(expandedMask, convInput, MLXArray.zeros(like: convInput))
        }

        let batch = convInput.dim(0)
        let dtype = convInput.dtype
        var convState = cache?[0]

        if convState == nil {
            if convKernelSize > 1 {
                convState = MLXArray.zeros([batch, convKernelSize - 1, convDim], dtype: dtype)
            } else {
                convState = MLXArray.zeros([batch, 0, convDim], dtype: dtype)
            }
        }

        let (convOutput, padded) = packedConv(convState!, convInput)

        if let cache {
            let end = padded.dim(1)
            let start = max(0, end - (convKernelSize - 1))
            cache[0] = padded[0..., start ..< end, 0...]
        }

        return convOutput
    }

    private func mambaForward(
        _ hiddenStates: MLXArray,
        mask: MLXArray?,
        cache: MambaCache?
    ) -> MLXArray {
        let projected = inProj(hiddenStates)
        let splits = split(
            projected, indices: [intermediateSize, intermediateSize + convDim], axis: -1)
        let gate = splits[0]
        let convInput = splits[1]
        let dt = splits[2]

        let convOutput = applyConv(convInput, mask: mask, cache: cache)
        let convSplits = split(
            convOutput,
            indices: [intermediateSize, intermediateSize + numGroups * ssmStateSize],
            axis: -1
        )

        var hidden = convSplits[0]
        var B = convSplits[1]
        var C = convSplits[2]

        hidden = hidden.reshaped([hidden.dim(0), hidden.dim(1), numHeads, headDim])
        B = B.reshaped([B.dim(0), B.dim(1), numGroups, ssmStateSize])
        C = C.reshaped([C.dim(0), C.dim(1), numGroups, ssmStateSize])

        let dtArray = dt.reshaped([dt.dim(0), dt.dim(1), numHeads])

        let previousState = cache?[1]
        let (y, nextState) = ssmUpdate(
            hiddenStates: hidden,
            ALog: aLog,
            B: B,
            C: C,
            D: D,
            dt: dtArray,
            dtBias: dtBias,
            state: previousState,
            timeStepLimit: timeStepLimit,
            mask: mask
        )

        if let cache {
            cache[1] = keepsFloat32State ? nextState.asType(.float32) : nextState
        }

        let flattenedY = y.flattened(start: 2)
        return outProj(norm(flattenedY, gate: gate))
    }

    /// CBv2 path with request-owned convolution and SSM state. The active
    /// requests are gathered into one rectangle for compute and split back
    /// into their owning recurrent transactions after the chunk.
    func cbv2Forward(
        _ hiddenStates: MLXArray,
        modelLayerIndex: Int,
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray {
        let batch = hiddenStates.dim(0)
        precondition(
            recurrentState.count == batch,
            "NemotronH CBv2 recurrent row count mismatch")

        // Request prefill: the projection reads its pair dictionary (`NemotronPrefillDenseDict`).
        let projected = NemotronPrefillDenseDict.apply(hiddenStates, inProj, kind: "in_proj")
            ?? inProj(hiddenStates)
        if NemotronSeedEarlyCommit.applies(modelLayerIndex: modelLayerIndex, rows: hiddenStates.dim(1)) {
            asyncEval(projected)
        }
        let splits = split(
            projected, indices: [intermediateSize, intermediateSize + convDim], axis: -1)
        let gate = splits[0]
        let convInput = splits[1]
        let dt = splits[2]

        // Fresh request (no row has any input state): zero state without
        // zero buffers, see `NemotronMambaZeroState`.
        let freshState = NemotronMambaZeroState.enabled
            && recurrentState.allSatisfy { $0.inputState(modelLayerIndex: modelLayerIndex) == nil }
        let convState: MLXArray
        let ssmState: MLXArray?
        if freshState {
            convState = NemotronMambaZeroState.view(
                [batch, max(0, convKernelSize - 1), convDim], dtype: hiddenStates.dtype)
            ssmState = nil
        } else {
            var convRows: [MLXArray] = []
            var ssmRows: [MLXArray] = []
            convRows.reserveCapacity(batch)
            ssmRows.reserveCapacity(batch)
            for evaluation in recurrentState {
                let state = evaluation.inputState(modelLayerIndex: modelLayerIndex)
                convRows.append(
                    state?.conv
                        ?? MLXArray.zeros(
                            [1, max(0, convKernelSize - 1), convDim],
                            dtype: hiddenStates.dtype))
                ssmRows.append(
                    state?.ssm
                        ?? MLXArray.zeros(
                            [1, numHeads, headDim, ssmStateSize],
                            dtype: .float32))
            }
            convState =
                convRows.count == 1 ? convRows[0] : concatenated(convRows, axis: 0)
            ssmState =
                ssmRows.count == 1 ? ssmRows[0] : concatenated(ssmRows, axis: 0)
        }

        // Seed request prefill: conv, history and the scan's dtA / dtx in one
        // launch (`NemotronPrefillConvDt`, bitwise the stock graph).
        let front = NemotronPrefillConvDt.verified && NemotronHPrefillX5.active
            && hiddenStates.dim(1) > 8
            ? prefillConvDt(convState, convInput, dt,
                            writeB: NemotronPrefillConvDt.bf32Verified) : nil
        let convOutput: MLXArray
        let nextConvState: MLXArray
        if let front {
            convOutput = front.0
            nextConvState = front.1
        } else {
            let (packedOutput, padded) = packedConv(convState, convInput)
            convOutput = packedOutput
            let end = padded.dim(1)
            let history = padded[0..., max(0, end - (convKernelSize - 1)) ..< end, 0...]
            if hiddenStates.dim(1) > 8 {
                nextConvState = NemotronRequestConvHistory.copy(
                    [history], template: [("N", history.dim(1)), ("C", convDim)],
                    grid: (convDim, history.dim(1), batch), threadGroup: (256, 1, 1),
                    outputShapes: [history.shape], outputDTypes: [history.dtype])[0]
            } else {
                nextConvState = history
            }
        }
        let convSplits = split(
            convOutput,
            indices: [intermediateSize, intermediateSize + numGroups * ssmStateSize],
            axis: -1)
        let hidden = convSplits[0].reshaped(
            batch, convOutput.dim(1), numHeads, headDim)
        let B = convSplits[1].reshaped(
            batch, convOutput.dim(1), numGroups, ssmStateSize)
        let C = convSplits[2].reshaped(
            batch, convOutput.dim(1), numGroups, ssmStateSize)
        let dtArray = dt.reshaped(batch, dt.dim(1), numHeads)
        let (y, nextState) = convOutput.dim(1) > 1
            ? nemotronSSMAttn(
                x: hidden, ALog: aLog, B: B, C: C, D: D, dt: dtArray,
                dtBias: dtBias, state: ssmState, timeStepLimit: timeStepLimit,
                prepared: front.map { (dtA: $0.2, dtx: $0.3, bF32: $0.4) })
            : nemotronDecodeSSM(
            hiddenStates: hidden,
            ALog: aLog,
            B: B,
            C: C,
            D: D,
            dt: dtArray,
            dtBias: dtBias,
            state: ssmState
                ?? MLXArray.zeros([batch, numHeads, headDim, ssmStateSize], dtype: .float32),
            timeStepLimit: timeStepLimit)
        let nextSSMState = nextState.asType(.float32)

        for (row, evaluation) in recurrentState.enumerated() {
            do {
                try evaluation.stage(
                    modelLayerIndex: modelLayerIndex,
                    conv: nextConvState[row ..< row + 1],
                    ssm: nextSSMState[row ..< row + 1])
            } catch {
                preconditionFailure(
                    "NemotronH CBv2 recurrent stage failed at layer "
                        + "\(modelLayerIndex): \(error)")
            }
        }

        // The fp32 recurrent accumulator is persistent state, not an
        // activation-width promotion. mlx-lm narrows SSM output back to the
        // input dtype before gated normalization and the output projection.
        let normalized = norm(
            y.asType(hiddenStates.dtype).flattened(start: 2),
            gate: gate)
        return NemotronPrefillDenseDict.apply(normalized, outProj, kind: "out_proj") ?? outProj(normalized)
    }

    // Protocol conformance
    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        mambaForward(x, mask: ssmMask, cache: cache as? MambaCache)
    }
}

// MARK: - Attention

class NemotronHAttention: Module, NemotronHMixer {
    let args: NemotronHConfiguration
    let scale: Float
    let numHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    // NOTE: NemotronH attention does NOT use RoPE (unlike most transformer models)
    // The Python implementation at mlx_lm/models/nemotron_h.py lines 247-274 shows
    // direct attention without position embeddings

    init(_ args: NemotronHConfiguration) {
        self.args = args
        self.numHeads = args.numAttentionHeads
        self.numKeyValueHeads = args.numKeyValueHeads
        self.headDim = args.headDim ?? (args.hiddenSize / args.numAttentionHeads)
        self.scale = pow(Float(headDim), -0.5)

        let dim = args.hiddenSize
        let attentionBias = args.attentionBias

        self._wq.wrappedValue = Linear(dim, numHeads * headDim, bias: attentionBias)
        self._wk.wrappedValue = Linear(dim, numKeyValueHeads * headDim, bias: attentionBias)
        self._wv.wrappedValue = Linear(dim, numKeyValueHeads * headDim, bias: attentionBias)
        self._wo.wrappedValue = Linear(numHeads * headDim, dim, bias: attentionBias)

        super.init()
    }

    private func attentionForward(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let queries = wq(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        let keys = wk(x).reshaped(B, L, numKeyValueHeads, headDim).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, numKeyValueHeads, headDim).transposed(0, 2, 1, 3)

        // No RoPE applied - NemotronH attention uses direct attention without position embeddings

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }

    /// The Q/K/V projections `cbv2Forward` takes of `x`; reads no cache.
    func cbv2Projections(
        _ x: MLXArray, captureMTP: Bool = false, targetQKV: Bool = false, prepared: MLXArray? = nil
    ) -> (MLXArray, MLXArray, MLXArray) {
        let length = x.dim(1)
        if targetQKV, numHeads == 32, numKeyValueHeads == 2, headDim == 128,
           let outputs = NemotronTargetQKV.apply(
               x, query: wq, key: wk, value: wv, captureMTP: captureMTP, prepared: prepared) {
            return NemotronMoELeader.selfTest && length == 1
                ? (nemotronBitwiseOrNaN(outputs.0, wq(x)), nemotronBitwiseOrNaN(outputs.1, wk(x)),
                   nemotronBitwiseOrNaN(outputs.2, wv(x)))
                : outputs
        }
        return (
            captureMTP ? nemotronMTPLinearRows(x, wq) : wq(x),
            captureMTP ? nemotronMTPLinearRows(x, wk) : wk(x),
            captureMTP ? nemotronMTPLinearRows(x, wv) : wv(x))
    }

    /// `projections`, when given, are `cbv2Projections` of this same `x`
    /// computed earlier (the MTP draft head queues them before its cache is final).
    func cbv2Forward(
        _ x: MLXArray,
        cache: any CBv2AttendingLayerCache,
        captureMTP: Bool = false,
        targetQKV: Bool = false,
        prepared: MLXArray? = nil,
        projections precomputed: (MLXArray, MLXArray, MLXArray)? = nil
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        precondition(precomputed.map { $0.0.dim(0) == batch && $0.0.dim(1) == length } ?? true,
                     "NemotronH attention projections do not match their input rows")
        // No RoPE here: nothing reads this cache's position offsets.
        (cache as? CBv2LayerCache)?.deferPositionOffsets = true
        // CBV2_KV_IN_PLACE_APPEND: on the captured [1, 3] verify window the
        // fused Q/K/V launch also stores the K/V rows straight into this
        // layer's contiguous cache (the slice-update copies of K and V go
        // away); the serialized SDPA calls then read the cache behind that
        // launch. Same stored bytes, same attention.
        if precomputed == nil, captureMTP, targetQKV, batch == 1, length == 3,
           NemotronMTPAttentionDirect.enabled,
           numHeads == 32, numKeyValueHeads == 2, headDim == 128,
           let layerCache = cache as? CBv2LayerCache,
           let target = layerCache.mtpInPlaceKVTarget(
               count: length, dtype: x.dtype, headDim: headDim),
           let qkv = NemotronTargetQKV.applyWritingKV(
               x, query: wq, key: wk, value: wv, prepared: prepared,
               kCache: target.keys, vCache: target.values,
               offset: target.offset, capacity: target.capacity)
        {
            var projectedQueries = qkv.0
            if NemotronMoELeader.selfTest {
                // The cache rows just written must be bitwise the K/V rows
                // the stock slice update would store; poison on a difference.
                let span = target.offset ..< (target.offset + length)
                let stored = [target.keys, target.values].map {
                    depends(input: $0[0..., 0..., span, 0...], dependencies: [qkv.0])
                }
                let expected = [qkv.1, qkv.2].map {
                    $0.reshaped(batch, length, numKeyValueHeads, headDim).transposed(0, 2, 1, 3)
                }
                let same = all(stored[0].view(dtype: .uint16) .== expected[0].view(dtype: .uint16))
                    .&& all(stored[1].view(dtype: .uint16) .== expected[1].view(dtype: .uint16))
                // Poison the projection, not its transposed view, so `queries`
                // keeps the layout of the production path below.
                projectedQueries = which(same, projectedQueries, MLXArray(Float.nan).asType(projectedQueries.dtype))
            }
            let queries = projectedQueries.reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)
            let rows = layerCache.updateAndAttendSerialQueryRowsWritten(
                queries: queries, written: qkv.0, scale: scale, sinks: nil,
                attendRows: NemotronMTPVerifySDPA.rows)
            return NemotronMTPAttentionDirect.project(rows, wo)
        }
        let (q, k, v) = precomputed ?? cbv2Projections(
            x, captureMTP: captureMTP, targetQKV: targetQKV, prepared: prepared)
        let queries = q.reshaped(
            batch, length, numHeads, headDim
        ).transposed(0, 2, 1, 3)
        let keys = k.reshaped(
            batch, length, numKeyValueHeads, headDim
        ).transposed(0, 2, 1, 3)
        let values = v.reshaped(
            batch, length, numKeyValueHeads, headDim
        ).transposed(0, 2, 1, 3)

        // Keep parity with the selected MLX artifact's serial oracle: current
        // mlx-lm NemotronH attention does not apply RoPE despite rope_theta.
        // Serialized MTP verify: o_proj reads the per-query outputs in place;
        // the per-query SDPA calls run as one launch once qualified at load.
        if captureMTP, batch == 1, length > 1, NemotronMTPAttentionDirect.enabled,
           let layerCache = cache as? CBv2LayerCache,
           let rows = layerCache.updateAndAttendSerialQueryRows(
               queries: queries, keys: keys, values: values, scale: scale, sinks: nil,
               attendRows: NemotronMTPVerifySDPA.rows) {
            return NemotronMTPAttentionDirect.project(rows, wo)
        }
        let output = cache.updateAndAttend(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            sinks: nil)
        let merged = output.transposed(0, 2, 1, 3).reshaped(batch, length, -1)
        return captureMTP ? nemotronMTPLinearRows(merged, wo) : nemotronR1Linear(merged, wo)
    }

    // Protocol conformance
    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        attentionForward(x, mask: attentionMask, cache: cache)
    }
}

// MARK: - MLP

class NemotronHMLP: Module, UnaryLayer, NemotronHMixer {
    /// DARKBLOOM_NEMOTRON35_ORDINARY_R1_SHARED_UP=0 keeps stock up_proj + relu2.
    private static let ordinaryR1SharedUp =
        ProcessInfo.processInfo.environment["DARKBLOOM_NEMOTRON35_ORDINARY_R1_SHARED_UP"] != "0"

    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ args: NemotronHConfiguration, intermediateSize: Int? = nil) {
        let intermediate = intermediateSize ?? args.intermediateSize
        self._upProj.wrappedValue = Linear(args.hiddenSize, intermediate, bias: args.mlpBias)
        self._downProj.wrappedValue = Linear(intermediate, args.hiddenSize, bias: args.mlpBias)
        super.init()
    }

    /// The ordinary M = 1 shared up + relu2 rows [1, 3712] of `callAsFunction`, or nil when that route does not
    /// apply (`NemotronDraftColaunch` then keeps `callAsFunction`).
    func draftUpActivation(_ x: MLXArray) -> MLXArray? {
        guard Self.ordinaryR1SharedUp, type(of: self) == NemotronHMLP.self,
              type(of: upProj) == QuantizedLinear.self,
              type(of: downProj) == QuantizedLinear.self,
              x.shape == [1, 1, 2688], x.dtype == .bfloat16,
              NemotronMTPExecution.batchedM1,
              let up = upProj as? QuantizedLinear, up.bias == nil,
              up.weight.shape == [3712, 336], up.scales.shape == [3712, 42],
              let offsets = up.biases, offsets.shape == [3712, 42],
              Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
              !NemotronMoELeader.selfTest
        else { return nil }
        return NemotronMultiRowQMV.apply(x.reshaped(1, 2688), up, relu2: true)
    }

    /// `nemotronR1Linear(activated, downProj)` through the R = 1 kernel's `dep` twin (`NemotronDraftColaunch`),
    /// or nil when `nemotronR1Linear` would not take that kernel.
    func draftDown(_ activated: MLXArray, dep: MLXArray) -> MLXArray? {
        guard NemotronMultiRowQMV.r1Route, activated.shape == [1, 3712], activated.dtype == .bfloat16,
              type(of: downProj) == QuantizedLinear.self, let q = downProj as? QuantizedLinear,
              q.bias == nil, !NemotronMoELeader.selfTest,
              let y = NemotronMultiRowQMV.apply(activated, q, dep: dep)
        else { return nil }
        return y.reshaped(1, 1, q.weight.dim(0))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Ordinary M=1 shared up (draft one-token tail): the multi-row QMV at
        // R = 1 repeats stock qmv (K = 2688 is not qmv_fast) and its epilogue
        // rounds to BF16 before the BF16 maximum/multiply, as stock relu2 does.
        // Other geometries and unknown overrides retain the original entry.
        if Self.ordinaryR1SharedUp, type(of: self) == NemotronHMLP.self,
           type(of: upProj) == QuantizedLinear.self,
           type(of: downProj) == QuantizedLinear.self,
           x.shape == [1, 1, 2688], x.dtype == .bfloat16,
           NemotronMTPExecution.batchedM1,
           let up = upProj as? QuantizedLinear, up.bias == nil,
           up.weight.shape == [3712, 336], up.scales.shape == [3712, 42],
           let offsets = up.biases, offsets.shape == [3712, 42],
           Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
           var activated = NemotronMultiRowQMV.apply(
               x.reshaped(1, 2688), up, relu2: true) {
            if NemotronMoELeader.selfTest {
                activated = nemotronBitwiseOrNaN(activated, relu2(upProj(x)).reshaped(1, 3712))
            }
            return nemotronR1Linear(activated.reshaped(1, 1, 3712), downProj)
        }
        // Request prefill: up projection and relu2 in one launch (`NemotronPrefillSharedUpGather`).
        // Request prefill: the down projection reads the decode copy (`NemotronPrefillSharedDownCopy`).
        if let activated = NemotronPrefillDenseDict.apply(x, upProj, kind: "up_proj", relu2: true)
            ?? NemotronPrefillSharedUpGather.apply(x, upProj) {
            return NemotronPrefillSharedDownCopy.apply(activated, downProj) ?? downProj(activated)
        }
        let up = upProj(x)
        // Request prefill: the stock relu2 in one launch (`NemotronPrefillSharedRelu2`).
        if let activated = NemotronPrefillSharedRelu2.apply(up) {
            return NemotronPrefillSharedDownCopy.apply(activated, downProj) ?? downProj(activated)
        }
        return downProj(relu2(up))
    }

    // Protocol conformance
    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        callAsFunction(x)
    }
}

// MARK: - MoE Gate

enum NemotronHRouterSelection {
    static let sigmoidHeader = """
        inline float nemotron_h_router_sigmoid(float x) {
            auto y = 1 / (1 + metal::exp(metal::abs(x)));
            return (x < 0) ? y : 1 - y;
        }
        """

    static let kernel = MLXFast.metalKernel(
        name: "nemotron_h_router_select",
        inputNames: ["gates", "bias"], outputNames: ["indices", "weights"],
        source: """
            const uint lane = thread_index_in_simdgroup;
            const uint row = threadgroup_position_in_grid.y;
            threadgroup float original[128];
            threadgroup float normalized[6];
            float corrected[4];
            bool selected[4] = {false, false, false, false};
            float chosen[6];
            for (uint i = 0; i < 4; ++i) {
                const uint id = lane + 32 * i;
                original[id] = nemotron_h_router_sigmoid(float(gates[row * 128 + id]));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = 0; i < 4; ++i) {
                const uint id = lane + 32 * i;
                corrected[i] = original[id] + bias[id];
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
                    chosen[slot] = original[winner];
                }
                for (uint i = 0; i < 4; ++i) {
                    if (lane + 32 * i == winner) { selected[i] = true; }
                }
            }
            if (lane == 0) {
                // Match row_reduce_small's value + total order from positive zero.
                float total = 0.0f;
                for (uint slot = 0; slot < 6; ++slot) { total = chosen[slot] + total; }
                const float denominator = total + 1e-20f;
                for (uint slot = 0; slot < 6; ++slot) {
                    normalized[slot] = chosen[slot] / denominator;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane < 6) { weights[row * 6 + lane] = normalized[lane] * 2.5f; }
            """,
        header: sigmoidHeader, ensureRowContiguous: true)

    /// Prefill rows (L > 8) take the same one-launch selection instead of the
    /// stock sigmoid / bias / argPartition / takeAlong / sum / divide / scale
    /// chain: each row is the same 128 scores, the same descending pick with
    /// the lowest id on ties (the stock arg-sort of a 128-wide row), the same
    /// sequential six-term sum and the same divide and scale, so indices and
    /// weights are bitwise the stock chain's at any row count.
    /// NEMOTRON_PREFILL_ROUTER_SELECT=0 keeps the stock chain for L > 8.
    static let prefillEnabled =
        ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_ROUTER_SELECT"] != "0"

    static func apply(
        gates: MLXArray, eSCB: MLXArray, topK: Int, nGroup: Int, topkGroup: Int,
        routedScalingFactor: Float, normTopkProb: Bool
    ) -> (MLXArray, MLXArray)? {
        guard Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
            gates.ndim == 3, gates.dim(0) == 1, gates.dim(1) >= 1,
            gates.dim(1) <= 8 || (prefillEnabled && NemotronHPrefillX5.active),
            gates.dim(2) == 128, gates.dtype == .bfloat16,
            eSCB.shape == [128], eSCB.dtype == .float32,
            topK == 6, nGroup == 1, topkGroup == 1,
            normTopkProb, routedScalingFactor == 2.5
        else { return nil }
        let shape = [1, gates.dim(1), 6]
        let result = kernel(
            [gates, eSCB], grid: (32, gates.dim(1), 1), threadGroup: (32, 1, 1),
            outputShapes: [shape, shape], outputDTypes: [.uint32, .float32])
        return (result[0], result[1])
    }
}

private func groupExpertSelect(
    gates: MLXArray,
    eSCB: MLXArray,  // e_score_correction_bias
    topK: Int,
    nGroup: Int,
    topkGroup: Int,
    routedScalingFactor: Float,
    normTopkProb: Bool
) -> (MLXArray, MLXArray) {
    if let selected = NemotronHRouterSelection.apply(
        gates: gates, eSCB: eSCB, topK: topK, nGroup: nGroup, topkGroup: topkGroup,
        routedScalingFactor: routedScalingFactor, normTopkProb: normTopkProb) {
        return selected
    }
    let (bsz, seqLen) = (gates.dim(0), gates.dim(1))

    // Original scores using sigmoid
    let origScores = sigmoid(gates.asType(.float32))
    var scores = origScores + eSCB

    // Group-based selection if n_group > 1
    if nGroup > 1 {
        let numExperts = scores.dim(-1)
        let expertsPerGroup = numExperts / nGroup

        // Reshape to [batch, seq, n_group, experts_per_group]
        let groupScores = scores.reshaped(bsz, seqLen, nGroup, -1)

        // Get top-2 per group and sum for group scores (using sorted to get top values)
        let topKGroupScores = sorted(groupScores, axis: -1)[.ellipsis, ..<2].sum(
            axis: -1, keepDims: true)

        // Keep only top topkGroup groups (zero out the rest)
        let k = nGroup - topkGroup
        var groupIdx = argPartition(topKGroupScores, kth: k - 1, axis: -2)[.ellipsis, ..<k, 0...]
        groupIdx = broadcast(groupIdx, to: [bsz, seqLen, k, expertsPerGroup])

        // Zero out scores from non-selected groups
        scores = putAlong(groupScores, groupIdx, values: MLXArray(0.0), axis: -2)

        // Flatten back
        scores = flattened(scores, start: -2, end: -1)
    }

    // Get top-k experts
    let inds = argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
    var finalScores = takeAlong(origScores, inds, axis: -1)

    // Normalize if needed
    if topK > 1 && normTopkProb {
        let denominator = finalScores.sum(axis: -1, keepDims: true) + 1e-20
        finalScores = finalScores / denominator
    }

    // Apply scaling factor
    finalScores = finalScores * routedScalingFactor

    return (inds, finalScores)
}

class NemotronHMoEGate: Module {
    let topK: Int
    let nGroup: Int
    let topkGroup: Int
    let routedScalingFactor: Float
    let normTopkProb: Bool

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eSCB: MLXArray

    init(_ args: NemotronHConfiguration) {
        self.topK = args.numExpertsPerTok
        self.nGroup = args.nGroup
        self.topkGroup = args.topkGroup
        self.routedScalingFactor = args.routedScalingFactor
        self.normTopkProb = args.normTopkProb

        self._weight.wrappedValue = MLXArray.zeros([args.nRoutedExperts, args.hiddenSize])
        self._eSCB.wrappedValue = MLXArray.zeros([args.nRoutedExperts])

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let gates = MLX.matmul(x, weight.transposed())
        return groupExpertSelect(
            gates: gates,
            eSCB: eSCB,
            topK: topK,
            nGroup: nGroup,
            topkGroup: topkGroup,
            routedScalingFactor: routedScalingFactor,
            normTopkProb: normTopkProb
        )
    }

    /// Keep each router projection at native M=1, but perform the row-local
    /// sigmoid/selection/normalization over the complete verification window.
    /// The multi-row kernel repeats the M=1 gemv per row in one launch.
    func mtpForwardRows(_ x: MLXArray) -> (MLXArray, MLXArray) {
        if topK == 6, nGroup == 1, topkGroup == 1, normTopkProb, routedScalingFactor == 2.5,
           let fused = NemotronMTPRouterRowsSelect.run(x, weight, bias: eSCB) {
            return fused
        }
        let gates = NemotronMTPRouterRows.gates(x, weight)
            ?? nemotronMTPMapRows(x) { MLX.matmul($0, weight.transposed()) }
        return groupExpertSelect(gates: gates, eSCB: eSCB, topK: topK,
            nGroup: nGroup, topkGroup: topkGroup,
            routedScalingFactor: routedScalingFactor, normTopkProb: normTopkProb)
    }
}

// MARK: - SwitchMLP for NemotronH (uses relu2 instead of silu/glu)

class NemotronHSwitchMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: SwitchLinear
    @ModuleInfo(key: "fc2") var fc2: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int

    init(inputDims: Int, hiddenDims: Int, numExperts: Int) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts

        self._fc1.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: false)
        self._fc2.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: false)

        super.init()
    }

    /// Multi-token MoE (sorted gather route) fused with the caller's
    /// weighted reduction. Same graph as `callAsFunction` followed by
    /// `(y * scores[..., None]).sum(-2).asType(y.dtype)`, except that
    /// (1) the inverse permutation is a scatter instead of a second argsort
    /// (a permutation has exactly one inverse), (2) relu2 keeps the stock typed
    /// Maximum/Multiply at the fc1 store or in one separate pass, and
    /// (3) the unsort gather, fp32 promotion, score multiply, K-way sum and
    /// narrowing are one kernel that reproduces col_reduce_small's order
    /// (lsize.y == K <= 8: row j is op(v_j, 0), then total = op(row_j, total)
    /// for j = 1..K-1). With x5 enabled, (4) argsort + inverse + sorted ids
    /// come from one stable counting sort and (5) where stock takes the
    /// sorted-rhs NAX gather, fc1/fc2 run its expert-run-aligned clone.
    /// Every value is bitwise identical to the unfused graph.
    func fusedSortedCombine(_ x: MLXArray, _ indices: MLXArray, _ scores: MLXArray) -> MLXArray? {
        fusedSortedCombine(x, indices, scores, addend: nil)?.out
    }

    /// `fusedSortedCombine`, and with `addend` (the shared-expert output,
    /// same shape and dtype as the result) inside the request prefill, the
    /// MoE block's `y + addend` in the combine launch; `added` says whether it
    /// was applied.
    func fusedSortedCombine(_ x: MLXArray, _ indices: MLXArray, _ scores: MLXArray, addend: MLXArray?,
                            join: (residual: MLXArray, norm: RMSNorm)? = nil)
        -> (out: MLXArray, added: Bool, normalized: MLXArray?)? {
        let k = indices.dim(-1)
        guard indices.size > 64, (1...8).contains(k), scores.dtype == .float32,
              scores.shape == indices.shape,
              x.dtype == .bfloat16 || x.dtype == .float16, x.dim(-1) == inputDims,
              inputDims % 4 == 0
        else { return nil }
        let tokens = indices.size / k
        let flat = indices.flattened()
        let kernels = NemotronHPrefillKernels.shared
        let order: MLXArray, inverse: MLXArray, idx: MLXArray
        var runTiles: NemotronHRunTiles? = nil
        var orderK: MLXArray? = nil
        if NemotronPrefillCountingSort.verified, NemotronHPrefillX5.active,
           let sorted = nemotronCountingSortChunked(flat, experts: numExperts, k: k) {
            (order, inverse, idx) = (sorted.order, sorted.inverse, sorted.sorted)
            runTiles = sorted.tiles
            orderK = sorted.orderK
        } else if NemotronHPrefillX5.enabled, let sorted = nemotronCountingSort(flat, experts: numExperts) {
            (order, inverse, idx) = (sorted.order, sorted.inverse, sorted.sorted)
            runTiles = sorted.tiles
        } else {
            order = argSort(flat)
            inverse = kernels.inversePermutation(
                [order], template: [("T", order.dtype)],
                grid: (order.size, 1, 1), threadGroup: (256, 1, 1),
                outputShapes: [[order.size]], outputDTypes: [order.dtype])[0]
            idx = flat[order]
        }
        let xs = MLX.expandedDimensions(x, axes: [-2, -3])
            .flattened(start: 0, end: -3)[orderK ?? order.floorDivide(k)]
        func gather(_ lin: SwitchLinear, _ input: MLXArray) -> MLXArray {
            if let tiles = runTiles, let q = lin as? QuantizedSwitchLinear,
               let fused = nemotronRunGather(input, q, tiles: tiles) {
                return fused
            }
            return lin(input, idx, sortedIndices: true)
        }
        var y: MLXArray
        if let tiles = runTiles, let rowsK = orderK, let q = fc1 as? QuantizedSwitchLinear,
            let activated = NemotronPrefillDirectFC1.apply(
                x.reshaped(tokens, 1, inputDims), q, rows: rowsK, tiles: tiles) {
            // Seed only: fc1 reads the token rows directly; `xs` stays lazy.
            y = activated
        } else if let tiles = runTiles, let q = fc1 as? QuantizedSwitchLinear,
            let activated = NemotronSeedRetile.fc1Plain(xs, q, tiles: tiles)
                ?? nemotronRunGather(xs, q, tiles: tiles, relu2: true) {
            y = activated
        } else {
            y = gather(fc1, xs)
            // Stock relu2 promotes against an int scalar; fuse only when that
            // keeps the activation dtype (graph metadata, never evaluated).
            if MLX.maximum(y, MLXArray(0)).dtype == y.dtype {
                y = kernels.relu2(
                    [y], template: [("T", y.dtype)],
                    grid: (y.size, 1, 1), threadGroup: (256, 1, 1),
                    outputShapes: [y.shape], outputDTypes: [y.dtype])[0]
            } else {
                y = relu2(y)
            }
        }
        if let tiles = runTiles, let q = fc2 as? QuantizedSwitchLinear,
           let fromCopy = NemotronPrefillFC2Copy.apply(y, q, tiles: tiles) {
            y = fromCopy
        } else if let tiles = runTiles, let q = fc2 as? QuantizedSwitchLinear,
                  let retiled = NemotronSeedRetile.fc2Plain(y, q, tiles: tiles) {
            y = retiled
        } else {
            y = gather(fc2, y)
        }
        guard y.dim(-1) == inputDims, y.size == indices.size * inputDims else { return nil }
        var outShape = indices.shape
        outShape[outShape.count - 1] = inputDims
        if let addend, NemotronPrefillCombineAdd.verified, NemotronHPrefillX5.active,
           addend.shape == outShape, addend.dtype == y.dtype {
            // terrapinelf's MoE join: the block's residual add and next norm in this launch.
            if let join, NemotronPrefillMoEJoin.verified,
               let joined = NemotronHResidualRMSNorm.applyMoEJoin(
                   join.residual, y: y, inverse: inverse, scores: scores, addend: addend, k: k,
                   norm: join.norm) {
                return (joined.0, true, joined.1)
            }
            return ((NemotronPrefillCombineAdd.vec4Verified ? kernels.combineAdd4 : kernels.combineAdd)(
                [y, inverse, scores, inputDims, addend],
                template: [("T", y.dtype), ("K", k)],
                grid: (inputDims / 4, tokens, 1), threadGroup: (NemotronPrefillCombineAdd.vec4Verified ? 64 : 32, 1, 1),
                outputShapes: [outShape], outputDTypes: [y.dtype])[0], true, nil)
        }
        return (kernels.combine(
            [y, inverse, scores, inputDims],
            template: [("T", y.dtype), ("K", k)],
            grid: (inputDims / 4, tokens, 1), threadGroup: (32, 1, 1),
            outputShapes: [outShape], outputDTypes: [y.dtype])[0], false, nil)
    }

    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        let doSort = indices.size > 64

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        var y = fc1(x, idx, sortedIndices: doSort)
        y = relu2(y)
        y = fc2(y, idx, sortedIndices: doSort)

        if doSort {
            y = scatterUnsort(x: y, invOrder: inverseOrder, shape: indices.shape)
        }

        return MLX.squeezed(y, axis: -2)
    }
}

// MARK: - Exact multi-token MoE element kernels

private final class NemotronHPrefillKernels: Sendable {
    static let shared = NemotronHPrefillKernels()
    let inversePermutation: MLXFast.MLXFastKernel
    let countingSort: MLXFast.MLXFastKernel
    let countingSortChunked: MLXFast.MLXFastKernel
    let relu2: MLXFast.MLXFastKernel
    let combine: MLXFast.MLXFastKernel
    let combineAdd: MLXFast.MLXFastKernel
    let combineAdd4: MLXFast.MLXFastKernel
    private init() {
        // Stable counting sort of expert ids (MLX argsort is a stable merge
        // sort, so ties keep ascending flat index): one threadgroup of E*S
        // threads; E workers per segment count each route once into a shared
        // histogram. The original serial exclusive scan over (e, s) gives
        // each scatter thread its first slot, preserving stable input order.
        countingSort = MLXFast.metalKernel(
            name: "nemotron_moe_counting_sort",
            inputNames: ["flat", "n", "tmax"], outputNames: ["order", "inv", "sorted", "tiles"],
            source: """
                constexpr int S = THREADS / E;
                threadgroup uint counts[THREADS];
                threadgroup atomic_uint histogram[THREADS];
                uint tid = thread_position_in_threadgroup.x;
                uint e = tid / S;
                uint s = tid % S;
                int chunk = (n + S - 1) / S;
                int lo = min(n, int(s) * chunk);
                int hi = min(n, lo + chunk);
                atomic_store_explicit(&histogram[tid], 0u, memory_order_relaxed);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                // Transpose only the count-phase work assignment: each of S
                // input segments has E workers reading disjoint positions.
                // The scatter below retains its original (expert, segment)
                // assignment and increasing input order.
                uint countSegment = tid / E;
                uint countLane = tid % E;
                int countLo = min(n, int(countSegment) * chunk);
                int countHi = min(n, countLo + chunk);
                for (int i = countLo + int(countLane); i < countHi; i += E) {
                    uint expert = flat[i];
                    if (expert < uint(E)) {
                        atomic_fetch_add_explicit(
                            &histogram[expert * S + countSegment], 1u, memory_order_relaxed);
                    }
                }
                // All increments finish before any bin is loaded. The
                // existing publication barrier then protects the plain uint
                // counts consumed by the unchanged serial prefix scan.
                threadgroup_barrier(mem_flags::mem_threadgroup);
                counts[tid] = atomic_load_explicit(&histogram[tid], memory_order_relaxed);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0) {
                    uint acc = 0;
                    for (int q = 0; q < THREADS; ++q) {
                        uint v = counts[q];
                        counts[q] = acc;
                        acc += v;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0) {
                    // Expert-run tiles of BMT rows: (first row, rows, expert).
                    int t = 0;
                    for (int ee = 0; ee < E; ++ee) {
                        int start = int(counts[ee * S]);
                        int end = (ee + 1 < E) ? int(counts[(ee + 1) * S]) : n;
                        for (int r = start; r < end && t < tmax;) {
                            const int remaining = end - r;
                            // Where the original batch used 32-row tiles, preserve its
                            // 48-row padded work for a 33...48 tail instead of rounding
                            // that tail up to a 64-row NAX body.
                            const int take = BMT == 64 && n / E < 64
                                && remaining > 32 && remaining <= 48
                                ? 32 : min(BMT, remaining);
                            tiles[3 * t] = r;
                            tiles[3 * t + 1] = take;
                            tiles[3 * t + 2] = ee;
                            r += take;
                            t++;
                        }
                    }
                    for (; t < tmax; ++t) {
                        tiles[3 * t] = 0;
                        tiles[3 * t + 1] = 0;
                        tiles[3 * t + 2] = 0;
                    }
                }
                uint pos = counts[tid];
                for (int i = lo; i < hi; ++i) {
                    if (flat[i] == e) {
                        order[pos] = uint32_t(i);
                        inv[i] = uint32_t(pos);
                        sorted[pos] = uint32_t(e);
                        pos++;
                    }
                }
            """)
        // The same stable counting sort with each of the 32 simdgroups owning
        // one input chunk: per-(expert, chunk) counts, one integer exclusive
        // scan in (expert, chunk) order, and an in-order scatter 32 routes at
        // a time (rank among equal experts from the block's lanes), so every
        // route lands at the serial kernel's position. Also writes order / KK
        // (the token row each sorted route gathers). Seed only, after
        // `NemotronPrefillCountingSort.qualify`.
        countingSortChunked = MLXFast.metalKernel(
            name: "nemotron_moe_counting_sort_chunked",
            inputNames: ["flat", "n", "tmax"], outputNames: ["order", "inv", "sorted", "tiles", "orderk"],
            source: """
                // Stable counting sort, chunk-parallel: simdgroup w owns flat
                // chunk w. Position of route i (expert e, chunk w) = #routes
                // with expert < e + #routes of e in chunks < w + #routes of e
                // earlier in chunk w: the stable order of the serial kernel.
                constexpr int W = THREADS / 32;
                constexpr int PER = (E * W) / THREADS;
                threadgroup uint tf[NMAX];
                threadgroup atomic_uint table[E * W];
                threadgroup uint sgsum[32];
                threadgroup int tsum[32];
                const uint tid = thread_position_in_threadgroup.x;
                const uint lane = tid % 32;
                const uint sg = tid / 32;
                const int chunk = (n + W - 1) / W;
                const int lo = min(n, int(sg) * chunk);
                const int hi = min(n, lo + chunk);
                for (int q = int(tid); q < E * W; q += THREADS) {
                    atomic_store_explicit(&table[q], 0u, memory_order_relaxed);
                }
                for (int i = int(tid); i < n; i += THREADS) {
                    tf[i] = flat[i];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int i = lo + int(lane); i < hi; i += 32) {
                    const uint expert = tf[i];
                    if (expert < uint(E)) {
                        atomic_fetch_add_explicit(&table[expert * W + sg], 1u, memory_order_relaxed);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                {
                    // Exclusive scan of the (e, w) counts in index order (integers).
                    uint v[PER];
                    uint local = 0;
                    for (int k = 0; k < PER; ++k) {
                        v[k] = atomic_load_explicit(&table[tid * PER + k], memory_order_relaxed);
                        local += v[k];
                    }
                    const uint pre = simd_prefix_exclusive_sum(local);
                    if (lane == 31) { sgsum[sg] = pre + local; }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (sg == 0) {
                        const uint w = sgsum[lane];
                        sgsum[lane] = simd_prefix_exclusive_sum(w);
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    uint base = pre + sgsum[sg];
                    for (int k = 0; k < PER; ++k) {
                        atomic_store_explicit(&table[tid * PER + k], base, memory_order_relaxed);
                        base += v[k];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                {
                    // Expert-run tiles of BMT rows: (first row, rows, expert), in
                    // expert order, capped at tmax, then zero rows: the serial
                    // loop's slots, each expert writing its own run of tiles.
                    int start = 0, end = 0, nt = 0;
                    if (tid < uint(E)) {
                        start = int(atomic_load_explicit(&table[tid * W], memory_order_relaxed));
                        end = (tid + 1 < uint(E))
                            ? int(atomic_load_explicit(&table[(tid + 1) * W], memory_order_relaxed)) : n;
                        // The serial kernel's split: BMT rows, except that with
                        // BMT 64 and n / E < 64 a 33...48-row remainder takes 32.
                        for (int r = start; r < end;) {
                            const int remaining = end - r;
                            const int take = BMT == 64 && n / E < 64
                                && remaining > 32 && remaining <= 48
                                ? 32 : min(BMT, remaining);
                            r += take;
                            nt++;
                        }
                    }
                    const int pre = simd_prefix_exclusive_sum(nt);
                    if (lane == 31) { tsum[sg] = pre + nt; }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (sg == 0) {
                        const int w = tsum[lane];
                        const int tot = simd_sum(w);
                        tsum[lane] = simd_prefix_exclusive_sum(w);
                        if (lane == 0) { sgsum[0] = uint(tot); }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (tid < uint(E)) {
                        int t = pre + tsum[sg];
                        for (int r = start; r < end && t < tmax;) {
                            const int remaining = end - r;
                            const int take = BMT == 64 && n / E < 64
                                && remaining > 32 && remaining <= 48
                                ? 32 : min(BMT, remaining);
                            tiles[3 * t] = r;
                            tiles[3 * t + 1] = take;
                            tiles[3 * t + 2] = int(tid);
                            r += take;
                            t++;
                        }
                    }
                    for (int t = int(sgsum[0]) + int(tid); t < tmax; t += THREADS) {
                        tiles[3 * t] = 0;
                        tiles[3 * t + 1] = 0;
                        tiles[3 * t + 2] = 0;
                    }
                }
                // Scatter: 32 routes at a time per chunk, in input order; a
                // route's rank among equal experts earlier in the block comes
                // from the other lanes, the block's last such lane advances
                // the (e, w) cursor.
                for (int b = lo; b < hi; b += 32) {
                    const int i = b + int(lane);
                    const bool live = i < hi;
                    const uint e = live ? tf[i] : 0xffffffffu;
                    uint rank = 0;
                    uint total = 0;
                    for (uint l = 0; l < 32; ++l) {
                        const uint o = simd_shuffle(e, ushort(l));
                        const uint m = (o == e) ? 1u : 0u;
                        total += m;
                        rank += (l < lane) ? m : 0u;
                    }
                    uint base = 0;
                    if (live && e < uint(E)) {
                        base = atomic_load_explicit(&table[e * W + sg], memory_order_relaxed);
                        const uint pos = base + rank;
                        order[pos] = uint32_t(i);
                        inv[i] = uint32_t(pos);
                        sorted[pos] = uint32_t(e);
                        orderk[pos] = uint32_t(i) / uint32_t(KK);
                    }
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    if (live && e < uint(E) && rank + 1 == total) {
                        atomic_store_explicit(&table[e * W + sg], base + total, memory_order_relaxed);
                    }
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                }
""")
        inversePermutation = MLXFast.metalKernel(
            name: "nemotron_inverse_permutation",
            inputNames: ["order"], outputNames: ["inv"],
            source: """
                uint p = thread_position_in_grid.x;
                inv[order[p]] = static_cast<T>(p);
            """)
        // Stock relu2: Maximum(x, 0) (NaN-propagating) then Multiply(m, m), in T.
        relu2 = MLXFast.metalKernel(
            name: "nemotron_relu2",
            inputNames: ["x"], outputNames: ["out"],
            source: """
                uint i = thread_position_in_grid.x;
                T v = x[i];
                T zero = static_cast<T>(0);
                T m = metal::isnan(v) ? v : (v > zero ? v : zero);
                out[i] = m * m;
            """)
        // y: [tokens*K, N] expert outputs in sorted order; inv[t*K+k] is the
        // sorted row of (t, k); scores: [tokens*K] fp32. Reproduces
        // take(y, inv) -> astype(fp32) -> multiply(scores) -> sum(axis=-2)
        // (col_reduce_small) -> astype(T). The +0.0f is the reduce's
        // op(v, init); separate statements keep each product rounded once.
        combine = MLXFast.metalKernel(
            name: "nemotron_moe_unsort_combine",
            inputNames: ["y", "inv", "scores", "N"], outputNames: ["out"],
            source: """
                uint c = thread_position_in_grid.x * 4;
                uint t = thread_position_in_grid.y;
                float total[4];
                for (int k = 0; k < K; ++k) {
                    uint src = inv[t * K + k];
                    float s = scores[t * K + k];
                    const device T* row = y + size_t(src) * N + c;
                    for (int j = 0; j < 4; ++j) {
                        float v = static_cast<float>(row[j]);
                        float p = v * s;
                        float a = p + 0.0f;
                        if (k == 0) {
                            total[j] = a;
                        } else {
                            total[j] = a + total[j];
                        }
                    }
                }
                device T* o = out + size_t(t) * N + c;
                for (int j = 0; j < 4; ++j) {
                    o[j] = static_cast<T>(total[j]);
                }
            """)
        // `combine` followed by the MoE block's `y + shared` (the stock
        // T + T add) on each output element; seed only, after
        // `NemotronPrefillCombineAdd.qualify`.
        // combineAdd with each row's four values read as one 4-element vector
        // (y, addend) and written as one (out); the statements per value are
        // combineAdd's. Rows are contiguous (row-contiguous inputs).
        combineAdd4 = MLXFast.metalKernel(
            name: "nemotron_moe_unsort_combine_add4",
            inputNames: ["y", "inv", "scores", "N", "addend"], outputNames: ["out"],
            source: """
                uint c = thread_position_in_grid.x * 4;
                uint t = thread_position_in_grid.y;
                float total[4];
                for (int k = 0; k < K; ++k) {
                    uint src = inv[t * K + k];
                    float s = scores[t * K + k];
                    const vec<T, 4> row = *((const device vec<T, 4>*)(y + size_t(src) * N + c));
                    for (int j = 0; j < 4; ++j) {
                        float v = static_cast<float>(row[j]);
                        float p = v * s;
                        float a = p + 0.0f;
                        if (k == 0) {
                            total[j] = a;
                        } else {
                            total[j] = a + total[j];
                        }
                    }
                }
                const vec<T, 4> a4 = *((const device vec<T, 4>*)(addend + size_t(t) * N + c));
                vec<T, 4> o4;
                for (int j = 0; j < 4; ++j) {
                    const T r = static_cast<T>(total[j]);
                    o4[j] = static_cast<T>(static_cast<float>(r) + static_cast<float>(a4[j]));
                }
                *((device vec<T, 4>*)(out + size_t(t) * N + c)) = o4;
            """)
        combineAdd = MLXFast.metalKernel(
            name: "nemotron_moe_unsort_combine_add",
            inputNames: ["y", "inv", "scores", "N", "addend"], outputNames: ["out"],
            source: """
                uint c = thread_position_in_grid.x * 4;
                uint t = thread_position_in_grid.y;
                float total[4];
                for (int k = 0; k < K; ++k) {
                    uint src = inv[t * K + k];
                    float s = scores[t * K + k];
                    const device T* row = y + size_t(src) * N + c;
                    for (int j = 0; j < 4; ++j) {
                        float v = static_cast<float>(row[j]);
                        float p = v * s;
                        float a = p + 0.0f;
                        if (k == 0) {
                            total[j] = a;
                        } else {
                            total[j] = a + total[j];
                        }
                    }
                }
                device T* o = out + size_t(t) * N + c;
                const device T* a = addend + size_t(t) * N + c;
                for (int j = 0; j < 4; ++j) {
                    // astype(T), then the stock T + T add (float add, one rounding).
                    const T r = static_cast<T>(total[j]);
                    o[j] = static_cast<T>(static_cast<float>(r) + static_cast<float>(a[j]));
                }
            """)
    }
}

/// Group expert runs into at most 64 rows. The launch uses the existing
/// 32-row body for short descriptors, preserving its small-run work while
/// letting busy experts share one weight load over a larger row segment.
func nemotronGatherTileRows(rows: Int, experts: Int) -> Int { 64 }

/// Expert-run tiles [count, 3] int32 of `bm` rows: (first sorted row, rows,
/// expert); trailing slots have rows == 0.
struct NemotronHRunTiles {
    let tiles: MLXArray
    let count: Int
    let bm: Int
    let rows: Int
}

/// (order, inverse, sorted) identical to (argSort(flat), argSort(order),
/// flat[order]) for uint32 expert ids < experts, plus the expert-run tiles;
/// nil when unsupported.
private func nemotronCountingSort(_ flat: MLXArray, experts: Int)
    -> (order: MLXArray, inverse: MLXArray, sorted: MLXArray, tiles: NemotronHRunTiles)?
{
    let threads = 1024
    guard flat.ndim == 1, flat.dtype == .uint32, experts > 0, experts <= threads,
          threads % experts == 0, flat.size > 0, flat.size < (1 << 24)
    else { return nil }
    let n = flat.size
    let bm = nemotronGatherTileRows(rows: n, experts: experts)
    // Retain the original capacity bound: an adaptive run uses no more
    // descriptors than its original 32/64-row partition, including split tails.
    let capacityRows = n / experts < 64 ? 32 : 64
    let count = (n + capacityRows - 1) / capacityRows + experts
    let out = NemotronHPrefillKernels.shared.countingSort(
        [flat, n, count], template: [("E", experts), ("THREADS", threads), ("BMT", bm)],
        grid: (threads, 1, 1), threadGroup: (threads, 1, 1),
        outputShapes: [[n], [n], [n], [count, 3]],
        outputDTypes: [.uint32, .uint32, .uint32, .int32])
    return (out[0], out[1], out[2], NemotronHRunTiles(tiles: out[3], count: count, bm: bm, rows: n))
}

/// `nemotronCountingSort` through the chunk-parallel kernel, plus
/// order / k; nil outside its shapes (n <= 3072, experts a multiple of 32).
func nemotronCountingSortChunked(_ flat: MLXArray, experts: Int, k: Int)
    -> (order: MLXArray, inverse: MLXArray, sorted: MLXArray, tiles: NemotronHRunTiles,
        orderK: MLXArray)?
{
    let threads = 1024
    guard flat.ndim == 1, flat.dtype == .uint32, experts > 0, experts % 32 == 0,
          experts * 32 <= 4 * threads, (experts * 32) % threads == 0,
          flat.size > 0, flat.size <= 3072, k > 0
    else { return nil }
    let n = flat.size
    let bm = nemotronGatherTileRows(rows: n, experts: experts)
    // The same descriptor capacity as `nemotronCountingSort`.
    let capacityRows = n / experts < 64 ? 32 : 64
    let count = (n + capacityRows - 1) / capacityRows + experts
    let out = NemotronHPrefillKernels.shared.countingSortChunked(
        [flat, n, count],
        template: [("E", experts), ("THREADS", threads), ("BMT", bm), ("NMAX", 3072), ("KK", k)],
        grid: (threads, 1, 1), threadGroup: (threads, 1, 1),
        outputShapes: [[n], [n], [n], [count, 3], [n]],
        outputDTypes: [.uint32, .uint32, .uint32, .int32, .uint32])
    return (out[0], out[1], out[2], NemotronHRunTiles(tiles: out[3], count: count, bm: bm, rows: n),
            out[4])
}

/// Seed-only MoE combine with the shared-expert add (`combineAdd`).
/// `qualify` checks it bitwise against `combine` then the stock add on
/// seeded random expert rows, routes, scores and addends.
/// NEMOTRON_PREFILL_COMBINE_ADD=0 keeps the two launches.
enum NemotronPrefillCombineAdd {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_COMBINE_ADD"] != "0"
    nonisolated(unsafe) static var verified = false

    static func qualify() {
        verified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron prefill combine add: \(message)\n".utf8))
        }
        guard enabled, NemotronHPrefillX5.enabled else { return report("off (setting)") }
        let kernels = NemotronHPrefillKernels.shared
        var checked = 0
        do {
            let matches = try withError { () -> Bool in
                for (index, (tokens, k, n)) in [(488, 6, 2688), (97, 6, 2688), (16, 2, 64)].enumerated() {
                    let seed = UInt64(0xc0a0_0000 + index)
                    let y = MLXRandom.normal([tokens * k, n], dtype: .bfloat16, scale: 4,
                                             key: MLXRandom.key(seed))
                    let inv = argSort(MLXRandom.uniform(0 ..< 1, [tokens * k], key: MLXRandom.key(seed + 1)))
                        .asType(.uint32)
                    let scores = MLXRandom.uniform(0 ..< 1, [1, tokens, k], key: MLXRandom.key(seed + 2))
                    let addend = MLXRandom.normal([1, tokens, n], dtype: .bfloat16, scale: 4,
                                                  key: MLXRandom.key(seed + 3))
                    let combined = kernels.combine(
                        [y, inv, scores, n], template: [("T", DType.bfloat16), ("K", k)],
                        grid: (n / 4, tokens, 1), threadGroup: (32, 1, 1),
                        outputShapes: [[1, tokens, n]], outputDTypes: [.bfloat16])[0]
                    let stock = combined + addend
                    let fused = kernels.combineAdd(
                        [y, inv, scores, n, addend], template: [("T", DType.bfloat16), ("K", k)],
                        grid: (n / 4, tokens, 1), threadGroup: (32, 1, 1),
                        outputShapes: [[1, tokens, n]], outputDTypes: [.bfloat16])[0]
                    let same = (fused.view(dtype: .uint16) .== stock.view(dtype: .uint16)).all()
                    if !same.item(Bool.self) { return false }
                    checked += 1
                }
                return true
            }
            verified = matches
        } catch {
            return report("off (\(error))")
        }
        report(verified ? "on (\(checked) cases bitwise equal)"
                        : "off (outputs differ from combine + add after \(checked) cases)")
        qualifyVec4()
    }

    /// `combineAdd4` (vector loads and stores) against `combineAdd`, bit for
    /// bit. NEMOTRON_PREFILL_COMBINE_ADD_VEC=0 keeps `combineAdd`.
    nonisolated(unsafe) static var vec4Verified = false
    static func qualifyVec4() {
        vec4Verified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron prefill combine add vec4: \(message)\n".utf8))
        }
        guard verified, ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_COMBINE_ADD_VEC"] != "0" else {
            return report("off (setting)")
        }
        let kernels = NemotronHPrefillKernels.shared
        var same = MLXArray(true)
        for (index, (tokens, k, n)) in [(488, 6, 2688), (97, 6, 2688), (16, 2, 64)].enumerated() {
            let seed = UInt64(0xc0a4_0000 + index)
            let y = MLXRandom.normal([tokens * k, n], dtype: .bfloat16, scale: 4, key: MLXRandom.key(seed))
            let inv = argSort(MLXRandom.uniform(0 ..< 1, [tokens * k], key: MLXRandom.key(seed + 1))).asType(.uint32)
            let scores = MLXRandom.uniform(0 ..< 1, [1, tokens, k], key: MLXRandom.key(seed + 2))
            let addend = MLXRandom.normal([1, tokens, n], dtype: .bfloat16, scale: 4, key: MLXRandom.key(seed + 3))
            func go(_ kernel: MLXFast.MLXFastKernel, _ tg: Int) -> MLXArray {
                kernel([y, inv, scores, n, addend], template: [("T", DType.bfloat16), ("K", k)],
                       grid: (n / 4, tokens, 1), threadGroup: (tg, 1, 1),
                       outputShapes: [[1, tokens, n]], outputDTypes: [.bfloat16])[0]
            }
            same = same .&& (go(kernels.combineAdd, 32).view(dtype: .uint16)
                .== go(kernels.combineAdd4, 64).view(dtype: .uint16)).all()
            eval(same)
        }
        vec4Verified = same.item(Bool.self)
        report(vec4Verified ? "on (3 cases bitwise equal)" : "off (differs from combineAdd)")
    }
}

/// Seed-only (request prefill, `NemotronHPrefillX5.active`) shared-expert
/// relu2: the stock `maximum(up, 0)` (the int scalar cast to T, then the
/// NaN-propagating Maximum) and `m * m` as one `relu2` launch, the kernel the
/// routed experts use. Replaces three launches per MoE layer and two passes
/// over the [L, 3712] activation. `qualify` checks it bitwise against the
/// stock chain on the target's shared up-projection of seeded rows, with
/// signed zeros, infinities, NaN and subnormals injected.
/// NEMOTRON_PREFILL_SHARED_RELU2=0 keeps the stock chain.
enum NemotronPrefillSharedRelu2 {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_SHARED_RELU2"] != "0"
    nonisolated(unsafe) static var verified = false

    /// relu2 of a request-prefill shared up-projection in one launch; nil
    /// outside the request prefill or the kernel's shapes and dtypes.
    static func apply(_ up: MLXArray) -> MLXArray? {
        guard verified, NemotronHPrefillX5.active else { return nil }
        return launch(up)
    }

    private static func launch(_ up: MLXArray) -> MLXArray? {
        // Stock relu2 promotes against an int scalar; fuse only when that
        // keeps the activation dtype (graph metadata, never evaluated).
        guard up.ndim == 3, up.dim(0) == 1, up.dim(1) > 8,
              up.dtype == .bfloat16 || up.dtype == .float16,
              MLX.maximum(up, MLXArray(0)).dtype == up.dtype
        else { return nil }
        return NemotronHPrefillKernels.shared.relu2(
            [up], template: [("T", up.dtype)],
            grid: (up.size, 1, 1), threadGroup: (256, 1, 1),
            outputShapes: [up.shape], outputDTypes: [up.dtype])[0]
    }

    static func qualify(_ target: Module) {
        verified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron prefill shared relu2: \(message)\n".utf8))
        }
        guard enabled, NemotronHPrefillX5.enabled else { return report("off (setting)") }
        guard let moe = target.modules().compactMap({ $0 as? NemotronHMoE }).first,
              let mlp = moe.sharedExperts
        else { return report("off (no shared expert)") }
        // +-0, +-inf, NaN (both signs, a payload), subnormals, +-max, values
        // whose square overflows or lands in the subnormal range, +-1.
        let specials = MLXArray([
            0x0000, 0x8000, 0x7f80, 0xff80, 0x7fc0, 0xffc0, 0x7f81, 0x0001, 0x8001, 0x007f,
            0x0080, 0x7f7f, 0xff7f, 0x5f80, 0xdf80, 0x1f80, 0x9f80, 0x3f80, 0xbf80,
        ] as [UInt16]).view(dtype: .bfloat16)
        var checked = 0
        do {
            let matches = try withError { () -> Bool in
                for (index, rows) in [488, 300, 9].enumerated() {
                    let seed = UInt64(0x5e1a_0000 + index)
                    let x = MLXRandom.normal([1, rows, moe.switchMLP.inputDims], scale: 2,
                                             key: MLXRandom.key(seed)).asType(.bfloat16)
                    var up = mlp.upProj(x)
                    let width = up.dim(2)
                    up = concatenated([specials, up.reshaped(-1)[specials.size...]])
                        .reshaped(1, rows, width)
                    guard let fused = launch(up) else { return false }
                    let stock = relu2(up)
                    let same = (fused.view(dtype: .uint16) .== stock.view(dtype: .uint16)).all()
                    if !same.item(Bool.self) { return false }
                    checked += 1
                }
                return true
            }
            verified = matches
        } catch {
            return report("off (\(error))")
        }
        report(verified ? "on (\(checked) cases bitwise equal)"
                        : "off (outputs differ from maximum + multiply after \(checked) cases)")
    }
}

/// Seed-only (request prefill, `NemotronHPrefillX5.active`) chunk-parallel
/// counting sort. `qualify` runs it next to `nemotronCountingSort` on seeded
/// top-k routes (skewed and uniform, 2 to 512 tokens) and engages it only if
/// order, inverse, sorted ids and tiles are bitwise equal and order / k
/// equals the stock floor divide. NEMOTRON_PREFILL_COUNTING_SORT=0 keeps the
/// single-thread-scan kernel.
enum NemotronPrefillCountingSort {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_COUNTING_SORT"] != "0"
    nonisolated(unsafe) static var verified = false

    static func qualify(experts: Int, k: Int) {
        verified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron prefill counting sort: \(message)\n".utf8))
        }
        guard enabled, NemotronHPrefillX5.enabled else { return report("off (setting)") }
        var checked = 0
        do {
            let matches = try withError { () -> Bool in
                for (index, (tokens, skew)) in [(488, Float(2)), (488, 0), (512, 4), (512, 9), (300, 1), (300, 12), (9, 1), (2, 0)].enumerated() {
                    let seed = UInt64(0x50c7_0000 + index)
                    let logits = MLXRandom.normal([tokens, experts], key: MLXRandom.key(seed))
                        + skew * MLXRandom.normal([1, experts], key: MLXRandom.key(seed + 1))
                    let flat = argSort(-logits, axis: -1)[0..., ..<k].asType(.uint32).flattened()
                    guard let fast = nemotronCountingSortChunked(flat, experts: experts, k: k),
                          let stock = nemotronCountingSort(flat, experts: experts)
                    else { return false }
                    var same = MLXArray(true)
                    for (a, b) in [(fast.order, stock.order), (fast.inverse, stock.inverse),
                                   (fast.sorted, stock.sorted), (fast.orderK, stock.order.floorDivide(k))] {
                        same = same .&& (a .== b).all()
                    }
                    same = same .&& (fast.tiles.tiles .== stock.tiles.tiles).all()
                    if !same.item(Bool.self) { return false }
                    checked += 1
                }
                return true
            }
            verified = matches
        } catch {
            return report("off (\(error))")
        }
        report(verified ? "on (\(checked) cases bitwise equal)"
                        : "off (outputs differ from the serial-scan kernel after \(checked) cases)")
    }

    /// Qualify with the target's first MoE layer shape.
    static func qualify(_ target: Module) {
        for module in target.modules() {
            if let moe = module as? NemotronHMoE {
                return qualify(experts: moe.switchMLP.numExperts, k: moe.numExpertsPerTok)
            }
        }
        FileHandle.standardError.write(Data("nemotron prefill counting sort: off (no MoE layer)\n".utf8))
    }
}

/// Compile the multi-token kernels at model load (same names, template
/// arguments, dtypes and address spaces as the prefill calls: every array
/// input has >= 8 elements so none is passed as `constant`), so the first
/// timed prefill does not pay their Metal JIT. Values are discarded.
private func nemotronWarmUpPrefillKernels(
    expertsPerToken k: Int, experts: Int, ssmHeads: Int, ssmGroups: Int,
    ssmHeadDim: Int, ssmStateSize: Int, dtype: DType
) {
    guard (1...8).contains(k) else { return }
    let kernels = NemotronHPrefillKernels.shared
    let rows = 2 * k
    let order = argSort(MLXArray((0 ..< rows).map { Int32(rows - 1 - $0) }))
    var arrays: [MLXArray] = []
    arrays.append(kernels.inversePermutation(
        [order], template: [("T", order.dtype)],
        grid: (rows, 1, 1), threadGroup: (256, 1, 1),
        outputShapes: [[rows]], outputDTypes: [order.dtype])[0])
    let y = MLXArray.zeros([rows, 1, 8], dtype: dtype)
    arrays.append(kernels.relu2(
        [y], template: [("T", dtype)],
        grid: (y.size, 1, 1), threadGroup: (256, 1, 1),
        outputShapes: [y.shape], outputDTypes: [dtype])[0])
    arrays.append(kernels.combine(
        [y.reshaped(rows, 8), order, MLXArray.zeros([1, 2, k], dtype: .float32), 8],
        template: [("T", dtype), ("K", k)],
        grid: (2, 2, 1), threadGroup: (32, 1, 1),
        outputShapes: [[1, 2, 8]], outputDTypes: [dtype])[0])
    let cb = MLXArray.zeros([1, 8, 4, 4], dtype: dtype)
    let seg = MLXArray.zeros([1, 64, 4, 4], dtype: .float32)
    arrays.append(NemotronHScanKernels.shared.maskedDecay(
        [cb, seg, 4, 8], template: [("T", dtype)],
        grid: (4, 4, 64), threadGroup: (32, 8, 1),
        outputShapes: [[1, 64, 4, 4]], outputDTypes: [.float32])[0])
    if NemotronHPrefillX5.enabled {
        // Counting sort for small and large row sets, using the current descriptor bound.
        for n in [rows, 64 * experts] {
            if let cs = nemotronCountingSort(MLXArray.zeros([n], dtype: .uint32), experts: experts) {
                arrays.append(cs.order)
            }
        }
        if ssmGroups > 0, ssmHeads % ssmGroups == 0 {
            let dtA = MLXArray.zeros([1, 4, ssmHeads], dtype: .float32)
            let cbw = MLXArray.zeros([1, ssmGroups, 4, 4], dtype: dtype)
            arrays.append(contentsOf: NemotronHScanKernels.shared.scanDecay(
                [dtA, cbw, 4],
                template: [("CBT", dtype), ("H", ssmHeads), ("R", ssmHeads / ssmGroups)],
                grid: (1024, 1, ssmHeads), threadGroup: (1024, 1, 1),
                outputShapes: [[1, ssmHeads, 4, 4], [1, ssmHeads, 1, 4]],
                outputDTypes: [.float32, .float32]))
            if ssmStateSize == 128, ssmHeadDim % 32 == 0 {
                arrays.append(NemotronHScanKernels.shared.windowOut(
                    [MLXArray.zeros([1, ssmHeads, ssmHeadDim, ssmStateSize], dtype: .float32),
                     MLXArray.zeros([1, 2, ssmGroups, ssmStateSize], dtype: dtype),
                     MLXArray.zeros([1, ssmHeads, 2, ssmHeadDim], dtype: .float32),
                     MLXArray.zeros([1, 2, ssmHeads], dtype: .float32), 2],
                    template: [("CT", dtype), ("OT", dtype), ("H", ssmHeads),
                               ("R", ssmHeads / ssmGroups), ("DH", ssmHeadDim), ("TC", 32)],
                    grid: (ssmHeadDim / 32 * 128, 1, ssmHeads), threadGroup: (128, 1, 1),
                    outputShapes: [[1, 2, ssmHeads, ssmHeadDim]], outputDTypes: [dtype])[0])
            }
        }
    }
    if NemotronHPrefillX5.runGather {
        let xw = MLXArray.zeros([rows, 1, 64], dtype: dtype)
        let ww = MLXArray.zeros([1, 64, 8], dtype: .uint32)
        let sw = MLXArray.zeros([1, 64, 1], dtype: dtype)
        let tw = MLXArray([Int32(0), Int32(rows), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshaped(4, 3)
        for bm in [32, 64] {
            for relu2 in [false, true] where !relu2 || dtype == .bfloat16 {
                arrays.append(NemotronHRunGatherKernel.shared.kernel(
                    [xw, ww, sw, sw, tw, 64, 64],
                    template: [("T", dtype), ("GS", 64), ("BITS", 4), ("BM", bm), ("BN", 64), ("BK", 64),
                               ("ALIGN_N", true), ("ALIGN_K", true), ("RELU2", relu2)],
                    grid: (128, 4, 1), threadGroup: (128, 1, 1),
                    outputShapes: [[rows, 1, 64]], outputDTypes: [dtype])[0])
            }
        }
    }
    eval(arrays)
}

// MARK: - Run-aligned NAX gather (stock affine_gather_qmm_rhs_nax per row)

/// The part of the MLX JIT preamble (mlx-generated gemm_nax + quantized_nax)
/// that the stock affine_gather_qmm_rhs_nax uses, followed by its run-aligned
/// clone. Output stores add a typed ReLU2 specialization; the remaining
/// declarations retain the stock math. Unused declarations are dropped.
let nemotronRunGatherHeader = #"""
// Subset of the MLX JIT preamble that stock affine_gather_qmm_rhs_nax is
// compiled from (mlx-generated gemm_nax.cpp + quantized_nax.cpp). Only the
// output stores add the typed ReLU2 specialization; unused declarations are dropped.

#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#define STEEL_PRAGMA_NO_UNROLL _Pragma("clang loop unroll(disable)")

// steel/utils/type_traits.h
#include <metal_stdlib>

#pragma METAL internals : enable

namespace metal {

template <typename T>
struct is_empty : metal::bool_constant<__is_empty(T)> {};

#ifdef __cpp_variable_templates
template <typename T>
constexpr constant bool is_empty_v = is_empty<T>::value;
#endif

template <typename... Ts>
struct make_void {
  typedef void type;
};

template <typename... Ts>
using void_t = typename make_void<Ts...>::type;

template <class T>
struct is_static : metal::bool_constant<is_empty<remove_cv_t<T>>::value> {};

template <typename T>
struct pointer_element {};

template <typename T>
struct pointer_element<thread T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<device T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<constant T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<threadgroup T*> {
  using type = remove_cv_t<T>;
};

template <typename T>
using pointer_element_t = typename pointer_element<remove_cv_t<T>>::type;

} // namespace metal

#pragma METAL internals : disable

// steel/utils/integral_constant.h
#include <metal_stdlib>

#pragma METAL internals : enable

namespace mlx {
namespace steel {

///////////////////////////////////////////////////////////////////////////////
// Integral constant with casting
///////////////////////////////////////////////////////////////////////////////

template <typename T, T v>
struct integral_constant {
  static constexpr constant T value = v;
  using value_type = T;
  using type = integral_constant;

  METAL_FUNC constexpr operator value_type() const thread noexcept {
    return value;
  }
};

template <bool B>
using bool_constant = integral_constant<bool, B>;
using true_type = bool_constant<true>;
using false_type = bool_constant<false>;

template <class T>
struct is_integral : bool_constant<metal::is_integral<T>::value> {};

template <class T, T v>
struct is_integral<integral_constant<T, v>>
    : bool_constant<metal::is_integral<T>::value> {};

template <typename T>
constexpr constant bool is_integral_v = is_integral<T>::value;

template <int val>
using Int = integral_constant<int, val>;

///////////////////////////////////////////////////////////////////////////////
// Binary Operators on Integral constants
///////////////////////////////////////////////////////////////////////////////

#define integral_const_binop(__op__, __operator__)          \
  template <typename T, T tv, typename U, U uv>             \
  METAL_FUNC constexpr auto __operator__(                   \
      integral_constant<T, tv>, integral_constant<U, uv>) { \
    constexpr auto res = tv __op__ uv;                      \
    using res_t = metal::remove_addrspace_t<decltype(res)>; \
    return integral_constant<res_t, res>{};                 \
  }

integral_const_binop(+, operator+);
integral_const_binop(-, operator-);
integral_const_binop(*, operator*);
integral_const_binop(/, operator/);

integral_const_binop(==, operator==);
integral_const_binop(!=, operator!=);
integral_const_binop(<, operator<);
integral_const_binop(>, operator>);
integral_const_binop(<=, operator<=);
integral_const_binop(>=, operator>=);

integral_const_binop(&&, operator&&);
integral_const_binop(||, operator||);

template <typename T, typename = metal::enable_if_t<!is_integral_v<T>>>
METAL_FUNC constexpr auto operator||(true_type, T) {
  return true_type{};
}
template <typename T, typename = metal::enable_if_t<!is_integral_v<T>>>
METAL_FUNC constexpr auto operator||(T, true_type) {
  return true_type{};
}

template <typename T, typename = metal::enable_if_t<!is_integral_v<T>>>
METAL_FUNC constexpr auto operator&&(false_type, T) {
  return false_type{};
}

template <typename T, typename = metal::enable_if_t<!is_integral_v<T>>>
METAL_FUNC constexpr auto operator&&(T, false_type) {
  return false_type{};
}

// Dispatch utilities
template <typename F>
void dispatch_bool(bool v, F f) {
  if (v) {
    f(true_type{});
  } else {
    f(false_type{});
  }
}

template <int start, int stop, int step, typename F>
constexpr void const_for_loop(F f) {
  if constexpr (start < stop) {
    constexpr auto idx = Int<start>{};
    f(idx);
    const_for_loop<start + step, stop, step, F>(f);
  }
}

#undef integral_const_binop

///////////////////////////////////////////////////////////////////////////////
// Reduction operators
///////////////////////////////////////////////////////////////////////////////

template <typename T>
METAL_FUNC constexpr T sum(T x) {
  return x;
}

template <typename T, typename... Us>
METAL_FUNC constexpr auto sum(T x, Us... us) {
  return x + sum(us...);
}

} // namespace steel
} // namespace mlx

#pragma METAL internals : disable

// steel/gemm/nax.h
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>


#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// MMA helper
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

///////////////////////////////////////////////////////////////////////////////
// NAX Steel with new tiles
///////////////////////////////////////////////////////////////////////////////

struct BaseNAXFrag {
  STEEL_CONST short kFragRows = 16;
  STEEL_CONST short kFragCols = 16;

  STEEL_CONST short kElemsPerFrag = (kFragRows * kFragCols) / 32;

  STEEL_CONST short kElemRows = 2;
  STEEL_CONST short kElemCols = 4;

  STEEL_CONST short kElemRowsJump = 8;

  static_assert(
      kElemRows * kElemCols == kElemsPerFrag,
      "MMAFrag shape is not consistent with MMAFrag size");

  template <typename U>
  using dtype_frag_t = typename metal::vec<U, kElemsPerFrag>;

  METAL_FUNC static short2 get_coord() {
    const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
    const short qid = simd_lane_id >> 2;
    const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));
    const short fn = ((qid & 2) | (simd_lane_id & 1)) * 4;
    return short2{fn, fm};
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;

      if constexpr (metal::is_same_v<StrY, Int<1>>) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + c + j]);
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] =
              static_cast<T>(src[r * str_x + (c + j) * str_y]);
        }
      }
    }
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load_safe(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;
    auto ly = lim_y - sc.x;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((r < lx) && ((c + j) < ly)) {
          dst[i * kElemCols + j] =
              static_cast<T>(src[r * str_x + (c + j) * str_y]);
        } else {
          dst[i * kElemCols + j] = T(0);
        }
      }
    }
  }

  // Narrow exactly as the ordinary store, then apply the typed activation.
  template <bool RELU2, typename U, typename T>
  METAL_FUNC static constexpr U store_value(T value) {
    U h = static_cast<U>(value);
    if constexpr (RELU2) {
      U zero = static_cast<U>(0);
      U m = metal::isnan(h) ? h : (h > zero ? h : zero);
      return m * m;
    }
    return h;
  }

  template <
      bool RELU2 = false,
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      OffX off_x = {},
      OffY off_y = {}) {
    using U = pointer_element_t<DstPtrType>;

    const short2 sc = get_coord();
    dst += sc.y * str_x + sc.x * str_y;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;

      if constexpr (metal::is_same_v<StrY, Int<1>>) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[r * str_x + c + j] = store_value<RELU2, U>(src[i * kElemCols + j]);
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[r * str_x + (c + j) * str_y] =
              store_value<RELU2, U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  template <
      bool RELU2 = false,
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename StartX,
      typename StopX,
      typename StartY,
      typename StopY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store_slice(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      StartX start_x,
      StopX stop_x,
      StartY start_y,
      StopY stop_y,
      OffX off_x = Int<0>{},
      OffY off_y = Int<0>{}) {
    using U = pointer_element_t<DstPtrType>;

    const short2 sc = get_coord();

    const_for_loop<0, kElemRows, 1>([&](auto idx_row) {
      const auto r = off_x + idx_row * Int<kElemRowsJump>{};
      if (r >= stop_x - sc.y || r < start_x - sc.y) {
        return;
      }

      const_for_loop<0, kElemCols, 1>([&](auto idx_col) {
        const auto c = off_y + idx_col;
        if (c >= stop_y - sc.x || c < start_y - sc.x) {
          return;
        }

        const auto src_idx = idx_row * Int<kElemCols>{} + idx_col;
        dst[(r + sc.y) * str_x + (c + sc.x) * str_y] =
            store_value<RELU2, U>(src[src_idx]);
      });
    });
  }

  template <
      typename CType,
      typename AType,
      typename BType,
      bool transpose_a = false,
      bool transpose_b = false>
  METAL_FUNC static constexpr void mma(
      thread dtype_frag_t<CType>& Cn0,
      thread dtype_frag_t<CType>& Cn1,
      const thread dtype_frag_t<AType>& A,
      metal::bool_constant<transpose_a>,
      const thread dtype_frag_t<BType>& Bn0,
      const thread dtype_frag_t<BType>& Bn1,
      metal::bool_constant<transpose_b>) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16,
        32,
        16,
        transpose_a,
        transpose_b,
        true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

    // Create matmul op
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

    // Create matmul operands in registers
    auto ct_a =
        gemm_op
            .template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto ct_b =
        gemm_op
            .template get_right_input_cooperative_tensor<AType, BType, CType>();

    // Create matmul output in register
    auto ct_c = gemm_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ct_a)>,
        metal::remove_addrspace_t<decltype(ct_b)>,
        CType>();

    // Load A in to left operand registers
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_a[i] = A[i];
    }

    // Load B into right operand registers
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_b[i] = Bn0[i];
      ct_b[kElemsPerFrag + i] = Bn1[i];
    }

    // Load C into output registers (op handles accumulation)
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_c[i] = Cn0[i];
      ct_c[kElemsPerFrag + i] = Cn1[i];
    }

    // Do matmul
    gemm_op.run(ct_a, ct_b, ct_c);

    // Copy out results
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      Cn0[i] = ct_c[i];
      Cn1[i] = ct_c[kElemsPerFrag + i];
    }
  }

};

template <
    typename T,
    short kTileRows_,
    short kTileCols_,
    class NAXFrag_ = BaseNAXFrag>
struct NAXTile {
  using NAXFrag_t = NAXFrag_;
  using elem_type = T;

  STEEL_CONST short kFragRows = NAXFrag_t::kFragRows;
  STEEL_CONST short kFragCols = NAXFrag_t::kFragCols;
  STEEL_CONST short kElemsPerFrag = NAXFrag_t::kElemsPerFrag;

  STEEL_CONST short kTileRows = kTileRows_;
  STEEL_CONST short kTileCols = kTileCols_;

  STEEL_CONST short kRows = kTileRows * kFragRows;
  STEEL_CONST short kCols = kTileCols * kFragCols;

  STEEL_CONST short kNumFrags = kTileRows * kTileCols;
  STEEL_CONST short kElemsPerTile = kNumFrags * kElemsPerFrag;

  STEEL_CONST short kFragThrRows = NAXFrag_t::kElemRows;
  STEEL_CONST short kFragThrCols = NAXFrag_t::kElemCols;
  STEEL_CONST short kFragRowsJump = NAXFrag_t::kElemRowsJump;

  STEEL_CONST short kRowsPerThread = kTileRows * NAXFrag_t::kElemRows;
  STEEL_CONST short kColsPerThread = kTileCols * NAXFrag_t::kElemCols;

  typedef typename NAXFrag_t::template dtype_frag_t<T> frag_type;

  frag_type val_frags[kNumFrags]; // = {frag_type(0)};

  METAL_FUNC NAXTile() thread {}

  METAL_FUNC constexpr void clear() thread {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kNumFrags; ++i) {
      val_frags[i] = frag_type(0);
    }
  }

  METAL_FUNC constexpr thread frag_type& frag_at(const short i, const short j)
      thread {
    return val_frags[i * kTileCols + j];
  }

  template <int i, int j>
  METAL_FUNC constexpr thread frag_type& frag_at() thread {
    return val_frags[i * kTileCols + j];
  }

  template <int i, int j>
  METAL_FUNC constexpr const thread frag_type& frag_at() const thread {
    return val_frags[i * kTileCols + j];
  }

  template <bool transpose>
  METAL_FUNC constexpr thread frag_type& frag_at(
      const short i,
      const short j,
      metal::bool_constant<transpose>) thread {
    if constexpr (transpose) {
      return frag_at(j, i);
    } else {
      return frag_at(i, j);
    }
  }

  template <typename U, int str_x, int str_y>
  METAL_FUNC void load(const threadgroup U* src) thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load(
            frag_at<idx_row.value, idx_col.value>(),
            src,
            Int<str_x>{},
            Int<str_y>{},
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void load(const device U* src, const int ld) thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load(
            frag_at<idx_row.value, idx_col.value>(),
            src,
            ld,
            Int<1>{},
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <bool RELU2 = false, typename U>
  METAL_FUNC void store(device U* dst, const int ld) const thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::template store<RELU2>(
            frag_at<idx_row.value, idx_col.value>(),
            dst,
            ld,
            Int<1>{},
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void load_safe(
      const device U* src,
      const int ld,
      const short2 src_tile_dims) thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load_safe(
            frag_at<idx_row.value, idx_col.value>(),
            src,
            ld,
            Int<1>{},
            src_tile_dims.y,
            src_tile_dims.x,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <bool RELU2 = false, typename U>
  METAL_FUNC void store_slice(
      device U* dst,
      const int ld,
      const short2 start,
      const short2 stop) const thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::template store_slice<RELU2>(
            frag_at<idx_row.value, idx_col.value>(),
            dst,
            ld,
            Int<1>{},
            start.y,
            stop.y,
            start.x,
            stop.x,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }
};

template <
    class CTile,
    class ATile,
    class BTile,
    bool transpose_a,
    bool transpose_b>
METAL_FUNC void tile_matmad_nax(
    thread CTile& C,
    thread ATile& A,
    metal::bool_constant<transpose_a>,
    thread BTile& B,
    metal::bool_constant<transpose_b>) {
  // Static checks
  constexpr short TMa = transpose_a ? ATile::kTileCols : ATile::kTileRows;
  constexpr short TM = CTile::kTileRows;
  static_assert(TMa == TM, "MXU tile matmul: M dimensions do not match");

  constexpr short TNb = transpose_b ? BTile::kTileRows : BTile::kTileCols;
  constexpr short TN = CTile::kTileCols;
  static_assert(TNb == TN, "MXU tile matmul: N dimensions do not match");

  constexpr short TKa = transpose_a ? ATile::kTileRows : ATile::kTileCols;
  constexpr short TK = transpose_b ? BTile::kTileCols : BTile::kTileRows;
  static_assert(TKa == TK, "MXU tile matmul: K dimensions do not match");

  constexpr auto ta = metal::bool_constant<transpose_a>{};
  constexpr auto tb = metal::bool_constant<transpose_b>{};

  if constexpr (TN == 1 && TM % 2 == 0) {
    STEEL_PRAGMA_UNROLL
    for (short mm = 0; mm < TM; mm += 2) {
      STEEL_PRAGMA_UNROLL
      for (short nn = 0; nn < TN; ++nn) {
        STEEL_PRAGMA_UNROLL
        for (short kk = 0; kk < TK; ++kk) {
          CTile::NAXFrag_t::mma(
              C.frag_at(mm, nn),
              C.frag_at(mm + 1, nn),
              A.frag_at(mm, kk, ta),
              A.frag_at(mm + 1, kk, ta),
              metal::bool_constant<transpose_a>{},
              B.frag_at(kk, nn, tb),
              metal::bool_constant<transpose_b>{});
        }
      }
    }
  } else if constexpr (TN % 2 == 0) {
    STEEL_PRAGMA_UNROLL
    for (short mm = 0; mm < TM; ++mm) {
      STEEL_PRAGMA_UNROLL
      for (short nn = 0; nn < TN; nn += 2) {
        STEEL_PRAGMA_UNROLL
        for (short kk = 0; kk < TK; ++kk) {
          CTile::NAXFrag_t::mma(
              C.frag_at(mm, nn),
              C.frag_at(mm, nn + 1),
              A.frag_at(mm, kk, ta),
              metal::bool_constant<transpose_a>{},
              B.frag_at(kk, nn, tb),
              B.frag_at(kk, nn + 1, tb),
              metal::bool_constant<transpose_b>{});
        }
      }
    }
  }
}

} // namespace steel
} // namespace mlx

// quantized_nax.h
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;
using namespace mlx::steel;

using namespace metal;

#define MLX_MTL_CONST static constant constexpr const

MLX_MTL_CONST int SIMD_SIZE = 32;
MLX_MTL_CONST int QUAD_SIZE = 4;

template <int bits, int wsize = 8>
inline constexpr short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

template <int bits, int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

template <typename U, int N, int bits>
inline void
dequantize(const device uint8_t* w, U scale, U bias, threadgroup U* w_local) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  const float s = float(scale);
  const float b = float(bias);

  if (bits == 2) {
    float sc[4] = {s, s / 4.0f, s / 16.0f, s / 64.0f};
    for (int i = 0; i < (N / 4); i++) {
      w_local[4 * i] = static_cast<U>(sc[0] * (w[i] & 0x03) + b);
      w_local[4 * i + 1] = static_cast<U>(sc[1] * (w[i] & 0x0c) + b);
      w_local[4 * i + 2] = static_cast<U>(sc[2] * (w[i] & 0x30) + b);
      w_local[4 * i + 3] = static_cast<U>(sc[3] * (w[i] & 0xc0) + b);
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (N / 8); i++) {
      w_local += 8 * i;
      w += 3 * i;

      w_local[0] = static_cast<U>((w[0] & 0x7) * s + b);
      w_local[1] = static_cast<U>(((w[0] & 0x38) >> 3) * s + b);
      w_local[2] =
          static_cast<U>((((w[0] & 0xc0) >> 6) + ((w[1] & 0x1) << 2)) * s + b);
      w_local[3] = static_cast<U>(((w[1] & 0xe) >> 1) * s + b);
      w_local[4] = static_cast<U>(((w[1] & 0x70) >> 4) * s + b);
      w_local[5] =
          static_cast<U>((((w[1] & 0x80) >> 7) + ((w[2] & 0x3) << 1)) * s + b);
      w_local[6] = static_cast<U>(((w[2] & 0x1c) >> 2) * s + b);
      w_local[7] = static_cast<U>(((w[2] & 0xe0) >> 5) * s + b);
    }
  }

  else if (bits == 4) {
    float sc[2] = {s, s / 16.0f};
    for (int i = 0; i < (N / 2); i++) {
      w_local[2 * i] = static_cast<U>(sc[0] * (w[i] & 0x0f) + b);
      w_local[2 * i + 1] = static_cast<U>(sc[1] * (w[i] & 0xf0) + b);
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (N / 8); i++) {
      w_local += 8 * i;
      w += 5 * i;

      w_local[0] = static_cast<U>((w[0] & 0x1f) * s + b);
      w_local[1] =
          static_cast<U>((((w[0] & 0xe0) >> 5) + ((w[1] & 0x3) << 3)) * s + b);
      w_local[2] = static_cast<U>(((w[1] & 0x7c) >> 2) * s + b);
      w_local[3] =
          static_cast<U>((((w[1] & 0x80) >> 7) + ((w[2] & 0xf) << 1)) * s + b);
      w_local[4] =
          static_cast<U>((((w[2] & 0xf0) >> 4) + ((w[3] & 0x1) << 4)) * s + b);
      w_local[5] = static_cast<U>(((w[3] & 0x3e) >> 1) * s + b);
      w_local[6] =
          static_cast<U>((((w[3] & 0xc0) >> 6) + ((w[4] & 0x7) << 2)) * s + b);
      w_local[7] = static_cast<U>(((w[4] & 0xf8) >> 3) * s + b);
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (N / 4); i++) {
      w_local += 4 * i;
      w += 3 * i;
      w_local[0] = static_cast<U>((w[0] & 0x3f) * s + b);
      w_local[1] =
          static_cast<U>((((w[0] >> 6) & 0x03) + ((w[1] & 0x0f) << 2)) * s + b);
      w_local[2] =
          static_cast<U>((((w[1] >> 4) & 0x0f) + ((w[2] & 0x03) << 4)) * s + b);
      w_local[3] = static_cast<U>(((w[2] >> 2) & 0x3f) * s + b);
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      w_local[i] = static_cast<U>(s * w[i] + b);
    }
  }
}

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short group_size,
    short bits>
struct QuantizedBlockLoader {
  static_assert(
      BCOLS <= group_size,
      "The group size should be larger than the columns");
  static_assert(
      group_size % BCOLS == 0,
      "The group size should be divisible by the columns");
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  MLX_MTL_CONST short pack_factor = get_pack_factor<bits, 8>();
  MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack<bits>();
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  MLX_MTL_CONST short group_steps = group_size / BCOLS;

  const int src_ld;
  const int tile_stride;
  short group_step_cnt;
  const int group_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  const device T* scales;
  const device T* biases;

  QuantizedBlockLoader(
      const device uint8_t* src_,
      const device T* scales_,
      const device T* biases_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]) thread
      : src_ld(src_ld_),
        tile_stride(
            reduction_dim ? BCOLS_PACKED* bytes_per_pack
                          : BROWS * src_ld * bytes_per_pack / pack_factor),
        group_step_cnt(0),
        group_stride(BROWS* src_ld / group_size),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads* thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        scales(scales_ + bi * src_ld / group_size),
        biases(biases_ + bi * src_ld / group_size) {}

  void load_unsafe() const thread {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    T scale = *scales;
    T bias = *biases;
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
    }
  }

  void load_safe(short2 src_tile_dim) const thread {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    if (reduction_dim == 1 && bi >= src_tile_dim.x) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    if (reduction_dim == 0 && bi >= src_tile_dim.y) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    T scale = *scales;
    T bias = *biases;
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          (device uint8_t*)(src + i * bytes_per_pack),
          scale,
          bias,
          dst + i * pack_factor);
    }
  }

  void next() thread {
    src += tile_stride;
    if (reduction_dim == 1) {
      if (group_steps > 1) {
        group_step_cnt++;
        if (group_step_cnt == group_steps) {
          group_step_cnt = 0;
          scales++;
          biases++;
        }
      } else {
        scales++;
        biases++;
      }
    } else {
      scales += group_stride;
      biases += group_stride;
    }
  }
};

// Nemotron: affine_gather_qmm_rhs_nax (transpose = true) with tiles aligned
// to expert runs. tiles[t] = (first sorted row, rows <= BM, expert). Every
// tile holds exactly one expert segment, so the stock segment loop collapses
// to one pass; the loader, K loop and NAX fragment MMA sequence are stock.
// The optional typed ReLU2 follows the native output cast. Row bounds stay
// m_lo_lim = 0, m_hi_lim from the tile's row count.
template <
    typename T,
    int group_size,
    int bits,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool align_N,
    bool align_K,
    bool RELU2>
METAL_FUNC void nemotron_gather_qmm_runs_impl(
    const device T* x,
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device int* tiles,
    device T* y,
    const int N,
    const int K,
    threadgroup T* Ws,
    uint3 tid,
    uint simd_group_id,
    uint simd_lane_id) {
  const int tile_rows = tiles[tid.y * 3 + 1];
  if (tile_rows <= 0) {
    return;
  }
  const int row0 = tiles[tid.y * 3 + 0];
  const uint32_t index = uint32_t(tiles[tid.y * 3 + 2]);

  constexpr bool transpose = true;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using loader_w_t = QuantizedBlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      transpose,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int K_it = K / BK;
  const size_t stride_w = size_t(N) * K_w;
  const size_t stride_s = size_t(N) * K_g;
  const int y_col = tid.x * BN;
  const size_t y_col_long = size_t(y_col);

  const short tgp_bm = short(min(BM, tile_rows));
  const short tgp_bn = align_N ? BN : short(min(BN, N - y_col));

  const int k_remain = K - K_it * BK;
  const short2 tile_w = short2(k_remain, tgp_bn);

  auto wl = (const device uint8_t*)w;
  x += size_t(row0) * K;
  y += size_t(row0) * N + y_col_long;
  wl += y_col_long * K_w;
  scales += y_col_long * K_g;
  biases += y_col_long * K_g;

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);

  const short sgp_sm = min(SM, short(max(0, int(tgp_bm) - tm)));
  const short sgp_sn =
      align_N ? SN : min(SN, short(max(0, (N - (y_col + tn)))));

  const bool is_unaligned_sm = (sgp_sm != SM);
  const bool is_unaligned_bn = align_N ? false : (tgp_bn != BN);

  constexpr short BR = TN;
  constexpr short BC = TK;

  using AccumType = float;

  threadgroup_barrier(mem_flags::mem_none);

  const short m_lo_lim = 0;
  const short m_hi_lim = min(int(sgp_sm), max(0, int(tgp_bm) - tm));
  const bool sg_active = m_hi_lim > m_lo_lim;

  NAXTile<AccumType, TM, TN> Dtile;
  Dtile.clear();

  const device T* xn = x + tm * K;

  thread loader_w_t loader_w(
      wl + index * stride_w,
      scales + index * stride_s,
      biases + index * stride_s,
      K,
      Ws,
      simd_group_id,
      simd_lane_id);

  dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
    dispatch_bool(align_N || !is_unaligned_bn, [&](auto kAlignedN) {
      for (int k = 0; k < K_it; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if constexpr (kAlignedN.value) {
          loader_w.load_unsafe();
        } else {
          loader_w.load_safe(short2(BK, tgp_bn));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        STEEL_PRAGMA_NO_UNROLL
        for (int kk1 = 0; kk1 < BK; kk1 += SK) {
          if (sg_active) {
            NAXTile<T, TM, TK> Atile;
            NAXTile<T, BR, BC> Btile;

            volatile int compiler_barrier;

            if constexpr (kAlignedM.value) {
              Atile.load(xn + kk1, K);
            } else {
              Atile.load_safe(xn + kk1, K, short2(SK, sgp_sm));
            }

            Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);

            tile_matmad_nax(
                Dtile,
                Atile,
                metal::bool_constant<false>{},
                Btile,
                metal::bool_constant<transpose>{});

            (void)compiler_barrier;
          }
        }

        xn += BK;
        loader_w.next();
      }

      if (!align_K) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_w.load_safe(tile_w);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        STEEL_PRAGMA_NO_UNROLL
        for (int kk1 = 0; kk1 < BK; kk1 += SK) {
          if (sg_active) {
            NAXTile<T, TM, TK> Atile;
            NAXTile<T, BR, BC> Btile;

            volatile int compiler_barrier;

            const short psk = min(int(SK), max(0, (BK - kk1)));
            Atile.load_safe(xn + kk1, K, short2(psk, sgp_sm));

            Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);

            tile_matmad_nax(
                Dtile,
                Atile,
                metal::bool_constant<false>{},
                Btile,
                metal::bool_constant<transpose>{});

            (void)compiler_barrier;
          }
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      if constexpr (kAlignedN.value) {
        if (m_lo_lim == 0 && m_hi_lim == SM) {
          Dtile.template store<RELU2>(y + tm * N + tn, N);
        } else {
          Dtile.template store_slice<RELU2>(
              y + tm * N + tn, N, short2(0, m_lo_lim), short2(SN, m_hi_lim));
        }
      } else {
        Dtile.template store_slice<RELU2>(
            y + tm * N + tn,
            N,
            short2(0, m_lo_lim),
            short2(sgp_sn, m_hi_lim));
      }
    });
  });
}
"""#

private final class NemotronHRunGatherKernel: Sendable {
    static let shared = NemotronHRunGatherKernel()
    let kernel: MLXFast.MLXFastKernel
    private init() {
        kernel = MLXFast.metalKernel(
            name: "nemotron_gather_qmm_runs_adaptive_tiles",
            inputNames: ["x", "w", "scales", "biases", "tiles", "N", "K"],
            outputNames: ["y"],
            source: """
                threadgroup T Ws[BN * (BK + 16 / sizeof(T))];
                // Descriptor length is uniform across the threadgroup. Both bodies
                // retain the same K loop and independent per-row NAX fragment updates.
                if (BM == 64 && tiles[threadgroup_position_in_grid.y * 3 + 1] <= 32) {
                    nemotron_gather_qmm_runs_impl<T, GS, BITS, 32, BN, BK, 2, 2, ALIGN_N, ALIGN_K, RELU2>(
                        x, w, scales, biases, tiles, y, N, K, Ws,
                        threadgroup_position_in_grid,
                        simdgroup_index_in_threadgroup,
                        thread_index_in_simdgroup);
                } else {
                    nemotron_gather_qmm_runs_impl<T, GS, BITS, BM, BN, BK, 2, 2, ALIGN_N, ALIGN_K, RELU2>(
                        x, w, scales, biases, tiles, y, N, K, Ws,
                        threadgroup_position_in_grid,
                        simdgroup_index_in_threadgroup,
                        thread_index_in_simdgroup);
                }
            """,
            header: nemotronRunGatherHeader)
    }
}

/// Same rows as `lin(x, idx, sortedIndices: true)` with one threadgroup per
/// (N block, expert-run tile). Only where stock GatherQMM takes the
/// sorted-rhs route (M == 1, B >= 16, B / E >= 4) and gather_qmm_rhs picks
/// affine_gather_qmm_rhs_nax (NAX available, half-precision activations)
/// using the same per-row K fragment order; nil = use the stock op. ReLU2 is BF16-only and retains
/// the native Maximum promotion rule before selecting the fused store.
func nemotronRunGather(
    _ x: MLXArray, _ lin: QuantizedSwitchLinear, tiles: NemotronHRunTiles,
    relu2: Bool = false
) -> MLXArray? {
    guard NemotronHPrefillX5.runGather, lin.mode == .affine, lin.bits == 4, lin.groupSize == 64,
          x.ndim == 3, x.dim(1) == 1, x.dtype == .bfloat16 || x.dtype == .float16,
          !relu2 || (x.dtype == .bfloat16 && MLX.maximum(x, MLXArray(0)).dtype == x.dtype)
    else { return nil }
    let ops = lin.nemotronGatherOperands(for: x.dtype)
    guard !ops.hasLinearBias, let biases = ops.biases, ops.weight.dtype == .uint32,
          ops.weight.ndim == 3, ops.scales.dtype == x.dtype, biases.dtype == x.dtype
    else { return nil }
    let n = x.dim(0), k = x.dim(-1), experts = ops.weight.dim(0), nOut = ops.weight.dim(1)
    guard n == tiles.rows, n >= 16, n / experts >= 4,
          tiles.bm == nemotronGatherTileRows(rows: n, experts: experts),
          ops.weight.dim(2) * 8 == k, k % 64 == 0,
          ops.scales.shape == [experts, nOut, k / 64], biases.shape == ops.scales.shape
    else { return nil }
    return NemotronHRunGatherKernel.shared.kernel(
        [x, ops.weight, ops.scales, biases, tiles.tiles, nOut, k],
        template: [("T", x.dtype), ("GS", 64), ("BITS", 4), ("BM", tiles.bm), ("BN", 64), ("BK", 64),
                   ("ALIGN_N", nOut % 64 == 0), ("ALIGN_K", k % 64 == 0), ("RELU2", relu2)],
        grid: (((nOut + 63) / 64) * 128, tiles.count, 1), threadGroup: (128, 1, 1),
        outputShapes: [[n, 1, nOut]], outputDTypes: [x.dtype])[0]
}

/// Seed-only (request prefill, `NemotronHPrefillX5.active`) routed fc2: the
/// run gather reads the layer's zero-skip decode copy (`NemotronZSkip.Copy`,
/// tile-major codes with the pair index and FP32 pair table) instead of the
/// original bank. Each loader thread takes one code word lane (four outputs)
/// and eight k, rebuilds each k pair's bank byte from the two nibbles and
/// applies the stock `dequantize` statements to the pair's scale and bias (the
/// table holds their exact FP32 widenings), so the threadgroup weight block,
/// A loads, K loop and NAX MMA sequence are the run gather's. The seed then
/// needs the copies, not the original fc2 banks, which the decode residency
/// therefore requests in their place (`NemotronDecodeResidency.targetGroups`).
/// `qualify` compares it with `nemotronRunGather` on the bank bit for bit on
/// the target's MoE fc2 with seeded inputs and routes.
/// NEMOTRON_PREFILL_FC2_COPY=0 keeps the bank.
enum NemotronPrefillFC2Copy {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_FC2_COPY"] != "0"
    nonisolated(unsafe) static var verified = false
    /// Columns per threadgroup (each of the two SIMD groups across N takes half).
    static let columns = Int(ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_FC2_COPY_BN"] ?? "") ?? 64

    static let header: String = nemotronRunGatherHeader + "\n" + #"""
// Nemotron: the run gather reading the routed fc2 tile-major decode copy
// (codes, pair index and pair table) instead of the bank. tiles[t] = (first sorted row, rows <= BM, expert). Every
// tile holds exactly one expert segment, so the stock segment loop collapses
// to one pass; the loader, K loop and NAX fragment MMA sequence are stock.
// The optional typed ReLU2 follows the native output cast. Row bounds stay
// m_lo_lim = 0, m_hi_lim from the tile's row count.
template <
    typename T,
    int group_size,
    int bits,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool align_N,
    bool align_K,
    bool RELU2>
METAL_FUNC void nemotron_gather_qmm_runs_copy_impl(
    const device T* x,
    const device uint16_t* wk,
    const device uint16_t* sidx,
    const device float* table,
    const device int* tiles,
    device T* y,
    const int N,
    const int K,
    threadgroup T* Ws,
    uint3 tid,
    uint simd_group_id,
    uint simd_lane_id) {
  const int tile_rows = tiles[tid.y * 3 + 1];
  if (tile_rows <= 0) {
    return;
  }
  const int row0 = tiles[tid.y * 3 + 0];
  const uint32_t index = uint32_t(tiles[tid.y * 3 + 2]);

  constexpr bool transpose = true;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using loader_w_t = QuantizedBlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      transpose,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int K_it = K / BK;
  const size_t stride_w = size_t(N) * K_w;
  const size_t stride_s = size_t(N) * K_g;
  const int y_col = tid.x * BN;
  const size_t y_col_long = size_t(y_col);

  const short tgp_bm = short(min(BM, tile_rows));
  const short tgp_bn = align_N ? BN : short(min(BN, N - y_col));

  const int k_remain = K - K_it * BK;
  const short2 tile_w = short2(k_remain, tgp_bn);

  x += size_t(row0) * K;
  y += size_t(row0) * N + y_col_long;

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);

  const short sgp_sm = min(SM, short(max(0, int(tgp_bm) - tm)));
  const short sgp_sn =
      align_N ? SN : min(SN, short(max(0, (N - (y_col + tn)))));

  const bool is_unaligned_sm = (sgp_sm != SM);
  const bool is_unaligned_bn = align_N ? false : (tgp_bn != BN);

  constexpr short BR = TN;
  constexpr short BC = TK;

  using AccumType = float;

  threadgroup_barrier(mem_flags::mem_none);

  const short m_lo_lim = 0;
  const short m_hi_lim = min(int(sgp_sm), max(0, int(tgp_bm) - tm));
  const bool sg_active = m_hi_lim > m_lo_lim;

  NAXTile<AccumType, TM, TN> Dtile;
  Dtile.clear();

  const device T* xn = x + tm * K;

  // The weight block from the tile-major decode copy: thread t holds four
  // consecutive outputs (one code word lane) and CL_KPT consecutive k; each
  // k pair is put back into the bank's byte and dequantized with the stock
  // `dequantize` statements, so every Ws value is the stock loader's.
  constexpr int CL_KPT = BK * (BN / 4) / (WM * WN * SIMD_SIZE);
  const int cl_t = int(simd_group_id) * SIMD_SIZE + int(simd_lane_id);
  const int cl_m = cl_t % (BN / 4);
  const int cl_c = cl_t / (BN / 4);
  const int cl_n = y_col + 4 * cl_m;
  const device uint16_t* cl_codes = wk
      + ((size_t(index) * size_t(N / 128) + size_t(cl_n / 128)) * size_t(K) + size_t(cl_c * CL_KPT)) * 32
      + (cl_n % 128) / 4;
  const device uint16_t* cl_idx = sidx
      + (size_t(index) * size_t(N / 128) + size_t(cl_n / 128)) * size_t(K / group_size) * 128 + (cl_n % 128);
  threadgroup T* cl_dst = Ws + (4 * cl_m) * BK_padded + cl_c * CL_KPT;
  auto cl_load = [&]() {
    STEEL_PRAGMA_UNROLL
    for (short o = 0; o < 4; o++) {
      const uint pair = uint(cl_idx[o]);
      const float s = table[2 * pair];
      const float b = table[2 * pair + 1];
      float sc[2] = {s, s / 16.0f};
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < CL_KPT; j += 2) {
        const uint lo = (uint(cl_codes[j * 32]) >> (4 * o)) & 0xfu;
        const uint hi = (uint(cl_codes[(j + 1) * 32]) >> (4 * o)) & 0xfu;
        const uint8_t byte = uint8_t(lo | (hi << 4));
        cl_dst[o * BK_padded + j] = static_cast<T>(sc[0] * (byte & 0x0f) + b);
        cl_dst[o * BK_padded + j + 1] = static_cast<T>(sc[1] * (byte & 0xf0) + b);
      }
    }
  };

  dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
    dispatch_bool(align_N || !is_unaligned_bn, [&](auto kAlignedN) {
      for (int k = 0; k < K_it; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        cl_load();

        threadgroup_barrier(mem_flags::mem_threadgroup);

        STEEL_PRAGMA_NO_UNROLL
        for (int kk1 = 0; kk1 < BK; kk1 += SK) {
          if (sg_active) {
            NAXTile<T, TM, TK> Atile;
            NAXTile<T, BR, BC> Btile;

            volatile int compiler_barrier;

            if constexpr (kAlignedM.value) {
              Atile.load(xn + kk1, K);
            } else {
              Atile.load_safe(xn + kk1, K, short2(SK, sgp_sm));
            }

            Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);

            tile_matmad_nax(
                Dtile,
                Atile,
                metal::bool_constant<false>{},
                Btile,
                metal::bool_constant<transpose>{});

            (void)compiler_barrier;
          }
        }

        xn += BK;
        cl_codes += BK * 32;
        cl_idx += 128;
      }

      if (!align_K) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        cl_load();
        threadgroup_barrier(mem_flags::mem_threadgroup);

        STEEL_PRAGMA_NO_UNROLL
        for (int kk1 = 0; kk1 < BK; kk1 += SK) {
          if (sg_active) {
            NAXTile<T, TM, TK> Atile;
            NAXTile<T, BR, BC> Btile;

            volatile int compiler_barrier;

            const short psk = min(int(SK), max(0, (BK - kk1)));
            Atile.load_safe(xn + kk1, K, short2(psk, sgp_sm));

            Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);

            tile_matmad_nax(
                Dtile,
                Atile,
                metal::bool_constant<false>{},
                Btile,
                metal::bool_constant<transpose>{});

            (void)compiler_barrier;
          }
        }
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      if constexpr (kAlignedN.value) {
        if (m_lo_lim == 0 && m_hi_lim == SM) {
          Dtile.template store<RELU2>(y + tm * N + tn, N);
        } else {
          Dtile.template store_slice<RELU2>(
              y + tm * N + tn, N, short2(0, m_lo_lim), short2(SN, m_hi_lim));
        }
      } else {
        Dtile.template store_slice<RELU2>(
            y + tm * N + tn,
            N,
            short2(0, m_lo_lim),
            short2(sgp_sn, m_hi_lim));
      }
    });
  });
}
"""#

    private static let kernel = MLXFast.metalKernel(
        name: "nemotron_gather_qmm_runs_fc2_copy",
        inputNames: ["x", "wk", "sidx", "table", "tiles", "N", "K"],
        outputNames: ["y"],
        source: """
            threadgroup T Ws[BN * (BK + 16 / sizeof(T))];
            if (BM == 64 && tiles[threadgroup_position_in_grid.y * 3 + 1] <= 32) {
                nemotron_gather_qmm_runs_copy_impl<T, GS, BITS, 32, BN, BK, 2, 2, ALIGN_N, ALIGN_K, RELU2>(
                    x, wk, sidx, table, tiles, y, N, K, Ws,
                    threadgroup_position_in_grid,
                    simdgroup_index_in_threadgroup,
                    thread_index_in_simdgroup);
            } else {
                nemotron_gather_qmm_runs_copy_impl<T, GS, BITS, BM, BN, BK, 2, 2, ALIGN_N, ALIGN_K, RELU2>(
                    x, wk, sidx, table, tiles, y, N, K, Ws,
                    threadgroup_position_in_grid,
                    simdgroup_index_in_threadgroup,
                    thread_index_in_simdgroup);
            }
        """,
        header: header)

    static func apply(_ x: MLXArray, _ lin: QuantizedSwitchLinear, tiles: NemotronHRunTiles) -> MLXArray? {
        guard verified, NemotronHPrefillX5.active else { return nil }
        return NemotronSeedRetile.fc2Copy(x, lin, tiles: tiles) ?? launch(x, lin, tiles: tiles)
    }

    static func launch(_ x: MLXArray, _ lin: QuantizedSwitchLinear, tiles: NemotronHRunTiles) -> MLXArray? {
        guard NemotronHPrefillX5.runGather, lin.mode == .affine, lin.bits == 4, lin.groupSize == 64,
              x.ndim == 3, x.dim(1) == 1, x.dtype == .bfloat16,
              let copy = NemotronZSkip.copy(for: lin), copy.dict
        else { return nil }
        let ops = lin.nemotronGatherOperands(for: x.dtype)
        let n = x.dim(0), k = x.dim(-1), experts = ops.weight.dim(0), nOut = ops.weight.dim(1)
        guard n == tiles.rows, n >= 16, n / experts >= 4,
              tiles.bm == nemotronGatherTileRows(rows: n, experts: experts),
              ops.weight.dim(2) * 8 == k, k % 64 == 0, nOut % 128 == 0,
              copy.codes.shape == [experts, nOut / 128, k, 32], copy.codes.dtype == .uint16,
              copy.scales.shape == [experts, nOut / 128, k / 64, 128], copy.scales.dtype == .uint16,
              copy.biases.dtype == .float32
        else { return nil }
        let bn = Self.columns
        return kernel(
            [x, copy.codes, copy.scales, copy.biases, tiles.tiles, nOut, k],
            template: [("T", x.dtype), ("GS", 64), ("BITS", 4), ("BM", tiles.bm), ("BN", bn), ("BK", 64),
                       ("ALIGN_N", true), ("ALIGN_K", true), ("RELU2", false)],
            grid: (((nOut + bn - 1) / bn) * 128, tiles.count, 1), threadGroup: (128, 1, 1),
            outputShapes: [[n, 1, nOut]], outputDTypes: [x.dtype])[0]
    }

    static func qualify(_ target: Module) {
        verified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron prefill fc2 copy: \(message)\n".utf8))
        }
        guard enabled, NemotronHPrefillX5.runGather, NemotronPrefillCountingSort.verified else {
            return report("off (setting)")
        }
        let moes = target.modules().compactMap { $0 as? NemotronHMoE }
        guard let moe = moes.first,
              let fc2 = moe.switchMLP.fc2 as? QuantizedSwitchLinear
        else { return report("off (no quantized MoE fc2)") }
        guard moes.allSatisfy({ m in
            (m.switchMLP.fc2 as? QuantizedSwitchLinear).flatMap { NemotronZSkip.copy(for: $0) }?.dict == true
        }) else { return report("off (not every MoE fc2 has a dictionary copy)") }
        let experts = moe.switchMLP.numExperts, k = moe.numExpertsPerTok
        let width = fc2.nemotronGatherOperands(for: .bfloat16).weight.dim(2) * 8
        var checked = 0
        do {
            let matches = try withError { () -> Bool in
                for (index, (tokens, skew)) in [(488, Float(2)), (488, 0), (488, 9), (300, 4), (97, 1)].enumerated() {
                    let seed = UInt64(0xfc2c_0000 + index)
                    let logits = MLXRandom.normal([tokens, experts], key: MLXRandom.key(seed))
                        + skew * MLXRandom.normal([1, experts], key: MLXRandom.key(seed + 1))
                    let flat = argSort(-logits, axis: -1)[0..., ..<k].asType(.uint32).flattened()
                    guard let sorted = nemotronCountingSortChunked(flat, experts: experts, k: k) else {
                        return false
                    }
                    var act = MLXRandom.normal([tokens * k, 1, width], scale: 2, key: MLXRandom.key(seed + 2))
                    if index == 2 {
                        // ReLU2-like rows: mostly exact zeros.
                        act = MLX.maximum(act - 1.5, MLXArray(Float(0)))
                        act = act * act
                    }
                    let input = act.asType(.bfloat16)
                    for m in (index == 0 ? moes : [moe]) {
                        guard let lin = m.switchMLP.fc2 as? QuantizedSwitchLinear,
                              let stock = nemotronRunGather(input, lin, tiles: sorted.tiles),
                              let fused = launch(input, lin, tiles: sorted.tiles)
                        else { return false }
                        let same = (fused.view(dtype: .uint16) .== stock.view(dtype: .uint16)).all()
                        if !same.item(Bool.self) { return false }
                    }
                    checked += 1
                }
                return true
            }
            verified = matches
        } catch {
            return report("off (\(error))")
        }
        report(verified ? "on (\(checked) cases bitwise equal, every MoE layer in the first)"
                        : "off (outputs differ from the bank run gather after \(checked) cases)")
    }
}

/// Seed-only (request prefill, `NemotronHPrefillX5.active`) dense projections
/// through their pair dictionaries: each Mamba layer's input and output
/// projections and each MoE layer's shared-expert up projection (with its
/// typed ReLU2 store) run as a one-"expert" NAX run gather whose weight loader
/// reads each group's (scale, bias) through the tensor's `NemotronSBDict`
/// pair dictionary (uint16 index in the scales' layout, FP32 pair table of
/// exact BF16 widenings), the loader the fc1 dictionary gather uses
/// (`QuantizedBlockLoaderDict`). The run gather's tiles, K loop and NAX MMA
/// sequence are unchanged, so each output is the run gather's. The verify
/// already reads these dictionaries, so with a kind admitted nothing in the
/// request reads that kind's BF16 scales and biases, and the decode residency
/// maps the dictionaries in their place (`NemotronDecodeResidency.targetGroups`).
/// `qualify` compares each kind with the stock projection (`relu2(up(x))` for
/// the shared up) bit for bit on every layer's tensor with seeded rows; each
/// kind is admitted on its own. NEMOTRON_PREFILL_DENSE_DICT=0 keeps the stock launches.
enum NemotronPrefillDenseDict {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_DENSE_DICT"] != "0"
    /// The admitted kinds: "in_proj", "out_proj" (Mamba mixers), "up_proj" (shared experts, with ReLU2).
    nonisolated(unsafe) static var kinds: Set<String> = []

    private static let header: String? = {
        var h = nemotronRunGatherHeader
        let edits: [(String, String)] = [
            ("// Nemotron: affine_gather_qmm_rhs_nax (transpose = true) with tiles aligned\n",
             nemotronQuantizedBlockLoaderDictSource
                + "\n// Nemotron: affine_gather_qmm_rhs_nax (transpose = true) with tiles aligned\n"),
            ("METAL_FUNC void nemotron_gather_qmm_runs_impl(", "METAL_FUNC void nemotron_gather_qmm_runs_dict_impl("),
            ("    const device T* scales,\n    const device T* biases,\n    const device int* tiles,\n",
             "    const device uint16_t* scales,\n    const device float* biases,\n    const device int* tiles,\n"),
            ("  using loader_w_t = QuantizedBlockLoader<", "  using loader_w_t = QuantizedBlockLoaderDict<"),
            ("  biases += y_col_long * K_g;\n", ""),
            ("      biases + index * stride_s,\n", "      biases,\n"),
        ]
        for (old, new) in edits {
            guard h.components(separatedBy: old).count == 2 else { return nil }
            h = h.replacingOccurrences(of: old, with: new)
        }
        return h
    }()

    private static let kernel = header.map { header in
        MLXFast.metalKernel(
            name: "nemotron_gather_qmm_runs_dense_dict",
            inputNames: ["x", "w", "scales", "biases", "tiles", "N", "K"],
            outputNames: ["y"],
            source: """
                threadgroup T Ws[BN * (BK + 16 / sizeof(T))];
                nemotron_gather_qmm_runs_dict_impl<T, GS, BITS, BM, BN, BK, 2, 2, ALIGN_N, ALIGN_K, RELU2>(
                    x, w, scales, biases, tiles, y, N, K, Ws,
                    threadgroup_position_in_grid,
                    simdgroup_index_in_threadgroup,
                    thread_index_in_simdgroup);
            """,
            header: header)
    }

    static func apply(_ x: MLXArray, _ linear: Linear, kind: String, relu2: Bool = false) -> MLXArray? {
        guard NemotronHPrefillX5.active, !kinds.isEmpty, kinds.contains(kind) else { return nil }
        return launch(x, linear, relu2: relu2)
    }

    private static func launch(_ x: MLXArray, _ linear: Linear, relu2: Bool) -> MLXArray? {
        guard NemotronHPrefillX5.runGather, let kernel, let q = linear as? QuantizedLinear,
              type(of: linear) == QuantizedLinear.self, q.mode == .affine, q.bits == 4, q.groupSize == 64,
              q.bias == nil, x.ndim == 3, x.dim(0) == 1, x.dim(1) > 8, x.dtype == .bfloat16,
              !relu2 || MLX.maximum(x, MLXArray(0)).dtype == x.dtype,
              q.weight.dtype == .uint32, q.weight.ndim == 2,
              let d = NemotronSBDict.dict(q), d.index.shape == q.scales.shape, d.index.dtype == .uint16,
              d.table.dtype == .float32
        else { return nil }
        let rows = x.dim(1), k = x.dim(2), nOut = q.weight.dim(0)
        guard q.weight.dim(1) * 8 == k, k % 64 == 0, q.scales.shape == [nOut, k / 64] else { return nil }
        // Row tiles of 64 over the request's tokens, at least three descriptors
        // (as `NemotronPrefillSharedUpGather`); a zero-row descriptor is skipped.
        let used = (rows + 63) / 64
        let count = max(used, 3)
        var desc = [Int32]()
        desc.reserveCapacity(count * 3)
        for t in 0 ..< count {
            desc += t < used ? [Int32(t * 64), Int32(min(64, rows - t * 64)), 0] : [0, 0, 0]
        }
        let tiles = MLXArray(desc).reshaped(count, 3)
        let y = kernel(
            [x.reshaped(rows, 1, k), q.weight.expandedDimensions(axis: 0), d.index.expandedDimensions(axis: 0),
             d.table, tiles, nOut, k],
            template: [("T", x.dtype), ("GS", 64), ("BITS", 4), ("BM", 64), ("BN", 64), ("BK", 64),
                       ("ALIGN_N", nOut % 64 == 0), ("ALIGN_K", true), ("RELU2", relu2)],
            grid: (((nOut + 63) / 64) * 128, count, 1), threadGroup: (128, 1, 1),
            outputShapes: [[rows, 1, nOut]], outputDTypes: [x.dtype])[0]
        return y.reshaped(1, rows, nOut)
    }

    /// The target's dense projections of `kind`, in model order.
    static func linears(_ target: Module, kind: String) -> [Linear] {
        target.modules().compactMap { module -> Linear? in
            switch kind {
            case "in_proj": return (module as? NemotronHMamba2Mixer)?.inProj
            case "out_proj": return (module as? NemotronHMamba2Mixer)?.outProj
            case "up_proj": return (module as? NemotronHMoE)?.sharedExperts?.upProj
            default: return nil
            }
        }
    }

    static func qualify(_ target: Module) {
        kinds = []
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron prefill dense dict: \(message)\n".utf8))
        }
        guard enabled, NemotronHPrefillX5.runGather else { return report("off (setting)") }
        guard kernel != nil else { return report("off (kernel source)") }
        var admitted: Set<String> = []
        var notes: [String] = []
        for (kindIndex, kind) in ["in_proj", "out_proj", "up_proj"].enumerated() {
            let withRelu2 = kind == "up_proj"
            let all = linears(target, kind: kind)
            guard let first = all.first,
                  all.allSatisfy({ ($0 as? QuantizedLinear).flatMap { NemotronSBDict.dict($0) } != nil })
            else { notes.append("\(kind) off (no dictionary)"); continue }
            var checked = 0
            let matches = (try? withError { () -> Bool in
                for (index, rows) in [488, 488, 300, 97, 17].enumerated() {
                    let k = first.weight.dim(1) * 8
                    var a = MLXRandom.normal([1, rows, k], scale: 2,
                                             key: MLXRandom.key(UInt64(0xdd1c_0000 + 16 * kindIndex + index)))
                    if index == 1 { a = a * 40 }
                    let x = a.asType(.bfloat16)
                    for linear in (index == 0 ? all : [first]) {
                        guard let fused = launch(x, linear, relu2: withRelu2) else { return false }
                        let stock = withRelu2 ? relu2(linear(x)) : linear(x)
                        let same = (fused.view(dtype: .uint16) .== stock.view(dtype: .uint16)).all()
                        if !same.item(Bool.self) { return false }
                    }
                    checked += 1
                }
                return true
            }) ?? false
            if matches { admitted.insert(kind) }
            notes.append("\(kind) \(matches ? "on" : "off") (\(checked) cases\(matches ? " bitwise equal, every layer in the first" : ""))")
        }
        kinds = admitted
        report(notes.joined(separator: "; "))
    }
}

/// Seed-only (request prefill, `NemotronHPrefillX5.active`) shared-expert down
/// projection: the run gather of `NemotronPrefillFC2Copy` over a single
/// "expert" (as `NemotronPrefillSharedUpGather` runs the shared up), reading the
/// layer's shared-down zero-skip decode copy (`NemotronSharedDownZSkip`:
/// tile-major codes with tile-major BF16 scales and biases) instead of the
/// dense weights. Its loader widens each group's BF16 scale and bias to FP32
/// (the values the fc2 copy's pair table holds) and otherwise runs the fc2 copy
/// loader's statements, so the threadgroup weight block, A loads, K loop and
/// NAX MMA sequence are the run gather's. The verify already reads the copy
/// (co-launched with the routed fc2), so with this on nothing in the request
/// reads the dense shared down, and the decode residency maps the copy in its
/// place (`NemotronDecodeResidency.targetGroups`). `qualify` compares it with
/// `downProj(x)` bit for bit on every MoE layer's shared down with seeded
/// relu2-like rows. NEMOTRON_PREFILL_SHARED_DOWN_COPY=0 keeps the dense weights.
enum NemotronPrefillSharedDownCopy {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_SHARED_DOWN_COPY"] != "0"
    nonisolated(unsafe) static var verified = false

    /// The fc2 copy loader with the pair lookup replaced by the copy's BF16 scale and bias.
    private static let header: String? = {
        let swaps: [(String, String)] = [
            ("METAL_FUNC void nemotron_gather_qmm_runs_copy_impl(",
             "METAL_FUNC void nemotron_gather_qmm_runs_copy_sb_impl("),
            ("    const device uint16_t* sidx,\n    const device float* table,\n",
             "    const device T* sk,\n    const device T* bk,\n"),
            ("  const device uint16_t* cl_idx = sidx\n      + (size_t(index)",
             "  const size_t cl_sb = (size_t(index)"),
            ("      const uint pair = uint(cl_idx[o]);\n      const float s = table[2 * pair];\n      const float b = table[2 * pair + 1];\n",
             "      const float s = float(cl_s[o]);\n      const float b = float(cl_b[o]);\n"),
            ("        cl_idx += 128;\n",
             "        cl_s += 128;\n        cl_b += 128;\n"),
        ]
        var source = NemotronPrefillFC2Copy.header
        for (old, new) in swaps {
            guard source.components(separatedBy: old).count == 2 else { return nil }
            source = source.replacingOccurrences(of: old, with: new)
        }
        let pointers = "* size_t(K / group_size) * 128 + (cl_n % 128);\n"
        guard source.components(separatedBy: pointers).count == 2 else { return nil }
        return source.replacingOccurrences(of: pointers, with: pointers
            + "  const device T* cl_s = sk + cl_sb;\n  const device T* cl_b = bk + cl_sb;\n")
    }()

    private static let kernel = header.map { header in
        MLXFast.metalKernel(
            name: "nemotron_gather_qmm_runs_shared_down_copy",
            inputNames: ["x", "wk", "sk", "bk", "tiles", "N", "K"],
            outputNames: ["y"],
            source: """
                threadgroup T Ws[BN * (BK + 16 / sizeof(T))];
                nemotron_gather_qmm_runs_copy_sb_impl<T, GS, BITS, BM, BN, BK, 2, 2, true, true, false>(
                    x, wk, sk, bk, tiles, y, N, K, Ws,
                    threadgroup_position_in_grid,
                    simdgroup_index_in_threadgroup,
                    thread_index_in_simdgroup);
            """,
            header: header)
    }

    static func apply(_ x: MLXArray, _ down: Linear) -> MLXArray? {
        guard verified, NemotronHPrefillX5.active else { return nil }
        return launch(x, down)
    }

    private static func launch(_ x: MLXArray, _ down: Linear) -> MLXArray? {
        guard NemotronHPrefillX5.runGather, let kernel, let q = down as? QuantizedLinear,
              type(of: down) == QuantizedLinear.self, q.mode == .affine, q.bits == 4, q.groupSize == 64,
              q.bias == nil, x.ndim == 3, x.dim(0) == 1, x.dim(1) > 8, x.dtype == .bfloat16,
              let copy = NemotronSharedDownZSkip.copyArrays(for: down)
        else { return nil }
        let rows = x.dim(1), k = x.dim(2), nOut = q.weight.dim(0)
        guard q.weight.dim(1) * 8 == k, k % 64 == 0, nOut % 128 == 0,
              copy.codes.shape == [1, nOut / 128, k, 32], copy.codes.dtype == .uint16,
              copy.scales.shape == [1, nOut / 128, k / 64, 128], copy.scales.dtype == x.dtype,
              copy.biases.shape == copy.scales.shape, copy.biases.dtype == x.dtype
        else { return nil }
        // Row tiles of 64 over the request's tokens, at least three descriptors
        // (as `NemotronPrefillSharedUpGather`); a zero-row descriptor is skipped.
        let used = (rows + 63) / 64
        let count = max(used, 3)
        var desc = [Int32]()
        desc.reserveCapacity(count * 3)
        for t in 0 ..< count {
            desc += t < used ? [Int32(t * 64), Int32(min(64, rows - t * 64)), 0] : [0, 0, 0]
        }
        let tiles = MLXArray(desc).reshaped(count, 3)
        let y = kernel(
            [x.reshaped(rows, 1, k), copy.codes, copy.scales, copy.biases, tiles, nOut, k],
            template: [("T", x.dtype), ("GS", 64), ("BITS", 4), ("BM", 64), ("BN", 64), ("BK", 64)],
            grid: ((nOut / 64) * 128, count, 1), threadGroup: (128, 1, 1),
            outputShapes: [[rows, 1, nOut]], outputDTypes: [x.dtype])[0]
        return y.reshaped(1, rows, nOut)
    }

    static func qualify(_ target: Module) {
        verified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron prefill shared down copy: \(message)\n".utf8))
        }
        guard enabled, NemotronHPrefillX5.runGather else { return report("off (setting)") }
        guard kernel != nil else { return report("off (kernel source)") }
        let downs = target.modules().compactMap { ($0 as? NemotronHMoE)?.sharedExperts?.downProj }
        guard let first = downs.first, downs.allSatisfy({ NemotronSharedDownZSkip.copyArrays(for: $0) != nil })
        else { return report("off (not every shared down has a decode copy)") }
        var checked = 0
        do {
            let matches = try withError { () -> Bool in
                for (index, rows) in [488, 488, 300, 97, 17].enumerated() {
                    let k = first.weight.dim(1) * 8
                    var a = MLXRandom.normal([1, rows, k], scale: 2, key: MLXRandom.key(UInt64(0x5d0c_0000 + index)))
                    if index != 1 {
                        // ReLU2-like rows: mostly exact zeros.
                        a = MLX.maximum(a - 1.5, MLXArray(Float(0)))
                        a = a * a
                    }
                    let x = a.asType(.bfloat16)
                    for down in (index == 0 ? downs : [first]) {
                        guard let fused = launch(x, down) else { return false }
                        let stock = down(x)
                        let same = (fused.view(dtype: .uint16) .== stock.view(dtype: .uint16)).all()
                        if !same.item(Bool.self) { return false }
                    }
                    checked += 1
                }
                return true
            }
            verified = matches
        } catch {
            return report("off (\(error))")
        }
        report(verified ? "on (\(checked) cases bitwise equal, every MoE layer in the first)"
                        : "off (outputs differ from the quantized matmul after \(checked) cases)")
    }
}

/// Seed-only (request prefill, `NemotronHPrefillX5.active`) routed fc1: the
/// run gather reads each sorted row's token row of the MoE input through the
/// counting sort's `orderK` instead of the expanded `xs` copy, so the `xs`
/// gather stays lazy and is never launched. Every A-fragment element is the
/// same BF16 value at the same fragment position (rows past the tile are
/// zero, as `load_safe` writes them); the loader, K loop, NAX MMA sequence and
/// typed ReLU2 store are the run gather's. Carried from newjordan's
/// `8007c64b` (`nemotron_load_routed`), with the adaptive 32/64-row tile
/// dispatch of the current run gather. `qualify` compares it with
/// `nemotronRunGather` on `xs` bit for bit on the target's first MoE fc1
/// with seeded inputs and routes. NEMOTRON_PREFILL_DIRECT_FC1=0 keeps `xs`.
private let nemotronQuantizedBlockLoaderDictSource = #"""
// The bank loader reading each group's (scale, bias) through the pair
// dictionary: `scales` is the uint16 pair index (the scales' layout) and
// `biases` the FP32 pair table (exact BF16 widenings), so `scale` and `bias`
// are the bank's BF16 values and the dequantized block is the stock one.
template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short group_size,
    short bits>
struct QuantizedBlockLoaderDict {
  static_assert(
      BCOLS <= group_size,
      "The group size should be larger than the columns");
  static_assert(
      group_size % BCOLS == 0,
      "The group size should be divisible by the columns");
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  MLX_MTL_CONST short pack_factor = get_pack_factor<bits, 8>();
  MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack<bits>();
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  MLX_MTL_CONST short group_steps = group_size / BCOLS;

  const int src_ld;
  const int tile_stride;
  short group_step_cnt;
  const int group_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  const device uint16_t* scales;
  const device float* biases;

  QuantizedBlockLoaderDict(
      const device uint8_t* src_,
      const device uint16_t* scales_,
      const device float* biases_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]) thread
      : src_ld(src_ld_),
        tile_stride(
            reduction_dim ? BCOLS_PACKED* bytes_per_pack
                          : BROWS * src_ld * bytes_per_pack / pack_factor),
        group_step_cnt(0),
        group_stride(BROWS* src_ld / group_size),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads* thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        scales(scales_ + bi * src_ld / group_size),
        biases(biases_) {}

  void load_unsafe() const thread {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    const uint pair = uint(*scales);
    T scale = static_cast<T>(biases[2 * pair]);
    T bias = static_cast<T>(biases[2 * pair + 1]);
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
    }
  }

  void load_safe(short2 src_tile_dim) const thread {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    if (reduction_dim == 1 && bi >= src_tile_dim.x) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    if (reduction_dim == 0 && bi >= src_tile_dim.y) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    const uint pair = uint(*scales);
    T scale = static_cast<T>(biases[2 * pair]);
    T bias = static_cast<T>(biases[2 * pair + 1]);
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          (device uint8_t*)(src + i * bytes_per_pack),
          scale,
          bias,
          dst + i * pack_factor);
    }
  }

  void next() thread {
    src += tile_stride;
    if (reduction_dim == 1) {
      if (group_steps > 1) {
        group_step_cnt++;
        if (group_step_cnt == group_steps) {
          group_step_cnt = 0;
          scales++;
        }
      } else {
        scales++;
      }
    } else {
      scales += group_stride;
    }
  }
};
"""#

enum NemotronPrefillDirectFC1 {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_DIRECT_FC1"] != "0"
    nonisolated(unsafe) static var verified = false

    /// The run gather header with a row-indexed A-tile load; nil if any edit
    /// point is not found exactly once.
    private static let header: String? = {
        var source = nemotronRunGatherHeader
        func swap(_ old: String, _ new: String) -> Bool {
            guard source.components(separatedBy: old).count == 2 else { return false }
            source = source.replacingOccurrences(of: old, with: new)
            return true
        }
        let edits: [(String, String)] = [
            ("// Nemotron: affine_gather_qmm_rhs_nax (transpose = true) with tiles aligned\n", #"""
template <typename Tile, typename T>
METAL_FUNC void nemotron_load_routed(
    thread Tile& tile, const device T* const thread* rows, const int offset,
    const short valid_rows) {
  using Frag = typename Tile::NAXFrag_t;
  const short2 sc = Frag::get_coord();
  STEEL_PRAGMA_UNROLL
  for (short mr = 0; mr < Tile::kTileRows; ++mr) {
    STEEL_PRAGMA_UNROLL
    for (short mc = 0; mc < Tile::kTileCols; ++mc) {
      thread typename Tile::frag_type& values = tile.frag_at(mr, mc);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < Frag::kElemRows; ++i) {
        const short row = mr * Frag::kFragRows + sc.y + i * Frag::kElemRowsJump;
        if (row < valid_rows) {
          const device T* source_row = rows[mr * Frag::kElemRows + i] + offset;
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < Frag::kElemCols; ++j) {
            const short col = mc * Frag::kFragCols + sc.x + j;
            values[i * Frag::kElemCols + j] = static_cast<T>(source_row[col]);
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < Frag::kElemCols; ++j) {
            values[i * Frag::kElemCols + j] = T(0);
          }
        }
      }
    }
  }
}

// Nemotron: affine_gather_qmm_rhs_nax (transpose = true) with tiles aligned

"""#),
            ("METAL_FUNC void nemotron_gather_qmm_runs_impl(",
             "METAL_FUNC void nemotron_gather_qmm_runs_direct_impl("),
            ("const device int* tiles,\n    device T* y,",
             "const device int* tiles,\n    const device uint32_t* routed_rows,\n    device T* y,"),
            ("x += size_t(row0) * K;", "// x is the unexpanded token rows; routed_rows maps sorted rows to them."),
            ("const device T* xn = x + tm * K;", #"""
const device T* xn = x;
  const short2 row_coord = BaseNAXFrag::get_coord();
  const device T* routed[TM * BaseNAXFrag::kElemRows];
  STEEL_PRAGMA_UNROLL
  for (short mr = 0; mr < TM; ++mr) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < BaseNAXFrag::kElemRows; ++i) {
      const short row = mr * BaseNAXFrag::kFragRows + row_coord.y
          + i * BaseNAXFrag::kElemRowsJump;
      routed[mr * BaseNAXFrag::kElemRows + i] = row < sgp_sm
          ? x + size_t(routed_rows[row0 + tm + row]) * K : x;
    }
  }
"""#),
            ("Atile.load(xn + kk1, K);", "nemotron_load_routed(Atile, routed, int(xn - x) + kk1, sgp_sm);"),
            ("Atile.load_safe(xn + kk1, K, short2(SK, sgp_sm));",
             "nemotron_load_routed(Atile, routed, int(xn - x) + kk1, sgp_sm);"),
            ("Atile.load_safe(xn + kk1, K, short2(psk, sgp_sm));",
             "nemotron_load_routed(Atile, routed, int(xn - x) + kk1, sgp_sm);"),
        ]
        for (old, new) in edits {
            guard swap(old, new) else { return nil }
        }
        return source
    }()

    /// `header` with the pair-dictionary loader: the scales and biases operands
    /// become the fc1 bank's `NemotronSBDict` index and FP32 pair table.
    private static let dictHeader: String? = NemotronPrefillDirectFC1.header.flatMap { source -> String? in
        var h = source
        func swap(_ old: String, _ new: String) -> Bool {
            guard h.components(separatedBy: old).count == 2 else { return false }
            h = h.replacingOccurrences(of: old, with: new)
            return true
        }
        let edits: [(String, String)] = [
            ("template <typename Tile, typename T>\nMETAL_FUNC void nemotron_load_routed(", nemotronQuantizedBlockLoaderDictSource
                + "\ntemplate <typename Tile, typename T>\nMETAL_FUNC void nemotron_load_routed("),
            ("METAL_FUNC void nemotron_gather_qmm_runs_direct_impl(",
             "METAL_FUNC void nemotron_gather_qmm_runs_direct_dict_impl("),
            ("    const device T* scales,\n    const device T* biases,\n",
             "    const device uint16_t* scales,\n    const device float* biases,\n"),
            ("  using loader_w_t = QuantizedBlockLoader<", "  using loader_w_t = QuantizedBlockLoaderDict<"),
            ("  biases += y_col_long * K_g;\n", ""),
            ("      biases + index * stride_s,\n", "      biases,\n"),
        ]
        for (old, new) in edits {
            guard swap(old, new) else { return nil }
        }
        return h
    }

    private static let dictKernel: MLXFast.MLXFastKernel? = NemotronPrefillDirectFC1.dictHeader.map { header in
        MLXFast.metalKernel(
            name: "nemotron_gather_qmm_runs_direct_input_dict",
            inputNames: ["x", "w", "scales", "biases", "tiles", "routed_rows", "N", "K"],
            outputNames: ["y"],
            source: """
                threadgroup T Ws[BN * (BK + 16 / sizeof(T))];
                if (BM == 64 && tiles[threadgroup_position_in_grid.y * 3 + 1] <= 32) {
                    nemotron_gather_qmm_runs_direct_dict_impl<T, GS, BITS, 32, BN, BK, 2, 2, ALIGN_N, ALIGN_K, RELU2>(
                        x, w, scales, biases, tiles, routed_rows, y, N, K, Ws,
                        threadgroup_position_in_grid,
                        simdgroup_index_in_threadgroup,
                        thread_index_in_simdgroup);
                } else {
                    nemotron_gather_qmm_runs_direct_dict_impl<T, GS, BITS, BM, BN, BK, 2, 2, ALIGN_N, ALIGN_K, RELU2>(
                        x, w, scales, biases, tiles, routed_rows, y, N, K, Ws,
                        threadgroup_position_in_grid,
                        simdgroup_index_in_threadgroup,
                        thread_index_in_simdgroup);
                }
            """,
            header: header)
    }
    /// Set by `qualifyDict`: the dictionary kernel is used for every MoE fc1 with a dictionary.
    nonisolated(unsafe) static var dictVerified = false

    private static let kernel: MLXFast.MLXFastKernel? = NemotronPrefillDirectFC1.header.map { header in
        MLXFast.metalKernel(
            name: "nemotron_gather_qmm_runs_direct_input",
            inputNames: ["x", "w", "scales", "biases", "tiles", "routed_rows", "N", "K"],
            outputNames: ["y"],
            source: """
                threadgroup T Ws[BN * (BK + 16 / sizeof(T))];
                if (BM == 64 && tiles[threadgroup_position_in_grid.y * 3 + 1] <= 32) {
                    nemotron_gather_qmm_runs_direct_impl<T, GS, BITS, 32, BN, BK, 2, 2, ALIGN_N, ALIGN_K, RELU2>(
                        x, w, scales, biases, tiles, routed_rows, y, N, K, Ws,
                        threadgroup_position_in_grid,
                        simdgroup_index_in_threadgroup,
                        thread_index_in_simdgroup);
                } else {
                    nemotron_gather_qmm_runs_direct_impl<T, GS, BITS, BM, BN, BK, 2, 2, ALIGN_N, ALIGN_K, RELU2>(
                        x, w, scales, biases, tiles, routed_rows, y, N, K, Ws,
                        threadgroup_position_in_grid,
                        simdgroup_index_in_threadgroup,
                        thread_index_in_simdgroup);
                }
            """,
            header: header)
    }

    /// `nemotronRunGather(xs, lin, tiles:, relu2: true)` with `xs` =
    /// `x[rows]` (x: [tokens, 1, K]); nil outside the request prefill or the
    /// run gather's shapes.
    static func apply(_ x: MLXArray, _ lin: QuantizedSwitchLinear, rows: MLXArray,
                      tiles: NemotronHRunTiles) -> MLXArray? {
        guard verified, NemotronHPrefillX5.active else { return nil }
        return NemotronSeedRetile.fc1Routed(x, lin, rows: rows, tiles: tiles, dict: dictVerified)
            ?? launch(x, lin, rows: rows, tiles: tiles, useDict: dictVerified)
    }

    static func launch(_ x: MLXArray, _ lin: QuantizedSwitchLinear, rows: MLXArray,
                       tiles: NemotronHRunTiles, useDict: Bool = false) -> MLXArray? {
        guard let kernel, NemotronHPrefillX5.runGather,
              lin.mode == .affine, lin.bits == 4, lin.groupSize == 64,
              x.ndim == 3, x.dim(1) == 1, x.dtype == .bfloat16,
              MLX.maximum(x, MLXArray(0)).dtype == x.dtype,
              rows.ndim == 1, rows.dtype == .uint32, rows.size == tiles.rows
        else { return nil }
        let ops = lin.nemotronGatherOperands(for: x.dtype)
        guard !ops.hasLinearBias, let biases = ops.biases, ops.weight.dtype == .uint32,
              ops.weight.ndim == 3, ops.scales.dtype == x.dtype, biases.dtype == x.dtype
        else { return nil }
        let n = tiles.rows, k = x.dim(-1), experts = ops.weight.dim(0), nOut = ops.weight.dim(1)
        guard n >= 16, n / experts >= 4,
              tiles.bm == nemotronGatherTileRows(rows: n, experts: experts),
              ops.weight.dim(2) * 8 == k, k % 64 == 0,
              ops.scales.shape == [experts, nOut, k / 64], biases.shape == ops.scales.shape
        else { return nil }
        if useDict, let dictKernel, let d = NemotronSBDict.dict(lin), d.index.shape == ops.scales.shape,
           d.index.dtype == .uint16, d.table.dtype == .float32 {
            return dictKernel(
                [x, ops.weight, d.index, d.table, tiles.tiles, rows, nOut, k],
                template: [("T", x.dtype), ("GS", 64), ("BITS", 4), ("BM", tiles.bm), ("BN", 64), ("BK", 64),
                           ("ALIGN_N", nOut % 64 == 0), ("ALIGN_K", k % 64 == 0), ("RELU2", true)],
                grid: (((nOut + 63) / 64) * 128, tiles.count, 1), threadGroup: (128, 1, 1),
                outputShapes: [[n, 1, nOut]], outputDTypes: [x.dtype])[0]
        }
        return kernel(
            [x, ops.weight, ops.scales, biases, tiles.tiles, rows, nOut, k],
            template: [("T", x.dtype), ("GS", 64), ("BITS", 4), ("BM", tiles.bm), ("BN", 64), ("BK", 64),
                       ("ALIGN_N", nOut % 64 == 0), ("ALIGN_K", k % 64 == 0), ("RELU2", true)],
            grid: (((nOut + 63) / 64) * 128, tiles.count, 1), threadGroup: (128, 1, 1),
            outputShapes: [[n, 1, nOut]], outputDTypes: [x.dtype])[0]
    }

    static func qualify(_ target: Module) {
        verified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron prefill direct fc1: \(message)\n".utf8))
        }
        guard enabled, NemotronHPrefillX5.runGather, NemotronPrefillCountingSort.verified else {
            return report("off (setting)")
        }
        guard kernel != nil else { return report("off (run gather source not as expected)") }
        guard let moe = target.modules().compactMap({ $0 as? NemotronHMoE }).first,
              let fc1 = moe.switchMLP.fc1 as? QuantizedSwitchLinear
        else { return report("off (no quantized MoE fc1)") }
        let experts = moe.switchMLP.numExperts, k = moe.numExpertsPerTok
        let width = moe.switchMLP.inputDims
        var checked = 0
        do {
            let matches = try withError { () -> Bool in
                for (index, (tokens, skew)) in [(488, Float(2)), (488, 0), (488, 9), (300, 4), (97, 1)].enumerated() {
                    let seed = UInt64(0xd1f1_0000 + index)
                    let logits = MLXRandom.normal([tokens, experts], key: MLXRandom.key(seed))
                        + skew * MLXRandom.normal([1, experts], key: MLXRandom.key(seed + 1))
                    let flat = argSort(-logits, axis: -1)[0..., ..<k].asType(.uint32).flattened()
                    guard let sorted = nemotronCountingSortChunked(flat, experts: experts, k: k) else {
                        return false
                    }
                    let x = MLXRandom.normal([1, tokens, width], scale: 2, key: MLXRandom.key(seed + 2))
                        .asType(.bfloat16)
                    let xs = MLX.expandedDimensions(x, axes: [-2, -3])
                        .flattened(start: 0, end: -3)[sorted.orderK]
                    guard let stock = nemotronRunGather(xs, fc1, tiles: sorted.tiles, relu2: true),
                          let direct = launch(x.reshaped(tokens, 1, width), fc1, rows: sorted.orderK,
                                              tiles: sorted.tiles)
                    else { return false }
                    let same = (direct.view(dtype: .uint16) .== stock.view(dtype: .uint16)).all()
                    if !same.item(Bool.self) { return false }
                    checked += 1
                }
                return true
            }
            verified = matches
        } catch {
            return report("off (\(error))")
        }
        report(verified ? "on (\(checked) cases bitwise equal)"
                        : "off (outputs differ from the expanded-row run gather after \(checked) cases)")
    }
}

extension NemotronPrefillDirectFC1 {
    /// NEMOTRON_PREFILL_FC1_DICT=0 keeps the fc1 bank's BF16 scales and biases.
    static let dictEnabled = ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_FC1_DICT"] != "0"

    /// Load time, after `qualify`: every MoE fc1 must have a pair dictionary; then the dictionary
    /// kernel must equal the bank kernel bit for bit on seeded routes and inputs (every MoE layer in
    /// the first case). Only then does the request prefill's fc1 read the dictionary, and the decode
    /// residency map it in place of the bank's scales and biases.
    static func qualifyDict(_ target: Module) {
        dictVerified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron prefill fc1 dict: \(message)\n".utf8))
        }
        guard dictEnabled, verified, NemotronSBDict.enabled else { return report("off (setting)") }
        let moes = target.modules().compactMap { $0 as? NemotronHMoE }
        guard dictKernel != nil else { return report("off (direct gather source not as expected)") }
        guard let moe = moes.first, moes.allSatisfy({ m in
            guard let lin = m.switchMLP.fc1 as? QuantizedSwitchLinear, let d = NemotronSBDict.dict(lin) else {
                return false
            }
            return d.index.shape == lin.nemotronGatherOperands(for: .bfloat16).scales.shape
                && d.index.dtype == .uint16 && d.table.dtype == .float32
        }) else { return report("off (not every MoE fc1 has a pair dictionary)") }
        let experts = moe.switchMLP.numExperts, k = moe.numExpertsPerTok
        let width = moe.switchMLP.inputDims
        var checked = 0
        do {
            let matches = try withError { () -> Bool in
                for (index, (tokens, skew)) in [(488, Float(2)), (488, 0), (488, 9), (300, 4), (97, 1)].enumerated() {
                    let seed = UInt64(0xfc1d_0000 + index)
                    let logits = MLXRandom.normal([tokens, experts], key: MLXRandom.key(seed))
                        + skew * MLXRandom.normal([1, experts], key: MLXRandom.key(seed + 1))
                    let flat = argSort(-logits, axis: -1)[0..., ..<k].asType(.uint32).flattened()
                    guard let sorted = nemotronCountingSortChunked(flat, experts: experts, k: k) else {
                        return false
                    }
                    let x = MLXRandom.normal([1, tokens, width], scale: 2, key: MLXRandom.key(seed + 2))
                        .asType(.bfloat16).reshaped(tokens, 1, width)
                    for m in (index == 0 ? moes : [moe]) {
                        guard let lin = m.switchMLP.fc1 as? QuantizedSwitchLinear,
                              let bank = launch(x, lin, rows: sorted.orderK, tiles: sorted.tiles),
                              let dict = launch(x, lin, rows: sorted.orderK, tiles: sorted.tiles, useDict: true)
                        else { return false }
                        let same = (dict.view(dtype: .uint16) .== bank.view(dtype: .uint16)).all()
                        if !same.item(Bool.self) { return false }
                    }
                    checked += 1
                }
                return true
            }
            dictVerified = matches
        } catch {
            return report("off (\(error))")
        }
        report(dictVerified ? "on (\(checked) cases bitwise equal, every MoE layer in the first)"
                            : "off (outputs differ from the bank kernel after \(checked) cases)")
    }
}

/// Seed-only (request prefill, `NemotronHPrefillX5.active`): the shared
/// expert's up projection and relu2 as one run-aligned NAX gather launch
/// (`NemotronHRunGatherKernel`, typed ReLU2 store) over a single "expert" (the
/// shared up weights) whose row tiles cover the request's tokens, instead of
/// the quantized matmul and the relu2 launch. `qualify` compares it with
/// `relu2(upProj(x))` bit for bit on the target's shared up projection.
/// NEMOTRON_PREFILL_SHARED_UP_GATHER=0 keeps the two launches.
enum NemotronPrefillSharedUpGather {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_SHARED_UP_GATHER"] != "0"
    nonisolated(unsafe) static var verified = false

    static func apply(_ x: MLXArray, _ up: Linear) -> MLXArray? {
        guard verified, NemotronHPrefillX5.active else { return nil }
        return launch(x, up)
    }

    private static func launch(_ x: MLXArray, _ up: Linear) -> MLXArray? {
        guard NemotronHPrefillX5.runGather, let q = up as? QuantizedLinear, type(of: up) == QuantizedLinear.self,
              q.mode == .affine, q.bits == 4, q.groupSize == 64, q.bias == nil, let biases = q.biases,
              x.ndim == 3, x.dim(0) == 1, x.dim(1) > 8, x.dtype == .bfloat16,
              MLX.maximum(x, MLXArray(0)).dtype == x.dtype,
              q.weight.dtype == .uint32, q.weight.ndim == 2, q.scales.dtype == x.dtype, biases.dtype == x.dtype
        else { return nil }
        let rows = x.dim(1), k = x.dim(2), nOut = q.weight.dim(0)
        guard q.weight.dim(1) * 8 == k, k % 64 == 0, q.scales.shape == [nOut, k / 64],
              biases.shape == q.scales.shape
        else { return nil }
        // At least three descriptors (>= 8 elements, so `tiles` stays a device
        // buffer); a descriptor with zero rows is skipped by the kernel.
        let used = (rows + 63) / 64
        let count = max(used, 3)
        var desc = [Int32]()
        desc.reserveCapacity(count * 3)
        for t in 0 ..< count {
            desc += t < used ? [Int32(t * 64), Int32(min(64, rows - t * 64)), 0] : [0, 0, 0]
        }
        let tiles = MLXArray(desc).reshaped(count, 3)
        let y = NemotronHRunGatherKernel.shared.kernel(
            [x.reshaped(rows, 1, k), q.weight.expandedDimensions(axis: 0), q.scales.expandedDimensions(axis: 0),
             biases.expandedDimensions(axis: 0), tiles, nOut, k],
            template: [("T", x.dtype), ("GS", 64), ("BITS", 4), ("BM", 64), ("BN", 64), ("BK", 64),
                       ("ALIGN_N", nOut % 64 == 0), ("ALIGN_K", true), ("RELU2", true)],
            grid: (((nOut + 63) / 64) * 128, count, 1), threadGroup: (128, 1, 1),
            outputShapes: [[rows, 1, nOut]], outputDTypes: [x.dtype])[0]
        return y.reshaped(1, rows, nOut)
    }

    static func qualify(_ target: Module) {
        verified = false
        func report(_ message: String) {
            FileHandle.standardError.write(Data("nemotron prefill shared up gather: \(message)\n".utf8))
        }
        guard enabled, NemotronHPrefillX5.runGather else { return report("off (setting)") }
        guard let moe = target.modules().compactMap({ $0 as? NemotronHMoE }).first,
              let mlp = moe.sharedExperts
        else { return report("off (no shared expert)") }
        var checked = 0
        do {
            let matches = try withError { () -> Bool in
                for (index, rows) in [488, 300, 97, 64, 9 + 8].enumerated() {
                    let x = MLXRandom.normal([1, rows, moe.switchMLP.inputDims], scale: 2,
                                             key: MLXRandom.key(UInt64(0x5a6e_0000 + index))).asType(.bfloat16)
                    guard let fused = launch(x, mlp.upProj) else { return false }
                    let stock = relu2(mlp.upProj(x))
                    let same = (fused.view(dtype: .uint16) .== stock.view(dtype: .uint16)).all()
                    if !same.item(Bool.self) { return false }
                    checked += 1
                }
                return true
            }
            verified = matches
        } catch {
            return report("off (\(error))")
        }
        report(verified ? "on (\(checked) cases bitwise equal)"
                        : "off (outputs differ from the quantized matmul and relu2 after \(checked) cases)")
    }
}

// MARK: - MoE

class NemotronHMoE: Module, UnaryLayer, NemotronHMixer {
    let numExpertsPerTok: Int
    let hasSharedExperts: Bool
    let mtpCompiledRowsCache = NemotronH35MTPGraphCache()

    @ModuleInfo(key: "gate") var gate: NemotronHMoEGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: NemotronHSwitchMLP
    @ModuleInfo(key: "shared_experts") var sharedExperts: NemotronHMLP?

    init(_ args: NemotronHConfiguration) {
        self.numExpertsPerTok = args.numExpertsPerTok
        self.hasSharedExperts = args.nSharedExperts != nil && args.nSharedExperts! > 0

        self._gate.wrappedValue = NemotronHMoEGate(args)
        self._switchMLP.wrappedValue = NemotronHSwitchMLP(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.nRoutedExperts
        )

        if hasSharedExperts {
            self._sharedExperts.wrappedValue = NemotronHMLP(
                args,
                intermediateSize: args.moeSharedExpertIntermediateSize
            )
        }

        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        cbv2Forward(x, join: nil).out
    }

    /// `callAsFunction`; with `join` (the CBv2 block's residual input and the
    /// next block's norm) inside the request prefill, the combine launch also
    /// applies the block's residual add and that norm
    /// (`NemotronPrefillMoEJoin`): `out` is then the residual and
    /// `normalized` the norm's output.
    func cbv2Forward(_ x: MLXArray, join: (residual: MLXArray, norm: RMSNorm)?)
        -> (out: MLXArray, normalized: MLXArray?)
    {
        let (inds, scores) = gate(x)
        // Request prefill: the shared-expert add rides the combine launch.
        let shared: MLXArray? = NemotronPrefillCombineAdd.verified && NemotronHPrefillX5.active
            ? sharedExperts.map { $0(x) } : nil
        var y: MLXArray
        var added = false
        if let fused = switchMLP.fusedSortedCombine(x, inds, scores, addend: shared, join: join) {
            if let normalized = fused.normalized { return (fused.out, normalized) }
            y = fused.out
            added = fused.added
        } else {
            y = switchMLP(x, inds)
            y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)
        }

        if let sharedExperts, !added {
            y = y + (shared ?? sharedExperts(x))
        }

        return (y, nil)
    }

    // Protocol conformance
    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        callAsFunction(x)
    }
}

// MARK: - Decoder Block

private class NemotronHBlock: Module {
    let blockType: NemotronHBlockType

    @ModuleInfo(key: "norm") var norm: RMSNorm

    // Single mixer property with base Module type - concrete type assigned at init
    // This pattern is used by Jamba for feedForward: Module
    @ModuleInfo(key: "mixer") var mixer: Module

    init(_ args: NemotronHConfiguration, blockType: Character) {
        self.blockType = NemotronHBlockType(from: blockType)

        self._norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)

        // Assign the appropriate concrete mixer type
        switch self.blockType {
        case .mamba:
            _mixer.wrappedValue = NemotronHMamba2Mixer(args)
        case .attention:
            _mixer.wrappedValue = NemotronHAttention(args)
        case .mlp:
            _mixer.wrappedValue = NemotronHMLP(args)
        case .moe:
            _mixer.wrappedValue = NemotronHMoE(args)
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let hidden = norm(x)

        // Cast to protocol and call - all mixer types conform to NemotronHMixer
        let mixerFunc = mixer as! NemotronHMixer
        let output = mixerFunc(hidden, attentionMask: attentionMask, ssmMask: ssmMask, cache: cache)

        return x + output
    }

    func cbv2Forward(
        _ x: MLXArray,
        modelLayerIndex: Int,
        attentionCache: (any CBv2AttendingLayerCache)?,
        recurrentState: [CBv2RecurrentStateEvaluation],
        captureMTP: Bool = false,
        preparedNorm: MLXArray? = nil,
        preparedActivation: MLXArray? = nil,
        nextNorm: RMSNorm? = nil,
        nextPreparedMoE: Bool = false
    ) -> (MLXArray, MLXArray?, MLXArray?) {
        let hidden = preparedNorm
            ?? (captureMTP ? nemotronMTPNormRows(x) { norm($0) } : norm(x))
        let output: MLXArray
        switch blockType {
        case .mamba:
            precondition(
                attentionCache == nil,
                "NemotronH recurrent layer received attention KV")
            let mamba = mixer as! NemotronHMamba2Mixer
            output = captureMTP
                ? mamba.cbv2ForwardCaptured(hidden, modelLayerIndex: modelLayerIndex, recurrentState: recurrentState)
                : mamba.cbv2Forward(hidden, modelLayerIndex: modelLayerIndex, recurrentState: recurrentState)
        case .attention:
            guard let attentionCache else {
                preconditionFailure("NemotronH attention layer is missing CBv2 cache")
            }
            output = (mixer as! NemotronHAttention).cbv2Forward(
                hidden, cache: attentionCache, captureMTP: captureMTP, targetQKV: true)
        case .mlp:
            precondition(
                attentionCache == nil,
                "NemotronH MLP layer received attention KV")
            let mlp = mixer as! NemotronHMLP
            output = captureMTP ? mlp.mtpForwardRows(hidden) : mlp(hidden)
        case .moe:
            precondition(
                attentionCache == nil,
                "NemotronH MoE layer received attention KV")
            let moe = mixer as! NemotronHMoE
            if captureMTP {
                output = moe.mtpForwardRows(hidden, prepared: preparedActivation)
            } else {
                // Request prefill: the combine launch also takes this block's
                // join with the next norm (`NemotronPrefillMoEJoin`).
                let result = moe.cbv2Forward(hidden, join: nextNorm.map { (residual: x, norm: $0) })
                if let normalized = result.normalized { return (result.out, normalized, nil) }
                output = result.out
            }
        }
        if nextPreparedMoE, let nextNorm,
           let triple = NemotronHResidualRMSNorm.applyPrepared(x, output, norm: nextNorm) {
            return (triple.0, triple.1, triple.2)
        }
        if let nextNorm, let pair = NemotronHResidualRMSNorm.apply(x, output, norm: nextNorm) {
            return (pair.0, pair.1, nil)
        }
        return (x + output, nil, nil)
    }

    /// Captured verify block whose join with `nextNorm` (the next block's
    /// norm, or norm_f after the last block) is one launch producing the
    /// residual, the normalized rows and their prepared K=2688 activation.
    /// The mixers consume `preparedActivation` instead of re-preparing.
    func cbv2ForwardCapturedFused(
        _ x: MLXArray,
        modelLayerIndex: Int,
        attentionCache: (any CBv2AttendingLayerCache)?,
        recurrentState: [CBv2RecurrentStateEvaluation],
        preparedNorm: MLXArray?,
        preparedActivation: MLXArray?,
        nextNorm: RMSNorm
    ) -> (MLXArray, MLXArray, MLXArray?) {
        let hidden = preparedNorm ?? nemotronMTPNormRows(x) { norm($0) }
        let prepared = preparedNorm == nil ? nil : preparedActivation
        var output: MLXArray
        var shared: MLXArray? = nil
        switch blockType {
        case .mamba:
            precondition(
                attentionCache == nil,
                "NemotronH recurrent layer received attention KV")
            let mamba = mixer as! NemotronHMamba2Mixer
            if let joined = mamba.cbv2ForwardCapturedJoined(hidden,
                modelLayerIndex: modelLayerIndex, recurrentState: recurrentState,
                prepared: prepared, residual: x, nextNorm: nextNorm) {
                return (joined.0, joined.1, joined.2)
            }
            output = mamba.cbv2ForwardCaptured(
                hidden, modelLayerIndex: modelLayerIndex, recurrentState: recurrentState,
                prepared: prepared)
        case .attention:
            guard let attentionCache else {
                preconditionFailure("NemotronH attention layer is missing CBv2 cache")
            }
            output = (mixer as! NemotronHAttention).cbv2Forward(
                hidden, cache: attentionCache, captureMTP: true, targetQKV: true,
                prepared: prepared)
        case .mlp:
            precondition(
                attentionCache == nil,
                "NemotronH MLP layer received attention KV")
            output = (mixer as! NemotronHMLP).mtpForwardRows(hidden)
        case .moe:
            precondition(
                attentionCache == nil,
                "NemotronH MoE layer received attention KV")
            let moe = mixer as! NemotronHMoE
            if let joined = moe.mtpForwardRowsJoined(hidden, prepared: prepared,
                residual: x, nextNorm: nextNorm, layerIndex: modelLayerIndex) {
                return (joined.0, joined.1, joined.2)
            }
            if let parts = moe.mtpForwardRowsParts(hidden, prepared: prepared) {
                output = parts.0
                shared = parts.1
            } else {
                output = moe.mtpForwardRows(hidden, prepared: prepared)
            }
        }
        if let triple = NemotronHResidualRMSNorm.applyPreparedRows(
            x, output, shared: shared, norm: nextNorm) {
            return (triple.0, triple.1, triple.2)
        }
        // Original arithmetic: stock shared Add, residual Add, stock RMSNorm.
        if let shared { output = output + shared }
        let residual = x + output
        return (residual, nemotronMTPNormRows(residual) { nextNorm($0) }, nil)
    }
}

// MARK: - Backbone (matches Python's NemotronHModel which is stored as self.backbone)

private class NemotronHBackbone: Module {
    let args: NemotronHConfiguration

    @ModuleInfo(key: "embeddings") var embeddings: Embedding
    @ModuleInfo(key: "layers") var layers: [NemotronHBlock]
    @ModuleInfo(key: "norm_f") var normF: RMSNorm

    // Cache indices (into the cache list, not pattern indices)
    // Python: fa_idx counts Mamba layers before first Attention
    // Python: ssm_idx counts Attention layers before first Mamba
    let firstAttentionCacheIndex: Int?
    let firstMambaCacheIndex: Int?

    init(_ args: NemotronHConfiguration) {
        self.args = args
        precondition(args.vocabSize > 0)

        self._embeddings.wrappedValue = Embedding(
            embeddingCount: args.vocabSize, dimensions: args.hiddenSize)

        let pattern = Array(args.hybridOverridePattern)
        self._layers.wrappedValue = pattern.map { NemotronHBlock(args, blockType: $0) }

        self._normF.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.layerNormEpsilon)

        // Calculate cache indices (only Mamba + Attention have caches)
        // fa_idx: count Mamba layers (M) before the first Attention layer (*)
        var faIdx: Int? = nil
        var mambaCount = 0
        for char in pattern {
            if char == "*" {
                faIdx = mambaCount
                break
            } else if char == "M" {
                mambaCount += 1
            }
        }
        self.firstAttentionCacheIndex = faIdx

        // ssm_idx: count Attention layers (*) before the first Mamba layer (M)
        var ssmIdx: Int? = nil
        var attnCount = 0
        for char in pattern {
            if char == "M" {
                ssmIdx = attnCount
                break
            } else if char == "*" {
                attnCount += 1
            }
        }
        self.firstMambaCacheIndex = ssmIdx

        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hidden = embeddings(inputs)

        // Create attention mask using the first attention layer's cache
        let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode = {
            guard let cacheIdx = firstAttentionCacheIndex,
                let cache = cache,
                cacheIdx < cache.count
            else { return .none }
            return createAttentionMask(h: hidden, cache: cache[cacheIdx])
        }()

        // Create SSM mask using the first Mamba layer's cache
        // Python: ssm_mask = create_ssm_mask(hidden_states, cache[self.ssm_idx])
        let ssmMask: MLXArray? = {
            guard let cacheIdx = firstMambaCacheIndex,
                let cache = cache,
                cacheIdx < cache.count,
                let mambaCache = cache[cacheIdx] as? MambaCache
            else { return nil }
            return mambaCache.makeMask(N: hidden.dim(1))
        }()

        // Track which cache to use for each layer
        var cacheCounter = 0
        for layer in layers {
            let c: KVCache?
            if layer.blockType == .mamba || layer.blockType == .attention {
                c = cache?[cacheCounter]
                cacheCounter += 1
            } else {
                c = nil
            }

            hidden = layer(hidden, attentionMask: attentionMask, ssmMask: ssmMask, cache: c)
        }

        return normF(hidden)
    }

    func cbv2Forward(
        _ inputs: MLXArray,
        caches: [any CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation],
        captureMTP: Bool = false
    ) -> MLXArray {
        let observation = CBv2ForwardShapeObservation.isActive
            ? CBv2ForwardShapeObservation.beginTarget(liveBatchRows: inputs.dim(0), sequenceWidth: inputs.dim(1))
            : nil
        defer { observation?.end() }
        var hidden = embeddings(inputs)
        var preparedNorm: MLXArray? = nil
        var preparedActivation: MLXArray? = nil
        var attentionIndex = 0
        var submittedPrefillChunk: MLXArray? = nil
        let prefillChunks = inputs.dim(1) > 1 && !captureMTP
        let earlyEvalInterval = prefillChunks ? nemotronPrefillEvalInterval : nemotronEarlyEvalInterval
        for (modelLayerIndex, layer) in layers.enumerated() {
            let cache: (any CBv2AttendingLayerCache)?
            if layer.blockType == .attention {
                guard attentionIndex < caches.count else {
                    preconditionFailure(
                        "NemotronH CBv2 attention cache count is too small")
                }
                cache = caches[attentionIndex]
                attentionIndex += 1
            } else {
                cache = nil
            }
            let next = modelLayerIndex + 1
            // The layers that hand the fused residual+norm down are unchanged:
            // they stay tied to the default spacing, so a different eval
            // interval only adds boundaries and never alters the graph.
            // Seed request prefill: the residual + next-norm join also at the
            // early-eval boundaries (`NemotronPrefillBoundaryJoin`).
            let suppressedNorm = nemotronEarlyEvalInterval > 0 && next < layers.count
                && next % nemotronEarlyEvalInterval == 0
                && !(prefillChunks && NemotronPrefillBoundaryJoin.verified && NemotronHPrefillX5.active)
            let submitsEarly = earlyEvalInterval > 0 && next < layers.count
                && next % earlyEvalInterval == 0
            let nextNorm = next < layers.count && !suppressedNorm
                && (!captureMTP || NemotronMTPExecution.batchedNorm) ? layers[next].norm : nil
            let result = layer.cbv2Forward(
                hidden,
                modelLayerIndex: modelLayerIndex,
                attentionCache: cache,
                recurrentState: recurrentState,
                captureMTP: captureMTP,
                preparedNorm: preparedNorm,
                preparedActivation: preparedActivation,
                nextNorm: nextNorm,
                nextPreparedMoE: captureMTP && nextNorm != nil && next < layers.count
                    && layers[next].blockType == .moe)
            hidden = result.0
            preparedNorm = result.1
            preparedActivation = result.2
            // Start the GPU on the finished layers while the host builds the
            // rest of the graph. Evaluation order only; values are unchanged.
            if submitsEarly {
                if prefillChunks {
                    let group = (next - earlyEvalInterval)..<next
                    let states = recurrentState.flatMap { $0.stagedEvaluationRoots(in: group) }
                    asyncEval([hidden] + states)
                    // Every prompt prefill. A chunk runs for milliseconds while
                    // the host builds the next in well under one, so without a
                    // wait the host encodes, and allocates, every layer's
                    // intermediates before the first chunk completes. Waiting on
                    // the previous chunk returns its temporaries to MLX's buffer
                    // cache for the next chunk; this chunk is already queued, so
                    // the GPU never waits on the host. Evaluation order only;
                    // values are unchanged.
                    if let previous = submittedPrefillChunk { eval(previous) }
                    submittedPrefillChunk = hidden
                } else {
                    asyncEval(hidden)
                }
            } else if prefillChunks && next == 1 && next < layers.count
                && nemotronPrefillFirstSubmit {
                // Layer 0 alone, ahead of the first chunk. Not recorded as a
                // submitted chunk: the first chunk boundary still waits on
                // nothing and every later wait is unchanged.
                let states = recurrentState.flatMap { $0.stagedEvaluationRoots(in: 0 ..< 1) }
                asyncEval([hidden] + states)
            }
        }
        precondition(
            attentionIndex == caches.count,
            "NemotronH CBv2 attention cache count is too large")
        return captureMTP ? nemotronMTPNormRows(hidden) { normF($0) } : normF(hidden)
    }

    /// Captured verify forward ([1, 3] window) with one join launch per
    /// layer: every residual add, block norm (including the early-eval
    /// boundaries and norm_f) and K=2688 activation prepare share one kernel.
    /// The same early asyncEval layers as `cbv2Forward`; the submitted root
    /// is the join's residual output. Returns (norm_f rows, their prepared
    /// activation for the head), or nil before building anything when the
    /// window or weights are not the fused shape.
    func cbv2ForwardCapturedFused(
        _ inputs: MLXArray,
        caches: [any CBv2AttendingLayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> (hidden: MLXArray, prepared: MLXArray?)? {
        guard NemotronHResidualRMSNorm.plumbingAvailable,
              inputs.shape == [1, 3], let first = layers.first,
              type(of: first.norm) == RMSNorm.self, first.norm.weight.dtype == .bfloat16
        else { return nil }
        let observation = CBv2ForwardShapeObservation.isActive
            ? CBv2ForwardShapeObservation.beginTarget(liveBatchRows: inputs.dim(0), sequenceWidth: inputs.dim(1))
            : nil
        defer { observation?.end() }
        var hidden: MLXArray
        var preparedNorm: MLXArray?
        var preparedActivation: MLXArray?
        if let entry = NemotronHResidualRMSNorm.verifyEntry(inputs, embeddings, norm: first.norm) {
            hidden = entry.0
            preparedNorm = entry.1
            preparedActivation = entry.2
        } else {
            if let rows = NemotronHEmbeddingRows.apply(inputs, embeddings) {
                hidden = NemotronMoELeader.selfTest
                    ? nemotronBitwiseOrNaN(rows, embeddings(inputs)) : rows
            } else {
                hidden = embeddings(inputs)
            }
            let entry = NemotronHResidualRMSNorm.normPreparedRows(hidden, norm: first.norm)
            preparedNorm = entry?.0
            preparedActivation = entry?.1
        }
        var attentionIndex = 0
        for (modelLayerIndex, layer) in layers.enumerated() {
            let cache: (any CBv2AttendingLayerCache)?
            if layer.blockType == .attention {
                guard attentionIndex < caches.count else {
                    preconditionFailure(
                        "NemotronH CBv2 attention cache count is too small")
                }
                cache = caches[attentionIndex]
                attentionIndex += 1
            } else {
                cache = nil
            }
            let next = modelLayerIndex + 1
            let result = layer.cbv2ForwardCapturedFused(
                hidden,
                modelLayerIndex: modelLayerIndex,
                attentionCache: cache,
                recurrentState: recurrentState,
                preparedNorm: preparedNorm,
                preparedActivation: preparedActivation,
                nextNorm: next < layers.count ? layers[next].norm : normF)
            hidden = result.0
            preparedNorm = result.1
            preparedActivation = result.2
            // Submit after the listed layers (or on the interval) so the GPU
            // runs the verify while the host builds the rest of the chunk
            // (evaluation order only).
            let submit = nemotronVerifySubmitLayers.map { $0.contains(next) }
                ?? (nemotronVerifySubmitInterval > 0
                    && (next % nemotronVerifySubmitInterval == 0
                        || (next == 1 && nemotronVerifyFirstSubmit)))
            if submit && next < layers.count {
                asyncEval([hidden] + CBv2MTPSubmitMerge.take())
            }
        }
        precondition(
            attentionIndex == caches.count,
            "NemotronH CBv2 attention cache count is too large")
        return (preparedNorm!, preparedActivation)
    }
}

/// Holds one model's probed Mamba activation dtypes. Input-independent: the
/// probe reads loaded parameter types only, never request tokens or state.
private final class NemotronHActivationDTypeMemo: @unchecked Sendable {
    private let lock = NSLock()
    private var resolved: [Int: DType]?

    func value(_ resolve: () -> [Int: DType]) -> [Int: DType] {
        lock.lock()
        defer { lock.unlock() }
        if let resolved { return resolved }
        let value = resolve()
        resolved = value
        return value
    }
}

/// One loaded model's input-independent engine-construction value (layer
/// kinds, recurrent state geometry, probed dtypes), resolved on first use
/// after the weights are bound and kept: it reads configuration and loaded
/// parameter types only, never request tokens, bounds or state.
final class NemotronHLoadedMemo<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var resolved: Value?

    func value(_ resolve: () -> Value) -> Value {
        lock.lock()
        defer { lock.unlock() }
        if let resolved { return resolved }
        let value = resolve()
        resolved = value
        return value
    }
}

// MARK: - Main Model (matches Python's Model class)

public class NemotronHModel:
    Module, LLMModel, KVCacheDimensionProvider, LoRAModel
{
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "backbone") private var backbone: NemotronHBackbone
    let configuration: NemotronHConfiguration
    private let recurrentActivationDTypes = NemotronHActivationDTypeMemo()
    private let layerKindsMemo = NemotronHLoadedMemo<[CBv2LayerKind]>()
    private let recurrentStateSpecMemo = NemotronHLoadedMemo<CBv2RecurrentStateSpec>()

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public var loraLayers: [Module] {
        backbone.layers
    }

    public var cbv2LayerKinds: [CBv2LayerKind] {
        layerKindsMemo.value { configuration.cbv2LayerKinds }
    }

    /// Memoized like the activation dtypes it is built from: every engine
    /// and stepper adapter reads the same loaded-model geometry.
    public var cbv2RecurrentStateSpec: CBv2RecurrentStateSpec {
        recurrentStateSpecMemo.value {
            let activationDTypes = resolvedRecurrentActivationDTypes()
            return configuration.cbv2RecurrentStateSpec(
                activationDTypes: activationDTypes,
                fallbackActivationDType:
                    backbone.embeddings(MLXArray([0])).dtype)
        }
    }

    public var cbv2Capabilities: CBv2ModelCapabilities {
        configuration.cbv2Capabilities
    }

    public init(_ args: NemotronHConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabSize

        // kvHeads array: non-zero for attention layers, zero for others
        let pattern = Array(args.hybridOverridePattern)
        self.kvHeads = pattern.compactMap { char -> Int? in
            let blockType = NemotronHBlockType(from: char)
            if blockType == .mamba || blockType == .attention {
                return blockType == .attention ? args.numKeyValueHeads : 0
            }
            return nil
        }

        self._backbone.wrappedValue = NemotronHBackbone(args)

        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
        }
        Device.withDefaultDevice(.gpu) {
            nemotronWarmUpPrefillKernels(
                expertsPerToken: args.numExpertsPerTok, experts: args.nRoutedExperts,
                ssmHeads: args.mambaNumHeads, ssmGroups: args.nGroups,
                ssmHeadDim: args.mambaHeadDim, ssmStateSize: args.ssmStateSize, dtype: .bfloat16)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = backbone(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = backbone.embeddings.asLinear(out)
        }
        return out
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let pattern = Array(configuration.hybridOverridePattern)
        return pattern.compactMap { char -> KVCache? in
            let blockType = NemotronHBlockType(from: char)
            switch blockType {
            case .mamba:
                return MambaCache()
            case .attention:
                return KVCacheSimple()
            case .mlp, .moe:
                return nil  // No cache needed for MLP/MoE layers
            }
        }
    }

    /// The probe below depends only on loaded parameter types, which every
    /// reader sees after load, so it runs once per model rather than once per
    /// adapter (each engine build and each stepper).
    private func resolvedRecurrentActivationDTypes() -> [Int: DType] {
        recurrentActivationDTypes.value { probeRecurrentActivationDTypes() }
    }

    /// Resolve convolution-history widths from the loaded model itself.
    /// Quantized storage dtype is not the activation dtype, and retained
    /// fp32 parameters can promote later residuals. A lazy one-token graph
    /// through private caches exposes each Mamba layer's deterministic dtype
    /// without evaluating kernels or mutating request state.
    private func probeRecurrentActivationDTypes() -> [Int: DType] {
        let caches = newCache(parameters: nil)
        _ = backbone(
            MLXArray([0]).reshaped(1, 1),
            cache: caches)

        var result: [Int: DType] = [:]
        var cacheIndex = 0
        for (modelLayerIndex, blockType) in
            Array(configuration.hybridOverridePattern).enumerated()
        {
            let kind = NemotronHBlockType(from: blockType)
            guard kind == .mamba || kind == .attention else { continue }
            defer { cacheIndex += 1 }
            guard kind == .mamba,
                let cache = caches[cacheIndex] as? MambaCache,
                let conv = cache.state.first
            else { continue }
            result[modelLayerIndex] = conv.dtype
        }
        return result
    }

    /// Observe the loaded projections' native types using isolated lazy
    /// caches. This performs no eval and never installs probe state in a
    /// serving request. Packed integer embedding weights are not activation
    /// dtypes, and Mamba's fp32 path can promote later attention layers.
    func observedAttentionKVDTypes() -> [DType]? {
        let caches = newCache(parameters: nil)
        _ = backbone(MLXArray([0]).reshaped(1, 1), cache: caches)
        var result: [DType] = []
        var cacheIndex = 0
        for blockType in configuration.hybridOverridePattern {
            let kind = NemotronHBlockType(from: blockType)
            guard kind == .mamba || kind == .attention else { continue }
            defer { cacheIndex += 1 }
            guard kind == .attention else { continue }
            let state = caches[cacheIndex].state
            guard state.count == 2, state[0].dtype == state[1].dtype,
                [.float16, .bfloat16, .float32].contains(state[0].dtype)
            else { return nil }
            result.append(state[0].dtype)
        }
        return result.count == cbv2LayerKinds.count ? result : nil
    }

    public func newCacheV2(
        makeLayerCache: (_ layerIndex: Int, _ kind: CBv2LayerKind) throws ->
            any CBv2AttendingLayerCache
    ) rethrows -> [any CBv2AttendingLayerCache] {
        try cbv2LayerKinds.enumerated().map { storageIndex, kind in
            try makeLayerCache(kind.modelLayerIndex ?? storageIndex, kind)
        }
    }

    func cbv2Hidden(
        _ inputs: MLXArray,
        caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation],
        captureMTP: Bool = false
    ) -> MLXArray {
        let attending = caches.map { cache -> any CBv2AttendingLayerCache in
            guard let attending = cache as? any CBv2AttendingLayerCache else {
                preconditionFailure("NemotronH CBv2 target received legacy KV cache")
            }
            return attending
        }
        return backbone.cbv2Forward(
            inputs, caches: attending, recurrentState: recurrentState, captureMTP: captureMTP)
    }

    /// Captured verify only; nil (nothing built) unless the fused join applies.
    func cbv2HiddenCapturedFused(
        _ inputs: MLXArray,
        caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> (hidden: MLXArray, prepared: MLXArray?)? {
        guard NemotronHResidualRMSNorm.plumbingAvailable, inputs.shape == [1, 3] else { return nil }
        let attending = caches.map { cache -> any CBv2AttendingLayerCache in
            guard let attending = cache as? any CBv2AttendingLayerCache else {
                preconditionFailure("NemotronH CBv2 target received legacy KV cache")
            }
            return attending
        }
        return backbone.cbv2ForwardCapturedFused(
            inputs, caches: attending, recurrentState: recurrentState)
    }

    func logits(_ hidden: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hidden)
        }
        return backbone.embeddings.asLinear(hidden)
    }

    func prepareVerifyEntry() {
        guard let first = backbone.layers.first else { return }
        NemotronHResidualRMSNorm.qualifyVerifyEntry(backbone.embeddings, norm: first.norm)
    }

    func mtpEmbedding(_ tokens: MLXArray) -> MLXArray {
        guard let rows = NemotronHEmbeddingRows.apply(tokens, backbone.embeddings) else {
            return backbone.embeddings(tokens)
        }
        guard NemotronMoELeader.selfTest else { return rows }
        return nemotronBitwiseOrNaN(rows, backbone.embeddings(tokens))
    }

    func mtpShortlistLogits(_ hidden: MLXArray, ids: MLXArray) -> MLXArray {
        if let head = lmHead {
            if let quantized = head as? QuantizedLinear {
                var logits = quantizedMM(hidden, quantized.weight[ids],
                    scales: quantized.scales[ids], biases: quantized.biases.map { $0[ids] },
                    transpose: true, groupSize: quantized.groupSize, bits: quantized.bits,
                    mode: quantized.mode)
                if let bias = quantized.bias { logits = logits + bias[ids] }
                return logits
            }
            var logits = matmul(hidden, head.weight[ids].transposed(1, 0))
            if let bias = head.bias { logits = logits + bias[ids] }
            return logits
        }
        let embedding = backbone.embeddings
        if let quantized = embedding as? QuantizedEmbedding {
            return quantizedMM(hidden, quantized.weight[ids], scales: quantized.scales[ids],
                biases: quantized.biases.map { $0[ids] }, transpose: true,
                groupSize: quantized.groupSize, bits: quantized.bits, mode: quantized.mode)
        }
        return matmul(hidden, embedding.weight[ids].transposed(1, 0))
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()

        for (key, value) in weights {
            // Nemotron 3.5 target checkpoints may copy MTP declarations into
            // config.json without shipping the corresponding head. Keep the
            // serial target weight namespace explicit; a separately verified
            // MTP adapter owns mtp.* tensors when one is available.
            if key.hasPrefix("mtp.") {
                continue
            }
            var finalValue = value

            // Handle conv1d weight axis swap
            if key.contains("conv1d.weight"), value.dim(-1) != 1 {
                finalValue = value.swappedAxes(1, 2)
            }

            sanitized[key] = finalValue
        }

        // Stack experts: backbone.layers.{l}.mixer.experts.{e}.{proj}.weight -> backbone.layers.{l}.mixer.switch_mlp.{proj}.weight
        for l in 0 ..< configuration.numHiddenLayers {
            let prefix = "backbone.layers.\(l).mixer"
            for (m, n) in [("down_proj", "fc2"), ("up_proj", "fc1")] {
                if sanitized["\(prefix).experts.0.\(m).weight"] != nil {
                    let toJoin = (0 ..< configuration.nRoutedExperts).compactMap { e in
                        sanitized.removeValue(forKey: "\(prefix).experts.\(e).\(m).weight")
                    }
                    if !toJoin.isEmpty {
                        sanitized["\(prefix).switch_mlp.\(n).weight"] = MLX.stacked(toJoin)
                    }
                }
            }
        }

        return sanitized
    }

    /// Predicate for casting: some parameters should stay in float32
    public var castPredicate: ((String) -> Bool)? {
        { key in
            !key.contains("e_score_correction_bias") && !key.contains("A_log")
        }
    }
}

extension NemotronHModel: CBv2RecurrentLanguageModelForwardable {
    public func cbv2Forward(
        _ tokens: MLXArray,
        caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray {
        let hidden = cbv2Hidden(tokens, caches: caches, recurrentState: recurrentState)
        // A long single-row forward (the teacher-forced prompt prefill) is
        // only ever read at its last position. Project just the trailing
        // rows from the start of the second-to-last 64-row M tile: the
        // quantized matmul then sees M in 65...128 (> 32, so the same
        // qmm_nax bm=64 / qmm bm=32 kernel as the full M) and the last row's
        // tile holds exactly the same rows as in the full projection, so
        // those logits are bitwise identical. Shorter forwards (decode,
        // verify windows) keep every position.
        let length = hidden.dim(1)
        if hidden.dim(0) == 1, length > 128 {
            let start = ((length - 1) / 64) * 64 - 64
            return logits(hidden[0..., start..., 0...])
        }
        return logits(hidden)
    }
}

extension NemotronHModel: CBv2RecurrentLanguageModelPrefillForwardable {
    public var cbv2SupportsPackedPrefill: Bool { false }

    public func cbv2RecurrentPrefill(
        _ inputs: MLXArray,
        inputEmbedding: MLXArray?,
        cache: [KVCache]?,
        recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        precondition(inputEmbedding == nil, "NemotronH is text-only")
        precondition(positionIds == nil, "NemotronH serial parity omits RoPE positions")
        let hidden = cbv2Hidden(
            inputs, caches: cache ?? [], recurrentState: recurrentState)
        switch requirement {
        case .evaluationOnly:
            return hidden[0..., -1, 0 ..< 1]
        case .lastPositionLogits:
            return logits(hidden[0..., -1, 0...])
        }
    }
}

// MARK: - Configuration

public struct NemotronHConfiguration: Codable, Sendable {
    public var modelType: String = "nemotron_h"
    public var vocabSize: Int
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var attentionBias: Bool
    public var mambaNumHeads: Int
    public var mambaHeadDim: Int
    public var mambaProjBias: Bool
    public var ssmStateSize: Int
    public var convKernel: Int
    public var nGroups: Int
    public var intermediateSize: Int
    public var moeIntermediateSize: Int
    public var moeSharedExpertIntermediateSize: Int
    public var nRoutedExperts: Int
    public var nSharedExperts: Int?
    public var numExpertsPerTok: Int
    public var hybridOverridePattern: String
    public var layerNormEpsilon: Float
    public var mlpBias: Bool
    public var useBias: Bool
    public var useConvBias: Bool
    public var tieWordEmbeddings: Bool
    public var ropeTheta: Float
    public var headDim: Int?
    public var nGroup: Int
    public var topkGroup: Int
    public var normTopkProb: Bool
    public var routedScalingFactor: Float
    public var timeStepLimitMin: Float
    public var timeStepLimitMax: Float
    public var chunkSize: Int
    public var mambaSSMCacheDType: String
    public var numNextnPredictLayers: Int
    public var mtpLayersBlockType: [String]

    public var cbv2LayerKinds: [CBv2LayerKind] {
        Array(hybridOverridePattern).enumerated().compactMap {
            modelLayerIndex, blockType in
            guard blockType == "*" else { return nil }
            return CBv2LayerKind(
                attention: .full,
                headDim: headDim ?? (hiddenSize / numAttentionHeads),
                kvHeads: numKeyValueHeads,
                queryHeads: numAttentionHeads,
                modelLayerIndex: modelLayerIndex)
        }
    }

    public func cbv2RecurrentStateSpec(
        activationDType: DType = .bfloat16
    ) -> CBv2RecurrentStateSpec {
        cbv2RecurrentStateSpec(
            activationDTypes: [:],
            fallbackActivationDType: activationDType)
    }

    func cbv2RecurrentStateSpec(
        activationDTypes: [Int: DType],
        fallbackActivationDType: DType
    ) -> CBv2RecurrentStateSpec {
        let convDim =
            mambaNumHeads * mambaHeadDim + 2 * nGroups * ssmStateSize
        let layers = Array(hybridOverridePattern).enumerated().compactMap {
            modelLayerIndex, blockType -> CBv2RecurrentLayerStateSpec? in
            guard blockType == "M" else { return nil }
            return CBv2RecurrentLayerStateSpec(
                modelLayerIndex: modelLayerIndex,
                convShape: [1, max(0, convKernel - 1), convDim],
                convDType:
                    activationDTypes[modelLayerIndex]
                    ?? fallbackActivationDType,
                ssmShape: [1, mambaNumHeads, mambaHeadDim, ssmStateSize],
                ssmDType: .float32)
        }
        return CBv2RecurrentStateSpec(layers: layers)
    }

    public var cbv2Capabilities: CBv2ModelCapabilities {
        // Serial first: recurrent snapshots, MTP, packed prefill, paged KV,
        // and compiled decode stay fail-closed until independently qualified.
        .initialRecurrentTarget
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case attentionBias = "attention_bias"
        case mambaNumHeads = "mamba_num_heads"
        case mambaHeadDim = "mamba_head_dim"
        case mambaProjBias = "mamba_proj_bias"
        case ssmStateSize = "ssm_state_size"
        case convKernel = "conv_kernel"
        case nGroups = "n_groups"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case moeSharedExpertIntermediateSize = "moe_shared_expert_intermediate_size"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case hybridOverridePattern = "hybrid_override_pattern"
        case layerNormEpsilon = "layer_norm_epsilon"
        case mlpBias = "mlp_bias"
        case useBias = "use_bias"
        case useConvBias = "use_conv_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeTheta = "rope_theta"
        case headDim = "head_dim"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case normTopkProb = "norm_topk_prob"
        case routedScalingFactor = "routed_scaling_factor"
        case timeStepLimitMin = "time_step_limit_min"
        case timeStepLimitMax = "time_step_limit_max"
        case chunkSize = "chunk_size"
        case mambaSSMCacheDType = "mamba_ssm_cache_dtype"
        case numNextnPredictLayers = "num_nextn_predict_layers"
        case mtpLayersBlockType = "mtp_layers_block_type"
    }

    enum AliasCodingKeys: String, CodingKey {
        case layersBlockType = "layers_block_type"
        case normEpsilon = "norm_eps"
        case timeStepLimit = "time_step_limit"
        case timeStepMin = "time_step_min"
        case timeStepMax = "time_step_max"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let aliases = try decoder.container(keyedBy: AliasCodingKeys.self)

        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "nemotron_h"
        vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mambaNumHeads = try container.decode(Int.self, forKey: .mambaNumHeads)
        mambaHeadDim = try container.decode(Int.self, forKey: .mambaHeadDim)
        mambaProjBias = try container.decodeIfPresent(Bool.self, forKey: .mambaProjBias) ?? false
        ssmStateSize = try container.decode(Int.self, forKey: .ssmStateSize)
        convKernel = try container.decode(Int.self, forKey: .convKernel)
        nGroups = try container.decode(Int.self, forKey: .nGroups)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        moeIntermediateSize = try container.decode(Int.self, forKey: .moeIntermediateSize)
        moeSharedExpertIntermediateSize = try container.decode(
            Int.self, forKey: .moeSharedExpertIntermediateSize)
        nRoutedExperts = try container.decode(Int.self, forKey: .nRoutedExperts)
        nSharedExperts = try container.decodeIfPresent(Int.self, forKey: .nSharedExperts)
        numExpertsPerTok = try container.decode(Int.self, forKey: .numExpertsPerTok)
        layerNormEpsilon =
            try container.decodeIfPresent(Float.self, forKey: .layerNormEpsilon)
            ?? aliases.decodeIfPresent(Float.self, forKey: .normEpsilon)
            ?? 1e-5
        mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        useBias = try container.decodeIfPresent(Bool.self, forKey: .useBias) ?? false
        useConvBias = try container.decodeIfPresent(Bool.self, forKey: .useConvBias) ?? true
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        nGroup = try container.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
        topkGroup = try container.decodeIfPresent(Int.self, forKey: .topkGroup) ?? 1
        normTopkProb = try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        routedScalingFactor =
            try container.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0

        // Legacy Nemotron Nano uses compact hybrid_override_pattern symbols.
        // Nemotron 3.5 uses descriptive layers_block_type entries instead.
        if let patternString = try? container.decode(String.self, forKey: .hybridOverridePattern) {
            hybridOverridePattern = patternString
        } else if let patternArray = try? container.decode(
            [String].self, forKey: .hybridOverridePattern)
        {
            hybridOverridePattern = patternArray.joined()
        } else if let blockTypes = try? aliases.decode(
            [String].self, forKey: .layersBlockType)
        {
            guard blockTypes.count == numHiddenLayers else {
                throw DecodingError.dataCorruptedError(
                    forKey: .layersBlockType, in: aliases,
                    debugDescription:
                        "layers_block_type count \(blockTypes.count) does not match "
                        + "num_hidden_layers \(numHiddenLayers)")
            }
            hybridOverridePattern = try blockTypes.map { blockType in
                switch blockType {
                case "mamba": return "M"
                case "attention": return "*"
                case "mlp": return "-"
                case "moe": return "E"
                default:
                    throw DecodingError.dataCorruptedError(
                        forKey: .layersBlockType, in: aliases,
                        debugDescription: "unsupported NemotronH block type \(blockType)")
                }
            }.joined()
        } else {
            throw DecodingError.dataCorruptedError(
                forKey: .hybridOverridePattern, in: container,
                debugDescription:
                    "configuration requires hybrid_override_pattern or layers_block_type")
        }

        // Explicit execution limits are distinct from the published
        // timestep-initialization metadata. Reject malformed arrays before
        // indexing; an empty operator-writable config must not crash loading.
        if let limits = try aliases.decodeIfPresent([Float].self, forKey: .timeStepLimit) {
            guard (1...2).contains(limits.count) else {
                throw DecodingError.dataCorruptedError(forKey: .timeStepLimit, in: aliases,
                    debugDescription: "time_step_limit requires one or two values")
            }
            timeStepLimitMin = limits[0]
            timeStepLimitMax = limits.count > 1 ? limits[1] : limits[0]
        } else if let limits = try? container.decode([Float].self, forKey: .timeStepLimitMin) {
            // Compatibility with an early Swift fixture that placed the
            // two-element array under time_step_limit_min.
            guard (1...2).contains(limits.count) else {
                throw DecodingError.dataCorruptedError(forKey: .timeStepLimitMin, in: container,
                    debugDescription: "time_step_limit_min array requires one or two values")
            }
            timeStepLimitMin = limits[0]
            timeStepLimitMax = limits.count > 1 ? limits[1] : limits[0]
        } else {
            timeStepLimitMin =
                try container.decodeIfPresent(Float.self, forKey: .timeStepLimitMin)
                ?? aliases.decodeIfPresent(Float.self, forKey: .timeStepMin)
                ?? 0.0
            timeStepLimitMax =
                try container.decodeIfPresent(Float.self, forKey: .timeStepLimitMax)
                ?? aliases.decodeIfPresent(Float.self, forKey: .timeStepMax)
                ?? Float.infinity
        }
        guard timeStepLimitMin.isFinite, timeStepLimitMin >= 0,
            !timeStepLimitMax.isNaN, timeStepLimitMax >= timeStepLimitMin else {
            throw DecodingError.dataCorruptedError(forKey: .timeStepLimitMin, in: container,
                debugDescription: "timestep bounds must be nonnegative and ordered")
        }
        chunkSize = try container.decodeIfPresent(Int.self, forKey: .chunkSize) ?? 128
        mambaSSMCacheDType =
            try container.decodeIfPresent(String.self, forKey: .mambaSSMCacheDType) ?? "float32"
        numNextnPredictLayers =
            try container.decodeIfPresent(Int.self, forKey: .numNextnPredictLayers) ?? 0
        mtpLayersBlockType =
            try container.decodeIfPresent([String].self, forKey: .mtpLayersBlockType) ?? []
    }

    /// Memberwise initializer for testing
    public init(
        vocabSize: Int,
        hiddenSize: Int,
        numHiddenLayers: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        mambaNumHeads: Int,
        mambaHeadDim: Int,
        ssmStateSize: Int,
        convKernel: Int,
        nGroups: Int,
        intermediateSize: Int,
        moeIntermediateSize: Int,
        moeSharedExpertIntermediateSize: Int,
        nRoutedExperts: Int,
        numExpertsPerTok: Int,
        hybridOverridePattern: String,
        layerNormEpsilon: Float = 1e-5,
        attentionBias: Bool = false,
        mambaProjBias: Bool = false,
        mlpBias: Bool = false,
        useBias: Bool = false,
        useConvBias: Bool = true,
        tieWordEmbeddings: Bool = false,
        ropeTheta: Float = 10000.0,
        headDim: Int? = nil,
        nSharedExperts: Int? = nil,
        nGroup: Int = 1,
        topkGroup: Int = 1,
        normTopkProb: Bool = true,
        routedScalingFactor: Float = 1.0,
        timeStepLimitMin: Float = 0.0,
        timeStepLimitMax: Float = .infinity,
        chunkSize: Int = 128,
        mambaSSMCacheDType: String = "float32",
        numNextnPredictLayers: Int = 0,
        mtpLayersBlockType: [String] = []
    ) {
        self.modelType = "nemotron_h"
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.attentionBias = attentionBias
        self.mambaNumHeads = mambaNumHeads
        self.mambaHeadDim = mambaHeadDim
        self.mambaProjBias = mambaProjBias
        self.ssmStateSize = ssmStateSize
        self.convKernel = convKernel
        self.nGroups = nGroups
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.moeSharedExpertIntermediateSize = moeSharedExpertIntermediateSize
        self.nRoutedExperts = nRoutedExperts
        self.nSharedExperts = nSharedExperts
        self.numExpertsPerTok = numExpertsPerTok
        self.hybridOverridePattern = hybridOverridePattern
        self.layerNormEpsilon = layerNormEpsilon
        self.mlpBias = mlpBias
        self.useBias = useBias
        self.useConvBias = useConvBias
        self.tieWordEmbeddings = tieWordEmbeddings
        self.ropeTheta = ropeTheta
        self.headDim = headDim
        self.nGroup = nGroup
        self.topkGroup = topkGroup
        self.normTopkProb = normTopkProb
        self.routedScalingFactor = routedScalingFactor
        self.timeStepLimitMin = timeStepLimitMin
        self.timeStepLimitMax = timeStepLimitMax
        self.chunkSize = chunkSize
        self.mambaSSMCacheDType = mambaSSMCacheDType
        self.numNextnPredictLayers = numNextnPredictLayers
        self.mtpLayersBlockType = mtpLayersBlockType
    }
}
private let gauntletRedraw_32a5ef90_20261024T102246Z: Int = 0
private let gauntletRedraw_570f00bf_20261024T122202Z: Int = 0
