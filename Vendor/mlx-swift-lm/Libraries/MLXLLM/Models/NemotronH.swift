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

/// Captured verify only: also submit after the first layer.
/// `NEMOTRON_MTP_VERIFY_FIRST_SUBMIT=0` keeps the plain interval.
let nemotronVerifyFirstSubmit =
    ProcessInfo.processInfo.environment["NEMOTRON_MTP_VERIFY_FIRST_SUBMIT"] != "0"

/// Layers between early asyncEval calls on the MTP prefill path only. The seed
/// window is the first request after the engine drains MLX's buffer cache, so a
/// shorter chunk hands its transient live set back to that cache sooner.
/// `NEMOTRON_PREFILL_REEVAL=0` restores the default spacing.
let nemotronPrefillEvalInterval =
    ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_REEVAL"] == "0"
    ? nemotronEarlyEvalInterval : 2

// Input-independent eight-element device views, restricted to the draft's
// one-row custom kernels. No global request flag or target compiled slot.
enum NemotronDraftWeightAnchor {
    static let enabled =
        ProcessInfo.processInfo.environment["NEMOTRON_DRAFT_WEIGHT_ANCHOR"] != "0"
}

func nemotronWeightAnchor(_ a: MLXArray, rows: Int) -> MLXArray {
    guard rows == 1, NemotronDraftWeightAnchor.enabled else { return a }
    return NemotronWeightAnchors.shared.anchor(a, rows: rows)
}

final class NemotronWeightAnchors: @unchecked Sendable {
    static let shared = NemotronWeightAnchors()
    private let lock = NSLock()
    // Keyed by object identity; the entry retains the source so the
    // identifier cannot be reused while the entry exists.
    private var entries: [ObjectIdentifier: (source: MLXArray, anchor: MLXArray, singleRow: Bool)] =
        [:]

    /// A one-row launch binds an anchor only for arrays marked `singleRow`:
    /// the MTP head's own weights and the target head / embedding tables
    /// that only the draft's one-row kernels read here. Target weights at one
    /// row (the non-captured stepper's Q/K/V and shared up) keep full bindings.
    func anchor(_ a: MLXArray, rows: Int) -> MLXArray {
        guard let entry = lock.withLock({ entries[ObjectIdentifier(a)] }),
              rows > 1 || entry.singleRow else { return a }
        return entry.anchor
    }

    /// Mark already-prepared arrays as eligible for one-row launches.
    func allowSingleRow(_ arrays: [MLXArray]) {
        lock.withLock {
            for a in arrays {
                if var entry = entries[ObjectIdentifier(a)] {
                    entry.singleRow = true
                    entries[ObjectIdentifier(a)] = entry
                }
            }
        }
    }

    /// Only a row-contiguous array flattens to a view of its own buffer
    /// (prepare_reshape: row_contiguous -> shared buffer, no copy); any other
    /// layout would be copied whole, so it is never anchored. Size-1 dims are
    /// ignored, as in MLX's own contiguity check.
    private static func isRowContiguous(_ a: MLXArray) -> Bool {
        var expected = 1
        for (dim, stride) in zip(a.shape, a.strides).reversed() where dim != 1 {
            if stride != expected { return false }
            expected *= dim
        }
        return true
    }

    /// Input-independent, load-time: one view per row-contiguous weight,
    /// scales and biases array of every quantized linear, expert bank and
    /// embedding in `model`. Call sites decide what is anchored (decode-window
    /// kernels only). Never copies: non-row-contiguous arrays are skipped.
    func prepare(_ model: Module, singleRow: Bool) {
        guard NemotronDraftWeightAnchor.enabled else { return }
        var built: [(MLXArray, MLXArray)] = []
        var skipped = 0, bytes = 0
        for module in model.modules() where module is QuantizedLinear
            || module is QuantizedSwitchLinear || module is QuantizedEmbedding {
            for (key, value) in module.parameters().flattened()
            where ["weight", "scales", "biases"].contains(key) && value.size > 8 {
                eval(value)
                guard Self.isRowContiguous(value) else { skipped += 1; continue }
                built.append((value, value.reshaped(-1)[0 ..< 8]))
                bytes += value.nbytes
            }
        }
        eval(built.map { $0.1 })
        lock.withLock {
            for (source, view) in built {
                guard let sourceInfo = try? source.evaluatedBufferInfo(),
                      let viewInfo = try? view.evaluatedBufferInfo(),
                      sourceInfo.isRowContiguous, viewInfo.isRowContiguous,
                      sourceInfo.allocatedBytes == viewInfo.allocatedBytes,
                      sourceInfo.dataOffset == viewInfo.dataOffset,
                      viewInfo.dataElements == 8
                else { continue }
                entries[ObjectIdentifier(source)] = (source, view, singleRow)
            }
        }
        if ProcessInfo.processInfo.environment["NEMOTRON_WEIGHT_ANCHOR_LOG"] == "1" {
            FileHandle.standardError.write(Data((
                "nemotron weight anchors: \(built.count) anchored (\(bytes >> 20) MiB), "
                    + "\(skipped) skipped (not row-contiguous)\n").utf8))
        }
    }
}

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

    private static let kernel = MLXFast.metalKernel(
        name: "nemotron_h_residual_rms_bf16",
        inputNames: ["lhs", "rhs", "weight", "axis_size", "eps"],
        outputNames: ["residual", "normalized"],
        source: source, ensureRowContiguous: true)

    static let preparedMoEEnabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_RMS_PREPARED_MOE"] != "0"

    private static func preparedSource() -> String {
        let scratch = "threadgroup float local_inv_mean[1];"
        let store = "normalized[offset + i] = weight[lid * n_reads + i]"
        let product = "* static_cast<T>(thread_x[i] * local_inv_mean[0]);"
        precondition(source.components(separatedBy: scratch).count == 2)
        precondition(source.components(separatedBy: store).count == 2)
        precondition(source.components(separatedBy: product).count == 2)
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
    private static let preparedKernel = MLXFast.metalKernel(
        name: "nemotron_h_residual_rms_prepared_moe_bf16",
        inputNames: ["lhs", "rhs", "weight", "axis_size", "eps"],
        outputNames: ["residual", "normalized", "prepared"],
        source: preparedSource(), ensureRowContiguous: true)

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
        name: "nemotron_h_residual_sum_rms_prepared_bf16",
        inputNames: ["lhs", "rhs", "shared", "weight", "axis_size", "eps"],
        outputNames: ["residual", "normalized", "prepared"],
        source: sumPreparedSource(), ensureRowContiguous: true)
    private static let normPreparedKernel = MLXFast.metalKernel(
        name: "nemotron_h_rms_prepared_bf16",
        inputNames: ["lhs", "weight", "axis_size", "eps"],
        outputNames: ["normalized", "prepared"],
        source: normPreparedSource(), ensureRowContiguous: true)

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
        name: "nemotron_h_residual_sum_rms_bf16",
        inputNames: ["lhs", "rhs", "shared", "weight", "axis_size", "eps"],
        outputNames: ["residual", "normalized"],
        source: sumSource(), ensureRowContiguous: true)

    static func applySum(_ lhs: MLXArray, _ rhs: MLXArray, shared: MLXArray, norm: RMSNorm)
        -> (MLXArray, MLXArray)?
    {
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
        return kernel([tokens.reshaped(count)]
            + [q.weight, q.scales, biases].map { nemotronWeightAnchor($0, rows: count) },
            template: [("T", DType.bfloat16), ("V", vocabulary), ("BYTES", dims / 2),
                       ("G", dims / 64)],
            grid: (dims / 2, count, 1), threadGroup: (64, 1, 1),
            outputShapes: [[1, count, dims]], outputDTypes: [.bfloat16])[0]
    }
}

// MARK: - MambaRMSNormGated

private enum NemotronHGatedRMSKernel {
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

// MARK: - Prefill x5 switches

/// Prefill x5: stable counting sort with expert-run tiles, expert-run NAX
/// gather, the fused segsum scan and the fused scan-window output.
/// NEMOTRON_PREFILL_X5=0 restores the argsort / stock gather / masked-decay
/// graph.
enum NemotronHPrefillX5 {
    static let enabled = ProcessInfo.processInfo.environment["NEMOTRON_PREFILL_X5"] != "0"
    /// The run gather clones the NAX kernel; without NAX stock runs another one.
    static let runGather = enabled && GPU.gemma4ExpertQMMDiagnostics().naxAvailable
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
    private init() {
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
func nemotronSSMAttn(
    x: MLXArray, ALog: MLXArray, B: MLXArray, C: MLXArray, D: MLXArray,
    dt: MLXArray, dtBias: MLXArray, state: MLXArray,
    timeStepLimit: (Float, Float)
) -> (MLXArray, MLXArray) {
    let (b, l, h, dh) = x.shape4
    let (_, _, g, d) = B.shape4

    let dt = nemotronPrefillComputeDt(dt, dtBias, timeStepLimit)
    let repeats = h / g
    let A = -MLX.exp(ALog).asType(dt.dtype)
    let dtA = dt * A.reshaped(1, 1, -1)
    let dtx = dt.reshaped(b, l, h, 1) * x

    let step = 256
    var currentState = state
    var outputs = [MLXArray]()
    outputs.reserveCapacity((l + step - 1) / step)
    for start in stride(from: 0, to: l, by: step) {
        let end = min(start + step, l)
        let n = end - start
        let stepDtx = dtx[0..., start ..< end, 0..., 0...]
        let stepDtA = dtA[0..., start ..< end, 0...]
        var stepB = B[0..., start ..< end, 0..., 0...]
        let stepC = C[0..., start ..< end, 0..., 0...]

        stepB = MLX.transposed(stepB, axes: [0, 2, 3, 1])
        let CB = MLX.swappedAxes(stepC, 1, 2).matmul(stepB)

        let surrogateAttentionMatrix: MLXArray
        let decayLast: MLXArray
        if NemotronHPrefillX5.enabled, stepDtA.dtype == .float32, h % g == 0 {
            let fused = NemotronHScanKernels.shared.scanDecay(
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
        let yWindow = surrogateAttentionMatrix.matmul(stepDtx.swappedAxes(1, 2))

        // decay[..., -1:, :] == exp(segsum[..., -1:, :]) element-wise.
        let decay = decayLast.transposed(0, 3, 1, 2)
        var dtxdecay = stepDtx * decay
        dtxdecay = dtxdecay.swappedAxes(1, 2).swappedAxes(2, 3)
        let nextStateProduct: MLXArray
        if b == 1, h == 64, g == 8, dh == 64, d == 128, n >= 64,
           Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu {
            // Broadcast each B group over its heads, preserving the reduction
            // axis and input order without materializing eight copies.
            let groupedB = stepB.swappedAxes(2, 3).reshaped(b, g, 1, n, d)
            nextStateProduct = dtxdecay.reshaped(b, g, repeats, dh, n)
                .matmul(groupedB).reshaped(b, h, dh, d)
        } else {
            stepB = MLX.repeated(stepB, count: repeats, axis: 1).swappedAxes(2, 3)
            nextStateProduct = dtxdecay.matmul(stepB)
        }
        var nextState = nextStateProduct

        let previousState = currentState
        let expDtACumsum = MLX.exp(MLX.cumsum(stepDtA, axis: -2))
        nextState =
            nextState
            + expDtACumsum[0..., -1, 0..., .newAxis, .newAxis] * previousState
        if NemotronHPrefillX5.enabled, d == 128, dh % 32 == 0, h % g == 0,
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
    return (y, currentState)
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
    static let source = """
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
            """
    static let kernel = MLXFast.metalKernel(
        name: "nemotron_packed_conv_silu_dt",
        inputNames: ["oldState", "input", "weight", "bias", "dtIn", "dtBias", "dtLo", "dtHi"],
        outputNames: ["out", "padded", "dt"],
        source: source, ensureRowContiguous: false)
}

private enum NemotronPackedConvDtDecayKernel {
    private static let store = "dt[position * H + head] = metal::isnan(m) ? m : (m < hi ? m : hi);"
    static let kernel = MLXFast.metalKernel(
        name: "nemotron_packed_conv_silu_dt_decay",
        inputNames: ["oldState", "input", "weight", "bias", "dtIn", "dtBias", "dtLo", "dtHi", "A_log"],
        outputNames: ["out", "padded", "dt", "decay"],
        source: NemotronPackedConvDtKernel.source.replacingOccurrences(of: store, with: """
            float dtValue = metal::isnan(m) ? m : (m < hi ? m : hi);
            dt[position * H + head] = dtValue;
            auto A = -fast::exp(static_cast<float>(A_log[head * A_log_strides[0]]));
            decay[position * H + head] = fast::exp(A * dtValue);
            """), ensureRowContiguous: false)
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
    // Qualified only during official-worker model warm-up, before graph traces.
    var mtpConvDtDecayQualified = false
    var mtpConvDtDecayQualificationComplete = false
    static let convDtDecayEnabled =
        ProcessInfo.processInfo.environment["NEMOTRON_MTP_CONV_DT_DECAY"] != "0"

    func packedConvDtDecay(_ oldState: MLXArray, _ input: MLXArray, _ dt: MLXArray,
        qualifying: Bool = false)
        -> (convolution: MLXArray, padded: MLXArray, dt: MLXArray, decay: MLXArray)? {
        guard Self.convDtDecayEnabled, (qualifying || mtpConvDtDecayQualified),
            aLog.shape == [numHeads], [DType.float32, .bfloat16].contains(aLog.dtype),
            Self.convDtEnabled, convKernelSize == 4, oldState.shape == [1, 3, convDim],
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
        let arrays = NemotronPackedConvDtDecayKernel.kernel(
            [oldState, input, conv1d.weight, bias, dt, dtBias,
             MLXArray(timeStepLimit.0), MLXArray(timeStepLimit.1), aLog],
            template: [("T", DType.bfloat16), ("C", convDim), ("H", numHeads)],
            grid: (convDim + numHeads, length, 1), threadGroup: (256, 1, 1),
            outputShapes: [[1, length, convDim], [1, length + 3, convDim], [1, length, numHeads],
                           [1, length, numHeads]],
            outputDTypes: [.bfloat16, .bfloat16, .float32, .float32])
        return (arrays[0], arrays[1], arrays[2], arrays[3])
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

        let projected = inProj(hiddenStates)
        let splits = split(
            projected, indices: [intermediateSize, intermediateSize + convDim], axis: -1)
        let gate = splits[0]
        let convInput = splits[1]
        let dt = splits[2]

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
        let convState =
            convRows.count == 1 ? convRows[0] : concatenated(convRows, axis: 0)
        let ssmState =
            ssmRows.count == 1 ? ssmRows[0] : concatenated(ssmRows, axis: 0)

        let (convOutput, padded) = packedConv(convState, convInput)
        let end = padded.dim(1)
        let history = padded[0..., max(0, end - (convKernelSize - 1)) ..< end, 0...]
        let nextConvState: MLXArray
        if hiddenStates.dim(1) > 8 {
            nextConvState = NemotronRequestConvHistory.copy(
                [history], template: [("N", history.dim(1)), ("C", convDim)],
                grid: (convDim, history.dim(1), batch), threadGroup: (256, 1, 1),
                outputShapes: [history.shape], outputDTypes: [history.dtype])[0]
        } else {
            nextConvState = history
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
                dtBias: dtBias, state: ssmState, timeStepLimit: timeStepLimit)
            : nemotronDecodeSSM(
            hiddenStates: hidden,
            ALog: aLog,
            B: B,
            C: C,
            D: D,
            dt: dtArray,
            dtBias: dtBias,
            state: ssmState,
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
        return outProj(norm(
            y.asType(hiddenStates.dtype).flattened(start: 2),
            gate: gate))
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
        // Serialized MTP verify: o_proj reads the per-query outputs in place.
        if captureMTP, batch == 1, length > 1, NemotronMTPAttentionDirect.enabled,
           let layerCache = cache as? CBv2LayerCache,
           let rows = layerCache.updateAndAttendSerialQueryRows(
               queries: queries, keys: keys, values: values, scale: scale, sinks: nil) {
            return NemotronMTPAttentionDirect.project(rows, wo)
        }
        let output = cache.updateAndAttend(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            sinks: nil)
        let merged = output.transposed(0, 2, 1, 3).reshaped(batch, length, -1)
        return captureMTP ? nemotronMTPLinearRows(merged, wo) : wo(merged)
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
            return downProj(activated.reshaped(1, 1, 3712))
        }
        return downProj(relu2(upProj(x)))
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

    static func apply(
        gates: MLXArray, eSCB: MLXArray, topK: Int, nGroup: Int, topkGroup: Int,
        routedScalingFactor: Float, normTopkProb: Bool
    ) -> (MLXArray, MLXArray)? {
        guard Device.defaultDevice().deviceType == .gpu, StreamOrDevice.default == .gpu,
            gates.ndim == 3, gates.dim(0) == 1, (1...8).contains(gates.dim(1)),
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
        if NemotronHPrefillX5.enabled, let sorted = nemotronCountingSort(flat, experts: numExperts) {
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
            .flattened(start: 0, end: -3)[order.floorDivide(k)]
        func gather(_ lin: SwitchLinear, _ input: MLXArray) -> MLXArray {
            if let tiles = runTiles, let q = lin as? QuantizedSwitchLinear,
               let fused = nemotronRunGather(input, q, tiles: tiles) {
                return fused
            }
            return lin(input, idx, sortedIndices: true)
        }
        var y: MLXArray
        if let tiles = runTiles, let q = fc1 as? QuantizedSwitchLinear,
            let activated = nemotronRunGather(xs, q, tiles: tiles, relu2: true) {
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
        y = gather(fc2, y)
        guard y.dim(-1) == inputDims, y.size == indices.size * inputDims else { return nil }
        var outShape = indices.shape
        outShape[outShape.count - 1] = inputDims
        return kernels.combine(
            [y, inverse, scores, inputDims],
            template: [("T", y.dtype), ("K", k)],
            grid: (inputDims / 4, tokens, 1), threadGroup: (32, 1, 1),
            outputShapes: [outShape], outputDTypes: [y.dtype])[0]
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
    let relu2: MLXFast.MLXFastKernel
    let combine: MLXFast.MLXFastKernel
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
                        for (int r = start; r < end && t < tmax; r += BMT) {
                            tiles[3 * t] = r;
                            tiles[3 * t + 1] = min(BMT, end - r);
                            tiles[3 * t + 2] = ee;
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
    }
}

/// Rows per tile the stock gather_qmm_rhs_nax dispatch picks for M rows
/// and E experts (`bm = (M / E < 64) ? 32 : 64` in quantized.cpp).
func nemotronGatherTileRows(rows: Int, experts: Int) -> Int { rows / experts < 64 ? 32 : 64 }

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
    let count = (n + bm - 1) / bm + experts
    let out = NemotronHPrefillKernels.shared.countingSort(
        [flat, n, count], template: [("E", experts), ("THREADS", threads), ("BMT", bm)],
        grid: (threads, 1, 1), threadGroup: (threads, 1, 1),
        outputShapes: [[n], [n], [n], [count, 3]],
        outputDTypes: [.uint32, .uint32, .uint32, .int32])
    return (out[0], out[1], out[2], NemotronHRunTiles(tiles: out[3], count: count, bm: bm, rows: n))
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
        // Counting sort for both tile heights (bm 32: rows / E < 64, else 64).
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
private let nemotronRunGatherHeader = #"""
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
            name: "nemotron_gather_qmm_runs",
            inputNames: ["x", "w", "scales", "biases", "tiles", "N", "K"],
            outputNames: ["y"],
            source: """
                threadgroup T Ws[BN * (BK + 16 / sizeof(T))];
                nemotron_gather_qmm_runs_impl<T, GS, BITS, BM, BN, BK, 2, 2, ALIGN_N, ALIGN_K, RELU2>(
                    x, w, scales, biases, tiles, y, N, K, Ws,
                    threadgroup_position_in_grid,
                    simdgroup_index_in_threadgroup,
                    thread_index_in_simdgroup);
            """,
            header: nemotronRunGatherHeader)
    }
}

/// Same rows as `lin(x, idx, sortedIndices: true)` with one threadgroup per
/// (N block, expert-run tile). Only where stock GatherQMM takes the
/// sorted-rhs route (M == 1, B >= 16, B / E >= 4) and gather_qmm_rhs picks
/// affine_gather_qmm_rhs_nax (NAX available, half-precision activations)
/// with the same bm; nil = use the stock op. ReLU2 is BF16-only and retains
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
        let (inds, scores) = gate(x)
        var y: MLXArray
        if let fused = switchMLP.fusedSortedCombine(x, inds, scores) {
            y = fused
        } else {
            y = switchMLP(x, inds)
            y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)
        }

        if let sharedExperts {
            y = y + sharedExperts(x)
        }

        return y
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
            output = captureMTP
                ? moe.mtpForwardRows(hidden, prepared: preparedActivation) : moe(hidden)
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
            output = (mixer as! NemotronHMamba2Mixer).cbv2ForwardCaptured(
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
        captureMTP: Bool = false,
        recyclePrefillChunks: Bool = false,
        earlyEvalInterval: Int = nemotronEarlyEvalInterval
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
            let suppressedNorm = nemotronEarlyEvalInterval > 0 && next < layers.count
                && next % nemotronEarlyEvalInterval == 0
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
                if inputs.dim(1) > 1 && !captureMTP {
                    let group = (next - earlyEvalInterval)..<next
                    let states = recurrentState.flatMap { $0.stagedEvaluationRoots(in: group) }
                    asyncEval([hidden] + states)
                    // Engine prefill only (see `cbv2Hidden`). A chunk runs for
                    // milliseconds while the host builds the next in well under
                    // one, so without a wait the host encodes, and allocates,
                    // every layer's intermediates before the first chunk
                    // completes. Waiting on the previous chunk returns its
                    // temporaries to MLX's buffer cache for the next chunk; this
                    // chunk is already queued, so the GPU never waits on the
                    // host. Evaluation order only; values are unchanged.
                    if recyclePrefillChunks {
                        if let previous = submittedPrefillChunk { eval(previous) }
                        submittedPrefillChunk = hidden
                    }
                } else {
                    asyncEval(hidden)
                }
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
        var hidden = embeddings(inputs)
        let entry = NemotronHResidualRMSNorm.normPreparedRows(hidden, norm: first.norm)
        var preparedNorm: MLXArray? = entry?.0
        var preparedActivation: MLXArray? = entry?.1
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
            // Same submission layers as `cbv2Forward`, plus one after the
            // first layer so the GPU starts the verify while the host builds
            // the rest of the chunk (evaluation order only).
            if nemotronEarlyEvalInterval > 0 && next < layers.count
                && (next % nemotronEarlyEvalInterval == 0
                    || (next == 1 && nemotronVerifyFirstSubmit)) {
                asyncEval(hidden)
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

// MARK: - Main Model (matches Python's Model class)

public class NemotronHModel:
    Module, LLMModel, KVCacheDimensionProvider, LoRAModel
{
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "backbone") private var backbone: NemotronHBackbone
    let configuration: NemotronHConfiguration
    private let recurrentActivationDTypes = NemotronHActivationDTypeMemo()

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public var loraLayers: [Module] {
        backbone.layers
    }

    public var cbv2LayerKinds: [CBv2LayerKind] {
        configuration.cbv2LayerKinds
    }

    public var cbv2RecurrentStateSpec: CBv2RecurrentStateSpec {
        let activationDTypes = resolvedRecurrentActivationDTypes()
        return configuration.cbv2RecurrentStateSpec(
            activationDTypes: activationDTypes,
            fallbackActivationDType:
                backbone.embeddings(MLXArray([0])).dtype)
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
        captureMTP: Bool = false,
        recyclePrefillChunks: Bool = false,
        earlyEvalInterval: Int = nemotronEarlyEvalInterval
    ) -> MLXArray {
        let attending = caches.map { cache -> any CBv2AttendingLayerCache in
            guard let attending = cache as? any CBv2AttendingLayerCache else {
                preconditionFailure("NemotronH CBv2 target received legacy KV cache")
            }
            return attending
        }
        return backbone.cbv2Forward(
            inputs, caches: attending, recurrentState: recurrentState, captureMTP: captureMTP,
            recyclePrefillChunks: recyclePrefillChunks, earlyEvalInterval: earlyEvalInterval)
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

    /// Weight, scales and biases of the target head and embedding tables:
    /// the draft's one-row head screen and embedding rows read these.
    var draftSharedTables: [MLXArray] {
        [lmHead as Module?, backbone.embeddings as Module?].compactMap { $0 }.flatMap { module in
            module.parameters().flattened()
                .filter { ["weight", "scales", "biases"].contains($0.0) }.map { $0.1 }
        }
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
