import CoreFoundation
import Foundation
import MLXFastCore

/// Quantization expectations parsed from the pinned Nemotron 3.5 Lightning
/// 30B-A3B checkpoint's `quantization` / `quantization_config` blocks.
///
/// The target is UNIFORM: MLX affine, 4 bits, group size 64, and the contract
/// fixture records `mixed_precision: false`. There is no per-tensor override
/// table, so the three scalars ARE the whole block and a fourth key is
/// refused rather than half-read.
struct NemotronHTransformQuantizationSpec: Equatable {
    let groupSize: Int
    let bits: Int
    let mode: String
}

/// Transform-side structural validation of the Nemotron 3.5 Lightning
/// 30B-A3B tensor set.
///
/// The source checkpoint
/// (`EigenLabs/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit-mtp`) is already
/// MLX affine-quantized, so the transform passes tensors through unchanged.
/// This pass fails fast -- before the ~18.5 GB copy -- when the set it would
/// copy cannot satisfy the runtime loader:
///
/// - the selected namespace must be EXACTLY the public 763-tensor inventory,
///   tensor for tensor, at the exact dtype and shape;
/// - every quantized projection stored as packed U32 codes must ship BF16
///   `.scales` AND BF16 `.biases`, because affine needs both, with matching
///   leading dimensions;
/// - each packed width must match the group size and bit width the emitted
///   config.json declares: `in / 8` U32 columns and `in / 64` scale/bias
///   columns;
/// - compressed-tensors aliases, global-scale tensors and FP8 KV scales are
///   rejected outright;
/// - the `lm_head.*` triple MUST appear. `tie_word_embeddings` is false, so a
///   checkpoint without its own output projection is a different artifact;
/// - the 34 `mtp.*` tensors MUST appear. The speculative head is EMBEDDED in
///   this checkpoint; there is no separate head artifact, and a tree without
///   them has no speculative arm at all.
///
/// THE TOWER IS HYBRID, and that is the fact this file is built around. 52
/// layers on a published schedule of three kinds: 23 Mamba-2 mixers, 23
/// mixture-of-experts mixers and 6 full-attention mixers. Every layer carries
/// its own `norm.weight`; the kinds differ in which mixer tensors sit beside
/// it.
///
/// Deliberately independent of shard placement: this pass pins the complete
/// tensor namespace, dtype and shape, while the public inventory fixture
/// `fixtures/nemotron3_5_30b_a3b_tensor_inventory.json` pins placement.
enum NemotronHCheckpointValidation {
    struct ExpectedTensorMetadata: Equatable {
        let dtype: String
        let shape: [Int]
    }

    /// One entry of the published `layers_block_type` schedule.
    enum LayerKind: String, Equatable {
        case mamba
        case moe
        case attention
    }

    /// Frozen geometry of the pinned `nemotron_h` tower. Kept as literals
    /// rather than read from the config under validation: a validator that
    /// derives its expectations from the artifact it is checking cannot
    /// detect a changed artifact.
    enum PinnedGeometry {
        static let vocabSize = 131_072
        static let hiddenSize = 2_688
        static let layerCount = 52

        static let attentionHeads = 32
        static let keyValueHeads = 2
        static let headDim = 128

        static let mambaHeads = 64
        static let mambaHeadDim = 64
        static let ssmStateSize = 128
        /// `n_groups`: the shared key/value group count of the SSM. NOT the
        /// mixture-of-experts `n_group`, which is 1 on this target.
        static let ssmGroupCount = 8
        static let convKernel = 4

        static let routedExpertCount = 128
        static let moeIntermediateSize = 1_856
        static let sharedExpertIntermediateSize = 3_712

        /// `num_nextn_predict_layers`. The embedded head predicts one extra
        /// token per step, through the two blocks of `mtpLayersBlockType`.
        static let mtpPredictLayerCount = 1

        static let quantizationGroupSize = 64
        static let quantizationBits = 4
        static let quantizationMode = "affine"

        /// The published `layers_block_type`, one entry per layer.
        static let layersBlockType: [LayerKind] = [
            .mamba, .moe, .mamba, .moe, .mamba, .attention, .moe, .mamba,
            .moe, .mamba, .moe, .mamba, .attention, .moe, .mamba, .moe,
            .mamba, .moe, .mamba, .attention, .moe, .mamba, .moe, .mamba,
            .moe, .mamba, .attention, .moe, .mamba, .moe, .mamba, .moe,
            .mamba, .attention, .moe, .mamba, .moe, .mamba, .moe, .mamba,
            .moe, .mamba, .attention, .moe, .mamba, .moe, .mamba, .moe,
            .mamba, .moe, .mamba, .moe,
        ]

        /// The published `mtp_layers_block_type`. The embedded head is hybrid
        /// too: one attention block feeding one mixture-of-experts block.
        static let mtpLayersBlockType: [LayerKind] = [.attention, .moe]

        /// The Mamba-2 mixer's inner width, `mamba_num_heads *
        /// mamba_head_dim`. It is NOT `hidden_size * expand` on this target.
        static var mambaIntermediateSize: Int { mambaHeads * mambaHeadDim }

        /// The depthwise convolution runs over the inner stream plus the
        /// shared B and C group states.
        static var mambaConvDim: Int {
            mambaIntermediateSize + 2 * ssmGroupCount * ssmStateSize
        }

        /// The fused input projection emits the gate, the inner stream, the
        /// B and C group states, and one time step per Mamba head.
        static var mambaInputProjectionDim: Int {
            2 * mambaIntermediateSize + 2 * ssmGroupCount * ssmStateSize
                + mambaHeads
        }

        static var attentionDim: Int { attentionHeads * headDim }
        static var keyValueDim: Int { keyValueHeads * headDim }
    }

    /// 763 tensors: 726 under `backbone.`, 3 for the untied head and 34 for
    /// the embedded speculative head.
    static let expectedTensorCount = 763
    /// `backbone.embeddings` (3), `backbone.norm_f` (1) and `lm_head` (3).
    static let expectedTopLevelTensorCount = 7
    /// A Mamba-2 layer: the layer norm plus the 12 mixer tensors.
    static let expectedMambaLayerTensorCount = 13
    /// A mixture-of-experts layer: the layer norm plus the 14 mixer tensors.
    static let expectedMoELayerTensorCount = 15
    /// A full-attention layer: the layer norm plus the four projections.
    static let expectedAttentionLayerTensorCount = 13
    /// The head embedded under `mtp.*`.
    static let expectedMTPTensorCount = 34

    /// The three namespaces the transform selects. The pinned index holds
    /// nothing else -- there is no vision tower and no n-gram table on this
    /// target -- so the prefix test is a contract, not a filter.
    static let selectedPrefixes = ["backbone.", "lm_head.", "mtp."]

    private static let backbonePrefix = "backbone"
    private static let layerPrefix = "backbone.layers."
    static let mtpPrefix = "mtp"

    static func isSelectedKey(_ key: String) -> Bool {
        selectedPrefixes.contains { key.hasPrefix($0) }
    }

    /// Parses the checkpoint's quantization block(s).
    ///
    /// The pinned artifact publishes the SAME spec twice, as `quantization`
    /// and `quantization_config`; `SwiftTransform.makeRuntimeConfigData`
    /// accepts either form and requires them to agree when both are present,
    /// so this validator applies exactly that policy rather than a stricter
    /// one -- a transform that emits a config must not reject the checkpoint
    /// that config came from.
    static func quantizationSpec(
        fromConfigRoot root: [String: Any]
    ) throws -> NemotronHTransformQuantizationSpec {
        func parseBlock(
            _ key: String
        ) throws -> NemotronHTransformQuantizationSpec? {
            guard let value = root[key], !(value is NSNull) else {
                return nil
            }
            guard let block = value as? [String: Any] else {
                throw MLXFastError.invalidInput(
                    "Nemotron 3.5 config \(key) must be an object"
                )
            }
            let allowedKeys: Set<String> = ["group_size", "bits", "mode"]
            // UNIFORM. Every extra key is a per-tensor override this transform
            // would carry into a config the runtime then resolves one width
            // from. Refuse rather than emit a block whose extra half nothing
            // reads.
            let unexpectedKeys = Set(block.keys).subtracting(allowedKeys).sorted()
            guard unexpectedKeys.isEmpty else {
                throw MLXFastError.invalidInput(
                    "Nemotron 3.5 config \(key) carries \(unexpectedKeys.count) "
                        + "unsupported entr(ies) and this target is uniform, "
                        + "first: \(unexpectedKeys[0])"
                )
            }
            guard block["group_size"] != nil, block["bits"] != nil,
                  block["mode"] != nil
            else {
                throw MLXFastError.invalidInput(
                    "Nemotron 3.5 config \(key) must explicitly define "
                        + "group_size, bits, and mode"
                )
            }
            let groupSize = try intField("group_size", in: block)
            let bits = try intField("bits", in: block)
            let mode = try stringField("mode", in: block)
            guard mode == PinnedGeometry.quantizationMode,
                  groupSize == PinnedGeometry.quantizationGroupSize,
                  bits == PinnedGeometry.quantizationBits
            else {
                throw MLXFastError.invalidInput(
                    "Nemotron 3.5 quantization must be affine 4-bit "
                        + "group_size 64"
                )
            }
            return NemotronHTransformQuantizationSpec(
                groupSize: groupSize,
                bits: bits,
                mode: mode
            )
        }

        let quantization = try parseBlock("quantization")
        let quantizationConfig = try parseBlock("quantization_config")
        if let quantization, let quantizationConfig {
            guard quantization == quantizationConfig else {
                throw MLXFastError.invalidInput(
                    "Nemotron 3.5 config quantization and quantization_config "
                        + "must match exactly"
                )
            }
            return quantization
        }
        guard let spec = quantization ?? quantizationConfig else {
            throw MLXFastError.invalidInput(
                "Nemotron 3.5 config is missing both quantization and "
                    + "quantization_config"
            )
        }
        return spec
    }

    /// Exact metadata contract of the transformed checkpoint.
    ///
    /// DTYPES are read off the pinned artifact: packed codes `U32`, their
    /// scale and bias companions `BF16`, the router correction vectors `F32`,
    /// and every other unquantized parameter `BF16` (the checkpoint's own
    /// `dtype` is `bfloat16`). SHAPES are DERIVED from the pinned geometry
    /// above, so a geometry the tower does not have fails here rather than
    /// silently reproducing whatever the artifact holds.
    static func expectedTensorInventory() -> [String: ExpectedTensorMetadata] {
        let hidden = PinnedGeometry.hiddenSize
        let vocab = PinnedGeometry.vocabSize
        let groupSize = PinnedGeometry.quantizationGroupSize
        let bits = PinnedGeometry.quantizationBits

        var inventory: [String: ExpectedTensorMetadata] = [:]
        func add(_ name: String, _ dtype: TensorDType, _ shape: [Int]) {
            precondition(
                inventory[name] == nil,
                "duplicate expected Nemotron 3.5 tensor \(name)"
            )
            inventory[name] = ExpectedTensorMetadata(
                dtype: dtype.rawValue, shape: shape)
        }
        /// One affine-quantized projection: packed U32 codes plus the BF16
        /// scale and bias companions the affine scheme requires. `leading` is
        /// everything before the contracted axis, so the stacked expert
        /// tensors pass their experts axis through it.
        func addAffine(_ stem: String, leading: [Int], inFeatures: Int) {
            precondition(
                inFeatures.isMultiple(of: groupSize)
                    && (inFeatures * bits).isMultiple(of: 32),
                "Nemotron 3.5 tensor \(stem) is not packable at \(bits) bits"
            )
            add("\(stem).weight", .u32, leading + [inFeatures * bits / 32])
            add("\(stem).scales", .bf16, leading + [inFeatures / groupSize])
            add("\(stem).biases", .bf16, leading + [inFeatures / groupSize])
        }
        /// The Mamba-2 mixer. Its state parameters are per Mamba head, and the
        /// depthwise kernel is in MLX layout: (channels, kernel, 1).
        func addMambaMixer(_ prefix: String) {
            let inner = PinnedGeometry.mambaIntermediateSize
            let convDim = PinnedGeometry.mambaConvDim
            for parameter in ["A_log", "D", "dt_bias"] {
                add("\(prefix).\(parameter)", .bf16, [PinnedGeometry.mambaHeads])
            }
            add(
                "\(prefix).conv1d.weight", .bf16,
                [convDim, PinnedGeometry.convKernel, 1])
            // `use_conv_bias` is true on this target.
            add("\(prefix).conv1d.bias", .bf16, [convDim])
            add("\(prefix).norm.weight", .bf16, [inner])
            addAffine(
                "\(prefix).in_proj",
                leading: [PinnedGeometry.mambaInputProjectionDim],
                inFeatures: hidden)
            addAffine("\(prefix).out_proj", leading: [hidden], inFeatures: inner)
        }
        /// The mixture-of-experts mixer. The activation is `relu2`, so an
        /// expert is an up/down pair with no gate projection, and the experts
        /// are STACKED behind a leading experts axis, never split per expert.
        func addMoEMixer(_ prefix: String) {
            let experts = PinnedGeometry.routedExpertCount
            // The router stays in full precision: the checkpoint ships no
            // scales beside it, and its correction vector is F32.
            add("\(prefix).gate.weight", .bf16, [experts, hidden])
            add("\(prefix).gate.e_score_correction_bias", .f32, [experts])
            addAffine(
                "\(prefix).shared_experts.up_proj",
                leading: [PinnedGeometry.sharedExpertIntermediateSize],
                inFeatures: hidden)
            addAffine(
                "\(prefix).shared_experts.down_proj",
                leading: [hidden],
                inFeatures: PinnedGeometry.sharedExpertIntermediateSize)
            addAffine(
                "\(prefix).switch_mlp.fc1",
                leading: [experts, PinnedGeometry.moeIntermediateSize],
                inFeatures: hidden)
            addAffine(
                "\(prefix).switch_mlp.fc2",
                leading: [experts, hidden],
                inFeatures: PinnedGeometry.moeIntermediateSize)
        }
        /// The full-attention mixer. `attention_bias` is false and there are
        /// no query or key norms on this target.
        func addAttentionMixer(_ prefix: String) {
            addAffine(
                "\(prefix).q_proj",
                leading: [PinnedGeometry.attentionDim], inFeatures: hidden)
            for projection in ["k_proj", "v_proj"] {
                addAffine(
                    "\(prefix).\(projection)",
                    leading: [PinnedGeometry.keyValueDim], inFeatures: hidden)
            }
            addAffine(
                "\(prefix).o_proj",
                leading: [hidden], inFeatures: PinnedGeometry.attentionDim)
        }
        /// One block of either tower, selected by the published schedule.
        func addMixer(_ prefix: String, kind: LayerKind) {
            switch kind {
            case .mamba: addMambaMixer("\(prefix).mixer")
            case .moe: addMoEMixer("\(prefix).mixer")
            case .attention: addAttentionMixer("\(prefix).mixer")
            }
            add("\(prefix).norm.weight", .bf16, [hidden])
        }

        addAffine(
            "\(backbonePrefix).embeddings", leading: [vocab],
            inFeatures: hidden)
        add("\(backbonePrefix).norm_f.weight", .bf16, [hidden])
        // The head is UNTIED: `tie_word_embeddings` is false.
        addAffine("lm_head", leading: [vocab], inFeatures: hidden)

        for (layerIndex, kind) in PinnedGeometry.layersBlockType.enumerated() {
            addMixer("\(layerPrefix)\(layerIndex)", kind: kind)
        }

        // The head EMBEDDED in this checkpoint. It has no embedding table and
        // no output projection of its own; it rides the target's.
        for (layerIndex, kind) in PinnedGeometry.mtpLayersBlockType.enumerated() {
            let prefix = "\(mtpPrefix).layers.\(layerIndex)"
            addMixer(prefix, kind: kind)
            if layerIndex == 0 {
                // The FIRST prediction block joins the embedding of the
                // proposed token to the target's hidden state, so it carries
                // the two input norms and the joint projection.
                for norm in ["enorm", "hnorm"] {
                    add("\(prefix).\(norm).weight", .bf16, [hidden])
                }
                addAffine(
                    "\(prefix).eh_proj", leading: [hidden],
                    inFeatures: 2 * hidden)
            }
            if layerIndex == PinnedGeometry.mtpLayersBlockType.count - 1 {
                // The LAST prediction block normalizes before the shared head.
                add("\(prefix).final_layernorm.weight", .bf16, [hidden])
            }
        }

        precondition(
            inventory.count == expectedTensorCount,
            "Nemotron 3.5 inventory must contain \(expectedTensorCount) "
                + "tensors, built \(inventory.count)"
        )
        return inventory
    }

    static func validateSelectedTensors(
        selectedKeys: Set<String>,
        index: CheckpointIndex,
        headers: [String: SafetensorsHeader],
        quantization: NemotronHTransformQuantizationSpec
    ) throws {
        // Affine quantization legitimately ships `.biases`, so that suffix is
        // NOT forbidden here.
        let forbiddenSuffixes = [
            ".weight_packed",
            ".input_global_scale",
            ".weight_global_scale",
            ".k_scale",
            ".v_scale",
        ]
        if let forbiddenName = selectedKeys.sorted().first(where: { name in
            forbiddenSuffixes.contains { suffix in name.hasSuffix(suffix) }
        }) {
            throw MLXFastError.invalidInput(
                "Nemotron 3.5 MLX transform rejects compressed-tensors/"
                    + "global-scale and FP8 KV-scale tensor \(forbiddenName)"
            )
        }
        // The head is UNTIED on this target, so `lm_head.*` is a real triple
        // the transform MUST select.
        for component in ["weight", "scales", "biases"] {
            let name = "lm_head.\(component)"
            guard selectedKeys.contains(name) else {
                throw MLXFastError.invalidInput(
                    "Nemotron 3.5 has an untied output head and must select "
                        + "\(name)"
                )
            }
        }
        // The multi-token-prediction head is EMBEDDED in this checkpoint, so
        // the transform MUST select it: there is no separate head artifact,
        // and a tree without `mtp.*` has no speculative arm at all.
        let mtpSelected = selectedKeys.filter { name in
            name.hasPrefix("\(mtpPrefix).")
        }
        guard mtpSelected.count == expectedMTPTensorCount else {
            throw MLXFastError.invalidInput(
                "Nemotron 3.5 transform selected \(mtpSelected.count) embedded "
                    + "multi-token-prediction tensors, expected "
                    + "\(expectedMTPTensorCount)"
            )
        }

        for name in selectedKeys.sorted() where name.hasSuffix(".weight") {
            let stem = String(name.dropLast(".weight".count))
            let scalesName = "\(stem).scales"
            let biasesName = "\(stem).biases"
            guard selectedKeys.contains(scalesName)
                    || selectedKeys.contains(biasesName)
            else {
                continue
            }
            guard selectedKeys.contains(scalesName),
                  selectedKeys.contains(biasesName)
            else {
                throw MLXFastError.invalidInput(
                    "Nemotron 3.5 affine projection \(stem) must ship both "
                        + ".scales and .biases"
                )
            }
            let weightInfo = try tensorInfo(
                named: name, index: index, headers: headers)
            let scalesInfo = try tensorInfo(
                named: scalesName, index: index, headers: headers)
            let biasesInfo = try tensorInfo(
                named: biasesName, index: index, headers: headers)
            // Rank is >= 2 rather than == 2: the stacked expert tensors are
            // rank 3 (experts x out x packed-in).
            guard weightInfo.dtype == TensorDType.u32.rawValue,
                  scalesInfo.dtype == TensorDType.bf16.rawValue,
                  biasesInfo.dtype == TensorDType.bf16.rawValue,
                  weightInfo.shape.count >= 2,
                  scalesInfo.shape == biasesInfo.shape,
                  scalesInfo.shape.count == weightInfo.shape.count,
                  weightInfo.shape.dropLast() == scalesInfo.shape.dropLast(),
                  weightInfo.shape.allSatisfy({ $0 > 0 }),
                  scalesInfo.shape.allSatisfy({ $0 > 0 })
            else {
                throw MLXFastError.invalidInput(
                    "Nemotron 3.5 affine projection \(stem) has incompatible "
                        + "weight, scale, or bias metadata"
                )
            }

            guard let packedWidth = weightInfo.shape.last,
                  let groupCount = scalesInfo.shape.last
            else {
                throw MLXFastError.invalidInput(
                    "Nemotron 3.5 affine projection \(stem) has no packed axis"
                )
            }
            let (inputFeatures, inputOverflow) =
                groupCount.multipliedReportingOverflow(by: quantization.groupSize)
            let (packedBits, packedOverflow) =
                packedWidth.multipliedReportingOverflow(by: 32)
            guard !inputOverflow, !packedOverflow else {
                throw MLXFastError.invalidInput(
                    "Nemotron 3.5 projection \(stem) packed width overflows Int"
                )
            }
            let (expectedPackedBits, expectedOverflow) =
                inputFeatures.multipliedReportingOverflow(by: quantization.bits)
            guard !expectedOverflow, packedBits == expectedPackedBits else {
                throw MLXFastError.invalidInput(
                    "quantized Nemotron 3.5 projection \(stem) stored width "
                        + "\(packedWidth) does not match config quantization "
                        + "group_size \(quantization.groupSize) bits "
                        + "\(quantization.bits) for input dimension "
                        + "\(inputFeatures)"
                )
            }
        }

        try validateExactPublicInventory(
            selectedKeys: selectedKeys,
            index: index,
            headers: headers
        )
    }

    static func validateExactPublicInventory(
        selectedKeys: Set<String>,
        index: CheckpointIndex,
        headers: [String: SafetensorsHeader]
    ) throws {
        let expected = expectedTensorInventory()
        let expectedNames = Set(expected.keys)
        // This checkpoint is text-only: the index carries nothing the
        // transform drops, so the selected set, the index and the header names
        // are the same set. Every selected name must be indexed exactly once
        // and present in exactly one header.
        let headerNameList = headers.values.flatMap { $0.tensors.keys }
        let headerNames = Set(headerNameList)

        guard selectedKeys == expectedNames,
              expectedNames.isSubset(of: Set(index.weightMap.keys)),
              expectedNames.isSubset(of: headerNames),
              headerNameList.count == headerNames.count
        else {
            let missing = expectedNames.subtracting(selectedKeys).sorted()
            let extra = selectedKeys.subtracting(expectedNames).sorted()
            let unindexed = expectedNames
                .subtracting(Set(index.weightMap.keys)).sorted()
            throw MLXFastError.invalidInput(
                "Nemotron 3.5 checkpoint tensor inventory must match the exact "
                    + "public \(expectedTensorCount)-tensor contract "
                    + "(missing: \(missing.prefix(8).joined(separator: ", ")); "
                    + "extra: \(extra.prefix(8).joined(separator: ", ")); "
                    + "unindexed/duplicate header tensors: "
                    + "\(unindexed.prefix(8).joined(separator: ", ")))"
            )
        }

        for name in expected.keys.sorted() {
            guard let expectedMetadata = expected[name] else {
                preconditionFailure(
                    "missing expected Nemotron 3.5 metadata for \(name)")
            }
            let actual = try tensorInfo(
                named: name, index: index, headers: headers)
            guard actual.dtype == expectedMetadata.dtype,
                  actual.shape == expectedMetadata.shape
            else {
                throw MLXFastError.invalidInput(
                    "Nemotron 3.5 tensor \(name) dtype/shape \(actual.dtype) "
                        + "\(actual.shape) does not match exact public "
                        + "metadata \(expectedMetadata.dtype) "
                        + "\(expectedMetadata.shape)"
                )
            }
        }
    }

    private static func tensorInfo(
        named name: String,
        index: CheckpointIndex,
        headers: [String: SafetensorsHeader]
    ) throws -> SafetensorInfo {
        guard let shardName = index.weightMap[name],
              let info = headers[shardName]?.tensors[name]
        else {
            throw MLXFastError.invalidInput(
                "missing validated tensor metadata for \(name)")
        }
        return info
    }

    private static func intField(
        _ key: String, in object: [String: Any]
    ) throws -> Int {
        guard let number = object[key] as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID(),
              !CFNumberIsFloatType(number),
              let integer = Int(number.stringValue)
        else {
            throw MLXFastError.invalidInput(
                "Nemotron 3.5 quantization field \(key) must be a finite "
                    + "integer in Int range"
            )
        }
        return integer
    }

    private static func stringField(
        _ key: String, in object: [String: Any]
    ) throws -> String {
        guard let string = object[key] as? String else {
            throw MLXFastError.invalidInput(
                "Nemotron 3.5 quantization field \(key) must be a string"
            )
        }
        return string
    }
}
