import Foundation
import MLXFastCore
@testable import MLXFastTransform
import Testing

// Transform-side conformance for the Nemotron 3.5 Lightning 30B-A3B family:
// family detection against the legacy family in this tree, the uniform
// quantization contract, and the exact 763-tensor inventory.
//
// THE INVENTORY IS PINNED TWICE AND THE TWO ARE COMPARED. The validator
// DERIVES its expectations from the pinned geometry; the fixture
// `fixtures/nemotron3_5_30b_a3b_tensor_inventory.json` RECORDS what the
// published checkpoint holds. `theDerivedInventoryIsThePublishedInventory`
// below holds all 763 (name, dtype, shape) triples of the one against the
// other. A derivation that reproduced the artifact it reads would prove
// nothing, so the derivation reads no fixture at all -- it is arithmetic over
// the geometry block, and the fixture is the independent witness.

private typealias Geometry = NemotronHCheckpointValidation.PinnedGeometry

// MARK: - The derivation against the published inventory

/// THE HEADLINE TEST. Every derived tensor is a published tensor, at the
/// published dtype and the published shape, and there is no published tensor
/// the derivation missed.
@Test
func theDerivedInventoryIsThePublishedInventory() throws {
    let derived = NemotronHCheckpointValidation.expectedTensorInventory()
    let published = try nemotron35InventoryFixture().tensors

    #expect(derived.count == NemotronHCheckpointValidation.expectedTensorCount)
    #expect(derived.count == 763)
    #expect(published.count == derived.count)

    let missing = Set(published.keys).subtracting(derived.keys).sorted()
    let extra = Set(derived.keys).subtracting(published.keys).sorted()
    #expect(missing.isEmpty, "the derivation misses: \(missing.prefix(8))")
    #expect(extra.isEmpty, "the derivation invents: \(extra.prefix(8))")

    for name in published.keys.sorted() {
        guard let publishedRecord = published[name], let derivedRecord = derived[name] else {
            continue
        }
        #expect(
            derivedRecord.dtype == publishedRecord.dtype,
            Comment(rawValue: "\(name) dtype"))
        #expect(
            derivedRecord.shape == publishedRecord.shape,
            Comment(rawValue: "\(name) shape"))
    }
}

/// The dtype census is a second, independent reading of the same fact: a
/// derivation that got one family's element type wrong would move these
/// counts even if every name still matched.
@Test
func theDerivedInventoryReproducesThePublishedDtypeCensus() throws {
    let derived = NemotronHCheckpointValidation.expectedTensorInventory()
    let summary = try nemotron35InventoryFixture().summary

    var counts: [String: Int] = [:]
    for metadata in derived.values {
        counts[metadata.dtype, default: 0] += 1
    }
    #expect(counts == summary.dtypeCounts)
    // 173 packed weights, their 346 BF16 companions plus 220 other BF16
    // parameters, and 24 F32 router correction vectors.
    #expect(counts["U32"] == 173)
    #expect(counts["BF16"] == 566)
    #expect(counts["F32"] == 24)
}

/// The counts the validator publishes are the counts its own derivation
/// produces, kind by kind, and they sum to the pinned total.
@Test
func theDerivedInventoryMatchesThePinnedPerKindCounts() {
    let inventory = NemotronHCheckpointValidation.expectedTensorInventory()

    let topLevel = inventory.keys.filter {
        !$0.hasPrefix("backbone.layers.") && !$0.hasPrefix("mtp.")
    }
    #expect(
        topLevel.count
            == NemotronHCheckpointValidation.expectedTopLevelTensorCount)
    #expect(
        Set(topLevel) == [
            "backbone.embeddings.weight",
            "backbone.embeddings.scales",
            "backbone.embeddings.biases",
            "backbone.norm_f.weight",
            "lm_head.weight",
            "lm_head.scales",
            "lm_head.biases",
        ])

    var perKind: [NemotronHCheckpointValidation.LayerKind: Int] = [:]
    for (layerIndex, kind) in Geometry.layersBlockType.enumerated() {
        let prefix = "backbone.layers.\(layerIndex)."
        let count = inventory.keys.filter { $0.hasPrefix(prefix) }.count
        let expected: Int
        switch kind {
        case .mamba:
            expected = NemotronHCheckpointValidation.expectedMambaLayerTensorCount
        case .moe:
            expected = NemotronHCheckpointValidation.expectedMoELayerTensorCount
        case .attention:
            expected = NemotronHCheckpointValidation.expectedAttentionLayerTensorCount
        }
        #expect(count == expected, Comment(rawValue: "layer \(layerIndex)"))
        perKind[kind, default: 0] += 1
    }
    #expect(perKind[.mamba] == 23)
    #expect(perKind[.moe] == 23)
    #expect(perKind[.attention] == 6)

    let mtp = inventory.keys.filter { $0.hasPrefix("mtp.") }
    #expect(mtp.count == NemotronHCheckpointValidation.expectedMTPTensorCount)

    #expect(
        NemotronHCheckpointValidation.expectedTopLevelTensorCount
            + 23 * NemotronHCheckpointValidation.expectedMambaLayerTensorCount
            + 23 * NemotronHCheckpointValidation.expectedMoELayerTensorCount
            + 6 * NemotronHCheckpointValidation.expectedAttentionLayerTensorCount
            + NemotronHCheckpointValidation.expectedMTPTensorCount
            == NemotronHCheckpointValidation.expectedTensorCount)
}

/// Presence and absence are both facts of this checkpoint, asserted as such
/// rather than as tolerated extras.
@Test
func theDerivedInventoryEncodesTheUntiedHeadAndTheHybridSchedule() {
    let inventory = NemotronHCheckpointValidation.expectedTensorInventory()

    // The head is UNTIED, so it ships its own triple, and the embedded
    // speculative head ships NO embedding and NO head of its own.
    #expect(inventory["lm_head.weight"] != nil)
    #expect(inventory["mtp.embeddings.weight"] == nil)
    #expect(inventory["mtp.lm_head.weight"] == nil)
    // There IS a final norm on this tower, unlike the tree this repository was
    // seeded from.
    #expect(inventory["backbone.norm_f.weight"] != nil)

    for (layerIndex, kind) in Geometry.layersBlockType.enumerated() {
        let prefix = "backbone.layers.\(layerIndex).mixer"
        let comment = Comment(rawValue: "layer \(layerIndex)")
        #expect((inventory["\(prefix).A_log"] != nil) == (kind == .mamba), comment)
        #expect(
            (inventory["\(prefix).switch_mlp.fc1.weight"] != nil) == (kind == .moe),
            comment)
        #expect(
            (inventory["\(prefix).q_proj.weight"] != nil) == (kind == .attention),
            comment)
        // Every layer carries its own norm, whichever kind it is.
        #expect(inventory["backbone.layers.\(layerIndex).norm.weight"] != nil, comment)
    }

    // The head's first block joins the proposed token's embedding to the
    // target's hidden state; only the first block carries that machinery, and
    // only the last carries the final norm.
    #expect(inventory["mtp.layers.0.eh_proj.weight"] != nil)
    #expect(inventory["mtp.layers.0.enorm.weight"] != nil)
    #expect(inventory["mtp.layers.0.hnorm.weight"] != nil)
    #expect(inventory["mtp.layers.1.eh_proj.weight"] == nil)
    #expect(inventory["mtp.layers.1.final_layernorm.weight"] != nil)
    #expect(inventory["mtp.layers.0.final_layernorm.weight"] == nil)
}

/// The packing arithmetic, spelled out on one tensor of each family: hidden
/// 2688 packs to 2688 * 4 / 32 = 336 U32 columns with 2688 / 64 = 42 group
/// columns.
@Test
func theDerivedInventoryPacksEachPathAtTheUniformWidth() {
    let inventory = NemotronHCheckpointValidation.expectedTensorInventory()
    func shape(_ name: String) -> [Int] { inventory[name]?.shape ?? [] }

    #expect(shape("backbone.embeddings.weight") == [131_072, 336])
    #expect(shape("backbone.embeddings.scales") == [131_072, 42])
    #expect(shape("lm_head.weight") == [131_072, 336])

    // A Mamba-2 layer. The inner stream is 64 heads x 64 = 4096; the fused
    // convolution adds the two 8-group, 128-wide SSM states; the input
    // projection adds a second inner stream and one time step per head.
    #expect(Geometry.mambaIntermediateSize == 4_096)
    #expect(Geometry.mambaConvDim == 6_144)
    #expect(Geometry.mambaInputProjectionDim == 10_304)
    #expect(shape("backbone.layers.0.mixer.conv1d.weight") == [6_144, 4, 1])
    #expect(shape("backbone.layers.0.mixer.conv1d.bias") == [6_144])
    #expect(shape("backbone.layers.0.mixer.in_proj.weight") == [10_304, 336])
    #expect(shape("backbone.layers.0.mixer.out_proj.weight") == [2_688, 512])
    #expect(shape("backbone.layers.0.mixer.A_log") == [64])
    #expect(shape("backbone.layers.0.mixer.norm.weight") == [4_096])

    // A mixture-of-experts layer. Experts are STACKED behind a leading experts
    // axis, the router stays unquantized, and its correction vector is F32.
    #expect(shape("backbone.layers.1.mixer.switch_mlp.fc1.weight") == [128, 1_856, 336])
    #expect(shape("backbone.layers.1.mixer.switch_mlp.fc2.weight") == [128, 2_688, 232])
    #expect(shape("backbone.layers.1.mixer.gate.weight") == [128, 2_688])
    #expect(inventory["backbone.layers.1.mixer.gate.e_score_correction_bias"]?.dtype == "F32")
    #expect(shape("backbone.layers.1.mixer.shared_experts.up_proj.weight") == [3_712, 336])
    #expect(shape("backbone.layers.1.mixer.shared_experts.down_proj.weight") == [2_688, 464])

    // A full-attention layer: 32 query heads and 2 key/value heads, 128 wide.
    #expect(shape("backbone.layers.5.mixer.q_proj.weight") == [4_096, 336])
    #expect(shape("backbone.layers.5.mixer.k_proj.weight") == [256, 336])
    #expect(shape("backbone.layers.5.mixer.v_proj.weight") == [256, 336])
    #expect(shape("backbone.layers.5.mixer.o_proj.weight") == [2_688, 512])
    // No query or key norms on this target.
    #expect(inventory["backbone.layers.5.mixer.q_norm.weight"] == nil)

    // The head's joint projection reads embedding and hidden state side by
    // side, so it is twice hidden wide.
    #expect(shape("mtp.layers.0.eh_proj.weight") == [2_688, 672])
    #expect(shape("mtp.layers.0.eh_proj.scales") == [2_688, 84])
}

/// The trusted constants and the transform's own geometry describe ONE
/// checkpoint. They are deliberately duplicated -- `MLXFastCore` cannot import
/// the transform -- so the duplication is checked rather than trusted.
@Test
func thePinnedGeometryAgreesWithTheTrustedConstantsAndTheConfig() throws {
    #expect(Geometry.vocabSize == MLXFastConstants.vocabSize)
    #expect(Geometry.hiddenSize == MLXFastConstants.hiddenSize)
    #expect(Geometry.layerCount == MLXFastConstants.numHiddenLayers)
    #expect(Geometry.layersBlockType.count == Geometry.layerCount)

    let config = try nemotronConfigObject()
    #expect(config["vocab_size"] as? Int == Geometry.vocabSize)
    #expect(config["hidden_size"] as? Int == Geometry.hiddenSize)
    #expect(config["num_hidden_layers"] as? Int == Geometry.layerCount)
    #expect(config["num_attention_heads"] as? Int == Geometry.attentionHeads)
    #expect(config["num_key_value_heads"] as? Int == Geometry.keyValueHeads)
    #expect(config["head_dim"] as? Int == Geometry.headDim)
    #expect(config["mamba_num_heads"] as? Int == Geometry.mambaHeads)
    #expect(config["mamba_head_dim"] as? Int == Geometry.mambaHeadDim)
    #expect(config["ssm_state_size"] as? Int == Geometry.ssmStateSize)
    // `n_groups` is the SSM group count. `n_group` is the mixture-of-experts
    // group count and is 1; reading the wrong one would widen the convolution
    // by a factor of eight.
    #expect(config["n_groups"] as? Int == Geometry.ssmGroupCount)
    #expect(config["n_group"] as? Int == 1)
    #expect(config["conv_kernel"] as? Int == Geometry.convKernel)
    #expect(config["n_routed_experts"] as? Int == Geometry.routedExpertCount)
    #expect(config["moe_intermediate_size"] as? Int == Geometry.moeIntermediateSize)
    #expect(
        config["moe_shared_expert_intermediate_size"] as? Int
            == Geometry.sharedExpertIntermediateSize)
    #expect(
        config["num_nextn_predict_layers"] as? Int == Geometry.mtpPredictLayerCount)
    #expect(
        config["layers_block_type"] as? [String]
            == Geometry.layersBlockType.map(\.rawValue))
    #expect(
        config["mtp_layers_block_type"] as? [String]
            == Geometry.mtpLayersBlockType.map(\.rawValue))
    #expect(config["tie_word_embeddings"] as? Bool == false)
    #expect(config["use_conv_bias"] as? Bool == true)
}

// MARK: - Family detection

@Test
func nemotronIsDetectedByItsFlatModelType() throws {
    #expect(
        try SwiftTransform.detectModelFamily(
            sourceConfigRoot: ["model_type": "nemotron_h"]
        ) == .nemotronH
    )
    // The legacy Gemma family declares a nested text_config and routes there.
    #expect(
        try SwiftTransform.detectModelFamily(
            sourceConfigRoot: ["text_config": ["model_type": "gemma4_text"]]
        ) == .gemma4
    )
    // A config that is neither is refused rather than guessed at.
    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.detectModelFamily(sourceConfigRoot: ["model_type": "laguna"])
    }
    #expect(throws: MLXFastError.self) {
        _ = try SwiftTransform.detectModelFamily(sourceConfigRoot: [:])
    }
}

// MARK: - The quantization block

private func pinnedQuantizationBlock() -> [String: Any] {
    [
        "group_size": Geometry.quantizationGroupSize,
        "bits": Geometry.quantizationBits,
        "mode": Geometry.quantizationMode,
    ]
}

private func pinnedSourceConfig(
    quantization: [String: Any]? = nil,
    quantizationConfig: [String: Any]? = nil
) -> [String: Any] {
    let block = quantization ?? pinnedQuantizationBlock()
    return [
        "model_type": "nemotron_h",
        "num_hidden_layers": Geometry.layerCount,
        "quantization": block,
        "quantization_config": quantizationConfig ?? block,
    ]
}

@Test
func quantizationSpecParsesTheUniformBlock() throws {
    let spec = try NemotronHCheckpointValidation.quantizationSpec(
        fromConfigRoot: pinnedSourceConfig())
    #expect(spec.groupSize == 64)
    #expect(spec.bits == 4)
    #expect(spec.mode == "affine")
}

@Test
func quantizationSpecRequiresTheDuplicateBlocksToAgree() throws {
    var divergent = pinnedQuantizationBlock()
    divergent["bits"] = 8
    #expect(throws: (any Error).self) {
        _ = try NemotronHCheckpointValidation.quantizationSpec(
            fromConfigRoot: pinnedSourceConfig(quantizationConfig: divergent))
    }
    // One block alone is accepted -- the transform must not demand a duplicate
    // the checkpoint may legitimately not publish.
    var single = pinnedSourceConfig()
    single.removeValue(forKey: "quantization_config")
    #expect(throws: Never.self) {
        _ = try NemotronHCheckpointValidation.quantizationSpec(fromConfigRoot: single)
    }
    // And a config with neither block names no quantization at all.
    var neither = pinnedSourceConfig()
    neither.removeValue(forKey: "quantization")
    neither.removeValue(forKey: "quantization_config")
    #expect(throws: (any Error).self) {
        _ = try NemotronHCheckpointValidation.quantizationSpec(fromConfigRoot: neither)
    }
}

/// This target is UNIFORM, so the whole override surface is a REFUSAL surface:
/// an entry the emitter carried would be a width the runtime never reads.
@Test
func quantizationSpecRejectsEveryShapeOfBlockDrift() {
    func rejected(_ mutate: (inout [String: Any]) -> Void) -> Bool {
        var block = pinnedQuantizationBlock()
        mutate(&block)
        do {
            _ = try NemotronHCheckpointValidation.quantizationSpec(
                fromConfigRoot: pinnedSourceConfig(
                    quantization: block, quantizationConfig: block))
            return false
        } catch {
            return true
        }
    }

    // A per-tensor override of ANY shape, which this target does not have.
    #expect(
        rejected {
            $0["backbone.layers.3.mixer.switch_mlp.fc1"] = ["group_size": 64, "bits": 8]
        })
    #expect(
        rejected {
            $0["backbone.layers.3.mixer.switch_mlp.fc1"] = ["group_size": 64, "bits": 4]
        })
    // A non-object value under a tensor path is still an unexpected key.
    #expect(rejected { $0["backbone.layers.3.mixer.out_proj"] = 8 })
    // A changed fallback: this checkpoint is affine 4-bit group-64.
    #expect(rejected { $0["bits"] = 8 })
    #expect(rejected { $0["group_size"] = 32 })
    #expect(rejected { $0["mode"] = "nvfp4" })
    // A missing scalar is a refusal, not a default.
    #expect(rejected { $0.removeValue(forKey: "mode") })
    // And a scalar of the wrong JSON kind is not coerced.
    #expect(rejected { $0["bits"] = "4" })
}

// MARK: - The emitted runtime config

/// The flat source config is passed through whole, minus the duplicate
/// quantization block. The engine's MTP loader re-decodes the emitted file and
/// requires it to encode identically to the loaded target, so a dropped field
/// would refuse the embedded head.
@Test
func emittedRuntimeConfigPassesTheFlatSourceThroughAndDedupsQuantization() throws {
    let source = pinnedSourceConfig()
    let data = try SwiftTransform.makeRuntimeConfigData(
        sourceConfigRoot: source, family: .nemotronH)
    let emitted = try #require(
        try JSONSerialization.jsonObject(with: data) as? [String: Any])

    #expect(emitted["model_type"] as? String == "nemotron_h")
    #expect(emitted["num_hidden_layers"] as? Int == Geometry.layerCount)
    #expect(emitted["text_config"] == nil)
    #expect(emitted["quantization_config"] == nil)
    #expect(Set(emitted.keys) == Set(source.keys).subtracting(["quantization_config"]))

    let quantization = try #require(emitted["quantization"] as? [String: Any])
    #expect(quantization["group_size"] as? Int == 64)
    #expect(quantization["bits"] as? Int == 4)
    #expect(quantization["mode"] as? String == "affine")
    #expect(Set(quantization.keys) == ["group_size", "bits", "mode"])
}

// MARK: - validateSelectedTensors against a synthetic exact checkpoint

private func nemotronValidationMetadata(
    mutate: (inout [String: NemotronHCheckpointValidation.ExpectedTensorMetadata])
        -> Void = { _ in }
) -> (
    selectedKeys: Set<String>, index: CheckpointIndex,
    headers: [String: SafetensorsHeader]
) {
    var inventory = NemotronHCheckpointValidation.expectedTensorInventory()
    mutate(&inventory)

    let shardName = "model-00001-of-00001.safetensors"
    var weightMap: [String: String] = [:]
    var tensors: [String: SafetensorInfo] = [:]
    for (name, metadata) in inventory {
        weightMap[name] = shardName
        tensors[name] = SafetensorInfo(
            name: name,
            dtype: metadata.dtype,
            shape: metadata.shape,
            dataStart: 0,
            dataEnd: 1
        )
    }
    return (
        Set(inventory.keys),
        CheckpointIndex(raw: ["weight_map": weightMap], weightMap: weightMap),
        [
            shardName: SafetensorsHeader(
                headerLength: 8, metadata: ["format": "pt"], tensors: tensors)
        ]
    )
}

@Test
func transformAcceptsTheExactPublicNemotronInventory() throws {
    let metadata = nemotronValidationMetadata()
    #expect(
        metadata.selectedKeys.count
            == NemotronHCheckpointValidation.expectedTensorCount)
    try NemotronHCheckpointValidation.validateSelectedTensors(
        selectedKeys: metadata.selectedKeys,
        index: metadata.index,
        headers: metadata.headers,
        quantization: try NemotronHCheckpointValidation.quantizationSpec(
            fromConfigRoot: pinnedSourceConfig())
    )
}

@Test
func transformRejectsNemotronInventoryAndPackingDrift() throws {
    let quantization = try NemotronHCheckpointValidation.quantizationSpec(
        fromConfigRoot: pinnedSourceConfig())

    func rejected(
        _ mutate: (
            inout [String: NemotronHCheckpointValidation.ExpectedTensorMetadata]
        ) -> Void
    ) -> Bool {
        let metadata = nemotronValidationMetadata(mutate: mutate)
        do {
            try NemotronHCheckpointValidation.validateSelectedTensors(
                selectedKeys: metadata.selectedKeys,
                index: metadata.index,
                headers: metadata.headers,
                quantization: quantization
            )
            return false
        } catch {
            return true
        }
    }

    let quantized = NemotronHCheckpointValidation.ExpectedTensorMetadata.self
    // A packed weight stored at the wrong width. The name and the leading axis
    // are right and only the packed column count is wrong, which is what a
    // width mistake looks like.
    #expect(
        rejected {
            $0["backbone.layers.1.mixer.switch_mlp.fc1.weight"] =
                quantized.init(dtype: "U32", shape: [128, 1_856, 672])
        })
    // A MAMBA layer that ships attention projections after all: the layer
    // schedule is a tensor-inventory fact, not only a numerics one.
    #expect(
        rejected {
            for component in ["weight", "scales", "biases"] {
                $0["backbone.layers.0.mixer.q_proj.\(component)"] =
                    quantized.init(
                        dtype: component == "weight" ? "U32" : "BF16",
                        shape: component == "weight" ? [4_096, 336] : [4_096, 42])
            }
        })
    // An attention layer that drops a projection it must ship.
    #expect(rejected { $0.removeValue(forKey: "backbone.layers.5.mixer.o_proj.weight") })
    // The untied output head DROPPED. This target ships one, so its absence is
    // the error.
    #expect(rejected { $0.removeValue(forKey: "lm_head.weight") })
    // The embedded multi-token-prediction head DROPPED. There is no separate
    // head artifact on this track, so a tree without it has no speculative arm.
    #expect(rejected { $0.removeValue(forKey: "mtp.layers.0.eh_proj.weight") })
    // Compressed-tensors and FP8 KV-scale aliases.
    #expect(
        rejected {
            $0["backbone.layers.5.mixer.k_proj.k_scale"] =
                quantized.init(dtype: "BF16", shape: [1])
        })
    // A quantized projection missing its bias companion (affine needs both).
    #expect(rejected { $0.removeValue(forKey: "backbone.layers.5.mixer.q_proj.biases") })
    // Wrong dtype on a packed weight.
    #expect(
        rejected {
            $0["backbone.layers.5.mixer.q_proj.weight"] =
                quantized.init(dtype: "BF16", shape: [4_096, 336])
        })
    // Wrong dtype on the router's correction vector: it is F32, not BF16.
    #expect(
        rejected {
            $0["backbone.layers.1.mixer.gate.e_score_correction_bias"] =
                quantized.init(dtype: "BF16", shape: [128])
        })
    // Wrong shape on an unquantized norm.
    #expect(
        rejected {
            $0["backbone.layers.0.mixer.norm.weight"] =
                quantized.init(dtype: "BF16", shape: [2_688])
        })
    // The final norm DROPPED. This tower has one.
    #expect(rejected { $0.removeValue(forKey: "backbone.norm_f.weight") })
    // A stacked expert tensor flattened into rank 2.
    #expect(
        rejected {
            $0["backbone.layers.1.mixer.switch_mlp.fc1.weight"] =
                quantized.init(dtype: "U32", shape: [237_568, 336])
        })
}
