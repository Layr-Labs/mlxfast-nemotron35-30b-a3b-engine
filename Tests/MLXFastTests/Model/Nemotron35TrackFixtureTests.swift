import Foundation
import MLXFastCore
import Testing

// Contract tests for `fixtures/nemotron3_5_30b_a3b_mlx_v1_track.json`.
//
// The track IS ARMED AND PINS ITS GOLDENS. The timed pool holds the
// three live coding prompts, `live_goldens` names them, every MTP depth maps
// each of them to its oracle pin and the hidden oracle is the first live
// golden. `official_scoring_enabled` is true: it flipped in its own change,
// after one clean scored window. The goldens are organizer material,
// recorded on the ranked box and never in git. This suite asserts the pinned
// state as a COHERENT WHOLE, and it asserts the facts that do not wait for
// goldens: the target block, the tensor inventory, the reference manifest and
// the compiled constants all describing one checkpoint.
//
// This is a laptop-side JSON-shape test only. It does not exercise benchd's
// Rust contract parsers; those live in the bench repository.

@Suite("Nemotron 3.5 Lightning 30B-A3B track contract fixture")
struct Nemotron35TrackFixtureTests {

    /// The sentinel every unarmed organizer slot in this fixture carries.
    /// EXACT-MATCH ONLY -- never a prefix check.
    static let pendingOrganizerSentinel = "NEMOTRON3-5-30B-A3B-MLX-V1-PENDING-ORGANIZER"

    @Test("fixture parses as a JSON object")
    func fixtureParses() throws {
        let object = try nemotron35TrackContractObject()
        #expect(object["schema_version"] as? Int == 1)
    }

    @Test("track_id is pinned to the leaderboard / R2-prefix identity")
    func trackIdIsPinned() throws {
        let object = try nemotron35TrackContractObject()
        #expect(object["track_id"] as? String == "nemotron3.5-30b-a3b-mlx-v1")
    }

    @Test("benchmark_name matches the Yukon manifest name")
    func benchmarkNameMatchesManifest() throws {
        let object = try nemotron35TrackContractObject()
        #expect(object["benchmark_name"] as? String == "mlxfast-nemotron35-30b-a3b")
    }

    @Test("track_id is substring-clean against every retired MTP name")
    func trackIdIsCleanAgainstRetiredNames() throws {
        let object = try nemotron35TrackContractObject()
        let trackId = try #require(object["track_id"] as? String)
        let retired = [
            "MLXFAST_MTP_",
            "mtp-ranked",
            "measure-mtp-job",
            "mtp-weights",
            "laguna-xs-2.1-mtp",
            "gemma4-31b-it",
        ]
        for name in retired {
            #expect(
                !trackId.contains(name),
                "track_id must not substring-collide with retired name \(name)")
            #expect(
                !name.contains(trackId),
                "retired name \(name) must not substring-collide with track_id")
        }
    }

    /// THE PINNED STATE IS ONE STATE, so it is asserted in one test. Three
    /// distinct pool pins, a live list that names each of them once, a hidden
    /// oracle equal to the first live golden's pin and one oracle pin per live
    /// golden for every permitted MTP depth must travel together: any of them
    /// missing is a fixture that would let a ranked run start against
    /// material it does not have. Scoring is armed.
    @Test("the fixture is coherently armed: scoring, pool, live goldens, oracles")
    func theFixtureIsCoherentlyArmed() throws {
        let object = try nemotron35TrackContractObject()
        let prefix = "correctness_prompts/nemotron3.5-30b-a3b-mlx-v1/"
        func isPin(_ pin: [String: Any]) -> Bool {
            guard let path = pin["r2_path"] as? String, let sha256 = pin["sha256"] as? String,
                let bytes = pin["bytes"] as? Int
            else { return false }
            return path.hasPrefix(prefix) && path.hasSuffix(".golden.json") && sha256.count == 64
                && sha256.allSatisfy { "0123456789abcdef".contains($0) } && bytes > 0
        }
        #expect(object["official_scoring_enabled"] as? Bool == true)
        let pool = try #require(object["timed_prompt_pool"] as? [[String: Any]])
        #expect(pool.count == 3)
        #expect(pool.allSatisfy(isPin))
        #expect(Set(pool.compactMap { $0["sha256"] as? String }).count == pool.count)
        #expect(Set(pool.compactMap { $0["r2_path"] as? String }).count == pool.count)
        let live = try #require(object["live_goldens"] as? [String])
        #expect(live == ["pool-14", "pool-15", "pool-16"])
        #expect(object["live_golden"] == nil)
        let liveRows = try live.map { name in
            try #require(pool.first { ($0["r2_path"] as? String) == "\(prefix)\(name).golden.json" })
        }
        let pairs = try #require(object["official_pairs"] as? Int)
        #expect(pairs % live.count == 0)
        #expect(object["official_pair_combine"] as? String == "mean")
        // ROOT level -- benchd's `hidden_correctness_golden_pin_from_contract`
        // reads this key directly off the contract root, never a nested
        // wrapper.
        let golden = try #require(object["hidden_correctness_golden"] as? [String: Any])
        let firstRow = try #require(liveRows.first)
        #expect(golden["sha256"] as? String == firstRow["sha256"] as? String)
        #expect(golden["bytes"] as? Int == firstRow["bytes"] as? Int)
        #expect(golden["sha256"] as? String != Self.pendingOrganizerSentinel)
        #expect(object["hidden_material"] == nil)
        // Each depth's oracle for a prompt is that prompt's serial golden: the
        // MTP head only proposes tokens, so every depth emits the serial tape.
        let oracles = try #require(
            object["live_golden_speculative"] as? [String: [String: [String: Any]]])
        #expect(Set(oracles.keys) == Set((1...7).map { "mtp\($0)" }))
        for perPrompt in oracles.values {
            #expect(Set(perPrompt.keys) == Set(live))
            for (name, row) in zip(live, liveRows) {
                let pin = try #require(perPrompt[name])
                #expect(isPin(pin))
                #expect(pin["sha256"] as? String == row["sha256"] as? String)
                #expect(pin["bytes"] as? Int == row["bytes"] as? Int)
            }
        }
    }

    /// MODE FENCE. This track has exactly ONE speculative arm, the native MTP
    /// head embedded in the pinned target checkpoint. `serial` must be present
    /// because the baseline leg is pinned serial and is validated against this
    /// same list. `dflash` must be ABSENT: the DFlash arm is not part of this
    /// track and declaring it would arm a mode nothing here can run.
    @Test("allowed_modes declares serial and mtp only")
    func allowedModesDeclaresSerialAndMTPOnly() throws {
        let object = try nemotron35TrackContractObject()
        let modes = try #require(object["allowed_modes"] as? [String])
        #expect(modes == ["serial", "mtp"])
        #expect(!modes.contains("dflash"))
    }

    @Test("kv_backend is pinned explicitly to contiguous")
    func kvBackendIsContiguous() throws {
        let object = try nemotron35TrackContractObject()
        #expect(object["kv_backend"] as? String == "contiguous")
    }

    /// David ruling 2026-08-27: "Single-stream only". The scored width is 1,
    /// and the batched cohort adaptation is not pursued.
    @Test("scored_batch_size is pinned to the ruled single-stream width")
    func scoredBatchSizeIsOne() throws {
        let object = try nemotron35TrackContractObject()
        #expect(object["scored_batch_size"] as? Int == 1)
    }

    @Test("scored_exponents equals the ruled certify pair, exact field names")
    func scoredExponentsMatchesRuledPair() throws {
        let object = try nemotron35TrackContractObject()
        let exponents = try #require(object["scored_exponents"] as? [String: Any])
        // Field names must match benchd's `DeclaredScoredExponents` struct
        // exactly -- NOT the shorthand "prefill"/"decode" spelling, which
        // `ScoredExponents::certify` would treat as absent.
        let prefill = try #require(exponents["prefill_gain_exponent"] as? Double)
        let decode = try #require(exponents["decode_gain_exponent"] as? Double)
        #expect(prefill == 0.25)
        #expect(decode == 0.75)
        #expect(prefill.bitPattern == Double(0.25).bitPattern)
        #expect(decode.bitPattern == Double(0.75).bitPattern)
        #expect(
            exponents.count == 2,
            "scored_exponents must carry exactly the certify pair, no extra keys")
    }

    /// The window shape and the scoring weights are compiled into the harness
    /// as well as declared here, and the two must agree. The acceptance BANDS
    /// are deliberately not in this list: the fixture's bands are the track's
    /// own and the compiled constants are the shared local-mode ones.
    @Test("the contract and the compiled constants agree on the scored window")
    func contractAgreesWithCompiledConstants() throws {
        let object = try nemotron35TrackContractObject()
        #expect(object["vocab_size"] as? Int == MLXFastConstants.vocabSize)
        #expect(object["num_hidden_layers"] as? Int == MLXFastConstants.numHiddenLayers)
        #expect(
            object["golden_model_type"] as? String
                == MLXFastConstants.requiredGoldenModelType)
        // 488, the length every live coding prompt renders to.
        #expect(object["seed_tokens"] as? Int == MLXFastConstants.benchmarkDecodeSeedTokens)
        #expect(object["correctness_steps"] as? Int == MLXFastConstants.correctnessSteps)
        #expect(
            object["benchmark_decode_steps"] as? Int == MLXFastConstants.benchmarkDecodeSteps)
        #expect(
            object["local_submit_benchmark_decode_steps"] as? Int
                == MLXFastConstants.localSubmitBenchmarkDecodeSteps)
        #expect(object["score_decode_weight"] as? Double == MLXFastConstants.scoreDecodeWeight)
        #expect(object["score_prefill_weight"] as? Double == MLXFastConstants.scorePrefillWeight)
        #expect(
            object["decode_speedup_floor"] as? Double
                == MLXFastConstants.scoreDecodeSpeedupFloor)
        #expect(
            object["prefill_speedup_floor"] as? Double
                == MLXFastConstants.scorePrefillSpeedupFloor)
    }

    @Test("target reference-model pin is a real 40-hex revision, matching the compiled constants")
    func targetPinIsFortyHexAndMatchesConstants() throws {
        let object = try nemotron35TrackContractObject()
        let target = try #require(object["target"] as? [String: Any])
        let modelId = try #require(target["upstream_model_id"] as? String)
        let revision = try #require(target["upstream_revision"] as? String)
        #expect(modelId == nemotron35Repository)
        #expect(revision == nemotron35Revision)
        #expect(modelId == MLXFastConstants.referenceModelRepository)
        #expect(revision == MLXFastConstants.referenceModelRevision)
        #expect(isFortyLowercaseHex(revision))
    }

    @Test("target geometry matches the pinned checkpoint's own config.json")
    func targetGeometryMatchesPublishedConfig() throws {
        let object = try nemotron35TrackContractObject()
        let target = try #require(object["target"] as? [String: Any])
        let config = try nemotronConfigObject()

        #expect(target["model_type"] as? String == config["model_type"] as? String)
        #expect(target["model_type"] as? String == MLXFastConstants.requiredGoldenModelType)
        for key in [
            "num_hidden_layers", "hidden_size", "vocab_size",
            "max_position_embeddings", "num_attention_heads",
            "num_key_value_heads", "head_dim", "mamba_num_heads",
            "mamba_head_dim", "ssm_state_size", "conv_kernel", "n_groups",
            "chunk_size", "n_routed_experts", "num_experts_per_tok",
            "moe_intermediate_size", "n_shared_experts",
            "moe_shared_expert_intermediate_size", "intermediate_size",
            "n_group", "topk_group", "bos_token_id", "pad_token_id",
        ] {
            #expect(
                target[key] as? Int == config[key] as? Int,
                Comment(rawValue: key))
        }
        for key in [
            "attention_bias", "use_conv_bias", "mamba_proj_bias",
            "moe_shared_expert_overlap", "norm_topk_prob",
            "residual_in_fp32", "tie_word_embeddings",
        ] {
            #expect(
                target[key] as? Bool == config[key] as? Bool,
                Comment(rawValue: key))
        }
        // The head is UNTIED. Every `lm_head.*` expectation downstream rests
        // on this one flag, so it is asserted by value, not by agreement only.
        #expect(target["tie_word_embeddings"] as? Bool == false)
        #expect(target["eos_token_ids"] as? [Int] == config["eos_token_id"] as? [Int])
        #expect(target["architecture"] as? String == "NemotronHForCausalLM")
        #expect(config["architectures"] as? [String] == ["NemotronHForCausalLM"])
    }

    /// The declared schedule must BE the config's own, and the two derived
    /// summaries must be derived from it rather than typed beside it.
    @Test("the declared layer schedule is the config's own, and its summaries are derived")
    func targetLayerScheduleIsDerivedFromTheConfig() throws {
        let object = try nemotron35TrackContractObject()
        let target = try #require(object["target"] as? [String: Any])
        let config = try nemotronConfigObject()

        let blockType = try #require(config["layers_block_type"] as? [String])
        #expect(target["layers_block_type"] as? [String] == blockType)
        #expect(blockType.count == MLXFastConstants.numHiddenLayers)

        let counts = try #require(target["layer_type_counts"] as? [String: Int])
        var derivedCounts: [String: Int] = [:]
        for kind in blockType {
            derivedCounts[kind, default: 0] += 1
        }
        #expect(counts == derivedCounts)

        let derivedAttention = blockType.enumerated()
            .filter { $0.element == "attention" }
            .map(\.offset)
        #expect(target["attention_layer_indices"] as? [Int] == derivedAttention)
        #expect(derivedAttention.count == counts["attention"])
    }

    /// The pinned quantization is UNIFORM affine 4-bit group-64. The contract
    /// says so, the checkpoint's own config says so, and `mixed_precision` is
    /// false, which is what makes the transform's single-width check sound.
    @Test("target quantization agrees with the checkpoint and declares no mixed precision")
    func targetQuantizationIsUniformAffine() throws {
        let object = try nemotron35TrackContractObject()
        let target = try #require(object["target"] as? [String: Any])
        let quantization = try #require(target["quantization"] as? [String: Any])
        let config = try nemotronConfigObject()
        let configQuantization = try #require(config["quantization"] as? [String: Any])

        #expect(quantization["mode"] as? String == configQuantization["mode"] as? String)
        #expect(
            quantization["group_size"] as? Int == configQuantization["group_size"] as? Int)
        #expect(quantization["bits"] as? Int == configQuantization["bits"] as? Int)
        #expect(quantization["mode"] as? String == "affine")
        #expect(quantization["group_size"] as? Int == 64)
        #expect(quantization["bits"] as? Int == 4)
        #expect(quantization["mixed_precision"] as? Bool == false)
        // The checkpoint publishes the same block twice; they must agree.
        #expect(
            NSDictionary(dictionary: configQuantization)
                == NSDictionary(dictionary: try #require(config["quantization_config"] as? [String: Any])))
    }

    /// The config fixture is the published `config.json` VERBATIM. Three
    /// independent files record its digest, and they must all be the same
    /// digest: a fixture that agrees with only one of them is a fixture
    /// nobody can trust.
    @Test("the config fixture is the published bytes, and every pin of it agrees")
    func theConfigFixtureIsThePublishedBytes() throws {
        let data = try Data(contentsOf: nemotronConfigFixtureURL)
        #expect(data.count == nemotron35ConfigByteCount)
        #expect(sha256Hex(data) == nemotron35ConfigSHA256)

        let inventory = try nemotron35InventoryFixture()
        #expect(inventory.source.configSHA256 == nemotron35ConfigSHA256)
        #expect(inventory.source.configBytes == nemotron35ConfigByteCount)
        #expect(inventory.source.upstreamModelID == nemotron35Repository)
        #expect(inventory.source.upstreamRevision == nemotron35Revision)

        let manifest = try nemotron35ReferenceManifestRecords()
        let configRecord = try #require(manifest.first { $0.path == "config.json" })
        #expect(configRecord.sha256 == nemotron35ConfigSHA256)
        #expect(configRecord.byteCount == nemotron35ConfigByteCount)
    }

    /// The target's tensor counts are the INVENTORY's counts. The inventory is
    /// the file that enumerates all 763 tensors, so the contract's summary
    /// must be read off it rather than restated beside it.
    @Test("target tensor counts agree with the pinned tensor inventory")
    func targetTensorCountsAgreeWithInventory() throws {
        let object = try nemotron35TrackContractObject()
        let target = try #require(object["target"] as? [String: Any])
        let inventory = try nemotron35InventoryFixture()

        #expect(target["raw_checkpoint_tensor_count"] as? Int == inventory.summary.tensorCount)
        #expect(target["raw_checkpoint_shard_count"] as? Int == inventory.summary.shardCount)
        #expect(
            target["raw_checkpoint_total_size_bytes"] as? Int
                == inventory.summary.totalSizeBytes)
        #expect(target["backbone_tensor_count"] as? Int == inventory.summary.backboneTensorCount)
        #expect(target["lm_head_tensor_count"] as? Int == inventory.summary.lmHeadTensorCount)

        // And the inventory's own summary is summed from its own records.
        #expect(inventory.tensors.count == inventory.summary.tensorCount)
        #expect(inventory.shards.count == inventory.summary.shardCount)
        #expect(
            inventory.shards.map(\.tensorCount).reduce(0, +)
                == inventory.summary.tensorCount)
        #expect(
            inventory.tensors.keys.filter { $0.hasPrefix("backbone.") }.count
                == inventory.summary.backboneTensorCount)
        #expect(
            inventory.tensors.keys.filter { $0.hasPrefix("lm_head.") }.count
                == inventory.summary.lmHeadTensorCount)
        #expect(
            inventory.tensors.keys.filter { $0.hasPrefix("mtp.") }.count
                == inventory.summary.mtpTensorCount)
        #expect(
            inventory.summary.backboneTensorCount
                + inventory.summary.lmHeadTensorCount
                + inventory.summary.mtpTensorCount
                == inventory.summary.tensorCount)
    }

    /// The head is EMBEDDED. This track stages no head weight file: the MTP
    /// tensors ride inside the pinned target checkpoint under `mtp.*`, so
    /// there is no separate repository, no separate revision, and nothing for
    /// a head stager to fetch.
    @Test("mtp_head declares an embedded head, not a staged artifact")
    func mtpHeadIsEmbedded() throws {
        let object = try nemotron35TrackContractObject()
        let head = try #require(object["mtp_head"] as? [String: Any])
        #expect(head["role"] as? String == "spec_decode_mtp_head")
        #expect(head["source"] as? String == "embedded")
        #expect(head["tensor_prefix"] as? String == "mtp.")
        #expect(head["use_dedicated_embeddings"] as? Bool == false)
        // No staged-artifact keys may appear: those would imply a fetch.
        #expect(head["upstream_model_id"] == nil)
        #expect(head["upstream_revision"] == nil)
        // And no separate assistant declaration survives from an earlier track.
        #expect(object["assistant"] == nil)
    }

    /// The head's declared shape is three files' shared fact: the contract
    /// declares it, the inventory counts it, and the checkpoint's own config
    /// describes it. All three must say the same thing.
    @Test("the mtp_head block agrees with the inventory and the checkpoint config")
    func mtpHeadAgreesWithInventoryAndConfig() throws {
        let object = try nemotron35TrackContractObject()
        let head = try #require(object["mtp_head"] as? [String: Any])
        let inventory = try nemotron35InventoryFixture()
        let config = try nemotronConfigObject()

        #expect(head["tensor_count"] as? Int == inventory.summary.mtpTensorCount)
        #expect(head["tensor_count"] as? Int == 34)
        #expect(head["num_hidden_layers"] as? Int == config["num_nextn_predict_layers"] as? Int)
        #expect(head["layer_types"] as? [String] == config["mtp_layers_block_type"] as? [String])
        #expect(head["rope_theta"] as? Int == config["rope_theta"] as? Int)
        // HYBRID. The head is one attention block feeding one
        // mixture-of-experts block, the same two kinds the tower uses.
        #expect(head["hybrid"] as? Bool == true)
        let layerTypes = try #require(head["layer_types"] as? [String])
        #expect(layerTypes == ["attention", "moe"])
        let towerKinds = Set(try #require(config["layers_block_type"] as? [String]))
        #expect(Set(layerTypes).isSubset(of: towerKinds))

        // Every `mtp.*` tensor sits under one of the declared layer indices.
        let prefix = try #require(head["tensor_prefix"] as? String)
        let declaredLayers = try #require(head["num_hidden_layers"] as? Int)
        #expect(declaredLayers == 1)
        let mtpNames = inventory.tensors.keys.filter { $0.hasPrefix(prefix) }
        for name in mtpNames {
            #expect(
                (0 ..< layerTypes.count).contains(where: {
                    name.hasPrefix("\(prefix)layers.\($0).")
                }),
                Comment(rawValue: name))
        }
    }

    /// The draft depths the protocol permits and the envelope the driver is
    /// built against must not disagree: a permitted depth the envelope cannot
    /// carry is a request the worker would refuse at run time.
    @Test("permitted draft depths fit the declared MTP envelope")
    func permittedDraftDepthsFitTheEnvelope() throws {
        let object = try nemotron35TrackContractObject()
        let head = try #require(object["mtp_head"] as? [String: Any])
        let depths = try #require(head["permitted_draft_depths"] as? [Int])
        let proto = try #require(object["protocol"] as? [String: Any])
        let envelope = try #require(
            proto["mtp_envelope_constants_from_darkbloom"] as? [String: Any])
        let maxDraftTokens = try #require(envelope["max_draft_tokens"] as? Int)

        #expect(depths == Array(1 ... maxDraftTokens))
        #expect(envelope["max_speculative_batch"] as? Int == 1)
        #expect(envelope["verification_mode"] as? String == "rectangular_exact")
    }

    /// THE SHAPE PINS MOVE WITH THE BODY. A record count and a byte total that
    /// are not summed from the digests they must agree with are not pins, so
    /// the header's two declarations are checked against the records below it.
    @Test("the reference manifest has 11 records and header pins summed from them")
    func referenceManifestPinsAreSummedFromItsOwnRecords() throws {
        let records = try nemotron35ReferenceManifestRecords()
        let pins = try nemotron35ReferenceManifestHeaderPins()

        #expect(records.count == 11)
        #expect(pins.records == records.count)
        #expect(pins.bytes == records.map(\.byteCount).reduce(0, +))
        #expect(pins.bytes == 18_544_229_436)

        // Every record is a real 64-hex digest over a positive byte count, and
        // no path is pinned twice.
        for record in records {
            #expect(
                isSixtyFourLowercaseHex(record.sha256),
                Comment(rawValue: record.path))
            #expect(record.byteCount > 0, Comment(rawValue: record.path))
        }
        #expect(Set(records.map(\.path)).count == records.count)

        // The four weight shards the inventory enumerates are all pinned, and
        // their byte total is the inventory's own.
        let inventory = try nemotron35InventoryFixture()
        let shardRecords = try inventory.shards.map { shard in
            try #require(
                records.first { $0.path == shard.name },
                Comment(rawValue: shard.name))
        }
        #expect(shardRecords.count == inventory.summary.shardCount)
        // The safetensors files carry the tensor bytes plus their own headers,
        // so the pinned file bytes exceed the inventory's tensor-byte total.
        #expect(
            shardRecords.map(\.byteCount).reduce(0, +) > inventory.summary.totalSizeBytes)

        // The engine loads the index and the tokenizer from this same tree, so
        // both are pinned beside the shards.
        for path in ["model.safetensors.index.json", "tokenizer.json", "config.json"] {
            #expect(records.contains { $0.path == path }, Comment(rawValue: path))
        }
        let indexRecord = try #require(
            records.first { $0.path == "model.safetensors.index.json" })
        #expect(indexRecord.sha256 == inventory.source.indexSHA256)
        #expect(indexRecord.byteCount == inventory.source.indexBytes)
    }

    @Test("scoring_semantics records the ruled composite formula")
    func scoringSemanticsRecordsRuledFormula() throws {
        let object = try nemotron35TrackContractObject()
        let semantics = try #require(object["scoring_semantics"] as? [String: Any])
        let formula = try #require(semantics["formula"] as? String)
        #expect(formula.contains("0.25"))
        #expect(formula.contains("0.75"))
        #expect(formula.contains("prefill_gain"))
        #expect(formula.contains("decode_gain"))
    }

    /// The public contract records only the date of the ruling the composite
    /// formula comes from (2026-08-27), not a quote of it. The formula and
    /// the exponents are what bind, and they are asserted separately above.
    @Test("the public contract carries the ruling date, not a quote")
    func rulingDateWithoutQuote() throws {
        let object = try nemotron35TrackContractObject()
        let semantics = try #require(object["scoring_semantics"] as? [String: Any])
        #expect(semantics["ruling_verbatim"] == nil)
        #expect(semantics["ruling_date"] as? String == "2026-08-27")
        let survivors = allStrings(in: semantics).filter { $0.lowercased().contains("gemma") }
        #expect(
            survivors.isEmpty,
            "the Gemma-era ruling must not survive anywhere in scoring_semantics: \(survivors)")
    }

    /// The license block names the repository the weights actually come from.
    /// A license that verifies a different repository verifies nothing.
    @Test("the license block verifies the pinned target repository")
    func licenseVerifiesThePinnedRepository() throws {
        let object = try nemotron35TrackContractObject()
        let license = try #require(object["license"] as? [String: Any])
        #expect(license["verified_repositories"] as? [String] == [nemotron35Repository])
        let url = try #require(license["url"] as? String)
        #expect(url.contains(nemotron35Repository))
        #expect(url.contains(nemotron35Revision))
    }
}

/// Every string reachable inside a decoded JSON value -- object keys, string
/// values, and both recursively through nested objects and arrays -- so a whole
/// block can be scanned rather than one field of it.
private func allStrings(in value: Any) -> [String] {
    switch value {
    case let string as String:
        return [string]
    case let array as [Any]:
        return array.flatMap(allStrings(in:))
    case let object as [String: Any]:
        return object.flatMap { [$0.key] + allStrings(in: $0.value) }
    default:
        return []
    }
}

private func isFortyLowercaseHex(_ value: String) -> Bool {
    isLowercaseHex(value, count: 40)
}

private func isSixtyFourLowercaseHex(_ value: String) -> Bool {
    isLowercaseHex(value, count: 64)
}

private func isLowercaseHex(_ value: String, count: Int) -> Bool {
    value.utf8.count == count && value.utf8.allSatisfy { byte in
        (byte >= UInt8(ascii: "0") && byte <= UInt8(ascii: "9"))
            || (byte >= UInt8(ascii: "a") && byte <= UInt8(ascii: "f"))
    }
}
