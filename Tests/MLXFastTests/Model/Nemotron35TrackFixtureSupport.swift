import CryptoKit
import Foundation
import MLXFastCore

/// Fixture locations and pins for the track target,
/// `nemotron3.5-30b-a3b-mlx-v1`.
let nemotron35Repository =
    "EigenLabs/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit-mtp"
let nemotron35Revision = "153c469494b3f16a0eaa752dae25cfc9452ce5c6"

/// SHA256 and byte count of the checkpoint's own `config.json` exactly as
/// published at the pinned revision.
///
/// The checked-in `fixtures/nemotron3_5_30b_a3b_config.json` is that file
/// VERBATIM, not a normalized re-render, so the digest below is the fixture's
/// own digest and the tests assert both at once. The same pair appears in the
/// reference manifest and in the tensor inventory's `source` block; a test
/// holds all three together, because a fixture that agrees with only one of
/// them is a fixture nobody can trust.
let nemotron35ConfigSHA256 =
    "93dcd00b9caf4bc42f803550ecbb95d79a942dfc89314797ee33c0c20c8dbb67"
let nemotron35ConfigByteCount = 3_044

// Tests/MLXFastTests/Model/<this file> -> repository root is four levels up.
private let nemotron35RepositoryRoot = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()

let nemotronConfigFixtureURL = nemotron35RepositoryRoot
    .appendingPathComponent("fixtures/nemotron3_5_30b_a3b_config.json")

let nemotron35TrackContractURL = nemotron35RepositoryRoot
    .appendingPathComponent("fixtures/nemotron3_5_30b_a3b_mlx_v1_track.json")

let nemotron35TensorInventoryURL = nemotron35RepositoryRoot
    .appendingPathComponent("fixtures/nemotron3_5_30b_a3b_tensor_inventory.json")

let nemotron35ReferenceManifestURL = nemotron35RepositoryRoot
    .appendingPathComponent("fixtures/reference_nemotron3_5_30b_a3b_4bit.sha256")

func nemotron35TrackContractObject() throws -> [String: Any] {
    guard let object = try JSONSerialization.jsonObject(
        with: try Data(contentsOf: nemotron35TrackContractURL)
    ) as? [String: Any] else {
        throw MLXFastError.invalidInput(
            "Nemotron 3.5 track contract fixture must be a JSON object"
        )
    }
    return object
}

func nemotronConfigObject() throws -> [String: Any] {
    guard let object = try JSONSerialization.jsonObject(
        with: try Data(contentsOf: nemotronConfigFixtureURL)
    ) as? [String: Any] else {
        throw MLXFastError.invalidInput(
            "Nemotron 3.5 config fixture must be a JSON object"
        )
    }
    return object
}

/// One record of `fixtures/nemotron3_5_30b_a3b_tensor_inventory.json`.
struct Nemotron35TensorRecord: Decodable, Equatable {
    let dtype: String
    let shape: [Int]
    let shardIndex: Int

    enum CodingKeys: String, CodingKey {
        case dtype
        case shape
        case shardIndex = "shard_index"
    }
}

struct Nemotron35InventoryFixture: Decodable {
    struct Source: Decodable {
        let upstreamModelID: String
        let upstreamRevision: String
        let configBytes: Int
        let configSHA256: String
        let indexBytes: Int
        let indexSHA256: String

        enum CodingKeys: String, CodingKey {
            case upstreamModelID = "upstream_model_id"
            case upstreamRevision = "upstream_revision"
            case configBytes = "config_bytes"
            case configSHA256 = "config_sha256"
            case indexBytes = "index_bytes"
            case indexSHA256 = "index_sha256"
        }
    }

    struct Shard: Decodable {
        let index: Int
        let name: String
        let tensorCount: Int

        enum CodingKeys: String, CodingKey {
            case index
            case name
            case tensorCount = "tensor_count"
        }
    }

    struct Summary: Decodable {
        let tensorCount: Int
        let shardCount: Int
        let dtypeCounts: [String: Int]
        let mtpTensorCount: Int
        let backboneTensorCount: Int
        let lmHeadTensorCount: Int
        let totalSizeBytes: Int

        enum CodingKeys: String, CodingKey {
            case tensorCount = "tensor_count"
            case shardCount = "shard_count"
            case dtypeCounts = "dtype_counts"
            case mtpTensorCount = "mtp_tensor_count"
            case backboneTensorCount = "backbone_tensor_count"
            case lmHeadTensorCount = "lm_head_tensor_count"
            case totalSizeBytes = "total_size_bytes"
        }
    }

    let schemaVersion: Int
    let source: Source
    let shards: [Shard]
    let tensors: [String: Nemotron35TensorRecord]
    let summary: Summary

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case source
        case shards
        case tensors
        case summary
    }
}

func nemotron35InventoryFixture() throws -> Nemotron35InventoryFixture {
    try JSONDecoder().decode(
        Nemotron35InventoryFixture.self,
        from: Data(contentsOf: nemotron35TensorInventoryURL)
    )
}

/// One `<sha256> <byte_count> <relative_path>` record of the pinned reference
/// manifest. Comment and blank lines are skipped.
struct Nemotron35ManifestRecord: Equatable {
    let sha256: String
    let byteCount: Int
    let path: String
}

func nemotron35ReferenceManifestRecords() throws -> [Nemotron35ManifestRecord] {
    let text = try String(contentsOf: nemotron35ReferenceManifestURL, encoding: .utf8)
    return try text.split(separator: "\n").compactMap { rawLine -> Nemotron35ManifestRecord? in
        let line = rawLine.trimmingCharacters(in: .whitespaces)
        guard !line.isEmpty, !line.hasPrefix("#") else { return nil }
        let fields = line.split(separator: " ", omittingEmptySubsequences: true)
        guard fields.count == 3, let byteCount = Int(fields[1]) else {
            throw MLXFastError.invalidInput(
                "reference manifest record is not <sha256> <bytes> <path>: \(line)"
            )
        }
        return Nemotron35ManifestRecord(
            sha256: String(fields[0]),
            byteCount: byteCount,
            path: String(fields[2])
        )
    }
}

/// The two shape pins the manifest header declares, read back out of it. They
/// are asserted against the records they must be summed from: a count that is
/// not summed from the digests it must agree with is not a pin.
func nemotron35ReferenceManifestHeaderPins() throws -> (records: Int, bytes: Int) {
    let text = try String(contentsOf: nemotron35ReferenceManifestURL, encoding: .utf8)
    func pin(_ key: String) throws -> Int {
        guard let line = text.split(separator: "\n").first(where: { $0.contains(key) }),
              let value = line.split(separator: ":").last
                  .map({ $0.trimmingCharacters(in: .whitespaces) }),
              let number = Int(value)
        else {
            throw MLXFastError.invalidInput(
                "reference manifest header is missing \(key)"
            )
        }
        return number
    }
    return (
        try pin("MLXFAST_REFERENCE_MANIFEST_RECORDS"),
        try pin("MLXFAST_REFERENCE_MANIFEST_BYTES")
    )
}

/// Lowercase hex SHA256 of some bytes, the spelling every pin in this
/// repository is written in.
func sha256Hex(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}
