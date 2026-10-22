# Nemotron 3.5 Lightning 30B A3B port notes

This file records the public engineering shape of the port. It is not a
contract. `benchmark.json` and
`fixtures/nemotron3_5_30b_a3b_mlx_v1_track.json` hold the current values.

## 1. Track identity

- Track: `nemotron3.5-30b-a3b-mlx-v1`
- Manifest name: `mlxfast-nemotron35-30b-a3b`
- Checkpoint:
  `EigenLabs/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit-mtp`
- Revision: `153c469494b3f16a0eaa752dae25cfc9452ce5c6`

## 2. Model port

The fixture's `target` block is the model-facts authority. The port supports:

- 52 layers: 23 Mamba-2, 23 MoE, and 6 attention
- 32 query heads and 2 KV heads
- 128 routed experts, 6 selected per token
- one shared expert
- an embedded 34-tensor MTP head
- the `contiguous` KV backend

`Sources/MLXFastTransform/NemotronHCheckpointValidation.swift` derives the
expected tensor set and checks the checkpoint headers and pinned inventory.

`bench-worker` selects the runner from the checkpoint's `model_type`,
`nemotron_h`.

## 3. Vendored engine

`Vendor/mlx-swift-lm` and `Vendor/mlx-swift` contain plain files. They are not
submodules.

The public upstream bases and complete license attribution are in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). The fixture records the
engine upstream revision.

## 4. Editable surface

`benchmark.json` lists every editable path. The groups are:

1. the MTP declaration;
2. the offline transform;
3. the listed Nemotron model files;
4. the listed batching, cache, attention, RoPE, and server files;
5. the listed MLX and Metal kernels.

The manifest declares a 7,949,663-byte total cap, a 524,288-byte per-file cap,
and a 262,144-byte growth cap. No path is exempt.

## 5. MTP arm

The MTP head is embedded in the pinned checkpoint under `mtp.*`. No separate
head artifact is staged or submitted.

`mtp-head.manifest.json` accepts `"source": "pinned"` only. Participants may
select depth 1 through 7. Invalid depths are refused.

The head may be re-quantized on load and in memory. It may not be replaced.
Target-model quantization is frozen.

## 6. Scoring

The ranked run uses:

```text
composite = prefill_gain ^ 0.25 * decode_gain ^ 0.75
```

It measures three pairs and takes the arithmetic mean of the three per-pair
composites. Each pair has a serial control leg and a candidate leg. Both legs
run on the same machine.

The public measured-window dimensions are:

- one stream
- 488 seed tokens
- 128 checked decode steps
- one prompt per leg

The fixture is the authority for the live pool. It currently names
`pool-14`, `pool-15`, and `pool-16` and publishes the SHA-256 and byte count
for each pinned artifact.

The cached values documented in
[benchmark-window-freeze.md](benchmark-window-freeze.md) are local diagnostics
only. Ranked scoring uses the live control leg.

## 7. Current public state

- Official scoring is armed.
- The fixture pins the live pool.
- The ranked workflow targets the self-hosted labels in
  `.github/workflows/benchmark.yml`.
- The public captures are checked into `correctness_prompts/`.
- The paged KV backend is not declared; the track uses `contiguous`.

## 8. Reproducible checks

```bash
python3 tools/lint-benchmark-manifest.py
swift test --force-resolved-versions
swift build -c release --force-resolved-versions
tools/test-spec-declaration.sh
```

Model and Metal runtime checks need suitable Apple Silicon and a completed
setup.
