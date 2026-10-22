# The task — Nemotron 3.5 Lightning 30B A3B MLX

Make the Nemotron 3.5 Lightning 30B A3B model run faster on Apple Silicon.

The ranked track is `nemotron3.5-30b-a3b-mlx-v1`. Read
[README.md](README.md) for setup and
[the participant contract](docs/participant-contract.md) for the rules.

## What you optimize

You optimize the model engine, offline transform, MTP arm, and listed MLX and
Metal kernels.

The target checkpoint is
`EigenLabs/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit-mtp`. Its 52-layer
tower has 23 Mamba-2 layers, 23 sparse MoE layers, and 6 attention layers.

The MTP head is embedded under `mtp.*`. It proposes tokens. The target model
decides every emitted token.

## What you may change

`benchmark.json` `editablePaths` lists 94 entries and is the authority. The
main groups are:

1. `mtp-head.manifest.json`
2. `Sources/MLXFastTransform/`
3. The listed Nemotron model files
4. The listed batching, cache, attention, RoPE, and server files
5. The listed MLX and Metal kernels

The MTP declaration accepts `"source": "pinned"` only. You may re-quantize the
embedded head on load, in memory. You may not replace it or upload head
weights.

The target model's quantization is frozen. Do not re-quantize, re-represent, or
change the numerical format of target weights.

The ranked run scores one stream. The declaration parser accepts structural
integer depths from 0 through 8. An absent `spec` block, disabled speculation,
or depth 0 resolves to serial, including disabled depth 8. An enabled positive
depth must be in the fixture-permitted set 1 through 7, so enabled depth 8 is
refused.

## Run locally

```bash
./tools/fetch-benchd.sh
./setup.sh
.build/release/mlxfast-swift transform \
  --reference reference_weights/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit-mtp \
  --output weights
MLXFAST_ENGINE_BIN=.build/release/bench-worker \
MLXFAST_CORRECTNESS_GOLDEN_PATH=correctness_prompts/public_longcopy_gate_english_1024_256.json \
  ./benchmark.sh --local-iterate
```

The local test is serial unless `--mtp-depth N` is given.

## Scoring

```text
composite = prefill_gain ^ 0.25 * decode_gain ^ 0.75
gain = control_leg_seconds_per_token / candidate_leg_seconds_per_token
```

The ranked run measures three pairs. Each pair runs a serial control leg and a
candidate leg on the same machine. The score is the mean of the per-pair
composites.

The measured window uses one stream, 488 seed tokens, and 128 checked decode
steps. The KV backend is `contiguous`. The token-tolerance budget is 10%.

## Current state

- Official scoring is armed.
- The fixture pins `pool-14`, `pool-15`, and `pool-16`.
- The ranked workflow targets the self-hosted labels in
  `.github/workflows/benchmark.yml`.
- Hidden artifacts are organizer material and are not in git.

## Authorities

| Question | Authority |
|---|---|
| Editable paths, commands, and scoring values | `benchmark.json` |
| Pins, live pool, and scoring semantics | `fixtures/nemotron3_5_30b_a3b_mlx_v1_track.json` |
| Participant rules | `docs/participant-contract.md` |
| Measured behavior | Verified channel benchmarker |

The configuration files win over this summary.
