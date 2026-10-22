# mlxfast — Nemotron 3.5 Lightning 30B A3B MLX

This repository is the engine for the Nemotron 3.5 Lightning 30B A3B MLX
speed benchmark. The track id is `nemotron3.5-30b-a3b-mlx-v1`.

The engine is Swift and Metal code for Apple Silicon. The separate `benchd`
program owns timing, scoring, and correctness gates.

Official scoring is armed. The fixture pins the live pool. Ranked runs use one
stream and measure a serial control leg and a candidate leg on the same machine
in the same job. The ranked workflow targets the self-hosted labels declared in
`.github/workflows/benchmark.yml`.

## Requirements

- Apple Silicon
- macOS 14 or later
- Swift 6.3 through Xcode
- The Xcode Metal Toolchain
- CMake
- Git
- jq (`/usr/bin/jq` on macOS 15 and later; otherwise `brew install jq`)
- At least 45 GiB of free disk space before setup
- Enough unified memory for the 18,544,229,436-byte checkpoint and its working
  set

## Quickstart

Use the Yukon CLI to clone the public track:

```bash
yukon clone GumbiiDigital/mlxfast-nemotron35-30b-a3b
cd mlxfast-nemotron35-30b-a3b
```

Resolve the verified benchmarker:

```bash
./tools/fetch-benchd.sh
```

Build the binaries and Metal library, then download and verify the checkpoint:

```bash
./setup.sh
```

Transform the checkpoint:

```bash
.build/release/mlxfast-swift transform \
  --reference reference_weights/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit-mtp \
  --output weights
```

This local command uses the frozen transform. Ranked runs copy transformed
weights from the pinned reference workspace instead.

Run the short local check:

```bash
MLXFAST_ENGINE_BIN=.build/release/bench-worker \
MLXFAST_CORRECTNESS_GOLDEN_PATH=correctness_prompts/public_longcopy_gate_english_1024_256.json \
  ./benchmark.sh --local-iterate
```

### Local modes

| Mode | Decode steps | Minimum expected tokens | Notes |
|---|---:|---:|---|
| `--local-iterate` | 128 | 129 | One teacher-forced checked timing pass |
| `--local-submit` | 1023 | 1024 | One teacher-forced checked timing pass |

Both local modes check correctness and timing. A local pass does not establish
acceptance by the ranked paired gates.

There is no default golden path. `./benchmark.sh` rejects `--golden`,
`--weights`, and `--score-path`. Use `MLXFAST_CORRECTNESS_GOLDEN_PATH`,
`MLXFAST_WEIGHTS_PATH`, and `MLXFAST_SCORE_PATH` instead.

A correctness failure still reports timing through the mismatch-tolerant rerun
path.

Run the long local check at the declared draft depth:

```bash
depth="$(tools/spec-declaration.sh draft-len)"
mtp_args=()
if (( depth > 0 )); then
  mtp_args=(--mtp-depth "$depth")
fi

MLXFAST_ENGINE_BIN=.build/release/bench-worker \
MLXFAST_CORRECTNESS_GOLDEN_PATH=correctness_prompts/public_longcopy_gate_english_1024_1024.json \
  ./benchmark.sh --local-submit "${mtp_args[@]}"
```

The local test does not read the draft depth from `mtp-head.manifest.json`.
The command adds `--mtp-depth` only for a positive declared depth. A fresh
serial declaration therefore runs without the flag. Run local measurements
through `./benchmark.sh`.

## Public local captures

| File | Purpose |
|---|---|
| `correctness_prompts/public_longcopy_gate_english_1024.txt` | Public prompt text |
| `correctness_prompts/public_longcopy_gate_english_1024_256.json` | 488 prompt tokens and 256 expected tokens |
| `correctness_prompts/public_longcopy_gate_english_1024_1024.json` | 488 prompt tokens and 1024 expected tokens |

Both JSON files identify the pinned checkpoint through `model_provenance`.

## Repository structure

| Path | Purpose |
|---|---|
| `Sources/MLXFastTransform/` | Trusted offline transform; not editable |
| `Sources/MLXFastCLI/` | Trusted CLI |
| `Sources/MLXFastCore/` | Shared contracts |
| `Sources/MLXFastTrustedHarness/` | Trusted enforcement |
| `Vendor/mlx-swift/` | Vendored MLX and Metal sources |
| `Vendor/mlx-swift-lm/` | Vendored model engine and worker |
| `fixtures/` | Track contract and checkpoint manifests |
| `tools/` | Setup, checks, and measurement scripts |
| `benchmark.json` | Track manifest and editable-path authority |

The two vendor directories contain plain files, not submodules. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attribution and
licenses.

## What you may change

`benchmark.json` `editablePaths` lists 93 entries and is the authority. The
surface includes:

1. `mtp-head.manifest.json`
2. The listed Nemotron model files
3. The listed batching, cache, attention, RoPE, and server files
4. The listed MLX and Metal kernel files

Everything else is trusted. A submission cannot change measurement, scoring,
gates, fixtures, target weights, the offline transform, or the benchmarker.

### MTP head

The native MTP head is embedded in the pinned checkpoint under `mtp.*`.
`mtp-head.manifest.json` is an optional declaration and accepts
`"source": "pinned"` only. No submission carries head weights.

You may re-quantize the head on load, in memory. You may not replace it. The
target model's quantization is frozen.

The head loader is an editable path:
`Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/NemotronH35MTPLoading.swift`.
See [the participant contract](docs/participant-contract.md#44-in-memory-head-re-quantization).

### Draft depth

You may select a depth from 1 through 7:

```json
"spec": { "enabled": true, "num_speculative_tokens": 3 }
```

The declaration parser accepts structural integer depths from 0 through 8.
When speculation is enabled, a positive depth must be in the fixture's
permitted set 1 through 7. Enabled depth 8 is refused.

No `spec` block, `enabled: false`, or depth 0 selects serial mode. A disabled
depth 8 also resolves to serial. The run records `effective_spec` and
`effective_mean_draft_len`.

The candidate output is checked against the oracle for its declared depth
under the published token-tolerance gate.

### Byte budget

The manifest declares:

| Key | Value |
|---|---:|
| `maxTotalBytes` | 7,949,663 |
| `maxFileBytes` | 524,288 |
| `maxGrowthBytes` | 262,144 |
| `exemptPathMaxBytes` | 512,000,000 |
| `exemptPathMaxFileBytes` | 100,000,000 |

No path is exempt.

## Scoring

```text
composite = prefill_gain ^ 0.25 * decode_gain ^ 0.75
gain = control_leg_seconds_per_token / candidate_leg_seconds_per_token
```

Higher is better. `benchmark.json` sets `direction` to `+`.

The ranked run measures three pairs. Each pair uses one live prompt for timed
prefill, the next live prompt for decode, and two legs:

1. A serial control leg at the fixture's `baseline_reference_commit`.
2. The candidate leg at the declared draft depth.

The score is the arithmetic mean of the three per-pair composites. The fixture
sets `official_pairs = 3`. Each pair must independently clear
`prefill_speedup_floor = 0.95` and `decode_speedup_floor = 0.95`. The manifest
sets `decodeSpeedupCeiling = 5.0` and `prefillSpeedupCeiling = 1.5`. A ceiling
failure refuses the pair instead of clamping its score. The first failed pair
stops the run.

`official_pair_combine` is the authority for how pair composites combine. Its
current value is `mean`.

Calibration refuses when either axis coefficient of variation exceeds
1 percent. A control leg outside its calibration band seals no score.

The benchmarker seals an unrounded double. Equal sealed values tie, and the
platform decides improvement.

No file stores a baseline pair. Each control leg is measured live. The
machine's calibration file is a health band, not a score denominator. The KV
backend is `contiguous`.

### Measured window

| Quantity | Value |
|---|---:|
| Seed tokens | 488 |
| Checked decode steps | 128 |
| Streams | 1 |
| Timed prompts per leg | 2 roles: one prefill prompt and a different decode prompt |
| Warm-up prompt | 1 separate fixture-pinned prompt, never scored |
| Pairs | 3 |
| Legs per pair | 2 |

The benchmarker applies a 10% per-stream token-tolerance budget.

Both legs use the fixture's separate `warmup_golden` for warm-up. The prompt is
never scored. The parent seed clock runs from sending `free_decode_begin`
through the validated seed reply. The decode clock then runs through
`free_decode_run(N)` completion. Scored decode is `(seed window + decode
window) / N`; scored prefill comes from separate timed prefill requests.

The seed window must be no more than 1.40 times timed prefill per token. Guard
failures include `SPEEDUP-ABOVE-CEILING`, `SEED-WINDOW-ABOVE-LIMIT`,
`PROMPT-REUSE-REFUSED`, `PRIVATE-DIR-REQUIRED`, and `LEG-TEARDOWN-FAILED`.

Before each timed phase the benchmarker applies quiescence gates (one-minute load
average below 2.0, GPU utilization below 10%, 15-second polling, 900-second
timeout) and a cooling gate (GPU at or below 50 °C; refused when the GPU does not
cool down in time).
Ranked measurement refuses missing, unreadable, frozen, or implausible
temperature readings.

## Current public state

- `official_scoring_enabled` is `true`.
- The fixture pins `pool-14`, `pool-15`, and `pool-16` for scoring and
  `pool-17` for warm-up only.
- The fixture publishes each hidden artifact's SHA-256 and byte count.
- Every supported MTP depth maps to the pinned live pool.
- The ranked workflow targets the self-hosted labels in
  `.github/workflows/benchmark.yml`.
- Hidden artifacts are staged outside git.
- The public measured window is 488 seed tokens and 128 checked decode steps.

## Building after an edit

Always use the frozen dependency graph:

```bash
swift build -c release --force-resolved-versions
swift build -c release --force-resolved-versions \
  --scratch-path .build-worker --product bench-worker
tools/build-mlx-metallib.sh
tools/stage-bench-worker.sh
```

Kernel families with an `mlx-generated/*.cpp` twin compile at run time from
that generated source. Keep the readable Metal source and generated twin in
step.

`tools/build-mlx-metallib.sh` rebuilds ahead-of-time Metal sources. It runs
before staging so `.build/release/mlx.metallib` is current.

## Continuous integration

`.github/workflows/ci.yml` runs repository checks on `ubuntu-latest` and Swift
builds and tests on `macos-26`. Hosted CI does not download the target
checkpoint and does not run ranked measurement.

`.github/workflows/benchmark.yml` is the ranked workflow. Its hosted
surface-check job must pass before the self-hosted ranked job starts. Ranked
preflight refuses missing or invalid staged assets.

See [CI coverage](docs/ci-coverage.md).

## Submitting

Use Yukon for account and submission operations:

```bash
export PATH="${HOME}/.local/bin:${PATH}"
yukon login <api-key> --api https://api.yukon.org
yukon setup
yukon run
yukon submit --model "<exact model name>" --note-file submission-note.md
yukon submissions
```

`yukon submit` does not run or require a successful local test. Run the local
check yourself before you submit.

Ranked behavior is selected when `GITHUB_ACTIONS=true` or when any of
`MLXFAST_QWEN38_GOLDEN_DIR`, `MLXFAST_BASELINE_WORKSPACE`, or
`MLXFAST_BASELINE_CALIBRATION` is set. A partial ranked environment can refuse.
Otherwise `yukon run` uses the local path and returns `"score": null`.

A submission contains only editable paths. It rejects generated artifacts,
symlinks, scores, checkpoints, and changes outside the editable surface.

## Authorities

| Question | Authority |
|---|---|
| Editable paths, commands, and scoring values | `benchmark.json` |
| Pins, live pool, and scoring semantics | `fixtures/nemotron3_5_30b_a3b_mlx_v1_track.json` |
| Participant rules | `docs/participant-contract.md` |
| CI coverage | `docs/ci-coverage.md` |
| Measured behavior | Verified channel benchmarker |

The configuration files win over this explanation. The benchmarker wins on
measurement behavior.

## License and attribution

Repository code is licensed under [LICENSE](LICENSE). The pinned checkpoint
uses OpenMDW-1.1. This repository distributes no model weights.
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) contains complete third-party
attribution.
