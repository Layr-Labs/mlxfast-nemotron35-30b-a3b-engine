# Nemotron 3.5 Lightning 30B A3B participant contract

This document explains the rules for track
`nemotron3.5-30b-a3b-mlx-v1`. It does not override the configuration.

## 1. Order of authority

1. The verified benchmarker on the ranked workflow host
2. `benchmark.json` and
   `fixtures/nemotron3_5_30b_a3b_mlx_v1_track.json`
3. This document
4. `README.md` and `TASK.md`

`./tools/fetch-benchd.sh` resolves the public bench repository's `main`
distribution channel. It verifies the binary against the channel's
`benchd.manifest.json` SHA-256 and byte count. The default public fetch needs no
token.

## 2. What the track measures

The track measures Nemotron 3.5 Lightning 30B A3B inference on Apple Silicon.

The pinned checkpoint is:

```text
EigenLabs/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit-mtp
153c469494b3f16a0eaa752dae25cfc9452ce5c6
```

Important model facts:

| Property | Value |
|---|---|
| Architecture | `NemotronHForCausalLM` |
| Model type | `nemotron_h` |
| Layers | 52 |
| Layer counts | 23 Mamba-2, 23 MoE, 6 attention |
| Attention indices | 5, 12, 19, 26, 33, 42 |
| Attention | 32 query heads, 2 KV heads, head dimension 128 |
| Mamba-2 | 64 heads, head dimension 64, state 128, convolution 4, 8 groups |
| MoE | 128 routed experts, 6 per token, one shared expert |
| Mamba execution | Chunk size 128, recurrent state `float32` |
| MoE intermediate widths | 1856 routed, 3712 shared |
| RoPE | Full rotary factor 1.0, theta 10000 |
| Hidden size | 2688 |
| Vocabulary | 131072 |
| Context | 262144 |
| Quantization | Affine, group size 64, 4 bits |

The native MTP head is embedded under `mtp.*`. It has 34 tensors and one
`[attention, moe]` hidden layer. The arm is single-stream.

The KV backend is `contiguous`. An unsupported backend is refused.

## 3. What you may edit

`benchmark.json` `editablePaths` lists 94 entries and is the complete
authority. The surface has five groups:

1. `mtp-head.manifest.json`
2. `Sources/MLXFastTransform/`
3. The listed Nemotron model files
4. The listed batching, cache, attention, RoPE, and server files
5. The listed MLX and Metal kernel files

The two vendor directories contain plain files. They are not submodules.

Everything that verifies, measures, scores, or records trusted state is outside
the editable surface.

### 3.1 Optional path

`mtp-head.manifest.json` is optional in the declaration parser. If it is absent,
the embedded head is used with serial mode unless the request selects an allowed
speculative depth.

The archive overlay uses replace semantics. A missing optional declaration keeps
the trusted copy.

Ranked whole-surface static review is different: every path in `editablePaths`
must exist in the submission checkout. Optional status does not waive that
requirement on the ranked path.

### 3.2 Byte budget

| Key | Value |
|---|---:|
| `maxTotalBytes` | 7,949,663 |
| `maxFileBytes` | 524,288 |
| `maxGrowthBytes` | 262,144 |
| `exemptPathMaxBytes` | 512,000,000 |
| `exemptPathMaxFileBytes` | 100,000,000 |

No path is exempt.

### 3.3 Frozen target quantization

Do not re-quantize, re-represent, or change the numerical format of a target
weight. An editable transform does not permit a target-format change.

Passing the token-tolerance gate does not authorize target re-quantization.

The MTP head is the only exception. It may be re-quantized on load. It may not
be replaced.

Enforcement uses checkpoint integrity and transform validation, loaded
configuration equality against the target, and a full parameter update with
verification. Do not assume extra automatic loaded-target checks at each measured
window unless the current code path proves them.

### 3.4 Correctness goldens

Public correctness JSON files identify the pinned checkpoint through
`model_provenance`. The model-specific golden loader refuses a file with missing
or mismatched `model_provenance`.

## 4. Embedded MTP head

The checkpoint contains the only accepted head. No head-weight file is staged
or submitted.

### 4.1 Declaration

`mtp-head.manifest.json` accepts `"source": "pinned"` only.

An absent declaration selects the pinned default. In a present JSON object, an
absent or non-string `source` also defaults to `"pinned"`. An unknown string
source, `"remote"`, and `"in_branch"` are refused. An unreadable declaration or
a top-level value that is not a JSON object is refused.

The declaration may select a draft depth:

```json
"spec": { "enabled": true, "num_speculative_tokens": 3 }
```

`tools/spec-declaration.sh` accepts structural integers from 0 through 8.
Disabled speculation, depth 0, or no `spec` block resolves to serial mode.

When speculation is enabled, a positive `num_speculative_tokens` must be a member
of the fixture's permitted set 1 through 7. Enabled depth 8 is refused. Disabled
depth 8 passes structural validation and still resolves to serial.

### 4.2 Declaration size fields

Missing `max_bytes`, or a value that does not bridge to `NSNumber`, defaults to
2 GiB. A bridged value is read through `NSNumber.intValue`. The result must be
positive and must not exceed 2 GiB. A declaration may lower the cap and may not
raise it.

Missing `bytes`, or a value that does not bridge to `NSNumber`, defaults to 0.
A bridged value is read through `NSNumber.intValue`, and 0 means the byte count
is unstated. A negative value or a value above `max_bytes` is refused. This
gate checks declared size, not loaded head content.

### 4.3 Digest limit

A declared SHA-256 is optional. The runner does not compare it with the loaded
head.

The head bytes are part of the target checkpoint. `./setup.sh` verifies the
downloaded checkpoint files against
`fixtures/reference_nemotron3_5_30b_a3b_4bit.sha256`.

### 4.4 In-memory head re-quantization

A re-quantization happens ON LOAD, in memory. It writes no checkpoint file.

The editable seam is:

```text
Vendor/mlx-swift-lm/Libraries/MLXLLM/Models/NemotronH35MTPLoading.swift
```

The loader reads the checkpoint `config.json`, requires quantization metadata for
scaled `mtp.*` paths, applies that checkpoint metadata during bind, then verifies
all weights after update. It does not apply separate numeric group-size, bit-width,
or override-count guards beyond what the checkpoint metadata supplies.

The module-path rule is:

| Module | Rule |
|---|---|
| `mtp.*` | Head re-quantization is allowed |
| `mtp.*` naming `embed_tokens` or `lm_head` | Refused |
| All target modules | Frozen |

The shared embeddings and output projection are target weights. They are not
part of the exception.

The head proposes tokens. The pinned target model decides every emitted token.

## 5. Scoring

### 5.1 Formula

```text
composite = prefill_gain ^ 0.25 * decode_gain ^ 0.75
gain = control_leg_seconds_per_token / candidate_leg_seconds_per_token
```

Higher is better. `benchmark.json` sets `direction` to `+`.

Live control seconds per token is the numerator in each gain. Candidate seconds
per token is the denominator.

The fixture sets `prefill_speedup_floor = 0.95` and
`decode_speedup_floor = 0.95`. The manifest restates the decode floor as
`decodeSpeedupFloor = 0.95`. The manifest sets `decodeSpeedupCeiling = 5.0` as
the declared decode gain ceiling, separate from the two gain floors.

The benchmarker seals an unrounded double. Equal sealed values tie, and the
platform decides whether a submission improves the current best.

Calibration refuses to record a band when either coefficient of variation exceeds
1 percent on prefill or decode.

In a ranked run, the variance limit is the calibration health band (section
5.3): every pair's control leg must fall inside it, or the run stops and writes
no score (section 5.2).

### 5.2 Paired measurement

The fixture sets `official_pairs = 3` and `official_pair_combine` to the
arithmetic mean. `official_pairs` must be a multiple of the live-pool size. The
current live pool has three prompts, so the current value satisfies that rule.

Pair `k` uses live prompt `(k - 1) mod N`, where `N` is the live-pool size.

Each pair has:

1. a serial control leg at the fixture's `baseline_reference_commit`;
2. the candidate leg at the declared depth.

Both legs run on the same machine in the same job. The official score is the
arithmetic mean of the three per-pair composites.

`official_pair_combine` is the authority for how pair composites combine. Its
current value is `mean`.

Every pair must independently pass the gain floors and the calibration health
bands on its control leg. The first failed pair stops the run. Passing pairs
cannot offset a failed pair.

The artifact retains each measured pair in `metrics.paired_legs`. It names the
combine rule in `metrics.official_pair_combine`. The reported means of gains and
timings are separate from that mean composite.

No file stores a baseline pair. The control leg is measured live.

The fixture's reference revision is public and can be inspected in repository
history. The staged reference workspace is organizer material and is not
available to participants.

### 5.3 Calibration health band

Each ranked machine has a calibration file for its control leg. It is a health
band, not a denominator. A stale or mismatched file can stop a run but cannot
move its score.

`tools/calibrate-box.sh` defaults to three passes. It refuses to write a file
when either coefficient of variation exceeds 1 percent.

A ranked run refuses a calibration file when any of these fail:

- `version` is not 3;
- `track_id` does not match this track;
- `box` does not match the runner identity when the runner requires it;
- `reference_commit` does not match the fixture's `baseline_reference_commit`;
- live-pool prompt coverage is incomplete;
- `captured_at` is not later than the reference commit date;
- any numeric value is not finite and positive;
- a band does not straddle 1 (`low < 1 < high`).

Complete field shape, shown as a non-loadable template. Replace every
placeholder and include one entry for each live-pool prompt before use:

```json
{
  "version": 3,
  "track_id": "nemotron3.5-30b-a3b-mlx-v1",
  "box": "<runner identity when required>",
  "reference_commit": "<fixture baseline_reference_commit>",
  "captured_at": "<ISO 8601 UTC>",
  "benchd_source_commit": "<40 hex>",
  "prompts": [
    {
      "prompt": "<live-pool name>",
      "passes": 3,
      "prefill_seconds_per_token_mean": 0.0101,
      "decode_seconds_per_token_mean": 0.0124,
      "prefill_cv": 0.005,
      "decode_cv": 0.004,
      "prefill_band_low": 0.95,
      "prefill_band_high": 1.05,
      "decode_band_low": 0.95,
      "decode_band_high": 1.03
    }
  ]
}
```

### 5.4 Measured window

| Quantity | Value |
|---|---:|
| Seed tokens | 488 |
| Checked decode steps | 128 |
| Streams | 1 |
| Timed prompts per leg | 1 |
| Pairs | 3 |
| Legs per pair | 2 |
| Warm-up prefill passes | 1 |

Timing uses the benchmarker's parent clock:

| Window | From | To |
|---|---|---|
| Prefill | `free_decode_begin` sent | validated seed reply |
| Decode | seed reply | `free_decode_run(N)` completion |

Decode seconds per token is the decode window divided by N. Seed prefill is not
part of decode.

Both legs receive one unmeasured prefill warm-up before the timed window.

For `free_decode_begin`, the worker processes all 488 seed tokens before it
replies. The reply includes the greedy seed token and `effective_spec`. The
scored runner advertises only `free_run_decode`.

Do not prefill before `free_decode_begin`, re-prefill the seed during
`free_decode_run`, or perform future decode work inside `free_decode_begin`.

### 5.5 Token fidelity

The benchmarker applies a 10% per-stream token-tolerance budget.

The candidate leg is checked against the oracle for its declared depth. The
serial control leg is checked against the serial tape. The gate permits output
within the published tolerance; it does not require token-for-token identity
with a serial trajectory.

Fixed verification mode is `rectangular_exact`. An unsupported batch width or
shape capability is refused before measurement.

### 5.6 Live pool and arming

`official_scoring_enabled` is `true`. A false or absent flag prevents an
official score.

The fixture names `pool-14`, `pool-15`, and `pool-16`. It publishes these pins:

| Name | SHA-256 | Bytes |
|---|---|---:|
| `pool-14` | `428f6d3335104372168a9727c478578bf5491d4a1599c10655ed5cf91a3a23a3` | 32193 |
| `pool-15` | `1490c02337134a410189b78fc32a392b9f8188eccf1a96e8bcc35b3b2ed73e40` | 30434 |
| `pool-16` | `78d742439dda7197471c0ea8e9b5c5133c608f49978c240bc4d8afe22e846b81` | 30738 |

Every permitted MTP depth maps to the same pinned live pool.

Ranked preflight refuses missing or mismatched staged tapes, malformed or
duplicate pins, pending-organizer sentinels, and extra staged JSON files.
It also refuses an empty `timed_prompt_pool`, an empty `live_goldens` list, or
a live name with no corresponding timed-pool entry.

`tools/nemotron35-30b-a3b-measure-and-score.sh` refuses a requested depth when
its oracle is missing. It also refuses a speculative declaration when the
resolved benchmarker lacks `--mtp-depth` or `--control-golden` support.

Hidden artifacts are provisioned by the organizer. No other hidden-artifact
metadata is part of the participant contract.

### 5.7 Thermal and quiescence gates

Before each timed phase, the benchmarker applies:

1. Quiescence: one-minute load average below 2.0 and GPU utilization below 10%,
   polled every 15 seconds, refused after 900 seconds.
2. Cooling: GPU at or below 50 °C, refused when the GPU does not cool down in time.

Missing, unreadable, frozen, or implausible ranked temperature input refuses
the run.

A control leg outside its calibration band seals no score.

### 5.8 Official composite

An official score requires a composite from accepted pairs. A diagnostic decode
median or other partial metric cannot substitute for it.

The benchmarker seals one unrounded double score. Equal sealed values tie.

## 6. Running the benchmark

### 6.1 Local

`benchmarkCommand` selects a local path when ranked environment variables are
absent. The local path transforms weights when needed and runs
`./benchmark.sh --local-submit`. It returns `"score": null`.

The public captures are:

- `correctness_prompts/public_longcopy_gate_english_1024_256.json`
- `correctness_prompts/public_longcopy_gate_english_1024_1024.json`

They use 488 prompt tokens and identify the pinned checkpoint through
`model_provenance`.

### 6.2 Pre-submit dry run

The manifest defines this `preSubmitCommand`:

```bash
./tools/nemotron35-30b-a3b-measure-and-score.sh --preflight-only
```

This command checks the declaration and the official-scoring arm gate. If
staged live goldens are available, it also checks their pins. It loads no model,
writes no score, and needs no reference workspace.

### 6.3 Ranked routing

Ranked behavior is selected by environment, not by host name alone:

- `GITHUB_ACTIONS=true`, or
- `MLXFAST_QWEN38_GOLDEN_DIR`, or
- `MLXFAST_BASELINE_WORKSPACE`, or
- `MLXFAST_BASELINE_CALIBRATION`.

A partial ranked environment can refuse instead of falling back to local mode.

### 6.4 Ranked workflow

`.github/workflows/benchmark.yml` is the ranked workflow. It targets the
self-hosted runner labels declared in that workflow.

The workflow runs hosted manifest and static-review checks before the
self-hosted ranked job. Ranked preflight verifies staged artifacts, the
reference workspace, and calibration data before measurement.

Reference-workspace checks include the pinned git revision, an executable worker,
a sibling `mlx.metallib` with its fingerprint sidecar, and that workspace's
transformed weights.

The candidate engine and transformed-weight paths must resolve inside the
candidate checkout. The measure script uses checkout-relative paths to resolve
the control leg's corresponding assets in the reference workspace.

The ranked job uses one resident worker per leg. The benchmarker boots and
stops each resident from that leg's tree.

The measure script holds the exclusive GPU lock across both legs.
`tools/resident-up.sh` refuses to boot a resident when that lock is not held.

Ranked runs reject an inherited `BENCH_WORKER_RESIDENT_SOCKET`, because sharing
one resident would corrupt control and candidate separation.

The official benchmarker writes `score.json`. The engine does not write the
score.

## 7. Pinned artifacts

| Artifact | Identity |
|---|---|
| Target model | `EigenLabs/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit-mtp@153c469494b3f16a0eaa752dae25cfc9452ce5c6` |
| Target manifest | `fixtures/reference_nemotron3_5_30b_a3b_4bit.sha256` |
| MTP head | Embedded under `mtp.*` |
| Engine upstream revision | `859f4d97e0d1ae9cce0883b044f764283841edfb` |
| Serial reference revision | `32b75d2e7d8fc1517528ff058a732f40fffa7ed4` |

The target manifest pins 11 files totaling 18,544,229,436 bytes. Participants
do not supply target weights.

## 8. Submission enforcement

The enforced layers include:

- archive overlay from trusted `editablePaths`;
- modifiable-surface checks;
- deterministic static review;
- byte-budget checks;
- sandboxed build, transform, and worker execution;
- ranked staged-asset and workspace checks.

See [submission-restriction-spec.md](submission-restriction-spec.md).

### 8.1 Prohibited techniques

- Changes outside `editablePaths`
- Hardcoded prompt text, hidden token identifiers, or answers
- Whole-request caches that only exploit harness repetition
- Timing shortcuts
- Protocol injection
- Network access
- Filesystem exfiltration
- Target-weight replacement or re-quantization

Input-independent caches and within-request KV reuse are allowed.

## 9. Submitting

Use Yukon for account and submission operations. See
[README.md](../README.md#submitting).

A submission contains editable paths only. Generated artifacts, symlinks,
scores, checkpoints, and out-of-surface changes are refused.

The ranked run is the scoring authority.

## 10. License

The target checkpoint uses OpenMDW-1.1. This repository distributes no model
weights. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## 11. Participant limitations

- Hidden pool artifacts are not in git and are unavailable locally.
- The staged serial reference workspace is unavailable to participants; its
  revision is in the fixture.
- Local runs do not produce an official score.
- The scored shape is single-stream with verification mode `rectangular_exact`.
- The supported KV backend is `contiguous`.
- Ranked results depend on the verified benchmarker and the ranked workflow
  host environment.
