# Agent guide — Nemotron 3.5 Lightning 30B A3B MLX

Read [README.md](README.md) first.

The ranked track is `nemotron3.5-30b-a3b-mlx-v1`.

## Goal

Make target-model prefill and decode faster on Apple Silicon. Keep output
within the published token-tolerance gate.

## Authorities

| Question | Authority |
|---|---|
| Editable paths, commands, and scoring values | `benchmark.json` |
| Pins, live pool, and scoring semantics | `fixtures/nemotron3_5_30b_a3b_mlx_v1_track.json` |
| Participant rules | `docs/participant-contract.md` |
| Measured behavior | Verified channel benchmarker |

The fixture and manifest hold the current values. The fixture wins if this
guide disagrees with it.

## Current state

- Official scoring is armed.
- The fixture pins the live pool and a separate, never-scored warm-up golden.
- The ranked workflow targets the self-hosted labels in
  `.github/workflows/benchmark.yml`.
- Scoring is paired and single-stream. Timed prefill and decode use different
  live prompts. Decode includes the seed window.
- The candidate leg runs at its declared MTP depth.
- The serial control leg runs at `baseline_reference_commit`.
- No file stores a baseline pair.
- The per-machine calibration file is a health band, not a denominator.

`Vendor/mlx-swift-lm` and `Vendor/mlx-swift` are vendored trees of plain
files. Edit only paths listed in `benchmark.json`. The offline transform is
trusted and is not editable. Do not push vendored changes to an upstream
repository.

## Local measurement

Use `./benchmark.sh`. It passes `--cool-gate` to the benchmarker.

```bash
MLXFAST_ENGINE_BIN=.build/release/bench-worker \
MLXFAST_CORRECTNESS_GOLDEN_PATH=correctness_prompts/public_longcopy_gate_english_1024_256.json \
  ./benchmark.sh --local-iterate
```

The local test is serial unless you pass `--mtp-depth N`. Use
`tools/spec-declaration.sh draft-len` to read the declared depth.

### Cool-down gate

The benchmarker gates prefill and decode separately. A pause for cooling is
normal. Do not stop a run only because it is waiting for temperature.

Before each timed phase it also waits for quiescence: one-minute load average
below 2.0 and GPU utilization below 10%, polled every 15 seconds, refused after
900 seconds. Cooling requires GPU at or below 50 °C, refused when the GPU does not cool down in time.
Missing, unreadable, frozen, or implausible ranked temperature input refuses the
run.

`benchd-bin/benchd --local-cool-gate-only` probes the cooling gate.
`./benchmark.sh --local-cool-gate-only` exits without probing.

`./setup.sh` installs a pinned `macmon` binary unless
`MLXFAST_SKIP_MACMON_INSTALL=1` is set. A local run without a working
temperature reader is not a trusted timing.

If cooling stalls, use the manual fan helper:

```bash
tools/fan-control.sh boost
tools/fan-control.sh normal
tools/fan-control.sh status
```

The write operations use `sudo`. Return the fans to `normal` after the run.

### Measurement discipline

- Measure on a cool, quiet machine.
- Record a same-machine baseline from the current branch tip.
- Allow the machine to cool between runs.
- Treat local results as directional. They do not predict the ranked score.
- Run only one model-holding command at a time.
- Check for an orphaned `bench-worker` after an aborted run.

On non-M5 hardware, a near-tie argmax can differ from the ranked hardware. If
an unmodified tree fails at the same token, the local-only
`MLXFAST_LOCAL_ALLOW_GOLDEN_DRIFT=1` option can publish a timing estimate. It
does not turn the correctness result into a pass.

## Building

Always preserve the resolved dependency graph:

```bash
swift build -c release --force-resolved-versions
swift build -c release --force-resolved-versions \
  --scratch-path .build-worker --product bench-worker
tools/stage-bench-worker.sh
```

The scored binary is `.build-worker/release/bench-worker`. The staging command
copies it and `mlx.metallib` to `.build/release`.

Never run `swift package update`. A bare build can rewrite
`Package.resolved`; always pass `--force-resolved-versions`.
Do not run bare `swift package resolve`; pass `--force-resolved-versions`.
`./setup.sh` refuses `Package.swift` or `Package.resolved` when either file
differs from `HEAD`.

### Kernel edits

Kernel families with an `mlx-generated/*.cpp` twin compile at run time from
the generated source string. Update the generated twin and the readable Metal
source together.

RoPE, RMSNorm, SDPA vector, and `arg_reduce` load from `mlx.metallib`. Rebuild
it after editing an ahead-of-time source:

```bash
tools/build-mlx-metallib.sh
```

The ranked hardware selects `_nax` variants. Keep the plain and `_nax`
implementations in step when both exist.

## Useful commands

```bash
swift test --force-resolved-versions
./tools/fetch-benchd.sh
./setup.sh
```

The root tests have no MLX runtime opt-in. Run vendored test targets separately
as described in [CI coverage](docs/ci-coverage.md).

## Where to optimize

- MTP draft and verify work
- Layer dispatch
- Mamba-2 convolution, scan, and recurrent state
- Quantized matmul and MoE gather-GEMM
- KV-cache handling
- Weight preparation and reuse
- MLX scheduling and synchronization

## Prohibited strategies

- Do not specialize for a public or hidden prompt.
- Do not hardcode prompt text, token identifiers, or answers.
- Do not cache a whole request only because a harness might repeat it.
- Do not use timing shortcuts, protocol injection, network access, or
  filesystem exfiltration.
- Do not change target-weight representation or quantization.

Input-independent caches and within-request KV reuse are allowed.

The MTP head is the only quantization exception. You may re-quantize the
embedded head on load and in memory. You may not replace it or upload head
weights. The target model decides every emitted token.

## Before submitting

Run:

```bash
swift test --force-resolved-versions
swift build -c release --force-resolved-versions
./setup.sh
./tools/fetch-benchd.sh
```

Then run a local benchmark through `./benchmark.sh`.

Use Yukon for account and submission operations. See [README.md](README.md).
