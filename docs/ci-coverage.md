# CI coverage

`.github/workflows/ci.yml` is the hosted CI pipeline. It checks repository
hygiene, builds the Swift package, and runs tests that need neither the target
checkpoint nor a real Metal GPU.

The ranked pipeline is `.github/workflows/benchmark.yml`.

## Hosted CI

The `lint` job runs on `ubuntu-latest`. It checks:

- workflow egress patterns
- workflow syntax
- tracked shell syntax
- `benchmark.json`
- the box-only test inventory
- benchmark wrapper environment behavior
- benchmarker fetch verification
- golden and track tooling
- resident-worker behavior
- draft-depth declarations
- ranked preflight inputs
- calibration command construction
- reference-workspace staging commands

The `swift` job runs on `macos-26`. It checks:

- Swift and Xcode versions
- submission-security assertions
- `swift build --build-tests --force-resolved-versions`
- first-party compiler warnings
- sandbox wrappers under macOS Seatbelt
- the frozen dependency graph
- `swift test --force-resolved-versions --no-parallel`

The workflow is the source of truth for the exact commands and assertion
floor.

## What hosted CI does not run

Hosted CI does not:

- download the target checkpoint
- build and exercise the full production Metal library on a physical GPU
- run model correctness against hidden artifacts
- measure prefill or decode
- produce a ranked score

The current generated box-only test inventory is empty. The inventory command
still runs so a newly gated test cannot appear without updating the record:

```bash
tools/ci-box-only-inventory.sh --check
```

`MLXFAST_RUN_MLX_RUNTIME_TESTS=1` currently enables no additional root tests.
No file under the root `Tests/` directory reads that variable. Root
`swift test` also does not select the vendored package's test targets.

Run the vendored package's tests separately:

```bash
swift test --package-path Vendor/mlx-swift-lm
```

Some vendored tests use their own opt-in variables and require a complete
`mlx.metallib`. For example, the full AOT parity tests read
`MLXLM_FULL_AOT_METALLIB=1`.

## Ranked workflow

`.github/workflows/benchmark.yml` starts with a hosted editable-surface and
static-review job. The self-hosted ranked job depends on that job.

The ranked job then:

1. checks the runner environment;
2. verifies staged assets and the reference workspace;
3. resolves the verified benchmarker;
4. builds or restores the scored worker;
5. copies the reference workspace's transformed weights (the transform is organizer-owned);
6. runs the official paired measurement;
7. uploads `score.json` when it exists.

The ranked workflow targets the self-hosted labels declared in that file.
Missing or invalid required assets cause a refusal.

## Limits of the checks

The workflow egress scan checks named patterns. It is not a proof that arbitrary
code cannot communicate.

The submission-security suite checks the enforced archive, path, file-type,
and byte-budget rules. It does not replace ranked correctness or measurement.

The ranked workflow and sandbox are the enforcement points for code that needs
the checkpoint, hidden artifacts, or physical Apple Silicon.
