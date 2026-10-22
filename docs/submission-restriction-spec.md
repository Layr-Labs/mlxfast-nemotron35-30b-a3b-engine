# Submission restrictions

This document summarizes the current enforced rules for participant
submissions. `benchmark.json` is the authority for paths and byte limits.

## 1. Editable surface

A submission may change only `benchmark.json` `editablePaths`.
`optionalEditablePaths` contains only `mtp-head.manifest.json`.

The trusted contract supplies the allowlist. A submission cannot widen its own
surface or make a required path optional.

Changes to the benchmarker, trusted harness, package manifests, fixtures,
scripts, workflows, or other unlisted paths are refused.

Ranked whole-surface review requires every editable path to exist in the
submission checkout. Optional declaration status does not waive that rule.

## 2. Archive overlay

`.github/scripts/overlay-editable-paths.sh` applies submitted paths to a
trusted checkout.

Rules:

- editable paths use replace semantics, not merge semantics;
- a missing required path is refused;
- a missing optional path keeps the trusted copy;
- symlinks are refused;
- non-regular files are refused;
- hardlinked files are refused;
- invalid or escaping paths are refused;
- paths that reach the measurement-harness surface are refused.

Invalid paths include empty paths, absolute paths, backslashes, a leading `:`
(git pathspec magic), and any `.` or `..` path component. The harness also
guards against case-folded and filesystem-alias escapes.

Setuid and setgid bits on source files can be stripped during copy. The guarantee
is that copied output has no setuid or setgid bit, not that source mode bits
always cause refusal.

Stale-clone diagnostics appear when required editable paths are missing from the
submission archive. The reviewer reports deletion hints and advises resync and
resubmit. Report-only stale-overlay warnings alone do not refuse a submission.

The script reads `editablePaths` and `optionalEditablePaths` from the trusted
contract.

## 3. Modifiable-surface gate

`.github/scripts/enforce-modifiable-surface.sh` compares the base and submitted
commits. Every changed path must equal an allowed file or be below an allowed
directory.

The allowlist comes from the base contract. An absent, unreadable, or empty
allowlist is a refusal.

## 4. Deterministic static review

`.github/scripts/submission-static-review-checks.sh` checks:

- contract syntax;
- track identity;
- editable-path syntax;
- file types;
- per-file bytes;
- total bytes;
- growth bytes;
- diff size;
- required review mode and commit identity.

In diff mode, limits and paths come from the base commit. Whole-surface mode is
an explicit opt-in for a trusted checkout.

Diff mode charges each changed file's full blob size plus its unified diff
against the total byte cap. Net growth is computed from committed base and head
blob sizes and is enforced only in diff mode. The checked-in ranked workflow
uses whole-surface review and does not run growth or diff-size accounting.

Review refuses an empty base commit, contradictory or missing review mode,
unresolved base or head SHAs, a work tree that is not checked out at the review
head, and an empty or mismatched track id.

The public automated policy in this repository is deterministic. The ranked
workflow runs whole-surface review before the ranked job.

## 5. Byte budget

The manifest declares:

| Limit | Bytes |
|---|---:|
| Total editable surface | 7,949,663 |
| One editable file | 524,288 |
| Growth | 262,144 |
| Exempt aggregate fallback | 512,000,000 |
| Exempt file fallback | 100,000,000 |

No path is exempt. The exempt fallback values remain configuration checks but
cannot bind while `exemptPaths` is absent.

Malformed limits are refused. Each limit must be a positive integer JSON
number. Null, string, boolean, and fractional values are rejected.
`maxFileBytes` must not exceed `maxTotalBytes`. Retired environment overrides
for these caps are not honored.

The hosted whole-surface static review enforces the per-file and total limits.
`Sources/MLXFastTrustedHarness/EditableSurfaceByteBudget.swift` has standalone
CLI and test callers in this repository. It has no automatic launch-time call
site here. A surface that resolves to no files is refused.

## 6. MTP declaration

The MTP head is embedded in the pinned checkpoint. A submission can carry the
declaration only.

- `"source": "pinned"` is accepted.
- `"source": "remote"` is refused.
- `"source": "in_branch"` is refused.
- The declaration cap is 2 GiB.
- A declared SHA-256 is optional and is not compared with the loaded head.
- No head weight file is accepted.

The head may be re-quantized only on load and in memory. Target-model
quantization is frozen.

## 7. Sandbox

The ranked workflow runs participant-controlled build and worker code in
restricted macOS sandbox profiles. It does not run the transform: it copies the
reference workspace's transformed weights. The transform profile below applies to
the local `mlxfast-swift transform` command.

The transform profile allows writes to the `--output` directory, sibling
`.<output-name>.*` staging paths, and a private `TMPDIR`. It denies process
forks and all process execution except the named CLI that `sandbox-exec`
starts.

The build profile allows child processes. It allows writes only to its named
build directories, required account temporary directories, and a private
`TMPDIR`.

The resident profile allows writes to its private `TMPDIR` and its named Unix
socket. It permits traffic only through that Unix socket. It denies process
forks and all process execution except the named worker that `sandbox-exec`
starts. All other network access is denied.

The exact rules are generated by `tools/seatbelt-profile.py` and applied by the
trusted wrappers.

The sandbox is one layer. Path allowlists, archive checks, byte limits, staged
asset checks, and workspace-divergence checks remain separate enforcement
layers.

## 8. Ranked enforcement

`.github/workflows/benchmark.yml` is the ranked workflow. It checks out
`github.sha` and runs whole-surface static review in the hosted job. It does not
call a separate overlay-validation script in that workflow file. Any separate
overlay stage outside that workflow is external and is not verified here.

Before measurement the self-hosted job:

1. checks the runner environment;
2. verifies staged assets and the reference workspace;
3. resolves the verified benchmarker;
4. builds and transforms in the sandbox;
5. runs the official paired measurement.

The ranked workflow holds no repository credential. Hidden artifacts are
provisioned by the organizer and verified against the public SHA-256 and byte
pins in the fixture.

## 9. Source and artifact integrity

The manifest linter refuses editable paths that overlap trusted source,
package manifests, tools, workflows, or the measurement harness.

The checked-in ranked workflow builds the checkout at `github.sha`; it does not
run the overlay script. Staged checkpoint and hidden artifacts are verified
before use. The official benchmarker writes the score.

CI and ranked workflow details are in [ci-coverage.md](ci-coverage.md).

## 10. Prohibited techniques

A submission is refused for:

- changes outside the editable surface;
- prompt- or answer-specific code;
- hidden token identifiers;
- whole-request replay caches that only exploit harness repetition;
- timing shortcuts;
- protocol injection;
- network access;
- filesystem exfiltration;
- target-weight re-quantization or replacement.

Input-independent caches and within-request KV reuse are allowed.
