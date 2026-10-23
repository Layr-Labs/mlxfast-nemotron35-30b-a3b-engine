#!/usr/bin/env bash
#
# test-sandboxed-cli.sh -- the transform confinement starts outside the CLI.
#
# WHAT IS UNDER TEST.
#   tools/seatbelt-profile.py  the one profile generator (the transform
#                              wrapper and the resident use it)
#   tools/sandboxed-cli.sh     the trusted wrapper that starts the CLI under
#                              sandbox-exec
#
# Part 1 is hermetic and runs on any host: the profile rules, and the
# wrapper's refusals before it starts anything. Part 2 needs macOS
# (/usr/bin/sandbox-exec) and a built CLI (MLXFAST_SWIFT_BIN, default
# .build/release/mlxfast-swift). It runs the real CLI with canary evaluator
# paths in a temporary directory. When either is missing, part 2 is reported
# as not run. No weights, no GPU, no network.
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
WRAPPER="${REPO_ROOT}/tools/sandboxed-cli.sh"
GENERATOR="${REPO_ROOT}/tools/seatbelt-profile.py"
WORK="$(cd "$(mktemp -d)" && pwd -P)"
trap 'chmod -R u+w "${WORK}" 2>/dev/null; rm -rf "${WORK}"' EXIT

passes=0
failures=0
pass() { passes=$((passes + 1)); echo "ok    $*"; }
fail() { failures=$((failures + 1)); echo "FAIL  $*" >&2; }

# The canary evaluator paths. Every one exists, so a deny is observable.
C="${WORK}/canary"
PRIVATE_AUDIT_DIR="/private/var/mlxfast-private-audit-test-$$"
mkdir -p "${C}/goldens" "${C}/benchd-bin" "${C}/baseline" "${C}/private" \
  "${C}/resident-root/job" "${C}/runner/_work/repo" \
  "${WORK}/home/actions-runner" "${WORK}/home/actions-runner-x" "${WORK}/home/unrelated"
for directory in actions-runner actions-runner-x unrelated; do
  for name in .credentials .credentials_rsaparams .runner; do
    printf 'runner state\n' > "${WORK}/home/${directory}/${name}"
  done
done
printf '{"weight_map":{"a":"model.safetensors"}}\n' > "${C}/goldens/canary.golden.json"
printf 'benchd\n' > "${C}/benchd-bin/benchd"
printf '{}\n' > "${C}/calibration.json"
canary_env=(
  MLXFAST_QWEN38_GOLDEN_DIR="${C}/goldens"
  MLXFAST_BASELINE_WORKSPACE="${C}/baseline"
  MLXFAST_BASELINE_CALIBRATION="${C}/calibration.json"
  BENCHD_BIN_DIR="${C}/benchd-bin"
  BENCHD_PRIVATE_AUDIT_PATH="${PRIVATE_AUDIT_DIR}/private-audit.json"
  MLXFAST_PRIVATE_DIR="${C}/private"
  MLXFAST_RESIDENT_LOG_ROOT="${C}/resident-root"
  RESIDENT_UP_LOG_DIR="${C}/resident-root/job"
  RUNNER_WORKSPACE="${C}/runner/_work/repo"
)

# --- part 1: the profile generator -------------------------------------------
state_rules=(
  "(deny ipc-posix-shm*)"
  "(deny ipc-sysv*)"
  "(deny ipc-posix-sem*)"
  "(deny user-preference-write)"
  "(deny mach-lookup)"
  "(deny signal)"
  "(allow signal (target self))"
  '(allow mach-lookup (global-name "com.apple.MTLCompilerService") (global-name "com.apple.system.opendirectoryd.libinfo") (global-name "com.apple.system.opendirectoryd.membership") (global-name "com.apple.logd") (global-name "com.apple.diagnosticd"))'
)
profile="$(env -i PATH="${PATH}" HOME="${WORK}/home" "${canary_env[@]}" python3 "${GENERATOR}" \
  --exec /bin/bash --tree "${WORK}/tree" --write-subpath "${WORK}/out" --write-prefix "${WORK}/.out." \
  --unix-socket "${WORK}/resident.sock" --official)"
rc=$?
if [[ "${rc}" -ne 0 ]]; then
  fail "generator: exit ${rc} with every evaluator path set"
else
  for path in "${C}/goldens" "${C}/baseline" "${C}/calibration.json" "${C}/benchd-bin" "${C}/private" \
    "${PRIVATE_AUDIT_DIR}" "${C}/resident-root" \
    "${WORK}/home/.cache/mlxfast-engine-build" "${C}/runner/.runner" "${C}/runner/.credentials"; do
    grep -qF "(deny file-read* file-write* (subpath \"${path}\"))" <<< "${profile}" \
      || fail "generator: no read and write deny for ${path}"
  done
  # The generator resolves the program path, and /bin is a symlink on Linux.
  bash_path="$(python3 -c 'import os; print(os.path.realpath("/bin/bash"))')"
  grep -qF "(allow process-exec (literal \"${bash_path}\"))" <<< "${profile}" \
    || fail "generator: the named program is not the one exec allowed"
  grep -qF "(allow file-write* (subpath \"${WORK}/out\"))" <<< "${profile}" \
    || fail "generator: the output tree is not writable"
  grep -qF "(allow file-write* (regex #\"^${WORK//./\\.}/\\.out\\.\"))" <<< "${profile}" \
    || fail "generator: the staging prefix is not an escaped regex"
  grep -qF "(allow network-bind network-inbound network-outbound (local unix-socket (path-literal \"${WORK}/resident.sock\")))" <<< "${profile}" \
    || fail "generator: the resident socket has no local network allowance"
  grep -qF "(allow file-write* (literal \"${WORK}/resident.sock\"))" <<< "${profile}" \
    || fail "generator: the resident socket has no file-write allowance"
  last_write_allow="$(awk '/^\(allow file-write\*/ { line = NR } END { print line + 0 }' <<< "${profile}")"
  flags_deny="$(grep -nFx '(deny file-write-flags)' <<< "${profile}" | cut -d: -f1)"
  acl_deny="$(grep -nFx '(deny file-write-acl)' <<< "${profile}" | cut -d: -f1)"
  first_evaluator_deny="$(grep -nF '(deny file-read* file-write* ' <<< "${profile}" | head -n 1 | cut -d: -f1)"
  if [[ -n "${flags_deny}" && -n "${acl_deny}" && -n "${first_evaluator_deny}" \
     && "${flags_deny}" -gt "${last_write_allow}" \
     && "${acl_deny}" -gt "${flags_deny}" \
     && "${acl_deny}" -lt "${first_evaluator_deny}" ]]; then
    pass "generator: runtime flag and ACL denies follow all write allowances and precede evaluator denies"
  else
    fail "generator: runtime flag and ACL deny order is wrong"
  fi
  [[ "$(tail -n 1 <<< "${profile}")" == "(deny file-read* file-write* "* ]] \
    || fail "generator: the evaluator denies are not the last rules"
  for rule in "(deny network*)" "(deny system-sched)" "(deny process-fork)" "(deny process-exec*)" "(deny file-write*)"; do
    grep -qF "${rule}" <<< "${profile}" || fail "generator: no ${rule}"
  done
  for rule in "${state_rules[@]}"; do
    grep -Fxq "${rule}" <<< "${profile}" || fail "generator: non-build profile has no exact ${rule}"
  done
  expected_home_runner_rules="$(
    for directory in actions-runner actions-runner-x; do
      for name in .credentials .credentials_rsaparams .runner; do
        printf '(deny file-read* file-write* (subpath "%s"))\n' "${WORK}/home/${directory}/${name}"
      done
    done
  )"
  actual_home_runner_rules="$(grep -F "(subpath \"${WORK}/home/actions-runner" <<< "${profile}" || true)"
  if [[ "${actual_home_runner_rules}" == "${expected_home_runner_rules}" ]] \
    && ! grep -qF "(subpath \"${WORK}/home/unrelated/" <<< "${profile}"; then
    pass "generator: the runtime profile denies six sibling-runner registration files in deterministic order, but not an unrelated directory"
  else
    fail "generator: the runtime profile sibling-runner denies are wrong: ${actual_home_runner_rules}"
  fi
  pass "generator: denies network, fork, foreign exec and all writes; allows the named tree and prefix; denies every evaluator path and resident log root last"
fi

fallback_profile="$(env -i PATH="${PATH}" HOME="${WORK}/home" RESIDENT_UP_LOG_DIR="${C}/resident-root/job" \
  python3 "${GENERATOR}" --exec /bin/bash --tree "${WORK}/tree")"
if grep -qF "(deny file-read* file-write* (subpath \"${C}/resident-root/job\"))" <<< "${fallback_profile}"; then
  pass "generator: RESIDENT_UP_LOG_DIR is the deny fallback when the stable root is unset"
else
  fail "generator: no resident log fallback deny when MLXFAST_RESIDENT_LOG_ROOT is unset"
fi

default_profile="$(env -i PATH="${PATH}" HOME="${WORK}/home" python3 "${GENERATOR}" \
  --exec /bin/bash --tree "${WORK}/tree")"
if grep -qF "(deny file-read* file-write* (subpath \"${WORK}/home/.mlxfast-resident-logs\"))" <<< "${default_profile}"; then
  pass "generator: the self-hosted resident log root is denied when both log variables are unset"
else
  fail "generator: no default resident log deny when both log variables are unset"
fi

audit_guards_ok=1
for guarded_audit in "/p1fix-audit.json" "${WORK}/home/p1fix-audit.json" "${WORK}/tree/p1fix-audit.json"; do
  guarded_profile="$(env -i PATH="${PATH}" HOME="${WORK}/home" BENCHD_PRIVATE_AUDIT_PATH="${guarded_audit}" \
    python3 "${GENERATOR}" --exec /bin/bash --tree "${WORK}/tree")"
  guarded_parent="$(dirname "${guarded_audit}")"
  grep -qF "(deny file-read* file-write* (subpath \"${guarded_audit}\"))" <<< "${guarded_profile}" \
    || audit_guards_ok=0
  if [[ "${guarded_parent}" != "${guarded_audit}" ]] \
    && grep -qF "(deny file-read* file-write* (subpath \"${guarded_parent}\"))" <<< "${guarded_profile}"; then
    audit_guards_ok=0
  fi
done
if [[ "${audit_guards_ok}" -eq 1 ]]; then
  pass "generator: unsafe audit parents deny only the audit file"
else
  fail "generator: an audit under root, HOME or a parent of the tree denied its parent"
fi

runner_temp="${WORK}/runner-temp"
runner_audit="${runner_temp}/private-audit.json"
runner_profile="$(env -i PATH="${PATH}" HOME="${WORK}/home" RUNNER_TEMP="${runner_temp}" \
  BENCHD_PRIVATE_AUDIT_PATH="${runner_audit}" \
  python3 "${GENERATOR}" --exec /bin/bash --tree "${WORK}/tree")"
if grep -qF "(deny file-read* file-write* (subpath \"${runner_audit}\"))" <<< "${runner_profile}" \
  && ! grep -qF "(deny file-read* file-write* (subpath \"${runner_temp}\"))" <<< "${runner_profile}"; then
  pass "generator: an audit under RUNNER_TEMP denies only the audit file"
else
  fail "generator: an audit under RUNNER_TEMP denied the temp parent"
fi

private_var_tmpdir="/private/var/folders/p1fix-$$/T"
private_var_audit="/private/var/p1fix-audit-$$.json"
private_var_profile="$(env -i PATH="${PATH}" HOME="${WORK}/home" TMPDIR="${private_var_tmpdir}" \
  BENCHD_PRIVATE_AUDIT_PATH="${private_var_audit}" \
  python3 "${GENERATOR}" --exec /bin/bash --tree "${WORK}/tree")"
if grep -qF "(deny file-read* file-write* (subpath \"${private_var_audit}\"))" <<< "${private_var_profile}" \
  && ! grep -qF '(deny file-read* file-write* (subpath "/private/var"))' <<< "${private_var_profile}"; then
  pass "generator: an audit whose /private/var parent contains TMPDIR denies only the file"
else
  fail "generator: an audit parent that contains TMPDIR was denied as a directory"
fi

dedicated_temp_root="/private/tmp/p1fix-audit-private-$$"
dedicated_temp_parent="${dedicated_temp_root}/audits"
dedicated_temp_audit="${dedicated_temp_parent}/private-audit.json"
dedicated_temp_profile="$(env -i PATH="${PATH}" HOME="${WORK}/home" \
  BENCHD_PRIVATE_AUDIT_PATH="${dedicated_temp_audit}" \
  python3 "${GENERATOR}" --exec /bin/bash --tree "${WORK}/tree")"
if grep -qF "(deny file-read* file-write* (subpath \"${dedicated_temp_parent}\"))" <<< "${dedicated_temp_profile}"; then
  pass "generator: a dedicated audit directory below /private/tmp denies its parent"
else
  fail "generator: a dedicated audit directory below /private/tmp did not deny its parent"
fi

private_profile="$(env -i PATH="${PATH}" HOME="${WORK}/home" \
  BENCHD_PRIVATE_AUDIT_PATH="${PRIVATE_AUDIT_DIR}/private-audit.json" \
  python3 "${GENERATOR}" --exec /bin/bash --tree "${WORK}/tree")"
if grep -qF "(deny file-read* file-write* (subpath \"${PRIVATE_AUDIT_DIR}\"))" <<< "${private_profile}"; then
  pass "generator: an audit under a dedicated private directory denies its parent"
else
  fail "generator: a dedicated private audit directory was not denied"
fi

build_profile="$(env -i PATH="${PATH}" HOME="${WORK}/home" "${canary_env[@]}" python3 "${GENERATOR}" \
  --build --tree "${WORK}/tree")"
for rule in "${state_rules[@]}"; do
  if grep -Fxq "${rule}" <<< "${build_profile}"; then
    fail "generator: build profile contains non-build rule ${rule}"
  fi
done
pass "generator: the seven non-build state rules are absent from --build"
for rule in "(deny file-write-flags)" "(deny file-write-acl)"; do
  if grep -Fxq "${rule}" <<< "${build_profile}"; then
    fail "generator: build profile contains runtime-only rule ${rule}"
  fi
done
pass "generator: file flag and ACL denies are absent from --build"

out="$(env -i PATH="${PATH}" HOME="${WORK}/home" "${canary_env[@]}" BENCHD_BIN_DIR= \
  python3 "${GENERATOR}" --exec /bin/bash --official 2>&1)"
rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"unset: BENCHD_BIN_DIR"* ]]; then
  pass "generator: an official profile with BENCHD_BIN_DIR unset refuses by name"
else
  fail "generator: an official profile with BENCHD_BIN_DIR unset did not refuse (rc ${rc}): ${out}"
fi

profile="$(env -i PATH="${PATH}" HOME="${WORK}/home" "${canary_env[@]}" python3 "${GENERATOR}" \
  --exec /bin/bash --tree "${C}/baseline/checkout")"
if grep -qF "(subpath \"${C}/baseline\"))" <<< "${profile}"; then
  fail "generator: the reference workspace is denied to a program from inside it"
else
  pass "generator: a program from inside the reference workspace can read that workspace"
fi

# --- part 1: the wrapper refuses before it starts anything ------------------
expect_wrapper_refusal() {
  local label="$1" needle="$2"
  shift 2
  out="$(env -i PATH="${PATH}" HOME="${WORK}/home" MLXFAST_SWIFT_BIN="${WORK}/no-cli" "${WRAPPER}" "$@" 2>&1)"
  rc=$?
  if [[ "${rc}" -eq 2 && "${out}" == *"${needle}"* ]]; then
    pass "wrapper: ${label}"
  else
    fail "wrapper: ${label}: rc ${rc}, wanted 2 and '${needle}'; got: ${out}"
  fi
}
expect_wrapper_refusal "an unknown verb refuses" "unknown verb 'benchmark'" benchmark --weights w
expect_wrapper_refusal "an unknown option refuses" "unknown option '--golden'" transform --reference r --output w --golden g
expect_wrapper_refusal "the --name=value form refuses" "unknown option '--output=w'" transform --reference r --output=w
expect_wrapper_refusal "a repeated option refuses" "--output is given twice" transform --reference r --output a --output b
expect_wrapper_refusal "transform without --output refuses" "transform: --output is required" transform --reference r
expect_wrapper_refusal "verify-transform without --weights refuses" "verify-transform: --weights is required" verify-transform --reference r
expect_wrapper_refusal "an option without a value refuses" "--index has no value" checkpoint-shards --index

# --- part 2: the real CLI under the real Seatbelt ---------------------------
CLI="${MLXFAST_SWIFT_BIN:-${REPO_ROOT}/.build/release/mlxfast-swift}"
if [[ ! -x /usr/bin/sandbox-exec ]]; then
  echo "test-sandboxed-cli.sh: part 2 NOT RUN: /usr/bin/sandbox-exec is not on this host"
elif [[ ! -x "${CLI}" ]]; then
  echo "test-sandboxed-cli.sh: part 2 NOT RUN: no CLI at ${CLI} (set MLXFAST_SWIFT_BIN)"
else
  official=(env -i PATH="${PATH}" HOME="${WORK}/home" TMPDIR="${WORK}/tmp" "${canary_env[@]}"
    MLXFAST_OFFICIAL_BENCHMARK_RUN=1 MLXFAST_SWIFT_BIN="${CLI}")
  mkdir -p "${WORK}/tmp"

  out="$("${official[@]}" "${WRAPPER}" transform --reference "${WORK}/no-reference" --output "${WORK}/wrapped" 2>&1)"
  if [[ "${out}" == *"no config.json found under ${WORK}/no-reference"* ]]; then
    pass "wrapper: an official transform under the wrapper starts the CLI"
  else
    fail "wrapper: an official transform under the wrapper did not start the CLI: ${out}"
  fi

  cp "${C}/goldens/canary.golden.json" "${WORK}/control-index.json"
  out="$("${official[@]}" "${WRAPPER}" checkpoint-shards --index "${WORK}/control-index.json" 2>/dev/null)"
  if [[ "${out}" == "model.safetensors" ]]; then
    pass "wrapper: the CLI reads an index outside the evaluator paths"
  else
    fail "wrapper: the CLI could not read the control index: ${out}"
  fi
  out="$("${official[@]}" "${WRAPPER}" checkpoint-shards --index "${C}/goldens/canary.golden.json" 2>&1)"
  if [[ $? -ne 0 && "${out}" == *"Operation not permitted"* ]]; then
    pass "wrapper: the CLI cannot read the same bytes in the golden directory (EPERM)"
  else
    fail "wrapper: the CLI read a canary golden: ${out}"
  fi

  default_resident_root="${REPO_ROOT}/.build/resident"
  default_root_canary="${default_resident_root}/brief24b-canary-index.json"
  mkdir -p "${default_resident_root}"
  printf '{"weight_map":{"a":"must-not-read.safetensors"}}\n' > "${default_root_canary}"
  out="$(env -i PATH="${PATH}" HOME="${WORK}/home" TMPDIR="${WORK}/tmp" \
    MLXFAST_SWIFT_BIN="${CLI}" "${WRAPPER}" checkpoint-shards --index "${default_root_canary}" 2>&1)"
  rc=$?
  rm -f "${default_root_canary}"
  if [[ "${rc}" -ne 0 && "${out}" == *"Operation not permitted"* ]]; then
    pass "wrapper: with the root unset, the local default resident root is denied"
  else
    fail "wrapper: the default resident root was readable (rc ${rc}): ${out}"
  fi

  before="$(find "${C}/benchd-bin" -mindepth 1 | sort)"
  "${official[@]}" "${WRAPPER}" transform --reference "${REPO_ROOT}" --output "${C}/benchd-bin/weights" > /dev/null 2>&1
  if [[ "$(find "${C}/benchd-bin" -mindepth 1 | sort)" == "${before}" ]]; then
    pass "wrapper: a transform whose --output is inside benchd-bin cannot write there"
  else
    fail "wrapper: a transform wrote into benchd-bin: $(find "${C}/benchd-bin" -mindepth 1 | tr '\n' ' ')"
  fi
fi

echo "test-sandboxed-cli.sh: ${passes} passed, ${failures} failed"
[[ "${failures}" -eq 0 ]]
