#!/usr/bin/env bash
# Prove setup and the ranked workflow refuse a trusted CLI symlink escape.
set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
SETUP="${REPO_ROOT}/setup.sh"
WORKFLOW="${REPO_ROOT}/.github/workflows/benchmark.yml"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
fails=0
pass() { printf 'test-trusted-cli-containment: PASS -- %s\n' "$*"; }
fail() { printf 'test-trusted-cli-containment: FAIL -- %s\n' "$*" >&2; fails=$((fails + 1)); }

baseline="${WORK}/reference"
outside="${WORK}/outside-cli"
mkdir -p "${baseline}/.build/release"
printf '#!/bin/sh\nexit 0\n' > "${outside}"
chmod +x "${outside}"
ln -s "${outside}" "${baseline}/.build/release/mlxfast-swift"

out="$(MLXFAST_BASELINE_WORKSPACE="${baseline}" \
  MLXFAST_TRUSTED_CLI="${baseline}/.build/release/mlxfast-swift" \
  "${SETUP}" 2>&1)"
rc=$?
if [[ "${rc}" -eq 1 && "${out}" == *"MLXFAST_TRUSTED_CLI resolves outside MLXFAST_BASELINE_WORKSPACE"* ]]; then
  pass "setup refuses a final-file symlink outside the reference workspace"
else
  fail "setup did not refuse the trusted CLI escape (rc ${rc}): ${out}"
fi

# These are literal workflow source lines, not strings to expand in this test.
# shellcheck disable=SC2016
for needle in \
  'baseline_real="$(python3 -c '\''import os, sys; print(os.path.realpath(sys.argv[1]))'\'' "${MLXFAST_BASELINE_WORKSPACE}")"' \
  'trusted_cli_real="$(python3 -c '\''import os, sys; print(os.path.realpath(sys.argv[1]))'\'' "${staged}/mlxfast-swift")"' \
  '"${baseline_real}"/*) : ;;' \
  'the trusted CLI resolves outside the reference workspace'
do
  if grep -Fq "${needle}" "${WORKFLOW}"; then
    pass "workflow keeps guard: ${needle}"
  else
    fail "workflow lost guard: ${needle}"
  fi
done

if [[ "${fails}" -eq 0 ]]; then
  printf 'test-trusted-cli-containment: all cases pass\n'
  exit 0
fi
printf 'test-trusted-cli-containment: %s failed\n' "${fails}" >&2
exit 1
