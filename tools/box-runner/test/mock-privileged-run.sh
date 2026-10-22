#!/usr/bin/env bash
# The supervisor's `rm JIT && ./run.sh` chain reaches this only after unlink.
set -u
[ -n "${ACTIONS_RUNNER_INPUT_JITCONFIG:-}" ] || exit 1
SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd -P)"
: > "${SELF_DIR}/job-ran"
