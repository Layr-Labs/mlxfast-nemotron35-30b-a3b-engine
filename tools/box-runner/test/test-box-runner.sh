#!/usr/bin/env bash
# test-box-runner.sh -- dry-run test suite for tools/box-runner/.
#
# Exercises supervisor.sh's spawn/stop/status logic and lib/pidtree.sh's
# kill-tree primitive against MOCK processes only (mock-run.sh,
# mock-orphan-child.sh, mock-gh-app-mint.sh, all in this directory) -- no
# GitHub network access, no GitHub App credential, and no real actions-runner
# binary. The default path needs no root. An opt-in root section exercises the
# real privilege drop when BOX_RUNNER_TEST_USER names a non-root user.
#
# REQUIRED regression coverage (see the task brief this suite was written
# against): section 1b proves the kill-tree mechanism actually reaps a
# DETACHED grandchild of the tracked job root -- the box-3 orphan bug that
# lib/pidtree.sh and supervisor.sh's `stop` entrypoint exist to fix. Section
# 1c is the negative control: it proves the bug is real by showing that
# killing ONLY the tracked root pid (the naive/original behavior) does NOT
# reap the grandchild, so 1b is not a vacuous pass. Section 2 repeats the
# same proof end-to-end through supervisor.sh's actual `run`/`stop`/`status`
# entrypoints, including calling `stop` from a SEPARATE process invocation
# than the one running the job loop -- the box-3 recovery scenario itself
# (an operator, or a restarted daemon, reaping a tree a now-dead process was
# tracking).
#
# Usage: tools/box-runner/test/test-box-runner.sh [-v]
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd -P)"
BOX_RUNNER_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null && pwd -P)"
VERBOSE=0
[[ "${1:-}" == "-v" ]] && VERBOSE=1

# shellcheck source=tools/box-runner/lib/pidtree.sh
. "${BOX_RUNNER_DIR}/lib/pidtree.sh"

# See tools/test-submission-security.sh for why this floor exists: it is the
# only thing that notices a section silently stopping (an early `exit`, a
# deleted block) under `set -uo pipefail` without `-e`, where a truncated
# suite otherwise reports "0 failed" and looks green.
EXPECTED_MIN_ASSERTIONS=54

WORK="$(mktemp -d "${TMPDIR:-/tmp}/box-runner-test.XXXXXX")"
# shellcheck disable=SC2329  # invoked indirectly via `trap cleanup EXIT` below
cleanup() {
  # Belt and suspenders: every test below is expected to clean up its own
  # processes: this just makes sure nothing outlives the suite even if one
  # of them fails partway through.
  local pf
  for pf in "${WORK}"/*/job.pid "${WORK}"/*/mock.pid "${WORK}"/*/orphan.pid \
            "${WORK}"/*-state/job.pid "${WORK}"/*cat-pid; do
    [ -f "${pf}" ] || continue
    kill -KILL "$(cat "${pf}" 2>/dev/null)" 2>/dev/null || true
  done
  for pf in "${WORK}"/*jit-path; do
    [ -f "${pf}" ] || continue
    case "$(cat "${pf}" 2>/dev/null)" in
      /tmp/nemotron35-jitconfig.*|/private/tmp/nemotron35-jitconfig.*)
        rm -f -- "$(cat "${pf}" 2>/dev/null)"
        ;;
    esac
  done
  rm -rf "${WORK}"
}
trap cleanup EXIT

PASSED=0
FAILED=0
FAILURES=()

pass() { PASSED=$((PASSED + 1)); [ "${VERBOSE}" -eq 1 ] && printf 'PASS  %s\n' "$1"; }
fail_case() {
  FAILED=$((FAILED + 1))
  FAILURES+=("$1")
  printf 'FAIL  %s\n' "$1"
}

assert_true() {
  local name="$1"; shift
  if "$@"; then pass "${name}"; else fail_case "${name} (expected success: $*)"; fi
}

assert_false() {
  local name="$1"; shift
  if "$@"; then fail_case "${name} (expected failure: $*)"; else pass "${name}"; fi
}

assert_eq() {
  local name="$1" want="$2" got="$3"
  if [ "${got}" = "${want}" ]; then
    pass "${name}"
  else
    fail_case "${name} (want '${want}', got '${got}')"
  fi
}

# assert_exit NAME WANT_STATUS WANT_SUBSTRING CMD...
assert_exit() {
  local name="$1" want="$2" pattern="$3"
  shift 3
  local out status
  out="$("$@" 2>&1)"
  status=$?
  [ "${VERBOSE}" -eq 1 ] && printf -- '--- %s (exit %d)\n%s\n' "${name}" "${status}" "${out}"
  if [ "${status}" != "${want}" ]; then
    fail_case "${name} (exit ${status}, wanted ${want}) :: ${out}"
    return
  fi
  if [ -n "${pattern}" ]; then
    case "${out}" in
      *"${pattern}"*) : ;;
      *) fail_case "${name} (missing '${pattern}' in output) :: ${out}"; return ;;
    esac
  fi
  pass "${name}"
}

# The four helpers below are only ever invoked indirectly, as the CMD... of
# assert_true/assert_false ("$@" further down each), so shellcheck cannot
# see the call site; SC2329 ("never invoked") is a false positive on all of
# them for that reason.
# shellcheck disable=SC2329
pid_alive() { kill -0 "$1" 2>/dev/null; }
# shellcheck disable=SC2329
pid_dead() { ! kill -0 "$1" 2>/dev/null; }
# shellcheck disable=SC2329
contains() { case "$1" in *"$2"*) return 0 ;; *) return 1 ;; esac; }
# list_has LIST NEEDLE -- LIST is newline-separated; true iff one line
# equals NEEDLE exactly (used for pid-list membership, where a substring
# match could false-positive on e.g. pid 12 matching pid 123).
# shellcheck disable=SC2329
list_has() { printf '%s\n' "$1" | grep -qx -- "$2"; }

wait_for_file() {
  local path="$1" timeout="${2:-5}" waited=0
  while [ ! -s "${path}" ]; do
    waited=$((waited + 1))
    [ "${waited}" -gt $((timeout * 10)) ] && return 1
    sleep 0.1
  done
  return 0
}

wait_for_pid_dead() {
  local pid="$1" timeout="${2:-5}" waited=0
  while kill -0 "${pid}" 2>/dev/null; do
    waited=$((waited + 1))
    [ "${waited}" -gt $((timeout * 10)) ] && return 1
    sleep 0.1
  done
  return 0
}

file_mode() {
  if [ "$(uname -s)" = "Darwin" ]; then
    stat -f '%Lp' "$1"
  else
    stat -c '%a' "$1"
  fi
}

file_uid() {
  if [ "$(uname -s)" = "Darwin" ]; then
    stat -f '%u' "$1"
  else
    stat -c '%u' "$1"
  fi
}

# =========================================================================
# Section 1: lib/pidtree.sh primitives, directly, against mock-run.sh
# =========================================================================

# start_mock_tree STATE_DIR -- starts mock-run.sh (which spawns a detached
# mock-orphan-child.sh grandchild) and blocks until both have recorded their
# own pids, printing mock-run.sh's pid on stdout.
start_mock_tree() {
  local state_dir="$1"
  mkdir -p "${state_dir}"
  "${SCRIPT_DIR}/mock-run.sh" "${state_dir}" \
    > "${state_dir}/mock-run.log" 2>&1 &
  wait_for_file "${state_dir}/mock.pid" 5 || { echo "mock.pid never appeared" >&2; return 1; }
  wait_for_file "${state_dir}/orphan.pid" 5 || { echo "orphan.pid never appeared" >&2; return 1; }
  cat "${state_dir}/mock.pid"
}

# --- 1a. pidtree_collect finds root + orphan while alive -------------------
t1_dir="${WORK}/t1"
root_pid="$(start_mock_tree "${t1_dir}")"
orphan_pid="$(cat "${t1_dir}/orphan.pid")"
collected="$(pidtree_collect "${root_pid}")"
assert_true "pidtree_collect includes root pid" list_has "${collected}" "${root_pid}"
assert_true "pidtree_collect includes orphan grandchild pid" list_has "${collected}" "${orphan_pid}"
pidtree_kill_tree "${root_pid}" 5 >/dev/null 2>&1 || true
nonexistent_collected="$(pidtree_collect 999999)"
assert_eq "pidtree_collect: nonexistent pid returns nothing" "" "${nonexistent_collected}"

# --- 1b. THE REQUIRED REGRESSION TEST: kill-tree reaps a detached orphan ---
t2_dir="${WORK}/t2"
root_pid="$(start_mock_tree "${t2_dir}")"
orphan_pid="$(cat "${t2_dir}/orphan.pid")"
assert_true "orphan-kill setup: root alive before stop" pid_alive "${root_pid}"
assert_true "orphan-kill setup: orphan alive before stop" pid_alive "${orphan_pid}"
assert_false "orphan-kill setup: orphan is NOT a direct child of the test shell (pgrep -P misses it from here)" \
  bash -c "pgrep -P \$\$ | grep -qx '${orphan_pid}'"

pidtree_kill_tree "${root_pid}" 5
kill_tree_status=$?
assert_eq "orphan-kill: pidtree_kill_tree reports success" "0" "${kill_tree_status}"
sleep 0.3
assert_true "orphan-kill: root pid dead after kill_tree" pid_dead "${root_pid}"
assert_true "orphan-kill: DETACHED ORPHAN pid dead after kill_tree" pid_dead "${orphan_pid}"

# --- 1c. Negative control: naive single-pid kill leaves the orphan alive --
#     (proves 1b is not vacuous -- this is what the box-3 bug looked like:
#     killing only the tracked pid, or relying on a process-group signal
#     that never reaches a re-sessioned descendant, leaves it running.)
t3_dir="${WORK}/t3"
root_pid="$(start_mock_tree "${t3_dir}")"
orphan_pid="$(cat "${t3_dir}/orphan.pid")"
kill -KILL "${root_pid}" 2>/dev/null
sleep 0.3
assert_true "negative control: naive kill removes only the root" pid_dead "${root_pid}"
assert_true "negative control: naive kill LEAVES the detached orphan alive (the bug)" pid_alive "${orphan_pid}"
kill -KILL "${orphan_pid}" 2>/dev/null || true # clean up what the negative control deliberately left running

# =========================================================================
# Section 2: supervisor.sh end-to-end (spawn / status / stop), test mode
# =========================================================================

# --- 2a. relative log/state paths are anchored at supervisor.sh ------------
relative_stage="${WORK}/relative-stage"
mkdir -p "${relative_stage}/lib"
cp "${BOX_RUNNER_DIR}/supervisor.sh" "${relative_stage}/supervisor.sh"
cp "${BOX_RUNNER_DIR}/lib/pidtree.sh" "${relative_stage}/lib/pidtree.sh"
(
  cd "${WORK}" || exit 1
  env LOG_DIR=./log STATE_DIR=./state BOX_RUNNER_CONFIG=/nonexistent \
    "${relative_stage}/supervisor.sh" status >/dev/null
)
assert_true "relative LOG_DIR resolves under SCRIPT_DIR" test -d "${relative_stage}/log"
assert_true "relative STATE_DIR resolves under SCRIPT_DIR" test -d "${relative_stage}/state"

run_dir="${WORK}/runner-dir"
mkdir -p "${run_dir}"
# Real copies, not symlinks: mock-run.sh locates mock-orphan-child.sh next
# to itself (mirroring how a real actions-runner install keeps its helper
# scripts beside run.sh), so both must live in RUNNER_DIR together.
cp "${SCRIPT_DIR}/mock-run.sh" "${run_dir}/run.sh"
cp "${SCRIPT_DIR}/mock-orphan-child.sh" "${run_dir}/mock-orphan-child.sh"
chmod +x "${run_dir}/run.sh" "${run_dir}/mock-orphan-child.sh"

# Record the path passed to cat when the runner reads its JIT config. The
# wrapper then delegates to the system cat unchanged.
mock_bin="${WORK}/mock-bin"
mock_jit_path_file="${WORK}/mock-jit-path"
mkdir -p "${mock_bin}"
# shellcheck disable=SC2016  # deferred: mock cat expands these at run time
printf '%s\n' \
  '#!/usr/bin/env bash' \
  'case "${1:-}" in' \
  '  *jitconfig*)' \
  '    printf "%s\n" "$1" > "${MOCK_JIT_PATH_FILE}"' \
  '    if [ "${MOCK_JIT_HOLD:-0}" = "1" ]; then' \
  '      printf "%s\n" "$$" > "${MOCK_JIT_CAT_PID_FILE}"' \
  '      waited=0' \
  '      while [ ! -e "${MOCK_JIT_RELEASE_FILE}" ] && [ "${waited}" -lt 300 ]; do' \
  '        sleep 0.1' \
  '        waited=$((waited + 1))' \
  '      done' \
  '    fi' \
  '    ;;' \
  'esac' \
  'exec /bin/cat "$@"' \
  > "${mock_bin}/cat"
chmod +x "${mock_bin}/cat"

e2e_state="${WORK}/e2e-state"
sup_log_dir="${WORK}/sup-log"
sup_state_dir="${WORK}/sup-state"
mkdir -p "${e2e_state}"

run_supervisor() {
  env \
    RUNNER_SU_ENABLED=0 \
    RUNNER_DIR="${run_dir}" \
    RUNNER_LABELS="nemotron35-30b-a3b-box" \
    GH_APP_MINT="${SCRIPT_DIR}/mock-gh-app-mint.sh" \
    PATH="${mock_bin}:${PATH}" \
    MOCK_JIT_PATH_FILE="${mock_jit_path_file}" \
    LOG_DIR="${sup_log_dir}" \
    STATE_DIR="${sup_state_dir}" \
    MOCK_STATE_DIR="${e2e_state}" \
    PAUSE_BETWEEN_JOBS=1 \
    BOX_RUNNER_MAX_ITERATIONS=1 \
    BOX_RUNNER_CONFIG=/nonexistent \
    "${BOX_RUNNER_DIR}/supervisor.sh" "$@"
}

# --- 2b. status is idle before anything has run -----------------------------
out="$(run_supervisor status)"
assert_eq "status: idle before any job" "idle" "${out}"

# --- 2c. spawn a job (background, single-iteration loop) --------------------
run_supervisor run > "${WORK}/supervisor-run.log" 2>&1 &
sup_pid=$!
wait_for_file "${sup_state_dir}/job.pid" 5
assert_true "supervisor: job.pid appears" test -s "${sup_state_dir}/job.pid"
wait_for_file "${e2e_state}/mock.pid" 5
wait_for_file "${e2e_state}/orphan.pid" 5
wait_for_file "${mock_jit_path_file}" 5
job_pid="$(cat "${sup_state_dir}/job.pid")"
mock_pid="$(cat "${e2e_state}/mock.pid")"
orphan_pid="$(cat "${e2e_state}/orphan.pid")"
jit_file="$(/bin/cat "${mock_jit_path_file}" 2>/dev/null)"
assert_true "supervisor: mock runner reads a temporary JIT config" test -n "${jit_file}"
# shellcheck disable=SC2016  # deferred: nested bash expands its arguments
assert_false "supervisor: JIT config path is outside STATE_DIR" \
  bash -c 'case "$1" in "$2"/*) exit 0;; *) exit 1;; esac' _ "${jit_file}" "${sup_state_dir}"
# The tracked root pid is whatever `exec` ultimately replaced spawn_job's
# background subshell with -- mock-run.sh itself when no timeout(1)/
# gtimeout(1) is on PATH, or a `timeout` WRAPPER process (a distinct pid)
# ahead of it when one is (as it is on this dev machine, via Homebrew
# coreutils). Both are correct: what matters is that the tracked pid's tree
# includes mock-run.sh, which is what the box-3 fix actually depends on.
job_tree="$(pidtree_collect "${job_pid}")"
assert_true "supervisor: tracked root's tree includes mock-run.sh's pid" list_has "${job_tree}" "${mock_pid}"
assert_true "supervisor: mock-run.sh alive while job in progress" pid_alive "${mock_pid}"
assert_true "supervisor: detached orphan alive while job in progress" pid_alive "${orphan_pid}"

# --- 2d. status while running lists the tree --------------------------------
status_out="$(run_supervisor status)"
# shellcheck disable=SC2016  # deliberate: $0 must expand inside the nested
# bash -c (bound there to status_out, passed as its NAME argument), not here.
assert_true "status while running: reports 'running'" bash -c 'case "$0" in running:*) exit 0;; *) exit 1;; esac' "${status_out}"
assert_true "status while running: lists the mock-run.sh pid" contains "${status_out}" "${mock_pid}"

# --- 2e. stop, called as a SEPARATE process invocation (the standalone
#     entrypoint -- the box-3 recovery case: an operator, or a restarted
#     `run`, reaping a tree a DIFFERENT process instance is tracking) -------
stop_out="$(run_supervisor stop)"
assert_true "stop: reports stopped" contains "${stop_out}" "stopped"
sleep 0.3
assert_true "stop: mock-run.sh pid dead" pid_dead "${mock_pid}"
assert_true "stop: DETACHED ORPHAN pid dead (kill-tree via standalone stop)" pid_dead "${orphan_pid}"
assert_false "stop: job.pid file removed" test -f "${sup_state_dir}/job.pid"
assert_false "stop: temporary JIT config is removed after the job ends" test -e "${jit_file}"

# --- 2f. status is idle again after stop ------------------------------------
final_status="$(run_supervisor status)"
assert_eq "status: idle again after stop" "idle" "${final_status}"

# Reap the backgrounded `run` loop itself. Its single job was already killed
# by `stop` above; BOX_RUNNER_MAX_ITERATIONS=1 makes the loop exit on its
# own within PAUSE_BETWEEN_JOBS=1s, so this is a bounded wait, not a hang.
kill "${sup_pid}" 2>/dev/null || true
wait "${sup_pid}" 2>/dev/null || true

# --- 2g. TERM removes a JIT file before the job can read or unlink it ------
held_state="${WORK}/held-state"
held_log_dir="${WORK}/held-log"
held_sup_state_dir="${WORK}/held-sup-state"
held_jit_path_file="${WORK}/held-jit-path"
held_cat_pid_file="${WORK}/held-cat-pid"
held_release_file="${WORK}/held-release"
mkdir -p "${held_state}"

env \
  RUNNER_SU_ENABLED=0 \
  RUNNER_DIR="${run_dir}" \
  RUNNER_LABELS="nemotron35-30b-a3b-box" \
  GH_APP_MINT="${SCRIPT_DIR}/mock-gh-app-mint.sh" \
  PATH="${mock_bin}:${PATH}" \
  MOCK_JIT_PATH_FILE="${held_jit_path_file}" \
  MOCK_JIT_HOLD=1 \
  MOCK_JIT_CAT_PID_FILE="${held_cat_pid_file}" \
  MOCK_JIT_RELEASE_FILE="${held_release_file}" \
  LOG_DIR="${held_log_dir}" \
  STATE_DIR="${held_sup_state_dir}" \
  MOCK_STATE_DIR="${held_state}" \
  PAUSE_BETWEEN_JOBS=0 \
  STOP_TERM_TIMEOUT_SECONDS=1 \
  BOX_RUNNER_MAX_ITERATIONS=1 \
  BOX_RUNNER_CONFIG=/nonexistent \
  "${BOX_RUNNER_DIR}/supervisor.sh" run > "${WORK}/held-supervisor.log" 2>&1 &
held_sup_pid=$!
wait_for_file "${held_jit_path_file}" 5
wait_for_file "${held_cat_pid_file}" 5
wait_for_file "${held_sup_state_dir}/job.pid" 5
held_jit_file="$(/bin/cat "${held_jit_path_file}" 2>/dev/null)"
held_cat_pid="$(/bin/cat "${held_cat_pid_file}" 2>/dev/null)"
held_job_pid="$(/bin/cat "${held_sup_state_dir}/job.pid" 2>/dev/null)"
assert_true "TERM cleanup: held JIT config path is recorded" test -n "${held_jit_file}"
assert_true "TERM cleanup: held JIT config exists before TERM" test -f "${held_jit_file}"
assert_eq "TERM cleanup: held JIT config mode is 600" "600" "$(file_mode "${held_jit_file}")"

kill -TERM "${held_sup_pid}"
held_term_bounded=1
if ! wait_for_pid_dead "${held_sup_pid}" 5; then
  held_term_bounded=0
  kill -KILL "${held_sup_pid}" 2>/dev/null || true
fi
wait "${held_sup_pid}" 2>/dev/null
held_term_status=$?
assert_eq "TERM cleanup: supervisor exits with TERM status" "143" "${held_term_status}"
assert_eq "TERM cleanup: supervisor exits within five seconds" "1" "${held_term_bounded}"
assert_false "TERM cleanup: EXIT trap removes held JIT config" test -e "${held_jit_file}"
held_cat_dead=1
wait_for_pid_dead "${held_cat_pid}" 2 || held_cat_dead=0
held_job_dead=1
wait_for_pid_dead "${held_job_pid}" 2 || held_job_dead=0
assert_eq "TERM cleanup: held mock cat process is dead" "1" "${held_cat_dead}"
assert_eq "TERM cleanup: tracked mock job process is dead" "1" "${held_job_dead}"

# =========================================================================
# Section 3: opt-in production privilege path
# =========================================================================

priv_user="${BOX_RUNNER_TEST_USER:-}"
if [ "$(id -u)" -ne 0 ] || [ -z "${priv_user}" ] || \
   ! id "${priv_user}" >/dev/null 2>&1 || [ "$(id -u "${priv_user}" 2>/dev/null)" = "0" ]; then
  printf 'SKIP: privileged JIT checks (need root and BOX_RUNNER_TEST_USER)\n'
else
  priv_uid="$(id -u "${priv_user}")"
  priv_run_dir="${WORK}/priv-runner-dir"
  priv_log_dir="${WORK}/priv-log"
  priv_state_dir="${WORK}/priv-state"
  priv_mock_bin="${WORK}/priv-mock-bin"
  priv_jit_path_file="${WORK}/priv-jit-path"
  priv_chown_release="${WORK}/priv-chown-release"
  priv_job_marker="${priv_run_dir}/job-ran"

  chmod 755 "${WORK}"
  mkdir -p "${priv_run_dir}" "${priv_log_dir}" "${priv_state_dir}" "${priv_mock_bin}"
  cp "${SCRIPT_DIR}/mock-privileged-run.sh" "${priv_run_dir}/run.sh"
  cp "${SCRIPT_DIR}/mock-chown.sh" "${priv_mock_bin}/chown"
  chmod 755 "${priv_run_dir}" "${priv_run_dir}/run.sh" "${priv_mock_bin}/chown"
  chown "${priv_user}" "${priv_run_dir}"
  chown root "${priv_state_dir}"
  chmod 700 "${priv_state_dir}"

  env \
    RUNNER_SU_ENABLED=1 \
    RUNNER_USER="${priv_user}" \
    RUNNER_DIR="${priv_run_dir}" \
    RUNNER_LABELS="nemotron35-30b-a3b-box" \
    GH_APP_MINT="${SCRIPT_DIR}/mock-gh-app-mint.sh" \
    PATH="${priv_mock_bin}:${PATH}" \
    REAL_CHOWN="$(command -v chown)" \
    MOCK_JIT_PATH_FILE="${priv_jit_path_file}" \
    MOCK_CHOWN_RELEASE_FILE="${priv_chown_release}" \
    LOG_DIR="${priv_log_dir}" \
    STATE_DIR="${priv_state_dir}" \
    PAUSE_BETWEEN_JOBS=0 \
    BOX_RUNNER_MAX_ITERATIONS=1 \
    BOX_RUNNER_CONFIG=/nonexistent \
    "${BOX_RUNNER_DIR}/supervisor.sh" run > "${WORK}/priv-supervisor.log" 2>&1 &
  priv_sup_pid=$!
  wait_for_file "${priv_jit_path_file}" 5
  priv_jit_file="$(/bin/cat "${priv_jit_path_file}" 2>/dev/null)"
  assert_eq "privileged JIT: owner is test user" "${priv_uid}" "$(file_uid "${priv_jit_file}")"
  assert_eq "privileged JIT: mode is 600" "600" "$(file_mode "${priv_jit_file}")"
  assert_false "privileged JIT: test user cannot write STATE_DIR" \
    su - "${priv_user}" -c "touch '${priv_state_dir}/forbidden'"

  : > "${priv_chown_release}"
  priv_run_bounded=1
  if ! wait_for_pid_dead "${priv_sup_pid}" 5; then
    priv_run_bounded=0
    kill -KILL "${priv_sup_pid}" 2>/dev/null || true
  fi
  wait "${priv_sup_pid}" 2>/dev/null
  priv_run_status=$?
  assert_eq "privileged JIT: supervisor run exits 0" "0" "${priv_run_status}"
  assert_eq "privileged JIT: run finishes within five seconds" "1" "${priv_run_bounded}"
  assert_true "privileged JIT: user job runs after unlink" test -f "${priv_job_marker}"
  assert_false "privileged JIT: user job unlinks config" test -e "${priv_jit_file}"
fi

# =========================================================================
# Section 4: fail-closed configuration (supervisor.sh)
# =========================================================================

assert_exit "supervisor: refuses with no RUNNER_DIR" 78 "RUNNER_DIR is required" \
  env RUNNER_SU_ENABLED=0 RUNNER_LABELS=x GH_APP_MINT="${SCRIPT_DIR}/mock-gh-app-mint.sh" \
      LOG_DIR="${WORK}/x1-log" STATE_DIR="${WORK}/x1-state" BOX_RUNNER_MAX_ITERATIONS=1 \
      BOX_RUNNER_CONFIG=/nonexistent "${BOX_RUNNER_DIR}/supervisor.sh" run

assert_exit "supervisor: refuses with missing run.sh" 78 "run.sh" \
  env RUNNER_SU_ENABLED=0 RUNNER_DIR="${WORK}/x2-empty" RUNNER_LABELS=x \
      GH_APP_MINT="${SCRIPT_DIR}/mock-gh-app-mint.sh" LOG_DIR="${WORK}/x2-log" \
      STATE_DIR="${WORK}/x2-state" BOX_RUNNER_MAX_ITERATIONS=1 BOX_RUNNER_CONFIG=/nonexistent \
      "${BOX_RUNNER_DIR}/supervisor.sh" run

assert_exit "supervisor: refuses with no RUNNER_LABELS" 78 "RUNNER_LABELS is required" \
  env RUNNER_SU_ENABLED=0 RUNNER_DIR="${run_dir}" \
      GH_APP_MINT="${SCRIPT_DIR}/mock-gh-app-mint.sh" LOG_DIR="${WORK}/x3-log" \
      STATE_DIR="${WORK}/x3-state" BOX_RUNNER_MAX_ITERATIONS=1 BOX_RUNNER_CONFIG=/nonexistent \
      "${BOX_RUNNER_DIR}/supervisor.sh" run

assert_exit "supervisor: refuses with missing gh-app-mint helper" 78 "gh-app mint helper missing" \
  env RUNNER_SU_ENABLED=0 RUNNER_DIR="${run_dir}" RUNNER_LABELS=x \
      GH_APP_MINT="${WORK}/does-not-exist.sh" LOG_DIR="${WORK}/x4-log" \
      STATE_DIR="${WORK}/x4-state" BOX_RUNNER_MAX_ITERATIONS=1 BOX_RUNNER_CONFIG=/nonexistent \
      "${BOX_RUNNER_DIR}/supervisor.sh" run

# =========================================================================
# Section 4: fail-closed configuration (gh-app-mint.sh)
# =========================================================================

assert_exit "gh-app-mint: refuses with no GH_APP_ID" 78 "GH_APP_ID is required" \
  env GH_APP_KEY="${WORK}/fake-key.pem" GH_RUNNER_REPO="owner/repo" \
      "${BOX_RUNNER_DIR}/gh-app-mint.sh" installation-id

echo "fake" > "${WORK}/fake-key.pem"
assert_exit "gh-app-mint: refuses with no GH_RUNNER_REPO" 78 "GH_RUNNER_REPO is required" \
  env GH_APP_ID=1 GH_APP_KEY="${WORK}/fake-key.pem" \
      "${BOX_RUNNER_DIR}/gh-app-mint.sh" installation-id

assert_exit "gh-app-mint: refuses with missing GH_APP_KEY file" 78 "GH_APP_KEY does not exist" \
  env GH_APP_ID=1 GH_APP_KEY="${WORK}/no-such-key.pem" GH_RUNNER_REPO="owner/repo" \
      "${BOX_RUNNER_DIR}/gh-app-mint.sh" installation-id

assert_exit "gh-app-mint: usage error on unknown subcommand" 64 "usage" \
  env GH_APP_ID=1 GH_APP_KEY="${WORK}/fake-key.pem" GH_RUNNER_REPO="owner/repo" \
      "${BOX_RUNNER_DIR}/gh-app-mint.sh" bogus-subcommand

# =========================================================================
# Section 5: install.sh pure helpers (isolated subshell: install.sh's own
# `set -euo pipefail` must never leak into this suite's shell, so each call
# below runs install.sh in a fresh `bash -c`, not via `source` here).
# =========================================================================

install_call() {
  bash -c '
    set -uo pipefail
    BOX_RUNNER_INSTALL_SOURCE_ONLY=1
    # shellcheck disable=SC1091
    . "$1/install.sh"
    shift
    "$@"
  ' _ "${BOX_RUNNER_DIR}" "$@"
}

assert_eq "install.sh: pinned sha256 for osx-arm64 is 64 hex chars" "64" \
  "$(install_call pinned_sha256 osx-arm64 | wc -c | tr -d ' ')"
assert_eq "install.sh: pinned sha256 for osx-x64 is 64 hex chars" "64" \
  "$(install_call pinned_sha256 osx-x64 | wc -c | tr -d ' ')"
assert_false "install.sh: pinned_sha256 refuses an unknown platform tag" \
  install_call pinned_sha256 linux-arm64

echo -n "box-runner" > "${WORK}/checksum-fixture.txt"
known_sha="$(shasum -a 256 "${WORK}/checksum-fixture.txt" 2>/dev/null | awk '{print $1}')"
assert_eq "install.sh: compute_sha256 matches shasum" "${known_sha}" \
  "$(install_call compute_sha256 "${WORK}/checksum-fixture.txt")"

# =========================================================================
# Section 6: install.sh FIRST INSTALL, end to end through main()
#
# Section 5 above only reaches install.sh's pure helpers: it sources the
# script with BOX_RUNNER_INSTALL_SOURCE_ONLY=1 and never runs main(), so
# nothing there can see a failure in the install path itself. That blind
# spot shipped a real one. install_actions_runner armed
# `trap 'rm -rf "${tmp_dir}"' RETURN` against its own `local tmp_dir`; bash
# RETURN traps are global, not function-scoped, so the trap outlived the
# function and fired a second time when main() returned, where `tmp_dir` is
# unbound -- fatal under install.sh's `set -euo pipefail`. Every first
# install on a fresh box therefore exited 1 with "tmp_dir: unbound variable"
# AFTER all four install steps had succeeded. Re-running took the idempotent
# early `return 0` that sits above the trap, so the second run exited 0 and
# made the first look like a fluke.
#
# So this section drives the WHOLE first-install path through main(), with
# the network and the tarball stubbed out (the same source-then-override
# pattern section 5 uses), and asserts the thing the bug got wrong: a
# successful first install exits 0. It runs against a STAGED COPY of
# install.sh so that the files main() writes -- box-runner.env, the rendered
# plist -- land in this suite's temp dir instead of in the repo.
# =========================================================================

install_stage="${WORK}/install-stage"
mkdir -p "${install_stage}"
cp "${BOX_RUNNER_DIR}/install.sh" \
   "${BOX_RUNNER_DIR}/box-runner.env.example" \
   "${BOX_RUNNER_DIR}/com.nemotron35.box-runner.plist.template" \
   "${install_stage}/"

install_runner_dir="${WORK}/install-runner-dir"
install_tmpdir="${WORK}/install-tmp"
mkdir -p "${install_tmpdir}"

# run_install_main -- install.sh's main(), start to finish, as a first
# install on a box that has none: platform forced to the ranked osx-arm64 so
# the result does not depend on the test host, curl(1) replaced by a stub
# that just materializes the tarball path, tar(1) replaced by one that lays
# down the single file the idempotency check looks for, and compute_sha256
# answering with the script's OWN pin so the verify step passes (a stub
# cannot forge bytes hashing to the real pin, and the pin itself is already
# under test in section 5). Everything else -- the trap, the directory
# creation, the env seeding, the plist render, main()'s return -- is the
# real code path.
# shellcheck disable=SC2329  # invoked indirectly, as assert_exit's CMD...
run_install_main() {
  # shellcheck disable=SC2016  # deliberate: every expansion in the nested
  # bash -c body below must happen THERE, against install.sh's own
  # variables (RUNNER_DIR, the stubs' arguments), not out here.
  env \
    RUNNER_DIR="${install_runner_dir}" \
    LOG_DIR="${WORK}/install-log" \
    STATE_DIR="${WORK}/install-state" \
    TMPDIR="${install_tmpdir}" \
    bash -c '
      set -uo pipefail
      BOX_RUNNER_INSTALL_SOURCE_ONLY=1
      # shellcheck disable=SC1091
      . "$1/install.sh"
      platform_tag() { printf "osx-arm64"; }
      curl() {
        local arg out="" prev=""
        for arg in "$@"; do
          [ "${prev}" = "-o" ] && out="${arg}"
          prev="${arg}"
        done
        [ -n "${out}" ] || return 1
        printf "stub-tarball" > "${out}"
      }
      compute_sha256() { pinned_sha256 osx-arm64; }
      tar() {
        mkdir -p "${RUNNER_DIR}"
        printf "#!/bin/sh\n" > "${RUNNER_DIR}/run.sh"
        chmod +x "${RUNNER_DIR}/run.sh"
      }
      main
    ' _ "${install_stage}"
}

# --- 6a. THE REGRESSION ASSERTION: a successful first install exits 0 ------
assert_exit "install.sh: first install through main() exits 0" 0 "done." run_install_main

# --- 6b. ...and it exited 0 because it did the work, not because it bailed -
# shellcheck disable=SC2016  # deliberate: ACTIONS_RUNNER_VERSION must expand
# inside install_call's shell, where install.sh has been sourced, not here.
assert_eq "install.sh: first install records the pinned runner version" \
  "$(install_call eval 'printf %s "${ACTIONS_RUNNER_VERSION}"')" \
  "$(cat "${install_runner_dir}/.box-runner-installed-version" 2>/dev/null)"
assert_true "install.sh: first install seeds box-runner.env" \
  test -f "${install_stage}/box-runner.env"
assert_true "install.sh: first install renders the plist" \
  test -f "${install_stage}/com.nemotron35.box-runner.plist"

# --- 6c. the cleanup the trap exists for still happened, exactly once ------
leftover_tmp="$(find "${install_tmpdir}" -maxdepth 1 -name 'box-runner-install.*' 2>/dev/null | wc -l | tr -d ' ')"
assert_eq "install.sh: first install leaves no temp download dir behind" "0" "${leftover_tmp}"

# --- 6d. the re-run path, which is what masked the bug ---------------------
#     It returns before the trap is ever armed, so it exited 0 even when the
#     first install did not: green here is NOT evidence 6a is green.
assert_exit "install.sh: idempotent re-run is a no-op and exits 0" 0 "idempotent no-op" run_install_main

# =========================================================================
printf '\n%d passed, %d failed (floor: %d assertions)\n' "${PASSED}" "${FAILED}" "${EXPECTED_MIN_ASSERTIONS}"
total=$((PASSED + FAILED))
if [ "${total}" -lt "${EXPECTED_MIN_ASSERTIONS}" ]; then
  printf 'FAIL  suite ran only %d assertions, floor is %d (a section stopped running silently)\n' "${total}" "${EXPECTED_MIN_ASSERTIONS}"
  exit 1
fi
if [ "${FAILED}" -gt 0 ]; then
  printf '\nFailures:\n'
  printf '  - %s\n' "${FAILURES[@]}"
  exit 1
fi
exit 0
