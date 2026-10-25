#!/usr/bin/env bash
# Native tests for tools/stop-leaked-residents.sh.
# No model, GPU, network, sudo, or process-table shim is used.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
STOP="${ROOT_DIR}/tools/stop-leaked-residents.sh"

if ! ps -A -o pid=,args= >/dev/null 2>&1; then
  echo "test-stop-leaked-residents: FAIL 0 -- cannot enumerate processes on this host" >&2
  exit 1
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/test-stop-leaked-residents.XXXXXX")"
JOB_ROOT="${WORK}/root"
LOCK="${WORK}/gpu.lock"
SOCKET_PATH="${WORK}/s.sock"
HELPERS=("")
LOCK_HOLDER_PID=""
fails=0
ALL_OUTPUT=""
mkdir -p "${JOB_ROOT}/.build/release" "${WORK}/outside" "${WORK}/lock-directory"
export RESIDENT_UP_LOCK_PATH="${LOCK}"

cleanup() {
  local args pid
  for pid in "${HELPERS[@]}"; do
    args="$(ps -ww -o args= -p "${pid}" 2>/dev/null || true)"
    [[ "${args}" != *"${WORK}"* ]] || kill -KILL "${pid}" 2>/dev/null || true
  done
  if [[ -n "${LOCK_HOLDER_PID}" ]]; then
    args="$(ps -ww -o args= -p "${LOCK_HOLDER_PID}" 2>/dev/null || true)"
    [[ "${args}" != *"${WORK}"* ]] || kill -KILL "${LOCK_HOLDER_PID}" 2>/dev/null || true
  fi
  for pid in "${HELPERS[@]}" "${LOCK_HOLDER_PID}"; do
    [[ -z "${pid}" ]] || wait "${pid}" 2>/dev/null || true
  done
  rm -rf "${WORK}"
}
trap cleanup EXIT

pass() { printf 'test-stop-leaked-residents: PASS %s\n' "$1"; }
fail() { printf 'test-stop-leaked-residents: FAIL %s\n' "$1" >&2; fails=$((fails + 1)); }
record() { ALL_OUTPUT="${ALL_OUTPUT}${1}"$'\n'; }

is_alive() {
  local state
  kill -0 "$1" 2>/dev/null || return 1
  state="$(ps -o stat= -p "$1" 2>/dev/null | tr -d '[:space:]' || true)"
  [[ "${state}" != Z* ]]
}

gone_within() {
  local pid="$1"
  local tries="${2:-30}"
  while is_alive "${pid}" && [[ ${tries} -gt 0 ]]; do
    sleep 0.1
    tries=$((tries - 1))
  done
  ! is_alive "${pid}"
}

HELPER="${WORK}/helper.pl"
cat > "${HELPER}" <<'PL'
use strict;
use warnings;
use POSIX qw(setsid);

my ($mode, $title, $child_file, $child_title, $sibling_file, $sibling_title) = @ARGV;
setsid() >= 0 or die "setsid failed: $!";

if ($mode eq "ignore") {
    $SIG{TERM} = "IGNORE";
}
if ($mode eq "group") {
    my $child = fork();
    defined $child or die "fork failed: $!";
    if ($child == 0) {
        $0 = $child_title;
        sleep 300;
        exit 0;
    }
    open my $fh, ">", $child_file or die "child pid file: $!";
    print {$fh} "$child\n";
    close $fh;
}
if ($mode eq "nonleader") {
    my $child = fork();
    defined $child or die "fork failed: $!";
    if ($child == 0) {
        $0 = $child_title;
        sleep 300;
        exit 0;
    }
    open my $child_fh, ">", $child_file or die "child pid file: $!";
    print {$child_fh} "$child\n";
    close $child_fh;

    my $sibling = fork();
    defined $sibling or die "fork failed: $!";
    if ($sibling == 0) {
        $0 = $sibling_title;
        sleep 300;
        exit 0;
    }
    open my $sibling_fh, ">", $sibling_file or die "sibling pid file: $!";
    print {$sibling_fh} "$sibling\n";
    close $sibling_fh;
}
$0 = $title;
sleep 300;
PL

check_title() {
  local pid="$1"
  local title="$2"
  local got=""
  local got_argv0=""
  local got_argv1=""
  local title_argv0=""
  local title_argv1=""
  local tries=50

  read -r title_argv0 title_argv1 _ <<< "${title}"
  while [[ ${tries} -gt 0 ]]; do
    got="$(ps -ww -o args= -p "${pid}" 2>/dev/null || true)"
    read -r got_argv0 got_argv1 _ <<< "${got}"
    [[ "${got_argv0}" == "${title_argv0}" && "${got_argv1}" == "${title_argv1}" && "${got}" == "${title}"* ]] && break
    sleep 0.1
    tries=$((tries - 1))
  done
  if [[ "${got_argv0}" != "${title_argv0}" || "${got_argv1}" != "${title_argv1}" || "${got}" != "${title}"* ]]; then
    echo "test-stop-leaked-residents: FAIL 0 -- helper process title does not match argv0, argv1, and full title prefix" >&2
    exit 1
  fi
}

start_helper() {
  local mode="$1"
  local title="$2"
  local child_file="${3:-}"
  local child_title="${4:-}"
  local sibling_file="${5:-}"
  local sibling_title="${6:-}"

  perl "${HELPER}" "${mode}" "${title}" "${child_file}" "${child_title}" "${sibling_file}" "${sibling_title}" &
  STARTED_PID=$!
  HELPERS+=("${STARTED_PID}")
  check_title "${STARTED_PID}" "${title}"
}

read_pid_file() {
  local path="$1"
  local tries=50
  while [[ ${tries} -gt 0 ]]; do
    [[ -s "${path}" ]] && break
    sleep 0.1
    tries=$((tries - 1))
  done
  sed -n '1p' "${path}" 2>/dev/null || true
}

run_stop() {
  "${STOP}" "$@" --root "${JOB_ROOT}" 2>&1
}

resident_title="${JOB_ROOT}/.build/release/bench-worker resident --socket ${SOCKET_PATH}"

# 1. No resident.
out="$(run_stop --mode preflight)"; rc=$?; record "${out}"
if [[ ${rc} -eq 0 && "${out}" == *"stop-leaked-residents: no leftover resident"* ]]; then
  pass "1 -- no resident"
else
  fail "1 -- no resident (rc=${rc})"
fi

# 2. One rooted resident is stopped and named by pid only.
start_helper normal "${resident_title}"
pid2="${STARTED_PID}"
out="$(run_stop --mode preflight)"; rc=$?; record "${out}"
if [[ ${rc} -eq 0 && "${out}" == *"::warning::stopping resident bench-worker left by an earlier job (pid ${pid2}, elapsed "* ]] && gone_within "${pid2}"; then
  pass "2 -- one rooted resident"
else
  fail "2 -- one rooted resident (rc=${rc}, pid=${pid2})"
fi

# 3. A rooted group leader and its child are both stopped.
child_file="${WORK}/group-child.pid"
child_title="${WORK}/group-child keep"
start_helper group "${resident_title}" "${child_file}" "${child_title}"
pid3="${STARTED_PID}"
child3="$(read_pid_file "${child_file}")"
[[ -z "${child3}" ]] || HELPERS+=("${child3}")
[[ -z "${child3}" ]] || check_title "${child3}" "${child_title}"
out="$(run_stop --mode preflight)"; rc=$?; record "${out}"
if [[ ${rc} -eq 0 && -n "${child3}" ]] && gone_within "${pid3}" && gone_within "${child3}"; then
  pass "3 -- rooted process group and child"
else
  fail "3 -- rooted process group and child (rc=${rc}, leader=${pid3}, child=${child3:-missing})"
fi

# 4. A rooted resident that ignores SIGTERM is killed on time.
start_helper ignore "${resident_title}"
pid4="${STARTED_PID}"
SECONDS=0
out="$(run_stop --mode preflight --stop-wait 1)"; rc=$?; elapsed_s=${SECONDS}; record "${out}"
if [[ ${rc} -eq 0 && ${elapsed_s} -le 6 ]] && gone_within "${pid4}"; then
  pass "4 -- SIGTERM ignore escalates to SIGKILL"
else
  fail "4 -- SIGTERM ignore escalates to SIGKILL (rc=${rc}, elapsed=${elapsed_s}s)"
fi

# 5. Three nonmatches survive and lookalikes produce one count-only warning.
baseline_out="$(run_stop --mode preflight)"; baseline_rc=$?; record "${baseline_out}"
baseline_count="$(printf '%s\n' "${baseline_out}" | awk '/::warning::found [0-9]+ other process\(es\) that look like a resident; not stopped/{count=$2} END{print count+0}')"
outside_title="${WORK}/outside/bench-worker resident --socket ${SOCKET_PATH}"
wrong_arg_title="${JOB_ROOT}/.build/release/bench-worker worker --socket ${SOCKET_PATH}"
later_words_title="${WORK}/outside/not-worker 300 bench-worker resident"
start_helper normal "${outside_title}"; pid5a="${STARTED_PID}"
start_helper normal "${wrong_arg_title}"; pid5b="${STARTED_PID}"
start_helper normal "${later_words_title}"; pid5c="${STARTED_PID}"
out="$(run_stop --mode preflight)"; rc=$?; record "${out}"
warning_count="$(printf '%s\n' "${out}" | awk '/::warning::found [0-9]+ other process\(es\) that look like a resident; not stopped/{count++} END{print count+0}')"
reported_count="$(printf '%s\n' "${out}" | awk '/::warning::found [0-9]+ other process\(es\) that look like a resident; not stopped/{count=$2} END{print count+0}')"
expected_count=$((baseline_count + 2))
if [[ ${baseline_rc} -eq 0 && ${rc} -eq 0 && ${warning_count} -eq 1 && ${reported_count} -eq ${expected_count} ]] && is_alive "${pid5a}" && is_alive "${pid5b}" && is_alive "${pid5c}"; then
  pass "5 -- three nonmatches survive"
else
  fail "5 -- three nonmatches survive (baseline_rc=${baseline_rc}, rc=${rc}, warnings=${warning_count}, count=${reported_count}, want=${expected_count})"
fi

# 6. A busy lock skips preflight and defers cleanup without killing.
ready="${WORK}/lock-ready"
python3 - "${LOCK}" "${ready}" <<'PY' &
import fcntl
import os
import sys
import time

fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o666)
fcntl.flock(fd, fcntl.LOCK_EX)
open(sys.argv[2], "w").close()
while True:
    time.sleep(1)
PY
LOCK_HOLDER_PID=$!
for _ in 1 2 3 4 5 6 7 8 9 10; do
  [[ -f "${ready}" ]] && break
  sleep 0.1
done
start_helper normal "${resident_title}"
pid6="${STARTED_PID}"
out_pre="$(run_stop --mode preflight)"; rc_pre=$?; record "${out_pre}"
out_clean="$(run_stop --mode cleanup)"; rc_clean=$?; record "${out_clean}"
want_pre="::warning::the GPU lock is busy (a measurement is running on this box); leftover check skipped"
want_clean="::warning::the GPU lock is busy; cleanup deferred to the next job's preflight"
if [[ ${rc_pre} -eq 0 && "${out_pre}" == "${want_pre}" && ${rc_clean} -eq 3 && "${out_clean}" == "${want_clean}" ]] && is_alive "${pid6}"; then
  pass "6 -- busy lock uses exact outcomes"
else
  fail "6 -- busy lock uses exact outcomes (preflight=${rc_pre}, cleanup=${rc_clean})"
fi
kill "${LOCK_HOLDER_PID}" 2>/dev/null || true
wait "${LOCK_HOLDER_PID}" 2>/dev/null || true
LOCK_HOLDER_PID=""
out="$(run_stop --mode preflight)"; rc=$?; record "${out}"

# 8. Unknown options, a bad mode, and removed --lock all exit 2.
out_unknown="$("${STOP}" "--unknown=${SOCKET_PATH}" 2>&1)"; rc_unknown=$?; record "${out_unknown}"
out_mode="$("${STOP}" --mode bad --root "${JOB_ROOT}" 2>&1)"; rc_mode=$?; record "${out_mode}"
out_lock="$("${STOP}" --mode preflight --root "${JOB_ROOT}" --lock "${LOCK}" 2>&1)"; rc_lock=$?; record "${out_lock}"
if [[ ${rc_unknown} -eq 2 && ${rc_mode} -eq 2 && ${rc_lock} -eq 2 ]]; then
  pass "8 -- bad arguments"
else
  fail "8 -- bad arguments (unknown=${rc_unknown}, mode=${rc_mode}, lock=${rc_lock})"
fi

# 9. Lock open errors refuse preflight and defer cleanup without a traceback.
out_pre="$(RESIDENT_UP_LOCK_PATH="${WORK}/lock-directory" run_stop --mode preflight)"; rc_pre=$?
out_clean="$(RESIDENT_UP_LOCK_PATH="${WORK}/lock-directory" run_stop --mode cleanup)"; rc_clean=$?
record "${out_pre}"
record "${out_clean}"
if [[ ${rc_pre} -eq 1 && "${out_pre}" == "::error::cannot open the GPU lock ("*"; refusing" && "${out_pre}" != *"Traceback"* && ${rc_clean} -eq 3 && "${out_clean}" == "::warning::cannot open the GPU lock ("*"; cleanup deferred to the next job's preflight" && "${out_clean}" != *"Traceback"* ]]; then
  pass "9 -- lock open errors are handled"
else
  fail "9 -- lock open errors are handled (preflight=${rc_pre}, cleanup=${rc_clean})"
fi

# 10. Default runner roots are box-wide; unrelated paths stay untouched.
home_root="${WORK}/home/actions-runner-test/_work/x"
else_root="${WORK}/elsewhere"
mkdir -p "${home_root}/.build/release" "${else_root}"
home_title="${home_root}/.build/release/bench-worker resident"
else_title="${else_root}/bench-worker resident"
start_helper normal "${home_title}"; pid10a="${STARTED_PID}"
start_helper normal "${else_title}"; pid10b="${STARTED_PID}"
out="$(HOME="${WORK}/home" MLXFAST_BASELINE_WORKSPACE="" "${STOP}" --mode preflight 2>&1)"; rc=$?; record "${out}"
if [[ ${rc} -eq 0 ]] && gone_within "${pid10a}" && is_alive "${pid10b}"; then
  pass "10 -- default runner roots"
else
  fail "10 -- default runner roots (rc=${rc})"
fi

# 11. A non-leader resident stops without its group owner or sibling.
owner_title="${WORK}/group-owner keep"
target_file="${WORK}/nonleader-target.pid"
sibling_file="${WORK}/nonleader-sibling.pid"
sibling_title="${WORK}/nonleader-sibling keep"
start_helper nonleader "${owner_title}" "${target_file}" "${resident_title}" "${sibling_file}" "${sibling_title}"
owner11="${STARTED_PID}"
target11="$(read_pid_file "${target_file}")"
sibling11="$(read_pid_file "${sibling_file}")"
[[ -z "${target11}" ]] || HELPERS+=("${target11}")
[[ -z "${sibling11}" ]] || HELPERS+=("${sibling11}")
[[ -z "${target11}" ]] || check_title "${target11}" "${resident_title}"
[[ -z "${sibling11}" ]] || check_title "${sibling11}" "${sibling_title}"
out="$(run_stop --mode preflight)"; rc=$?; record "${out}"
if [[ ${rc} -eq 0 && -n "${target11}" && -n "${sibling11}" ]] && gone_within "${target11}" && is_alive "${owner11}" && is_alive "${sibling11}"; then
  pass "11 -- non-leader stops alone"
else
  fail "11 -- non-leader stops alone (rc=${rc}, target=${target11:-missing}, sibling=${sibling11:-missing})"
fi

# 7. Public output from all cases hides the socket, root, and lock paths.
if [[ "${ALL_OUTPUT}" != *"${SOCKET_PATH}"* && "${ALL_OUTPUT}" != *"${JOB_ROOT}"* && "${ALL_OUTPUT}" != *"${LOCK}"* ]]; then
  pass "7 -- output hides all private paths"
else
  fail "7 -- output hides all private paths"
fi

if [[ ${fails} -ne 0 ]]; then
  printf 'test-stop-leaked-residents: %d case(s) failed\n' "${fails}" >&2
  exit 1
fi
echo "test-stop-leaked-residents: all 11 cases passed"
