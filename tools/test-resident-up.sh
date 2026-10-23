#!/usr/bin/env bash
# test-resident-up.sh -- prove tools/resident-up.sh without a model, a GPU or
# a network.
#
# WHAT THIS PROVES, with a stub bench-worker that binds its socket after a
# delay and answers the hello the way the real resident does:
#   1. a served window: the stub receives the contract argv (resident,
#      --weights, --speculative-protocol v1.1, --socket), the wrapped
#      command sees BENCH_WORKER_RESIDENT_SOCKET and the resident's identity
#      file, its exit code is returned, and after the window the stub, the
#      socket, the pidfile and the pgid file are all gone
#   2. --hello-identity reaches the stub and the command
#   3. the GPU lock must be HELD: nobody holding it refuses by name before any
#      boot; the refusal has no environment bypass
#   4. an unpinned iogpu wired limit refuses by name before any boot
#   5. a resident already up (live pidfile, or a socket that answers) refuses
#      by name, and the first window is left alone
#   6. a resident that dies before it is healthy fails the window, with its log
#   7. a resident that never binds fails on the health timeout
#      a resident that streams an oversized or never-ending hello is refused,
#      stopped, and cannot hold the GPU window forever
#   8. a resident that ignores SIGTERM is killed with the process group
#   9. the resident gets only the documented environment allowlist, including
#      model/runtime controls but excluding evaluator paths and ranked flags
#
# And the PER-LEG form benchd drives (--boot / --stop), against a synthetic
# tree:
#   10. --boot writes the socket path as the first line of --socket-out, a
#       <socket>.pid sidecar and a <socket>.ready marker, exits 0, and leaves
#       the resident RUNNING
#   11. --spec serial is authoritative over a tree whose mtp-head.manifest.json
#       declares a draft depth
#   12. the boot loads THAT TREE'S own weights, not an inherited path
#   13. a second --boot while a resident is up refuses box-wide, across trees
#   14. --stop ends the resident and removes all three files; a second --stop is
#       a no-op; the next leg boots afterwards
#   15. --stop repairs hostile nested modes and Darwin flags without following
#       links outside the private temp tree; a failed cleanup still fails and
#       keeps its sidecar
#   16. every flag without a value, every missing required flag, and every
#       spec/draft-length mismatch refuses BY NAME
#   17. RESIDENT_UP_LOG_DIR outside MLXFAST_RESIDENT_LOG_ROOT refuses by name
#   18. under the real sandbox-exec, private TMPDIR is outside
#       RESIDENT_UP_LOG_DIR; --stop and a failed boot remove it. macOS only.
#       A failed wrapper cleanup returns non-zero and keeps its sidecar.
#   19. the worker inherits a 256 MiB file-size limit, so a bad resident cannot
#       grow its retained stdout/stderr log without bound.
#   20. the worker does not inherit descriptors opened by the trusted parent,
#       including fd 9 above a lowered soft limit of 8 and high fd 200.
#
# The stub is a Python script, and sandbox-exec would refuse the interpreter
# exec it needs, so these cases run with RESIDENT_UP_SANDBOX=0. Case 4b proves
# that knob is refused on a self-hosted runner, and that a bad value refuses.
# Case 15 runs the real sandbox: its worker is a symlink to the python3
# interpreter (one exec, which the profile allows), and the interpreter runs
# the file `resident` in the tree, because the boot runs in the tree.
#
# Hermetic: bash (3.2 on macOS is enough) and python3.
#
# Usage: tools/test-resident-up.sh
# Exit:  0 all cases pass, 1 a case failed (printed with a FAIL prefix)
set -uo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
RESIDENT_UP="${ROOT_DIR}/tools/resident-up.sh"

# Under the temp root resident-up.sh uses, so its cleanup accepts the test trees.
WORK_ROOT="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"
WORK="$(mktemp -d "${WORK_ROOT%/}/test-resident-up.XXXXXX")"
trap 'cleanup' EXIT
LOCK_HOLDER_PID=""
cleanup() {
  [[ -n "${LOCK_HOLDER_PID}" ]] && kill "${LOCK_HOLDER_PID}" 2>/dev/null
  pkill -f "stub-bench-worker.py" 2>/dev/null
  # The per-leg cases stage the stub under the name benchd resolves, so it does
  # not match the pattern above.
  pkill -f "release/bench-worker resident" 2>/dev/null
  if [[ "$(uname -s)" == "Darwin" ]]; then
    if [[ -n "${PROTECTED_TMP:-}" ]]; then
      chmod 700 "${PROTECTED_TMP}/mode-000" "${PROTECTED_TMP}/mode-000/mode-000" 2>/dev/null || true
    fi
    [[ -z "${INSIDE_FLAG_FILE:-}" ]] || chflags nouchg "${INSIDE_FLAG_FILE}" 2>/dev/null || true
    [[ -z "${OUTSIDE_FILE:-}" ]] || chflags nouchg "${OUTSIDE_FILE}" 2>/dev/null || true
  fi
  rm -rf "${WORK}"
}

export RESIDENT_UP_SANDBOX=0
unset RUNNER_ENVIRONMENT

fails=0
pass() { printf 'test-resident-up: PASS -- %s\n' "$*"; }
fail() { printf 'test-resident-up: FAIL -- %s\n' "$*" >&2; fails=$((fails + 1)); }

# --- the stub resident --------------------------------------------------------
# Speaks the contract: validates argv, waits BENCH_WORKER_TEST_LOAD_S, binds the socket,
# then serves one connection at a time: on accept it sends the hello line
# (unprompted, ok:true, backend mlx-resident) and closes when the client does.
# BENCH_WORKER_TEST_DIE_BEFORE_BIND=1 exits 3 before binding.
# BENCH_WORKER_TEST_NEVER_BIND=1 sleeps forever without binding.
# BENCH_WORKER_TEST_IGNORE_TERM=1 ignores SIGTERM.
# BENCH_WORKER_TEST_HELLO_MODE=overflow streams past the hello byte cap;
# `deadline` streams slowly without a newline until the whole-probe deadline.
# The stub records its argv and,
# when asked, its environment through BENCH_WORKER_TEST_* paths.
STUB="${WORK}/stub-bench-worker.py"
cat > "${STUB}" <<'PY'
#!/usr/bin/env python3
import json, os, resource, signal, socket, sys, time
argv = sys.argv[1:]
with open(os.environ["BENCH_WORKER_TEST_ARGV_FILE"], "w") as f:
    f.write("\n".join(argv) + "\n")
if os.environ.get("BENCH_WORKER_TEST_ENV_FILE"):
    with open(os.environ["BENCH_WORKER_TEST_ENV_FILE"], "w") as f:
        for key, value in sorted(os.environ.items()):
            f.write(f"{key}={value}\n")
if os.environ.get("BENCH_WORKER_TEST_LIMIT_FILE"):
    with open(os.environ["BENCH_WORKER_TEST_LIMIT_FILE"], "w") as f:
        f.write(str(resource.getrlimit(resource.RLIMIT_FSIZE)[0]) + "\n")
if os.environ.get("BENCH_WORKER_TEST_NOFILE_FILE"):
    with open(os.environ["BENCH_WORKER_TEST_NOFILE_FILE"], "w") as f:
        f.write(str(resource.getrlimit(resource.RLIMIT_NOFILE)[0]) + "\n")
if os.environ.get("BENCH_WORKER_TEST_FD_FILE"):
    def fd_status(fd):
        try:
            os.fstat(fd)
            return "open"
        except OSError:
            return "closed"
    with open(os.environ["BENCH_WORKER_TEST_FD_FILE"], "w") as f:
        f.write(f"9={fd_status(9)}\n200={fd_status(200)}\n")
if not argv or argv[0] != "resident":
    print("stub: first argument must be resident", file=sys.stderr); sys.exit(2)
opts = {}
i = 1
while i < len(argv):
    if argv[i].startswith("--"):
        opts.setdefault(argv[i], []).append(argv[i + 1] if i + 1 < len(argv) else None); i += 2
    else:
        print(f"stub: stray argument {argv[i]}", file=sys.stderr); sys.exit(2)
for needed in ("--weights", "--socket", "--speculative-protocol"):
    if needed not in opts:
        print(f"stub: missing {needed}", file=sys.stderr); sys.exit(2)
if opts["--speculative-protocol"] != ["v1.1"]:
    print("stub: --speculative-protocol must be v1.1", file=sys.stderr); sys.exit(2)
if os.environ.get("BENCH_WORKER_TEST_IGNORE_TERM") == "1":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
else:
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
time.sleep(float(os.environ.get("BENCH_WORKER_TEST_LOAD_S", "1")))
if os.environ.get("BENCH_WORKER_TEST_DIE_BEFORE_BIND") == "1":
    print("stub: dying before bind, as asked", file=sys.stderr); sys.exit(3)
if os.environ.get("BENCH_WORKER_TEST_NEVER_BIND") == "1":
    while True:
        time.sleep(1)
path = opts["--socket"][0]
try:
    os.unlink(path)
except FileNotFoundError:
    pass
srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
srv.bind(path)
srv.listen(4)
hello = {"id": 0, "nonce": "stubnonce", "ok": True, "protocol_version": 1,
         "backend": "mlx-resident", "device": "stub", "spec_modes": ["serial", "mtp"],
         "runner": {"id": "layr/stub"}, "resident": {"pid": os.getpid(), "load_epoch": 1}}
while True:
    conn, _ = srv.accept()
    try:
        hello_mode = os.environ.get("BENCH_WORKER_TEST_HELLO_MODE", "normal")
        if hello_mode == "overflow":
            while True:
                conn.sendall(b"x" * 65536)
        elif hello_mode == "deadline":
            while True:
                conn.sendall(b"x")
                time.sleep(0.25)
        else:
            conn.sendall((json.dumps(hello) + "\n").encode())
            while conn.recv(65536):
                pass
    except OSError:
        pass
    finally:
        conn.close()
PY
chmod +x "${STUB}"

# Bash 3.2 needs more than eight descriptors for this script's own
# redirections. Lower only each Python child's soft limit instead, including
# the real launcher and the resident stub it execs.
REAL_PYTHON="$(command -v python3)"
LOW_LIMIT_BIN="${WORK}/low-limit-bin"
mkdir -p "${LOW_LIMIT_BIN}"
cat > "${LOW_LIMIT_BIN}/python3" <<'SH'
#!/bin/bash
ulimit -n 8 || exit 1
exec "${BENCH_WORKER_TEST_PYTHON}" "$@"
SH
chmod +x "${LOW_LIMIT_BIN}/python3"

# --- the lock, held by a background holder for the served cases --------------
LOCK="${WORK}/gpu.lock"
: > "${LOCK}"
hold_lock() {
  python3 - "${LOCK}" <<'PY' &
import fcntl, sys, time
f = open(sys.argv[1], "a+")
fcntl.flock(f, fcntl.LOCK_EX)
while True:
    time.sleep(1)
PY
  LOCK_HOLDER_PID=$!
  sleep 0.5
}
release_lock() {
  [[ -n "${LOCK_HOLDER_PID}" ]] && kill "${LOCK_HOLDER_PID}" 2>/dev/null
  wait "${LOCK_HOLDER_PID}" 2>/dev/null
  LOCK_HOLDER_PID=""
}

WEIGHTS="${WORK}/weights"; mkdir -p "${WEIGHTS}"
WIRED_OK="${WORK}/wired-ok.sh"; printf '#!/bin/sh\necho 112000\n' > "${WIRED_OK}"; chmod +x "${WIRED_OK}"
WIRED_ZERO="${WORK}/wired-zero.sh"; printf '#!/bin/sh\necho 0\n' > "${WIRED_ZERO}"; chmod +x "${WIRED_ZERO}"

# run_up LOGDIR [ENV=VAL...] -- ARGS... : drive the real script with the stub.
run_up() {
  local logdir="$1"; shift
  local envs=()
  while [[ $# -gt 0 && "$1" == *=* ]]; do envs+=("$1"); shift; done
  # The case's own ENV=VAL pairs come LAST, so a case can override a default.
  env MLXFAST_ENGINE_BIN="${STUB}" \
    MLXFAST_RESIDENT_LOG_ROOT="${logdir}" \
    RESIDENT_UP_LOG_DIR="${logdir}" \
    RESIDENT_UP_LOCK_PATH="${LOCK}" \
    RESIDENT_UP_WIRED_LIMIT_READER="${WIRED_OK}" \
    BENCH_WORKER_TEST_ARGV_FILE="${logdir}/stub-argv" \
    ${envs[@]+"${envs[@]}"} \
    "${RESIDENT_UP}" "$@" 2>&1
}

# --- 1. a served window ------------------------------------------------------
hold_lock
L1="${WORK}/case1"; mkdir -p "${L1}"
SOCK1="${WORK}/case1.sock"
# These stand in for trusted descriptors held by the measure job. Open them
# before lowering the soft limit so both remain open in this subshell.
out="$(
  exec 9>"${WORK}/trusted-parent-fd-9"
  exec 200>"${WORK}/trusted-parent-fd-200"
  PATH="${LOW_LIMIT_BIN}:${PATH}" BENCH_WORKER_TEST_PYTHON="${REAL_PYTHON}" \
    python3 -c 'import os, resource; assert resource.getrlimit(resource.RLIMIT_NOFILE)[0] == 8; os.fstat(9); os.fstat(200)' \
    || exit
  run_up "${L1}" PATH="${LOW_LIMIT_BIN}:${PATH}" BENCH_WORKER_TEST_PYTHON="${REAL_PYTHON}" \
    RESIDENT_UP_HEALTH_TIMEOUT_S=30 BENCH_WORKER_TEST_LIMIT_FILE="${L1}/file-limit" \
    BENCH_WORKER_TEST_NOFILE_FILE="${L1}/nofile-limit" BENCH_WORKER_TEST_FD_FILE="${L1}/fd-status" \
    --weights "${WEIGHTS}" --socket "${SOCK1}" -- \
    bash -c 'echo "socket=$BENCH_WORKER_RESIDENT_SOCKET hello=${BENCH_WORKER_RESIDENT_HELLO:-unset} identity=$RESIDENT_IDENTITY_FILE"; test -S "$BENCH_WORKER_RESIDENT_SOCKET" && echo socket-live; kill -0 "$RESIDENT_UP_PID" && echo resident-live; exit 7'
)"
rc=$?
if [[ "${rc}" -eq 7 ]]; then pass "the wrapped command's exit code (7) is returned"; else fail "expected exit 7, got ${rc}: ${out}"; fi
if [[ "${out}" == *"socket=${SOCK1} hello=unset identity=${L1}/resident-identity.json"* && "${out}" == *"socket-live"* && "${out}" == *"resident-live"* ]]; then
  pass "the command runs with BENCH_WORKER_RESIDENT_SOCKET, no hello identity by default, and a live resident"
else
  fail "the command did not see the window: ${out}"
fi
if [[ -f "${L1}/stub-argv" ]] && grep -qx -- "resident" "${L1}/stub-argv" && grep -qx -- "--speculative-protocol" "${L1}/stub-argv" \
   && grep -qx -- "v1.1" "${L1}/stub-argv" \
   && grep -qx -- "${SOCK1}" "${L1}/stub-argv"; then
  pass "the resident received the contract argv"
else
  fail "the resident argv is wrong: $(tr '\n' ' ' < "${L1}/stub-argv" 2>/dev/null)"
fi
if [[ "${out}" == *"resident healthy on ${SOCK1}"* ]]; then pass "the health probe read the resident's hello"; else fail "no health line: ${out}"; fi
if [[ "$(cat "${L1}/file-limit" 2>/dev/null)" == "268435456" ]]; then
  pass "the resident alone receives the 256 MiB file-size cap"
else
  fail "the resident file-size cap is not 256 MiB: $(cat "${L1}/file-limit" 2>/dev/null)"
fi
if [[ "$(cat "${L1}/nofile-limit" 2>/dev/null)" == "8" ]]; then
  pass "the resident test reaches the launcher with a soft descriptor limit of 8"
else
  fail "the resident soft descriptor limit is not 8: $(cat "${L1}/nofile-limit" 2>/dev/null)"
fi
if grep -qx '9=closed' "${L1}/fd-status" 2>/dev/null; then
  pass "the resident closes fd 9 even when its soft limit is 8"
else
  fail "the resident inherited trusted parent fd 9"
fi
if grep -qx '200=closed' "${L1}/fd-status" 2>/dev/null; then
  pass "the resident closes high trusted parent fd 200"
else
  fail "the resident inherited high trusted parent fd 200"
fi
if python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["hello"]["backend"]=="mlx-resident" and d["weight_owner"]=="bench-worker-resident"' "${L1}/resident-identity.json" 2>/dev/null; then
  pass "the identity file records the resident's hello"
else
  fail "identity file missing or wrong: $(cat "${L1}/resident-identity.json" 2>/dev/null)"
fi
sleep 0.5
if [[ ! -e "${SOCK1}" && ! -e "${L1}/resident.pid" && ! -e "${L1}/resident.pgid" ]] && ! pgrep -f "stub-bench-worker.py resident --weights ${WEIGHTS} --speculative-protocol v1.1 --socket ${SOCK1}" >/dev/null; then
  pass "after the window: resident gone, socket, pidfile and pgid file removed"
else
  fail "the window left something behind: $(ls "${L1}" "${SOCK1}" 2>&1 | tr '\n' ' ')"
fi

# --- 2. --hello-identity -----------------------------------------------------
L2="${WORK}/case2"; mkdir -p "${L2}"
SOCK2="${WORK}/case2.sock"
out="$(run_up "${L2}" RESIDENT_UP_HEALTH_TIMEOUT_S=30 --weights "${WEIGHTS}" --hello-identity --socket "${SOCK2}" -- \
  bash -c 'echo "hello=${BENCH_WORKER_RESIDENT_HELLO:-unset}"')"
if [[ "${out}" == *"hello=1"* ]]; then
  pass "--hello-identity exports BENCH_WORKER_RESIDENT_HELLO=1"
else
  fail "--hello-identity not honoured: ${out} / $(tr '\n' ' ' < "${L2}/stub-argv" 2>/dev/null)"
fi

# --- 5. a resident already up refuses ----------------------------------------
L5="${WORK}/case5"; mkdir -p "${L5}"
SOCK5="${WORK}/case5.sock"
FIRST_OUT="${WORK}/case5.first"
( run_up "${L5}" RESIDENT_UP_HEALTH_TIMEOUT_S=30 --weights "${WEIGHTS}" --socket "${SOCK5}" -- \
    bash -c 'sleep 6; echo first-window-finished' > "${FIRST_OUT}" 2>&1 ) &
FIRST_PID=$!
for _ in $(seq 1 40); do [[ -S "${SOCK5}" ]] && break; sleep 0.2; done
out="$(run_up "${L5}" RESIDENT_UP_HEALTH_TIMEOUT_S=30 --weights "${WEIGHTS}" --socket "${SOCK5}" -- true)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"REFUSED (resident-already-up)"* && "${out}" == *"Nothing has been loaded"* ]]; then
  pass "a second window while a resident is up refuses by name (same pidfile)"
else
  fail "a second window did not refuse by name (rc ${rc}): ${out}"
fi
L5b="${WORK}/case5b"; mkdir -p "${L5b}"
out="$(run_up "${L5b}" RESIDENT_UP_HEALTH_TIMEOUT_S=30 --weights "${WEIGHTS}" --socket "${SOCK5}" -- true)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"REFUSED (resident-already-up)"* && "${out}" == *"already answers on ${SOCK5}"* ]]; then
  pass "a second window against a socket that answers refuses by name (different log dir)"
else
  fail "a socket that answers did not refuse (rc ${rc}): ${out}"
fi
wait "${FIRST_PID}"
if grep -q "first-window-finished" "${FIRST_OUT}"; then
  pass "the first window was left alone and finished"
else
  fail "the first window was disturbed: $(cat "${FIRST_OUT}")"
fi

# --- 6. a resident that dies before it is healthy ----------------------------
L6="${WORK}/case6"; mkdir -p "${L6}"
out="$(run_up "${L6}" BENCH_WORKER_TEST_DIE_BEFORE_BIND=1 RESIDENT_UP_HEALTH_TIMEOUT_S=30 --weights "${WEIGHTS}" --socket "${WORK}/case6.sock" -- echo should-not-run)"; rc=$?
if [[ "${rc}" -eq 1 && "${out}" == *"exited before it was healthy"* && "${out}" == *"dying before bind"* && "${out}" != *"should-not-run"* ]]; then
  pass "a resident that dies before it is healthy fails the window with its log, and the command never runs"
else
  fail "an early death was not reported (rc ${rc}): ${out}"
fi

# --- 7. a resident that never binds ------------------------------------------
L7="${WORK}/case7"; mkdir -p "${L7}"
out="$(run_up "${L7}" BENCH_WORKER_TEST_NEVER_BIND=1 RESIDENT_UP_HEALTH_TIMEOUT_S=3 --weights "${WEIGHTS}" --socket "${WORK}/case7.sock" -- echo should-not-run)"; rc=$?
if [[ "${rc}" -eq 1 && "${out}" == *"not healthy within 3s"* && "${out}" != *"should-not-run"* ]] && ! pgrep -f "case7.sock" >/dev/null; then
  pass "a resident that never binds fails on the health timeout and is torn down"
else
  fail "the health timeout did not fire or the resident survived (rc ${rc}): ${out}"
fi

# --- 8. a resident that ignores SIGTERM --------------------------------------
L8="${WORK}/case8"; mkdir -p "${L8}"
out="$(run_up "${L8}" BENCH_WORKER_TEST_IGNORE_TERM=1 RESIDENT_UP_HEALTH_TIMEOUT_S=30 --weights "${WEIGHTS}" --socket "${WORK}/case8.sock" -- true)"; rc=$?
sleep 0.5
if [[ "${rc}" -eq 0 && "${out}" == *"killing process group"* ]] && ! pgrep -f "case8.sock" >/dev/null; then
  pass "a resident that ignores SIGTERM is killed with its process group; nothing survives"
else
  fail "SIGTERM escalation failed (rc ${rc}): ${out}; survivors: $(pgrep -fl case8.sock | tr '\n' ' ')"
fi

# --- 8b. a hostile hello cannot grow memory or hold the GPU window forever ---
L8B="${WORK}/case8b"; mkdir -p "${L8B}"
started="${SECONDS}"
out="$(run_up "${L8B}" BENCH_WORKER_TEST_HELLO_MODE=overflow RESIDENT_UP_HEALTH_TIMEOUT_S=60 --weights "${WEIGHTS}" --socket "${WORK}/case8b.sock" -- echo should-not-run)"; rc=$?
elapsed=$((SECONDS - started))
if [[ "${rc}" -eq 2 && "${out}" == *"REFUSED (resident-hello-overflow)"* \
   && "${out}" == *"resident was stopped"* && "${out}" != *"should-not-run"* \
   && "${elapsed}" -lt 30 ]] && ! pgrep -f "case8b.sock" >/dev/null; then
  pass "an oversized hello is refused at 1 MiB and the resident is stopped"
else
  fail "an oversized hello was not bounded and stopped (rc ${rc}, ${elapsed}s): ${out}"
fi

L8C="${WORK}/case8c"; mkdir -p "${L8C}"
started="${SECONDS}"
out="$(run_up "${L8C}" BENCH_WORKER_TEST_HELLO_MODE=deadline RESIDENT_UP_HEALTH_TIMEOUT_S=60 --weights "${WEIGHTS}" --socket "${WORK}/case8c.sock" -- echo should-not-run)"; rc=$?
elapsed=$((SECONDS - started))
if [[ "${rc}" -eq 2 && "${out}" == *"REFUSED (resident-hello-deadline)"* \
   && "${out}" == *"resident was stopped"* && "${out}" != *"should-not-run"* \
   && "${elapsed}" -ge 25 && "${elapsed}" -le 40 ]] && ! pgrep -f "case8c.sock" >/dev/null; then
  pass "a hello stream without a newline is refused at the 30s whole-probe deadline and the resident is stopped"
else
  fail "a never-ending hello did not meet the deadline and stop rule (rc ${rc}, ${elapsed}s): ${out}"
fi

# --- 9. the resident environment is an allowlist ----------------------------
L9="${WORK}/case9"
mkdir -p "${L9}" "${WORK}/resident-env-home" "${WORK}/resident-env-tmp"
ENV9="${L9}/resident-env"
out="$(run_up "${L9}" \
  HOME="${WORK}/resident-env-home" TMPDIR="${WORK}/resident-env-tmp" LANG=C LC_BRIEF24=kept \
  BENCH_WORKER_TEST_ENV_FILE="${ENV9}" BENCH_WORKER_TEST_ALLOWED=kept MLX_TEST_ALLOWED=kept \
  DARKBLOOM_NEMOTRON35_MTP_CAPTURE_VERIFY=0 CBV2_STEP_PROFILE=1 BENCH_VERBOSE=1 \
  MLXFAST_OFFICIAL_BENCHMARK_RUN=secret MLXFAST_QWEN38_GOLDEN_DIR=/secret/goldens \
  MLXFAST_BASELINE_WORKSPACE=/secret/baseline MLXFAST_BASELINE_CALIBRATION=/secret/calibration \
  BENCHD_BIN_DIR=/secret/benchd BENCHD_TEST_SECRET=secret GITHUB_TEST_SECRET=secret RUNNER_TEST_SECRET=secret \
  RESIDENT_UP_TEST_SECRET=secret \
  --weights "${WEIGHTS}" --socket "${WORK}/case9.sock" -- true)"; rc=$?
if [[ "${rc}" -eq 0 ]] \
   && grep -qF "HOME=${WORK}/resident-env-home" "${ENV9}" \
   && grep -qF "TMPDIR=${WORK}/resident-env-tmp/" "${ENV9}" \
   && grep -qx 'LANG=C' "${ENV9}" \
   && grep -qx 'LC_BRIEF24=kept' "${ENV9}" \
   && grep -qx 'BENCH_WORKER_TEST_ALLOWED=kept' "${ENV9}" \
   && grep -qx 'MLX_TEST_ALLOWED=kept' "${ENV9}" \
   && grep -qx 'DARKBLOOM_NEMOTRON35_MTP_CAPTURE_VERIFY=0' "${ENV9}" \
   && grep -qx 'CBV2_STEP_PROFILE=1' "${ENV9}" \
   && grep -qx 'BENCH_VERBOSE=1' "${ENV9}"; then
  pass "the resident keeps process basics and bench-worker/model runtime controls"
else
  fail "the resident lost an allowed environment variable (rc ${rc}): ${out}; env=$(cat "${ENV9}" 2>/dev/null | tr '\n' ' ')"
fi
if grep -Eq '^(MLXFAST_|BENCHD_|GITHUB_|RUNNER_|RESIDENT_UP_)' "${ENV9}" 2>/dev/null \
   || ! awk -F= '$1 !~ /^(PATH|HOME|TMPDIR|LANG|LC_[A-Za-z0-9_]*|BENCH_WORKER_[A-Za-z0-9_]*|MLX_[A-Za-z0-9_]*|DARKBLOOM_[A-Za-z0-9_]*|CBV2_[A-Za-z0-9_]*|BENCH_VERBOSE|__CF_USER_TEXT_ENCODING)$/ { bad=1 } END { exit bad }' "${ENV9}"; then
  fail "the resident received a variable outside the allowlist: $(cat "${ENV9}" 2>/dev/null | tr '\n' ' ')"
else
  pass "ranked flags, evaluator paths and runner state are absent from the resident environment"
fi

release_lock

# --- 3. the lock must be held -------------------------------------------------
L3="${WORK}/case3"; mkdir -p "${L3}"
out="$(run_up "${L3}" --weights "${WEIGHTS}" --socket "${WORK}/case3.sock" -- echo should-not-run)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"REFUSED (lock-not-held)"* && "${out}" == *"Nothing has been loaded"* && ! -f "${L3}/stub-argv" ]]; then
  pass "nobody holding the GPU lock refuses by name before any boot"
else
  fail "an unheld lock did not refuse (rc ${rc}): ${out}"
fi
out="$(run_up "${L3}" RESIDENT_UP_SKIP_LOCK=1 RESIDENT_UP_LOCK_CHECK=0 --weights "${WEIGHTS}" --socket "${WORK}/case3.sock" -- echo should-not-run)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"REFUSED (lock-not-held)"* ]]; then
  pass "the lock refusal has no environment bypass"
else
  fail "an environment variable relaxed the lock refusal (rc ${rc}): ${out}"
fi
out="$(run_up "${L3}" RESIDENT_UP_LOCK_PATH="${WORK}/no-such-lock" --weights "${WEIGHTS}" --socket "${WORK}/case3.sock" -- echo should-not-run)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"REFUSED (lock-missing)"* ]]; then
  pass "a missing lock file refuses by name"
else
  fail "a missing lock file did not refuse (rc ${rc}): ${out}"
fi

# --- 4. the wired limit must be pinned ---------------------------------------
hold_lock
L4="${WORK}/case4"; mkdir -p "${L4}"
out="$(run_up "${L4}" RESIDENT_UP_WIRED_LIMIT_READER="${WIRED_ZERO}" --weights "${WEIGHTS}" --socket "${WORK}/case4.sock" -- echo should-not-run)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"REFUSED (wired-limit-unpinned)"* && ! -f "${L4}/stub-argv" ]]; then
  pass "an unpinned iogpu wired limit refuses by name before any boot"
else
  fail "an unpinned wired limit did not refuse (rc ${rc}): ${out}"
fi
release_lock

# --- 4b. the resident sandbox cannot be switched off on a self-hosted runner --
hold_lock
L4B="${WORK}/case4b"; mkdir -p "${L4B}"
out="$(run_up "${L4B}" RUNNER_ENVIRONMENT=self-hosted --weights "${WEIGHTS}" --socket "${WORK}/case4b.sock" -- echo should-not-run)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"REFUSED (sandbox-required)"* && ! -f "${L4B}/stub-argv" ]]; then
  pass "RESIDENT_UP_SANDBOX=0 on a self-hosted runner refuses by name before any boot"
else
  fail "RESIDENT_UP_SANDBOX=0 on a self-hosted runner did not refuse (rc ${rc}): ${out}"
fi
out="$(run_up "${L4B}" RESIDENT_UP_SANDBOX=maybe --weights "${WEIGHTS}" --socket "${WORK}/case4b.sock" -- echo should-not-run)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"RESIDENT_UP_SANDBOX must be 0 or 1"* && ! -f "${L4B}/stub-argv" ]]; then
  pass "a RESIDENT_UP_SANDBOX value other than 0 or 1 refuses by name"
else
  fail "a bad RESIDENT_UP_SANDBOX value did not refuse (rc ${rc}): ${out}"
fi
release_lock

# --- argv refusals -------------------------------------------------------------
out="$("${RESIDENT_UP}" --weights "${WEIGHTS}" 2>&1)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"no command after '--'"* ]]; then pass "a missing command refuses"; else fail "missing command (rc ${rc}): ${out}"; fi
out="$("${RESIDENT_UP}" -- true 2>&1)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"--weights <dir> is required"* ]]; then pass "a missing --weights refuses"; else fail "missing weights (rc ${rc}): ${out}"; fi

# --- 17. the per-job log directory must stay inside the stable root ---------
OUTSIDE_LOG="${WORK}/outside-log"
LOG_ROOT="${WORK}/stable-log-root"
mkdir -p "${OUTSIDE_LOG}" "${LOG_ROOT}"
out="$(run_up "${OUTSIDE_LOG}" MLXFAST_RESIDENT_LOG_ROOT="${LOG_ROOT}" --weights "${WEIGHTS}" --socket "${WORK}/outside.sock" -- true)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"REFUSED (log-dir-outside-root)"* \
   && "${out}" == *"outside MLXFAST_RESIDENT_LOG_ROOT"* ]]; then
  pass "a RESIDENT_UP_LOG_DIR outside MLXFAST_RESIDENT_LOG_ROOT refuses by name"
else
  fail "an outside resident log directory was accepted (rc ${rc}): ${out}"
fi


# ===========================================================================
# THE PER-LEG FORM: --boot / --stop.
#
# benchd calls these once per leg, in THAT LEG'S OWN TREE, so the cases below
# build a synthetic tree: a copy of the real script, the stub staged at the
# fixed path benchd resolves (.build/release/bench-worker), the tree's own
# weights/, and an mtp-head.manifest.json declaring a draft depth -- which
# --spec serial must ignore.
# ===========================================================================
hold_lock

TREE="${WORK}/leg-tree"
mkdir -p "${TREE}/tools" "${TREE}/.build/release" "${TREE}/weights"
cp "${RESIDENT_UP}" "${TREE}/tools/resident-up.sh"
chmod +x "${TREE}/tools/resident-up.sh"
# The stub is staged UNDER THE NAME benchd resolves, so the box-wide
# "bench-worker resident" scan sees it exactly as it sees a real one.
cp "${STUB}" "${TREE}/.build/release/bench-worker"
chmod +x "${TREE}/.build/release/bench-worker"
# A manifest that declares a SPECULATIVE tree. --spec serial must not read it.
cat > "${TREE}/mtp-head.manifest.json" <<'MANIFEST'
{
  "source": "pinned",
  "spec": {
    "enabled": true,
    "num_speculative_tokens": 3
  }
}
MANIFEST

TREE_UP="${TREE}/tools/resident-up.sh"
BOOT_LOG="${WORK}/boot-log"; mkdir -p "${BOOT_LOG}"

# run_leg [ENV=VAL...] -- ARGS... : drive the tree's own copy.
run_leg() {
  local envs=()
  while [[ $# -gt 0 && "$1" == *=* ]]; do envs+=("$1"); shift; done
  env -u MLXFAST_ENGINE_BIN \
    MLXFAST_RESIDENT_LOG_ROOT="${BOOT_LOG}" \
    RESIDENT_UP_LOG_DIR="${BOOT_LOG}" \
    RESIDENT_UP_LOCK_PATH="${LOCK}" \
    RESIDENT_UP_WIRED_LIMIT_READER="${WIRED_OK}" \
    RESIDENT_UP_HEALTH_TIMEOUT_S=30 \
    BENCH_WORKER_TEST_ARGV_FILE="${BOOT_LOG}/stub-argv" \
    ${envs[@]+"${envs[@]}"} \
    "${TREE_UP}" "$@" 2>&1
}

# --- B1. --boot reports the socket, writes the sidecars, and leaves it up ----
SOCKET_OUT="${WORK}/leg1.socket"
out="$(run_leg --boot --spec serial --draft-len 0 --socket-out "${SOCKET_OUT}")"; rc=$?
boot_socket="$(head -n 1 "${SOCKET_OUT}" 2>/dev/null || true)"
if [[ "${rc}" -eq 0 && -n "${boot_socket}" && -S "${boot_socket}" ]]; then
  pass "--boot exits 0 and writes the socket path as the FIRST line of --socket-out"
else
  fail "--boot did not report a live socket (rc ${rc}, socket '${boot_socket}'): ${out}"
fi
boot_pid="$(cat "${boot_socket}.pid" 2>/dev/null || true)"
if [[ -f "${boot_socket}.pid" && "${boot_pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${boot_pid}" 2>/dev/null; then
  pass "--boot writes a <socket>.pid sidecar naming a LIVE resident (it is still running after the boot exits)"
else
  fail "--boot left no live pid sidecar (pid '${boot_pid}'): ${out}"
fi
if [[ -f "${boot_socket}.ready" ]]; then
  pass "--boot writes the <socket>.ready marker"
else
  fail "--boot wrote no ready marker beside ${boot_socket}"
fi

# --- B2. --spec serial is authoritative over the tree's manifest -------------
# The tree declares depth 3. benchd asked for the SERIAL control leg. The
# manifest must not reach the boot at all.
if grep -q '^serial 0 ' "${boot_socket}.ready" \
   && python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d["spec"]=="serial" and d["draft_len"]=="0" else 1)' \
        "${BOOT_LOG}/resident-identity.json" 2>/dev/null; then
  pass "--spec serial is authoritative: the tree's mtp-head.manifest.json declares depth 3 and the boot records serial 0"
else
  fail "--spec serial did not win over the manifest: ready='$(cat "${boot_socket}.ready" 2>/dev/null)' identity=$(cat "${BOOT_LOG}/resident-identity.json" 2>/dev/null | tr '\n' ' ')"
fi

# --- B3. the boot ran THIS TREE'S worker, not an inherited MLXFAST_ENGINE_BIN
if grep -q "${TREE}/weights" "${BOOT_LOG}/stub-argv" 2>/dev/null; then
  pass "--boot loaded the tree's OWN weights/ (a leg is defined by its tree)"
else
  fail "--boot did not use the tree's own weights: $(tr '\n' ' ' < "${BOOT_LOG}/stub-argv" 2>/dev/null)"
fi

# --- B4. a second boot while one is up refuses, box-wide ---------------------
# The two legs live in two trees with two pidfiles, so only a box-wide scan
# catches leg 2 booting before leg 1 was stopped -- which would double-load.
out="$(run_leg --boot --spec mtp --draft-len 2 --socket-out "${WORK}/leg2.socket")"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"REFUSED (resident-already-up)"* ]]; then
  pass "a second --boot while a resident is up refuses by name (box-wide, across trees)"
else
  fail "a second --boot did not refuse (rc ${rc}): ${out}"
fi

# --- B5. --stop ends the resident and removes all three files ----------------
out="$(run_leg --stop --socket "${boot_socket}")"; rc=$?
sleep 0.5
if [[ "${rc}" -eq 0 ]] && ! kill -0 "${boot_pid}" 2>/dev/null \
   && [[ ! -e "${boot_socket}" && ! -e "${boot_socket}.pid" && ! -e "${boot_socket}.ready" ]]; then
  pass "--stop ends the resident and removes the socket, the pid sidecar and the ready marker"
else
  fail "--stop left something behind (rc ${rc}): $(ls "${boot_socket}"* 2>&1 | tr '\n' ' ') ${out}"
fi

# --- B6. --stop is idempotent ------------------------------------------------
out="$(run_leg --stop --socket "${boot_socket}")"; rc=$?
if [[ "${rc}" -eq 0 && "${out}" == *"nothing to stop"* ]]; then
  pass "a second --stop is a no-op: exit 0, and it says there was nothing to stop"
else
  fail "a second --stop was not a no-op (rc ${rc}): ${out}"
fi

# --- B7. --stop repairs modes and flags without following outside links ------
PROTECTED_TMP="${WORK}/tmp.cleanup-protected"
PROTECTED_SOCKET="${WORK}/cleanup-protected.sock"
OUTSIDE_FILE="${WORK}/outside-file"
OUTSIDE_DIR="${WORK}/outside-dir"
mkdir -p "${PROTECTED_TMP}/mode-000/mode-000" "${OUTSIDE_DIR}"
INSIDE_FLAG_FILE="${PROTECTED_TMP}/mode-000/mode-000/file"
: > "${INSIDE_FLAG_FILE}"
printf 'outside\n' > "${OUTSIDE_FILE}"
printf 'outside child\n' > "${OUTSIDE_DIR}/child"
chmod 640 "${OUTSIDE_FILE}"
chmod 750 "${OUTSIDE_DIR}"
ln -s "${OUTSIDE_FILE}" "${PROTECTED_TMP}/outside-file-link"
ln -s "${OUTSIDE_DIR}" "${PROTECTED_TMP}/outside-dir-link"
# A hard link shares the outside file's inode: cleanup must not change its mode.
OUTSIDE_HARD="${WORK}/outside-hard"
printf 'outside hard\n' > "${OUTSIDE_HARD}"
chmod 640 "${OUTSIDE_HARD}"
ln "${OUTSIDE_HARD}" "${PROTECTED_TMP}/mode-000/outside-hard-link"
outside_flags_before=""
outside_acl_before=""
if [[ "$(uname -s)" == "Darwin" ]]; then
  chmod +a "everyone allow read" "${OUTSIDE_FILE}" || fail "could not set the outside ACL sentinel"
  chflags uchg "${OUTSIDE_FILE}" || fail "could not set the outside uchg sentinel"
  chflags uchg "${INSIDE_FLAG_FILE}" || fail "could not set the inside uchg test flag"
  outside_flags_before="$(stat -f '%Sf' "${OUTSIDE_FILE}")"
  outside_acl_before="$(ls -le "${OUTSIDE_FILE}" | tail -n +2)"
fi
outside_file_mode_before="$(python3 -c 'import os,sys; print(format(os.lstat(sys.argv[1]).st_mode & 0o777, "o"))' "${OUTSIDE_FILE}")"
outside_dir_mode_before="$(python3 -c 'import os,sys; print(format(os.lstat(sys.argv[1]).st_mode & 0o777, "o"))' "${OUTSIDE_DIR}")"
chmod 000 "${PROTECTED_TMP}/mode-000/mode-000"
chmod 000 "${PROTECTED_TMP}/mode-000"
printf '%s\n' "${PROTECTED_TMP}" > "${PROTECTED_SOCKET}.tmpdir"
out="$(run_leg --stop --socket "${PROTECTED_SOCKET}")"; rc=$?
outside_file_mode_after="$(python3 -c 'import os,sys; print(format(os.lstat(sys.argv[1]).st_mode & 0o777, "o"))' "${OUTSIDE_FILE}")"
outside_dir_mode_after="$(python3 -c 'import os,sys; print(format(os.lstat(sys.argv[1]).st_mode & 0o777, "o"))' "${OUTSIDE_DIR}")"
outside_hard_mode_after="$(python3 -c 'import os,sys; print(format(os.lstat(sys.argv[1]).st_mode & 0o777, "o"))' "${OUTSIDE_HARD}" 2>/dev/null || echo missing)"
outside_darwin_ok=1
if [[ "$(uname -s)" == "Darwin" ]]; then
  [[ "$(stat -f '%Sf' "${OUTSIDE_FILE}")" == "${outside_flags_before}" ]] || outside_darwin_ok=0
  [[ "$(ls -le "${OUTSIDE_FILE}" | tail -n +2)" == "${outside_acl_before}" ]] || outside_darwin_ok=0
fi
if [[ "${rc}" -eq 0 && ! -e "${PROTECTED_TMP}" && ! -e "${PROTECTED_SOCKET}.tmpdir" \
   && -f "${OUTSIDE_FILE}" && -f "${OUTSIDE_DIR}/child" \
   && "${outside_file_mode_after}" == "${outside_file_mode_before}" \
   && "${outside_dir_mode_after}" == "${outside_dir_mode_before}" \
   && "${outside_hard_mode_after}" == "640" \
   && "${outside_darwin_ok}" -eq 1 ]]; then
  pass "--stop removes nested mode-000 content and an inside uchg file without touching outside symlink or hard-link targets"
else
  fail "--stop cleanup was unsafe or incomplete (rc ${rc}): ${out}"
fi
if [[ "$(uname -s)" == "Darwin" ]]; then
  chflags nouchg "${OUTSIDE_FILE}" 2>/dev/null || true
  chmod -N "${OUTSIDE_FILE}" 2>/dev/null || true
fi
/bin/rm -rf "${PROTECTED_TMP}" "${PROTECTED_SOCKET}.tmpdir" "${OUTSIDE_FILE}" "${OUTSIDE_DIR}"

# --- B7b. a failed private-temp removal makes --stop fail -------------------
FAIL_TMP="${WORK}/tmp.cleanup-fail"
FAIL_SOCKET="${WORK}/cleanup-fail.sock"
FAIL_RM_BIN="${WORK}/fail-rm-bin"
mkdir -p "${FAIL_TMP}" "${FAIL_RM_BIN}"
printf '%s\n' "${FAIL_TMP}" > "${FAIL_SOCKET}.tmpdir"
cat > "${FAIL_RM_BIN}/rm" <<'SH'
#!/bin/sh
if [ "$1" = "-rf" ]; then
  exit 73
fi
exec /bin/rm "$@"
SH
chmod +x "${FAIL_RM_BIN}/rm"
out="$(run_leg PATH="${FAIL_RM_BIN}:${PATH}" --stop --socket "${FAIL_SOCKET}")"; rc=$?
if [[ "${rc}" -eq 1 && "${out}" == *"temporary directory cleanup failed"* \
   && -d "${FAIL_TMP}" && -f "${FAIL_SOCKET}.tmpdir" ]]; then
  pass "--stop fails and keeps the tmpdir sidecar when private-temp removal fails"
else
  fail "--stop ignored a private-temp removal failure (rc ${rc}): ${out}"
fi
/bin/rm -rf "${FAIL_TMP}" "${FAIL_SOCKET}.tmpdir" "${FAIL_RM_BIN}"

# --- B8. a boot after a stop works: the box-wide scan sees a clean box -------
SOCKET_OUT2="${WORK}/leg2.socket"
out="$(run_leg --boot --spec mtp --draft-len 4 --socket-out "${SOCKET_OUT2}")"; rc=$?
boot2_socket="$(head -n 1 "${SOCKET_OUT2}" 2>/dev/null || true)"
if [[ "${rc}" -eq 0 && -S "${boot2_socket}" ]] && grep -q '^mtp 4 ' "${boot2_socket}.ready"; then
  pass "the next leg boots after a stop, and records its own spec (mtp 4)"
else
  fail "the second leg did not boot (rc ${rc}): ${out}"
fi
run_leg --stop --socket "${boot2_socket}" >/dev/null 2>&1

# --- B8b. a boot that fails after the resident started leaves nothing up ----
# Two failure classes: an explicit die (the identity file cannot be written) and
# a plain set -e failure (the --socket-out directory cannot be made). Both exit
# through the --boot EXIT trap, which must stop the resident and remove the
# private hello file.
rm -f "${BOOT_LOG}/resident-identity.json"
mkdir -p "${BOOT_LOG}/resident-identity.json"
out="$(run_leg --boot --spec serial --draft-len 0 --socket-out "${WORK}/leg-fail1.socket")"; rc=$?
sleep 0.5
if [[ "${rc}" -ne 0 && "${out}" == *"could not write the resident identity file"* ]] \
   && ! pgrep -f "${TREE}/.build/release/bench-worker" >/dev/null \
   && ! compgen -G "${BOOT_LOG}/resident-hello.*" >/dev/null; then
  pass "a boot whose identity write fails stops the resident and removes the hello file"
else
  fail "a failed identity write left a resident or hello file (rc ${rc}): ${out}; $(pgrep -fl "${TREE}/.build/release/bench-worker" | tr '\n' ' ')"
fi
rmdir "${BOOT_LOG}/resident-identity.json"
pkill -f "${TREE}/.build/release/bench-worker" 2>/dev/null || true

: > "${WORK}/not-a-dir"
out="$(run_leg --boot --spec serial --draft-len 0 --socket-out "${WORK}/not-a-dir/leg.socket")"; rc=$?
sleep 0.5
if [[ "${rc}" -ne 0 ]] && ! pgrep -f "${TREE}/.build/release/bench-worker" >/dev/null \
   && ! compgen -G "${BOOT_LOG}/resident-hello.*" >/dev/null; then
  pass "a boot that fails under set -e after the resident started stops the resident"
else
  fail "a set -e boot failure left a resident or hello file (rc ${rc}): ${out}; $(pgrep -fl "${TREE}/.build/release/bench-worker" | tr '\n' ' ')"
fi
pkill -f "${TREE}/.build/release/bench-worker" 2>/dev/null || true
/bin/rm -f "${WORK}/not-a-dir"

# --- B9. a flag without a value refuses BY NAME ------------------------------
for flag in --spec --draft-len --socket-out --socket; do
  out="$("${TREE_UP}" --boot "${flag}" 2>&1)"; rc=$?
  if [[ "${rc}" -eq 2 && "${out}" == *"REFUSED (bad-argument)"* && "${out}" == *"${flag} needs"* ]]; then
    pass "${flag} without a value refuses by name"
  else
    fail "${flag} without a value did not refuse by name (rc ${rc}): ${out}"
  fi
done

# --- B10. --boot needs all three, and each refusal names the missing one -----
out="$("${TREE_UP}" --boot --draft-len 0 --socket-out "${WORK}/x" 2>&1)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"--boot needs --spec"* ]]; then pass "--boot without --spec refuses by name"; else fail "--boot without --spec (rc ${rc}): ${out}"; fi
out="$("${TREE_UP}" --boot --spec serial --socket-out "${WORK}/x" 2>&1)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"--boot needs --draft-len"* ]]; then pass "--boot without --draft-len refuses by name"; else fail "--boot without --draft-len (rc ${rc}): ${out}"; fi
out="$("${TREE_UP}" --boot --spec serial --draft-len 0 2>&1)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"--boot needs --socket-out"* ]]; then pass "--boot without --socket-out refuses by name"; else fail "--boot without --socket-out (rc ${rc}): ${out}"; fi
out="$("${TREE_UP}" --stop 2>&1)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"--stop needs --socket"* ]]; then pass "--stop without --socket refuses by name"; else fail "--stop without --socket (rc ${rc}): ${out}"; fi

# --- B11. spec and draft length must agree ----------------------------------
out="$("${TREE_UP}" --boot --spec serial --draft-len 3 --socket-out "${WORK}/x" 2>&1)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"serial requires --draft-len 0"* ]]; then
  pass "--spec serial with a draft length refuses: a serial control leg drafts nothing"
else
  fail "serial + draft-len 3 did not refuse (rc ${rc}): ${out}"
fi
out="$("${TREE_UP}" --boot --spec mtp --draft-len 0 --socket-out "${WORK}/x" 2>&1)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"mtp requires --draft-len between 1 and 7"* ]]; then
  pass "--spec mtp with draft length 0 refuses"
else
  fail "mtp + draft-len 0 did not refuse (rc ${rc}): ${out}"
fi
out="$("${TREE_UP}" --boot --spec sideways --draft-len 1 --socket-out "${WORK}/x" 2>&1)"; rc=$?
if [[ "${rc}" -eq 2 && "${out}" == *"--spec must be 'serial' or 'mtp'"* ]]; then
  pass "an unknown --spec value refuses by name"
else
  fail "an unknown --spec did not refuse (rc ${rc}): ${out}"
fi

# --- B12. sandbox TMPDIR is outside logs and does not outlive the resident ---
if [[ -x /usr/bin/sandbox-exec ]]; then
  SBT="${WORK}/sandbox-tree"
  SB_TMP_ROOT="${WORK}/sandbox-private"
  mkdir -p "${SBT}/tools" "${SBT}/.build/release" "${SBT}/weights" "${SB_TMP_ROOT}"
  cp "${RESIDENT_UP}" "${ROOT_DIR}/tools/seatbelt-profile.py" "${SBT}/tools/"
  chmod +x "${SBT}/tools/resident-up.sh"
  # A framework build's bin/python3 starts Python.app with posix_spawn, which
  # the profile denies, so the link names Python.app's binary when it exists.
  ln -s "$(python3 -c 'import os, sys; app = os.path.join(sys.base_prefix, "Resources/Python.app/Contents/MacOS/Python"); print(app if os.path.exists(app) else os.path.realpath(sys.executable))')" "${SBT}/.build/release/bench-worker"
  cat > "${SBT}/resident" <<'PY'
import json, os, signal, socket, sys
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
if os.environ.get("BENCH_WORKER_TEST_DIE_BEFORE_BIND") == "1":
    sys.exit(3)
path = sys.argv[sys.argv.index("--socket") + 1]
srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
srv.bind(path)
srv.listen(4)
hello = {"id": 0, "ok": True, "protocol_version": 1, "backend": "mlx-resident",
         "device": "stub", "spec_modes": ["serial", "mtp"], "runner": {"id": "layr/stub"},
         "resident": {"pid": os.getpid(), "load_epoch": 1}}
while True:
    conn, _ = srv.accept()
    try:
        conn.sendall((json.dumps(hello) + "\n").encode())
        while conn.recv(65536):
            pass
    except OSError:
        pass
    finally:
        conn.close()
PY
  run_sandboxed() {
    env -u MLXFAST_ENGINE_BIN -u RESIDENT_UP_LOG_DIR -u MLXFAST_RESIDENT_LOG_ROOT \
      RESIDENT_UP_SANDBOX=1 \
      RUNNER_TEMP="${SB_TMP_ROOT}" \
      RESIDENT_UP_LOCK_PATH="${LOCK}" \
      RESIDENT_UP_WIRED_LIMIT_READER="${WIRED_OK}" \
      RESIDENT_UP_HEALTH_TIMEOUT_S=30 \
      "$@" 2>&1
  }
  sb_tmp_dirs() { find "${SB_TMP_ROOT}" -maxdepth 1 -name 'tmp.*' 2>/dev/null; }
  STALE_BOOT_TMP="${SB_TMP_ROOT}/tmp.resident-up.stale"
  mkdir -p "${STALE_BOOT_TMP}"
  out="$(run_sandboxed "${SBT}/tools/resident-up.sh" --boot --spec serial --draft-len 0 --socket-out "${WORK}/sb.socket")"; rc=$?
  sb_socket="$(head -n 1 "${WORK}/sb.socket" 2>/dev/null || true)"
  sb_tmp="$(head -n 1 "${sb_socket}.tmpdir" 2>/dev/null || true)"
  if [[ "${rc}" -eq 0 && -S "${sb_socket}" && -d "${sb_tmp}" \
     && "${sb_tmp}" == "${SB_TMP_ROOT}/tmp."* \
     && "${sb_tmp}" != "${SBT}/.build/resident/"* \
     && "$(python3 -c 'import os,sys;print(format(os.stat(sys.argv[1]).st_mode & 0o777, "o"))' "${sb_tmp}" 2>/dev/null)" == "700" ]]; then
    pass "--boot uses a mode-0700 tmp.* directory outside RESIDENT_UP_LOG_DIR"
  else
    fail "--boot under sandbox-exec did not use a private sibling TMPDIR (rc ${rc}, socket '${sb_socket}', tmp '${sb_tmp}'): ${out}"
  fi
  if [[ ! -e "${STALE_BOOT_TMP}" ]]; then
    pass "--boot removes a stale tmp.resident-up.* tree before creating its private temp tree"
  else
    fail "--boot left the stale resident temp tree ${STALE_BOOT_TMP}"
  fi
  out="$(run_sandboxed "${SBT}/tools/resident-up.sh" --stop --socket "${sb_socket}")"; rc=$?
  if [[ "${rc}" -eq 0 && -z "$(sb_tmp_dirs)" && ! -e "${sb_socket}.tmpdir" ]]; then
    pass "--stop removes the sandboxed resident's tmp.* directory and its sidecar"
  else
    fail "--stop left a tmp.* directory behind (rc ${rc}): $(sb_tmp_dirs | tr '\n' ' ') ${out}"
  fi
  STALE_FAIL_TMP="${SB_TMP_ROOT}/tmp.resident-up.cleanup-fail"
  FAIL_STALE_RM_BIN="${WORK}/fail-stale-rm-bin"
  mkdir -p "${STALE_FAIL_TMP}" "${FAIL_STALE_RM_BIN}"
  cat > "${FAIL_STALE_RM_BIN}/rm" <<'SH'
#!/bin/sh
if [ "$1" = "-rf" ]; then
  exit 73
fi
exec /bin/rm "$@"
SH
  chmod +x "${FAIL_STALE_RM_BIN}/rm"
  out="$(run_sandboxed PATH="${FAIL_STALE_RM_BIN}:${PATH}" "${SBT}/tools/resident-up.sh" --boot --spec serial --draft-len 0 --socket-out "${WORK}/sb-stale-fail.socket")"; rc=$?
  if [[ "${rc}" -eq 2 && "${out}" == *"REFUSED (stale-resident-tmp)"* \
     && -d "${STALE_FAIL_TMP}" && ! -e "${WORK}/sb-stale-fail.socket" ]]; then
    pass "--boot refuses by name when a stale resident temp tree cannot be removed"
  else
    fail "--boot did not refuse a stale resident temp tree that survived cleanup (rc ${rc}): ${out}"
  fi
  /bin/rm -rf "${STALE_FAIL_TMP}" "${FAIL_STALE_RM_BIN}"
  out="$(run_sandboxed BENCH_WORKER_TEST_DIE_BEFORE_BIND=1 "${SBT}/tools/resident-up.sh" --boot --spec serial --draft-len 0 --socket-out "${WORK}/sb2.socket")"; rc=$?
  if [[ "${rc}" -eq 1 && -z "$(sb_tmp_dirs)" ]]; then
    pass "a sandboxed boot that fails leaves no tmp.* directory behind"
  else
    fail "a failed sandboxed boot left a tmp.* directory behind (rc ${rc}): $(sb_tmp_dirs | tr '\n' ' ') ${out}"
  fi

  WRAP_SOCKET="${WORK}/wrapper-cleanup.sock"
  FAIL_WRAP_RM_BIN="${WORK}/fail-wrap-rm-bin"
  mkdir -p "${FAIL_WRAP_RM_BIN}"
  cat > "${FAIL_WRAP_RM_BIN}/rm" <<'SH'
#!/bin/sh
if [ "$1" = "-rf" ]; then
  exit 73
fi
exec /bin/rm "$@"
SH
  chmod +x "${FAIL_WRAP_RM_BIN}/rm"
  out="$(run_sandboxed PATH="${FAIL_WRAP_RM_BIN}:${PATH}" "${SBT}/tools/resident-up.sh" --weights "${SBT}/weights" --socket "${WRAP_SOCKET}" -- true)"; rc=$?
  wrap_tmp="$(head -n 1 "${WRAP_SOCKET}.tmpdir" 2>/dev/null || true)"
  if [[ "${rc}" -eq 1 && "${out}" == *"temporary directory"* \
     && -n "${wrap_tmp}" && -d "${wrap_tmp}" && -f "${WRAP_SOCKET}.tmpdir" ]]; then
    pass "a failed wrapper cleanup returns non-zero and keeps the tmpdir sidecar for retry"
  else
    fail "the wrapper ignored a private-temp cleanup failure (rc ${rc}, tmp '${wrap_tmp}'): ${out}"
  fi
  /bin/rm -rf "${wrap_tmp}" "${WRAP_SOCKET}.tmpdir" "${FAIL_WRAP_RM_BIN}"
else
  pass "sandbox TMPDIR case skipped: no /usr/bin/sandbox-exec on this host"
fi
release_lock

if [[ "${fails}" -eq 0 ]]; then
  printf 'test-resident-up: all cases pass\n'
  exit 0
fi
printf 'test-resident-up: %s case(s) failed\n' "${fails}" >&2
exit 1
