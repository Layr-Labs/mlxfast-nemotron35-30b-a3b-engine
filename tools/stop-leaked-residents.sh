#!/usr/bin/env bash
# Stop resident bench-workers that a cancelled ranked job left behind.
#
# On 2026-10-23, a ranked job lost its runner connection during Benchmark.
# Its resident bench-worker lived for more than 50 minutes after cancellation.
# The next job then refused to boot because it found that old resident.
#
# This script takes the same box-wide GPU lock as the measurement script.
# Holding that lock proves that no live measurement owns a resident.
# One Python process holds the lock, checks process identity, and sends signals.
# Only current-user residents rooted in an approved job tree can be stopped.
# Identity checks narrow, but cannot remove, a PID-reuse race on macOS.
# No command line or private path is printed.
set -euo pipefail

usage() {
  echo "usage: tools/stop-leaked-residents.sh --mode preflight|cleanup [--root PATH ...] [--stop-wait SECONDS]" >&2
}

bad_args() {
  echo "::error::$1" >&2
  usage
  exit 2
}

MODE=""
STOP_WAIT=10
ROOT_COUNT=0
ROOTS=("")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      [[ $# -ge 2 ]] || bad_args "--mode needs a value"
      [[ -z "${MODE}" ]] || bad_args "--mode was given more than once"
      MODE="$2"
      shift 2
      ;;
    --root)
      [[ $# -ge 2 ]] || bad_args "--root needs a value"
      [[ -n "$2" ]] || bad_args "--root must not be empty"
      ROOTS[ROOT_COUNT]="$2"
      ROOT_COUNT=$((ROOT_COUNT + 1))
      shift 2
      ;;
    --stop-wait)
      [[ $# -ge 2 ]] || bad_args "--stop-wait needs a value"
      STOP_WAIT="$2"
      shift 2
      ;;
    *)
      bad_args "unknown argument"
      ;;
  esac
done

case "${MODE}" in
  preflight|cleanup) : ;;
  "") bad_args "--mode is required" ;;
  *) bad_args "--mode must be preflight or cleanup" ;;
esac
[[ "${STOP_WAIT}" =~ ^[0-9]+$ ]] || bad_args "--stop-wait must be a non-negative integer"
command -v python3 >/dev/null 2>&1 || bad_args "python3 is required to take the GPU lock"

LOCK_PATH="${RESIDENT_UP_LOCK_PATH:-/tmp/mtplx-gpu-exclusive.lock}"
exec python3 - "${MODE}" "${LOCK_PATH}" "${STOP_WAIT}" "${ROOT_COUNT}" "${ROOTS[@]}" <<'PY'
import errno
import fcntl
import glob
import os
import signal
import subprocess
import sys
import time


mode = sys.argv[1]
lock_path = sys.argv[2]
stop_wait = int(sys.argv[3])
root_count = int(sys.argv[4])
current_uid = os.getuid()


def real_roots():
    if root_count:
        paths = sys.argv[5:]
    else:
        paths = []
        home = os.environ.get("HOME", "")
        if home:
            paths.extend(glob.glob(os.path.join(home, "actions-runner*", "_work")))
        baseline = os.environ.get("MLXFAST_BASELINE_WORKSPACE", "")
        if baseline and os.path.isdir(baseline):
            paths.append(baseline)

    roots = []
    for path in paths:
        if root_count or os.path.isdir(path):
            root = os.path.realpath(path)
            if root not in roots:
                roots.append(root)
    return roots


roots = real_roots()
if not roots:
    print("stop-leaked-residents: no job roots on this host; nothing to check")
    raise SystemExit(0)


def lock_busy():
    if mode == "preflight":
        sys.stderr.write(
            "::warning::the GPU lock is busy (a measurement is running on this box); leftover check skipped\n"
        )
        raise SystemExit(0)
    sys.stderr.write(
        "::warning::the GPU lock is busy; cleanup deferred to the next job's preflight\n"
    )
    raise SystemExit(3)


def lock_error(exc):
    message = os.strerror(exc.errno) if exc.errno else "unknown error"
    if mode == "preflight":
        sys.stderr.write(
            "::error::cannot open the GPU lock (%s); refusing\n" % message
        )
        raise SystemExit(1)
    sys.stderr.write(
        "::warning::cannot open the GPU lock (%s); cleanup deferred to the next job's preflight\n"
        % message
    )
    raise SystemExit(3)


try:
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
except OSError as exc:
    lock_error(exc)

try:
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError as exc:
    if exc.errno in (errno.EACCES, errno.EAGAIN):
        lock_busy()
    lock_error(exc)


def parse_process_line(line):
    fields = line.strip().split(None, 9)
    if len(fields) != 10:
        return None
    try:
        pid = int(fields[0])
        pgid = int(fields[1])
        uid = int(fields[2])
    except ValueError:
        return None
    return {
        "pid": pid,
        "pgid": pgid,
        "uid": uid,
        "stat": fields[3],
        "lstart": " ".join(fields[4:9]),
        "args": fields[9],
    }


def run_ps(arguments):
    try:
        process = subprocess.Popen(
            ["ps"] + arguments,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
        )
        output, _ = process.communicate()
    except OSError:
        return None
    if process.returncode != 0:
        return None
    return output


def all_processes():
    output = run_ps(
        ["-A", "-ww", "-o", "pid=,pgid=,uid=,stat=,lstart=,args="]
    )
    if output is None:
        sys.stderr.write("::error::could not scan current-user resident processes\n")
        raise SystemExit(1)
    processes = []
    for line in output.splitlines():
        process = parse_process_line(line)
        if process is not None:
            processes.append(process)
    return processes


def read_process(pid):
    output = run_ps(
        ["-ww", "-p", str(pid), "-o", "pid=,pgid=,uid=,stat=,lstart=,args="]
    )
    if not output:
        return None
    for line in output.splitlines():
        process = parse_process_line(line)
        if process is not None and process["pid"] == pid:
            return process
    return None


def is_dead(process):
    return process is None or process["stat"].startswith("Z")


def identity(process):
    return (process["uid"], process["lstart"], process["args"])


def same_live_process(recorded):
    current = read_process(recorded["pid"])
    if is_dead(current):
        return None
    if identity(current) != identity(recorded):
        return None
    return current


def in_job_root(argv0):
    if not os.path.isabs(argv0) or os.path.basename(argv0) != "bench-worker":
        return False
    path = os.path.realpath(argv0)
    for root in roots:
        try:
            if os.path.commonpath([path, root]) == root:
                return True
        except ValueError:
            pass
    return False


def is_rooted_resident(process):
    if process["uid"] != current_uid or is_dead(process):
        return False
    argv = process["args"].split()
    return len(argv) >= 2 and argv[1] == "resident" and in_job_root(argv[0])


def scan_residents(warn_lookalikes):
    targets = []
    lookalikes = 0
    for process in all_processes():
        if process["uid"] != current_uid or is_dead(process):
            continue
        if is_rooted_resident(process):
            targets.append(process)
        elif "bench-worker resident" in process["args"]:
            lookalikes += 1
    if warn_lookalikes and lookalikes:
        sys.stderr.write(
            "::warning::found %d other process(es) that look like a resident; not stopped\n"
            % lookalikes
        )
    return targets


def elapsed(pid):
    output = run_ps(["-p", str(pid), "-o", "etime="])
    value = output.strip() if output else ""
    return value or "unknown"


def signal_if_same(recorded, wanted_signal):
    if same_live_process(recorded) is None:
        return
    try:
        os.kill(recorded["pid"], wanted_signal)
    except ProcessLookupError:
        pass
    except OSError:
        pass


def live_records(records):
    return [record for record in records if same_live_process(record) is not None]


def wait_for_records(records, seconds):
    deadline = time.monotonic() + seconds
    while live_records(records):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)
    return True


targets = scan_residents(True)
if not targets:
    print("stop-leaked-residents: no leftover resident")
    raise SystemExit(0)

stopped = 0
for target in targets:
    current_target = same_live_process(target)
    if current_target is None:
        continue

    members = [current_target]
    if current_target["pgid"] == current_target["pid"]:
        snapshot = all_processes()
        snapshot_target = next(
            (
                process
                for process in snapshot
                if process["pid"] == current_target["pid"]
                and identity(process) == identity(current_target)
                and not is_dead(process)
            ),
            None,
        )
        if snapshot_target is None:
            continue
        members = [
            member
            for member in snapshot
            if member["pgid"] == snapshot_target["pgid"]
            and member["uid"] == current_uid
            and not is_dead(member)
        ]

    print(
        "::warning::stopping resident bench-worker left by an earlier job "
        "(pid %d, elapsed %s)" % (target["pid"], elapsed(target["pid"]))
    )
    for member in members:
        signal_if_same(member, signal.SIGTERM)

    if not wait_for_records(members, stop_wait):
        remaining = live_records(members)
        for member in remaining:
            signal_if_same(member, signal.SIGKILL)
        if not wait_for_records(remaining, 5):
            sys.stderr.write(
                "::error::resident pid %d survived SIGKILL; the box needs an operator\n"
                % target["pid"]
            )
            raise SystemExit(1)
    stopped += 1

remaining_targets = scan_residents(False)
if remaining_targets:
    sys.stderr.write(
        "::error::resident bench-worker match remains after cleanup (pid %d)\n"
        % remaining_targets[0]["pid"]
    )
    raise SystemExit(1)

print("stop-leaked-residents: stopped %d leftover resident(s)" % stopped)
PY
