#!/usr/bin/env bash
#
# test-benchmark-facade-env.sh -- the benchd facade exports the track id.
#
# `benchd iterate` REQUIRES MLXFAST_QWEN_MTP_TRACK_ID in every mode (its
# `-{platform}-v{N}` suffix keys the official baseline pair; the official path
# resolves it before the gates-only branch), and this repository once exported
# it nowhere, so `./benchmark.sh --local-iterate` and `--official` both refused
# with "no track_id". tools/benchmark.sh now reads benchmark.json's `trackId`
# and exports it before dispatch. This suite pins that, by driving the REAL
# facade with a STUB benchd that records the environment and argv it was
# spawned with. Hermetic: no toolchain, weights, GPU, benchd binary or network.
#
# Cases:
#   1. --local-iterate: the stub sees MLXFAST_QWEN_MTP_TRACK_ID == benchmark.json
#      trackId, non-empty, on an `iterate --mode local-iterate` argv. Its
#      temporary contract omits paired-only guards and is removed on exit.
#   2. --official (gates-only env): the same export on `--mode official`.
#      It receives the full fixture unchanged.
#   3. a manifest with NO trackId: the facade refuses (exit 1, names the field)
#      and benchd is never spawned.
#   4. a caller pre-sets a DIFFERENT MLXFAST_QWEN_MTP_TRACK_ID: refused, never
#      overridden, benchd never spawned.
#   5. a caller pre-sets the SAME value: accepted.
#
# It then pins the RANKED path's WINDOW wiring on
# tools/nemotron35-30b-a3b-measure-and-score.sh, with a stub bench-worker that
# binds a socket and answers a hello the way the real resident does:
#   6. THE RANKED PATH BOOTS NO RESIDENT AND EXPORTS NO SOCKET. It used to boot
#      one from the CANDIDATE tree and hand benchd the socket; under the paired
#      design the REFERENCE leg then attached to the candidate's resident and
#      was refused (public run 34230122059). benchd boots one resident PER LEG
#      now, from that leg's own tree, so the measure script must hand it
#      nothing -- and it must still take the GPU lock, because the box needs one
#      loader whoever boots it.
#   7. an INHERITED BENCH_WORKER_RESIDENT_SOCKET is refused by name, and benchd
#      is never dispatched: a socket arriving through the runner service
#      environment would reproduce the same failure.
#   8. --preflight-only boots no resident and takes no lock: a pre-GPU dry run
#      loads nothing, so it must not open a GPU window.
#
# Cases 9-13 pin the PAIRED plumbing (David ruling 2026-09-08). A ranked run
# measures a serial-control leg on the organizer's reference tree and a
# candidate leg on this checkout, and the score is the LIVE ratio:
#   9.  the dispatch carries --baseline-workspace and --baseline-calibration
#       from MLXFAST_BASELINE_WORKSPACE / MLXFAST_BASELINE_CALIBRATION, and
#       --engine is RELATIVE to the checkout root, because benchd re-roots that
#       path under the reference workspace to find the control leg's worker.
#   10. MLXFAST_BASELINE_WORKSPACE unset: refused by name, benchd never spawned.
#   11. MLXFAST_BASELINE_CALIBRATION unset: refused by name, benchd never spawned.
#   12. --preflight-only does NOT require either: it is benchmark.json's
#       preSubmitCommand, which a participant runs off the box.
#   13. an engine OUTSIDE the checkout is refused: it has no relative form, so
#       the control leg would run the candidate's own binary.
#   14. a WEIGHTS directory outside the checkout is refused for the same reason:
#       both legs would read the candidate workspace's weights.
#   15. --box is passed ONLY when RUNNER_NAME is absent. benchd lets RUNNER_NAME
#       win over the flag, so on a runner the flag is dead argv; on a hand run
#       it is the only box identity benchd has, and it comes from the
#       calibration file's own `box`.
#
# Cases 16-19 pin the live_goldens list. A ranked run measures each live
# golden, and benchd matches the golden flags by position:
#   16. every live golden reaches benchd as --golden, --golden-sha256,
#       --golden-bytes, --control-golden, --control-golden-sha256 and
#       --control-golden-bytes, one of each per golden, in fixture order.
#   17. an empty live_goldens list is refused by name.
#   18. a live golden with no timed_prompt_pool entry is refused by name.
#   19. a declared depth with no oracle for one live golden is refused by name.
#   20. with RESIDENT_UP_LOG_DIR unset, benchd gets a resident log directory
#       in the job's temporary directory, outside both trees, because the
#       ranked job account cannot write the reference workspace.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FACADE="${REPO_ROOT}/tools/benchmark.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

failures=0
fail() {
  echo "FAIL: $*" >&2
  failures=$((failures + 1))
}

command -v jq >/dev/null 2>&1 || { echo "test-benchmark-facade-env.sh: jq is required" >&2; exit 1; }

EXPECTED="$(jq -r '.trackId // empty' "${REPO_ROOT}/benchmark.json")"
if [[ -z "${EXPECTED}" ]]; then
  echo "test-benchmark-facade-env.sh: benchmark.json carries no trackId; nothing to pin" >&2
  exit 1
fi

# The stub benchd: records the track id env (or __UNSET__) and its argv, emits
# an empty JSON payload the way benchd would on stdout, exits 0.
STUB="${WORK}/benchd-stub"
cat > "${STUB}" <<'STUBEOF'
#!/usr/bin/env bash
# A `--help` probe records nothing: it asks what this benchd accepts, it is
# not a run.
for arg in "$@"; do
  if [[ "${arg}" == "--help" ]]; then
    echo "usage: benchd iterate --engine <bin> [--control-golden <path>]"
    exit 0
  fi
done
printf '%s\n' "${MLXFAST_QWEN_MTP_TRACK_ID-__UNSET__}" > "${STUB_CAPTURE_ENV}"
printf '%s\n' "$@" > "${STUB_CAPTURE_ARGV}"
previous=""
for arg in "$@"; do
  if [[ "${previous}" == "--contract" ]]; then
    printf '%s\n' "${arg}" > "${STUB_CAPTURE_CONTRACT_PATH}"
    cp "${arg}" "${STUB_CAPTURE_CONTRACT}"
    break
  fi
  previous="${arg}"
done
echo '{}'
exit 0
STUBEOF
chmod 755 "${STUB}"

GOLDEN="${WORK}/golden.json"
echo '{}' > "${GOLDEN}"

# run_facade CASE FACADE_PATH [facade args...] -- runs a facade with the stub
# benchd; captures land in ${WORK}/<case>.env / .argv / .out; sets rc.
run_facade() {
  local case_name="$1" facade="$2"
  shift 2
  rm -f "${WORK}/${case_name}.env" "${WORK}/${case_name}.argv"
  env -u MLXFAST_QWEN_MTP_TRACK_ID \
    STUB_CAPTURE_ENV="${WORK}/${case_name}.env" \
    STUB_CAPTURE_ARGV="${WORK}/${case_name}.argv" \
    STUB_CAPTURE_CONTRACT="${WORK}/${case_name}.contract.json" \
    STUB_CAPTURE_CONTRACT_PATH="${WORK}/${case_name}.contract-path" \
    BENCHD="${STUB}" \
    MLXFAST_ENGINE_BIN="${STUB}" \
    MLXFAST_CORRECTNESS_GOLDEN_PATH="${GOLDEN}" \
    MLXFAST_SCORE_PATH="${WORK}/${case_name}.score.json" \
    MLXFAST_INTEGRITY_PATH="${WORK}/${case_name}.integrity.json" \
    "${EXTRA_ENV[@]}" \
    "${facade}" "$@" > "${WORK}/${case_name}.out" 2>&1
  rc=$?
}

# Case 1: --local-iterate exports the manifest's trackId.
EXTRA_ENV=(MLXFAST_NO_SANDBOX=1)
run_facade case1 "${FACADE}" --local-iterate
if [[ "${rc}" -ne 0 ]]; then
  fail "case 1: facade exited ${rc} with the stub benchd; output: $(cat "${WORK}/case1.out")"
fi
if [[ ! -f "${WORK}/case1.env" ]]; then
  fail "case 1: benchd was never spawned"
elif [[ "$(cat "${WORK}/case1.env")" != "${EXPECTED}" ]]; then
  fail "case 1: benchd saw MLXFAST_QWEN_MTP_TRACK_ID='$(cat "${WORK}/case1.env")', expected '${EXPECTED}'"
fi
if [[ -f "${WORK}/case1.argv" ]]; then
  if [[ "$(head -n 1 "${WORK}/case1.argv")" != "iterate" ]]; then
    fail "case 1: benchd subcommand is not iterate: $(head -n 1 "${WORK}/case1.argv")"
  fi
  if ! grep -qx -- '--mode' "${WORK}/case1.argv" || ! grep -qx -- 'local-iterate' "${WORK}/case1.argv"; then
    fail "case 1: argv does not carry --mode local-iterate: $(tr '\n' ' ' < "${WORK}/case1.argv")"
  fi
fi
if [[ "$(jq -r 'has("warmup_golden") or has("decode_golden_offset")' "${WORK}/case1.contract.json")" != "false" ]]; then
  fail "case 1: local contract still carries a paired-only guard"
fi
case1_contract_path="$(cat "${WORK}/case1.contract-path")"
if [[ -e "${case1_contract_path}" ]]; then
  fail "case 1: local temporary contract remains after the run: ${case1_contract_path}"
fi

# Case 1b: local-submit uses the same temporary, stripped contract path.
EXTRA_ENV=(MLXFAST_NO_SANDBOX=1)
run_facade case1b "${FACADE}" --local-submit
if [[ "${rc}" -ne 0 ]]; then
  fail "case 1b: local-submit exited ${rc}; output: $(cat "${WORK}/case1b.out")"
elif ! grep -qx -- 'local-submit' "${WORK}/case1b.argv"; then
  fail "case 1b: argv does not carry --mode local-submit"
elif [[ "$(jq -r 'has("warmup_golden") or has("decode_golden_offset")' "${WORK}/case1b.contract.json")" != "false" ]]; then
  fail "case 1b: local-submit contract still carries a paired-only guard"
elif [[ -e "$(cat "${WORK}/case1b.contract-path")" ]]; then
  fail "case 1b: local-submit temporary contract remains after the run"
fi

# Case 2: --official, gates-only env (the seam-1 run the ranked box takes).
EXTRA_ENV=(MLXFAST_BENCHMARK_SKIP_TIMED=1 MLXFAST_BENCHMARK_CHECK_GATES=1)
run_facade case2 "${FACADE}" --official
if [[ "${rc}" -ne 0 ]]; then
  fail "case 2: facade exited ${rc} in official mode; output: $(cat "${WORK}/case2.out")"
fi
if [[ ! -f "${WORK}/case2.env" ]]; then
  fail "case 2: benchd was never spawned in official mode"
elif [[ "$(cat "${WORK}/case2.env")" != "${EXPECTED}" ]]; then
  fail "case 2: official mode: benchd saw MLXFAST_QWEN_MTP_TRACK_ID='$(cat "${WORK}/case2.env")', expected '${EXPECTED}'"
fi
if [[ -f "${WORK}/case2.argv" ]] && ! grep -qx -- 'official' "${WORK}/case2.argv"; then
  fail "case 2: argv does not carry --mode official: $(tr '\n' ' ' < "${WORK}/case2.argv")"
fi
if ! cmp -s "${REPO_ROOT}/fixtures/nemotron3_5_30b_a3b_mlx_v1_track.json" "${WORK}/case2.contract.json"; then
  fail "case 2: official mode did not receive the full fixture unchanged"
fi
if [[ "$(cat "${WORK}/case2.contract-path")" != "${REPO_ROOT}/tools/../fixtures/nemotron3_5_30b_a3b_mlx_v1_track.json" ]]; then
  fail "case 2: official mode received a copied contract instead of the authoritative fixture"
fi

# Case 3: a manifest with NO trackId refuses before benchd is spawned. The
# facade is copied into a throwaway tree whose benchmark.json lacks the field;
# it resolves the manifest relative to its own location, so nothing else moves.
mkdir -p "${WORK}/notrack/tools"
cp "${FACADE}" "${WORK}/notrack/tools/benchmark.sh"
jq 'del(.trackId)' "${REPO_ROOT}/benchmark.json" > "${WORK}/notrack/benchmark.json"
EXTRA_ENV=(MLXFAST_NO_SANDBOX=1)
run_facade case3 "${WORK}/notrack/tools/benchmark.sh" --local-iterate
if [[ "${rc}" -eq 0 ]]; then
  fail "case 3: facade ran with a manifest that carries no trackId"
fi
if ! grep -q 'trackId' "${WORK}/case3.out"; then
  fail "case 3: refusal does not name trackId; got: $(cat "${WORK}/case3.out")"
fi
if [[ -f "${WORK}/case3.env" ]]; then
  fail "case 3: benchd was spawned despite the missing trackId"
fi

# Case 4: a caller pre-sets a DIFFERENT track id -> refused, not overridden.
EXTRA_ENV=(MLXFAST_NO_SANDBOX=1 MLXFAST_QWEN_MTP_TRACK_ID=some-other-track-mlx-v9)
run_facade case4 "${FACADE}" --local-iterate
if [[ "${rc}" -eq 0 ]]; then
  fail "case 4: facade ran with a caller-set track id that differs from benchmark.json"
fi
if ! grep -q 'MLXFAST_QWEN_MTP_TRACK_ID' "${WORK}/case4.out"; then
  fail "case 4: refusal does not name the variable; got: $(cat "${WORK}/case4.out")"
fi
if [[ -f "${WORK}/case4.env" ]]; then
  fail "case 4: benchd was spawned despite the track id mismatch"
fi

# Case 5: a caller pre-sets the SAME value -> accepted, same export.
EXTRA_ENV=(MLXFAST_NO_SANDBOX=1 "MLXFAST_QWEN_MTP_TRACK_ID=${EXPECTED}")
run_facade case5 "${FACADE}" --local-iterate
if [[ "${rc}" -ne 0 ]]; then
  fail "case 5: facade refused a caller-set track id equal to benchmark.json's; output: $(cat "${WORK}/case5.out")"
fi
if [[ ! -f "${WORK}/case5.env" ]] || [[ "$(cat "${WORK}/case5.env")" != "${EXPECTED}" ]]; then
  fail "case 5: benchd did not see the expected track id"
fi

# ---------------------------------------------------------------------------
# Cases 6-8: the ranked measure script opens the GPU window and NOTHING ELSE.
#
# Hermetic root: the REAL measure script and the REAL resident script, copied
# beside the REAL benchmark.json and track fixture so both resolve their own
# paths inside the sandbox. Nothing loads: the bench-worker is a stub that
# binds the socket and answers the hello, and benchd is a stub too.
# ---------------------------------------------------------------------------
MEASURE_REL="tools/nemotron35-30b-a3b-measure-and-score.sh"
RES_ROOT="${WORK}/measure-root"
mkdir -p "${RES_ROOT}/tools" "${RES_ROOT}/fixtures" "${RES_ROOT}/weights"
cp "${REPO_ROOT}/${MEASURE_REL}" "${RES_ROOT}/tools/"
cp "${REPO_ROOT}/tools/resident-up.sh" "${RES_ROOT}/tools/"
# The measure script resolves the declared draft depth through the trusted
# declaration reader (tools/spec-declaration.sh); stage it beside the script.
cp "${REPO_ROOT}/tools/spec-declaration.sh" "${RES_ROOT}/tools/"
chmod +x "${RES_ROOT}/tools/"*.sh
cp "${REPO_ROOT}/benchmark.json" "${RES_ROOT}/benchmark.json"
cp "${REPO_ROOT}/fixtures/nemotron3_5_30b_a3b_mlx_v1_track.json" "${RES_ROOT}/fixtures/"

# THE GOLDEN MATERIAL, SYNTHESIZED INTO THIS SUITE'S OWN COPY OF THE FIXTURE.
#
# A freshly stamped track ships an EMPTY live_goldens and an EMPTY
# timed_prompt_pool: the real tapes are organizer material, published in R2 and
# staged on the box, and they do not exist for this track yet. The measure
# script refuses without them, and rightly so -- but that refusal is not what
# these cases are about, so the copy in ${RES_ROOT} gets a synthetic live
# golden, a pool entry pinned to the file staged beside it, and a per-depth
# oracle for every contract-permitted draft depth. The shipped fixture is never
# written. When the fixture DOES carry a pool, nothing below changes it.
#
# The arm state is synthesized the same way and for the same reason: an unarmed
# track refuses at --preflight-only's arm gate before it reaches any of the
# wiring these cases pin.
GOLDEN_DIR="${WORK}/goldens"
mkdir -p "${GOLDEN_DIR}"
python3 - "${RES_ROOT}/fixtures/nemotron3_5_30b_a3b_mlx_v1_track.json" "${GOLDEN_DIR}" <<'SYNTHEOF'
import hashlib, json, os, sys

contract_path, golden_dir = sys.argv[1:3]
contract = json.load(open(contract_path, encoding="utf-8"))
if not contract.get("live_goldens"):
    track = contract["track_id"]

    def stage(name):
        path = os.path.join(golden_dir, name + ".golden.json")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"synthetic_golden": name}) + "\n")
        data = open(path, "rb").read()
        return {
            "r2_path": "correctness_prompts/%s/%s.golden.json" % (track, name),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }

    live = ["synthetic-live-1", "synthetic-live-2"]
    contract["live_goldens"] = live
    contract["timed_prompt_pool"] = [stage(name) for name in live]
    contract["live_golden_speculative"] = {
        "mtp%d" % depth: {name: stage("%s.mtp%d" % (name, depth)) for name in live}
        for depth in contract.get("mtp_head", {}).get("permitted_draft_depths", [])
    }
# Armed in both shapes: a fixture that pins its goldens but does not score yet
# refuses at the arm gate too.
contract["official_scoring_enabled"] = True
if contract.get("warmup_golden"):
    warmup = contract["warmup_golden"]
    base = warmup["r2_path"].rsplit("/", 1)[-1]
    path = os.path.join(golden_dir, base)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"synthetic_warmup": base}) + "\n")
    data = open(path, "rb").read()
    warmup["sha256"] = hashlib.sha256(data).hexdigest()
    warmup["bytes"] = len(data)
with open(contract_path, "w", encoding="utf-8") as fh:
    json.dump(contract, fh, indent=2)
    fh.write("\n")
SYNTHEOF

# Each live golden has to exist under the name the fixture resolves to. When the
# block above staged it, it is already there with the bytes its pin names; when
# the fixture carried its own pool, an empty object is enough, because the bytes
# are never read here (benchd re-verifies the pin on a real run).
LIVE_GOLDEN_NAMES=()
while IFS= read -r name; do
  LIVE_GOLDEN_NAMES+=("${name}")
done < <(jq -r '.live_goldens[]' "${RES_ROOT}/fixtures/nemotron3_5_30b_a3b_mlx_v1_track.json")
for name in "${LIVE_GOLDEN_NAMES[@]}"; do
  [[ -f "${GOLDEN_DIR}/${name}.golden.json" ]] \
    || echo '{}' > "${GOLDEN_DIR}/${name}.golden.json"
done

# The stub benchd for these cases: it records whether the resident socket
# reached its environment, plus its argv, and answers an `iterate --help` probe
# with a usage line instead of recording it as a run.
MEASURE_STUB="${WORK}/benchd-measure-stub"
cat > "${MEASURE_STUB}" <<'MEASURESTUBEOF'
#!/usr/bin/env bash
for arg in "$@"; do
  if [[ "${arg}" == "--help" ]]; then
    echo "    --control-golden <path>"
    echo "    --warmup-golden <path>"
    exit 0
  fi
done
printf '%s\n' "${BENCH_WORKER_RESIDENT_SOCKET-__UNSET__}" > "${STUB_CAPTURE_ENV}"
printf '%s\n' "$@" > "${STUB_CAPTURE_ARGV}"
printf '%s\n' "${RESIDENT_UP_LOG_DIR-__UNSET__}" > "${STUB_CAPTURE_ARGV}.resident-log-dir"
echo '{}'
exit 0
MEASURESTUBEOF
chmod 755 "${MEASURE_STUB}"

# The stub bench-worker resident: records its argv, binds the socket, and sends
# the unprompted ok:true hello on every connection, as Engine Protocol v1 does.
WORKER_STUB="${WORK}/stub-bench-worker.py"
cat > "${WORKER_STUB}" <<'WORKERSTUBEOF'
#!/usr/bin/env python3
import json, os, signal, socket, sys
argv = sys.argv[1:]
with open(os.environ["STUB_ARGV_FILE"], "w") as f:
    f.write("\n".join(argv) + "\n")
if not argv or argv[0] != "resident":
    print("stub: first argument must be resident", file=sys.stderr); sys.exit(2)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
path = argv[argv.index("--socket") + 1]
try:
    os.unlink(path)
except FileNotFoundError:
    pass
srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
srv.bind(path)
srv.listen(4)
hello = {"id": 0, "nonce": "stubnonce", "ok": True, "protocol_version": 1,
         "backend": "mlx-resident", "device": "stub", "spec_modes": ["serial", "mtp"],
         "runner": {"id": "layr/stub"}, "resident": {"pid": os.getpid()}}
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
WORKERSTUBEOF
chmod +x "${WORKER_STUB}"

# THE STUB ENGINE LIVES INSIDE THE HERMETIC ROOT, at the FIXED path
# tools/stage-bench-worker.sh writes and benchd resolves. That is not a
# convenience: the measure script hands benchd a checkout-RELATIVE --engine so
# benchd can re-root it under the reference workspace, and an engine outside the
# checkout has no relative form (case 13 pins that refusal).
mkdir -p "${RES_ROOT}/.build/release"
cp "${WORKER_STUB}" "${RES_ROOT}/.build/release/bench-worker"
chmod +x "${RES_ROOT}/.build/release/bench-worker"
STAGED_WORKER_STUB="${RES_ROOT}/.build/release/bench-worker"

# THE PAIRED LEG PAIR. The measure script refuses a real run without both, so
# every measure case needs them staged. Nothing here is measured: the reference
# tree only has to exist and the calibration file only has to be a file --
# tools/ranked-box-preflight.sh is what validates their contents, and
# tools/test-ranked-box-preflight-env.sh is what pins that.
BASELINE_WORKSPACE="${WORK}/reference"
mkdir -p "${BASELINE_WORKSPACE}"
BASELINE_CALIBRATION="${WORK}/baseline-calibration.json"
CALIBRATION_BOX_NAME="synthetic-ranked-box"
printf '{"version":1,"box":"%s"}\n' "${CALIBRATION_BOX_NAME}" > "${BASELINE_CALIBRATION}"

WIRED_OK="${WORK}/wired-ok.sh"
printf '#!/bin/sh\necho 112000\n' > "${WIRED_OK}"
chmod +x "${WIRED_OK}"

# run_measure CASE [ENV=VAL...] [args...] -- drives the REAL measure script in
# the hermetic root. Captures land in ${WORK}/<case>.env / .argv / .out, and the
# resident's own argv in ${WORK}/<case>.resident-argv when one was booted.
run_measure() {
  local case_name="$1"
  shift
  rm -f "${WORK}/${case_name}.env" "${WORK}/${case_name}.argv" "${WORK}/${case_name}.resident-argv"
  local logdir="${WORK}/${case_name}.residentlog"
  mkdir -p "${logdir}"
  local extra=()
  while [[ $# -gt 0 && "$1" == *=* ]]; do extra+=("$1"); shift; done
  env -u BENCH_WORKER_RESIDENT_SOCKET -u MLXFAST_QWEN_MTP_TRACK_ID -u RUNNER_NAME \
    STUB_CAPTURE_ENV="${WORK}/${case_name}.env" \
    STUB_CAPTURE_ARGV="${WORK}/${case_name}.argv" \
    STUB_ARGV_FILE="${WORK}/${case_name}.resident-argv" \
    BENCHD="${MEASURE_STUB}" \
    MLXFAST_ENGINE_BIN="${STAGED_WORKER_STUB}" \
    MLXFAST_QWEN38_GOLDEN_DIR="${GOLDEN_DIR}" \
    MLXFAST_BASELINE_WORKSPACE="${BASELINE_WORKSPACE}" \
    MLXFAST_BASELINE_CALIBRATION="${BASELINE_CALIBRATION}" \
    MLXFAST_SCORE_PATH="${WORK}/${case_name}.score.json" \
    RESIDENT_UP_LOCK_PATH="${WORK}/${case_name}.gpu.lock" \
    RESIDENT_UP_LOG_DIR="${logdir}" \
    RESIDENT_UP_WIRED_LIMIT_READER="${WIRED_OK}" \
    RESIDENT_UP_HEALTH_TIMEOUT_S=60 \
    MLXFAST_GPU_LOCK_TIMEOUT_S=60 \
    ${extra[@]+"${extra[@]}"} \
    "${RES_ROOT}/${MEASURE_REL}" "$@" > "${WORK}/${case_name}.out" 2>&1
  rc=$?
}

# Case 6: the ranked path boots NOTHING, exports NO socket, and still locks.
run_measure case6
if [[ "${rc}" -ne 0 ]]; then
  fail "case 6: the measure script exited ${rc}; output: $(cat "${WORK}/case6.out")"
fi
if [[ ! -f "${WORK}/case6.env" ]]; then
  fail "case 6: benchd was never dispatched; output: $(cat "${WORK}/case6.out")"
elif [[ "$(cat "${WORK}/case6.env")" != "__UNSET__" ]]; then
  fail "case 6: BENCH_WORKER_RESIDENT_SOCKET reached benchd ($(cat "${WORK}/case6.env")); benchd boots ONE resident per leg from that leg's own tree, and a socket from here would attach the REFERENCE leg to the CANDIDATE's resident -- public run 34230122059"
fi
if [[ -f "${WORK}/case6.resident-argv" ]]; then
  fail "case 6: the measure script booted a resident: $(tr '\n' ' ' < "${WORK}/case6.resident-argv"); the legs' residents are benchd's to boot"
fi
# The lock is still this script's to hold: a resident holds ~113 GB whoever
# booted it, so the box must have exactly one loader for the whole measurement.
if [[ ! -e "${WORK}/case6.gpu.lock" ]]; then
  fail "case 6: the GPU lock was never taken; every per-leg boot benchd makes refuses when nobody holds it"
fi
if [[ -f "${WORK}/case6.argv" ]] && ! grep -qx -- 'official' "${WORK}/case6.argv"; then
  fail "case 6: the dispatch is not --mode official: $(tr '\n' ' ' < "${WORK}/case6.argv")"
fi

# Case 7: an inherited socket is refused by name, before benchd is dispatched.
run_measure case7 "BENCH_WORKER_RESIDENT_SOCKET=${WORK}/inherited.sock"
if [[ "${rc}" -eq 0 ]]; then
  fail "case 7: the measure script ran with an inherited BENCH_WORKER_RESIDENT_SOCKET"
elif ! grep -q "BENCH_WORKER_RESIDENT_SOCKET is set" "${WORK}/case7.out"; then
  fail "case 7: the refusal does not name the variable; got: $(cat "${WORK}/case7.out")"
fi
if [[ -f "${WORK}/case7.argv" ]]; then
  fail "case 7: benchd was dispatched despite the inherited socket"
fi
if [[ -f "${WORK}/case7.resident-argv" ]]; then
  fail "case 7: a resident was booted despite the inherited socket"
fi

# Case 8: a pre-GPU dry run opens no window.
run_measure case8 --preflight-only
if [[ "${rc}" -ne 0 ]]; then
  fail "case 8: --preflight-only exited ${rc}; output: $(cat "${WORK}/case8.out")"
fi
if [[ -f "${WORK}/case8.resident-argv" ]]; then
  fail "case 8: --preflight-only booted a resident: $(tr '\n' ' ' < "${WORK}/case8.resident-argv")"
fi
if [[ -e "${WORK}/case8.gpu.lock" ]]; then
  fail "case 8: --preflight-only took the GPU lock; a dry run loads nothing and must not open a window"
fi
if [[ -f "${WORK}/case8.argv" ]] && ! grep -qx -- 'validate-golden' "${WORK}/case8.argv"; then
  fail "case 8: --preflight-only did not dispatch validate-golden: $(tr '\n' ' ' < "${WORK}/case8.argv")"
fi

# ---------------------------------------------------------------------------
# Cases 9-13: the paired plumbing.
# ---------------------------------------------------------------------------

# argv_pair FILE FLAG VALUE -- the argv carries `FLAG VALUE` adjacently.
argv_pair() {
  python3 - "$1" "$2" "$3" <<'PYEOF'
import sys
path, flag, value = sys.argv[1], sys.argv[2], sys.argv[3]
args = open(path, encoding="utf-8").read().splitlines()
for i, arg in enumerate(args):
    if arg == flag and i + 1 < len(args) and args[i + 1] == value:
        raise SystemExit(0)
raise SystemExit(1)
PYEOF
}

# argv_value FILE FLAG -- the value that follows FLAG, or nothing.
argv_value() {
  python3 - "$1" "$2" <<'PYEOF'
import sys
path, flag = sys.argv[1], sys.argv[2]
args = open(path, encoding="utf-8").read().splitlines()
for i, arg in enumerate(args):
    if arg == flag and i + 1 < len(args):
        print(args[i + 1])
        break
PYEOF
}

# Case 9: the dispatch carries both names, and --engine is checkout-RELATIVE.
run_measure case9
if [[ "${rc}" -ne 0 ]]; then
  fail "case 9: the measure script exited ${rc} with both paired names set; output: $(cat "${WORK}/case9.out")"
elif [[ ! -f "${WORK}/case9.argv" ]]; then
  fail "case 9: benchd was never dispatched; output: $(cat "${WORK}/case9.out")"
else
  argv_pair "${WORK}/case9.argv" --baseline-workspace "${BASELINE_WORKSPACE}" \
    || fail "case 9: argv carries no --baseline-workspace ${BASELINE_WORKSPACE}"
  argv_pair "${WORK}/case9.argv" --baseline-calibration "${BASELINE_CALIBRATION}" \
    || fail "case 9: argv carries no --baseline-calibration ${BASELINE_CALIBRATION}"
  # RELATIVE, not absolute. benchd re-roots this path under the reference
  # workspace to find the control leg's worker; an absolute path would name
  # this checkout's binary and make the candidate its own control -- a score
  # of 1 by construction.
  if [[ -f "${WORK}/case9.env" && "$(cat "${WORK}/case9.env")" != "__UNSET__" ]]; then
    fail "case 9: the paired dispatch carried BENCH_WORKER_RESIDENT_SOCKET: $(cat "${WORK}/case9.env")"
  fi
  engine_value="$(argv_value "${WORK}/case9.argv" --engine)"
  if [[ -z "${engine_value}" ]]; then
    fail "case 9: argv carries no --engine"
  elif [[ "${engine_value}" == /* ]]; then
    fail "case 9: --engine is absolute (${engine_value}); benchd re-roots this path under the reference workspace, so an absolute path runs the candidate binary as its own control"
  fi
fi

# Case 10: the reference tree is required on a real run.
run_measure case10 "MLXFAST_BASELINE_WORKSPACE="
if [[ "${rc}" -eq 0 ]]; then
  fail "case 10: the measure script ran a real measurement with no reference tree"
elif ! grep -q "MLXFAST_BASELINE_WORKSPACE is unset" "${WORK}/case10.out"; then
  fail "case 10: the refusal does not name the variable; got: $(cat "${WORK}/case10.out")"
fi
if [[ -f "${WORK}/case10.argv" ]]; then
  fail "case 10: benchd was dispatched despite the missing reference tree"
fi

# Case 11: the calibration file is required on a real run.
run_measure case11 "MLXFAST_BASELINE_CALIBRATION="
if [[ "${rc}" -eq 0 ]]; then
  fail "case 11: the measure script ran a real measurement with no health band"
elif ! grep -q "MLXFAST_BASELINE_CALIBRATION is unset" "${WORK}/case11.out"; then
  fail "case 11: the refusal does not name the variable; got: $(cat "${WORK}/case11.out")"
fi
if [[ -f "${WORK}/case11.argv" ]]; then
  fail "case 11: benchd was dispatched despite the missing calibration file"
fi

# Case 12: --preflight-only requires neither. It is benchmark.json's
# preSubmitCommand, and a participant runs it where no reference tree exists.
run_measure case12 "MLXFAST_BASELINE_WORKSPACE=" "MLXFAST_BASELINE_CALIBRATION=" --preflight-only
if [[ "${rc}" -ne 0 ]]; then
  fail "case 12: --preflight-only refused off the box, where no reference tree can be staged; output: $(cat "${WORK}/case12.out")"
fi

# Case 13: an engine outside the checkout has no relative form.
OUTSIDE_ENGINE="${WORK}/outside-bench-worker"
cp "${WORKER_STUB}" "${OUTSIDE_ENGINE}"
chmod +x "${OUTSIDE_ENGINE}"
run_measure case13 "MLXFAST_ENGINE_BIN=${OUTSIDE_ENGINE}"
if [[ "${rc}" -eq 0 ]]; then
  fail "case 13: an engine outside the checkout was accepted; the control leg would run the candidate's own binary"
elif ! grep -q "outside this checkout" "${WORK}/case13.out"; then
  fail "case 13: the refusal does not explain the re-rooting; got: $(cat "${WORK}/case13.out")"
fi

# Case 14: weights outside the checkout have no relative form either.
OUTSIDE_WEIGHTS="${WORK}/outside-weights"
mkdir -p "${OUTSIDE_WEIGHTS}"
run_measure case14 "MLXFAST_WEIGHTS_PATH=${OUTSIDE_WEIGHTS}"
if [[ "${rc}" -eq 0 ]]; then
  fail "case 14: weights outside the checkout were accepted; both legs would read the candidate workspace's weights"
elif ! grep -q "outside this checkout" "${WORK}/case14.out"; then
  fail "case 14: the refusal does not explain the re-rooting; got: $(cat "${WORK}/case14.out")"
fi

# Case 15a: on a runner, --box is NOT passed. benchd reads RUNNER_NAME itself
# and lets it win over the flag, so passing it would be argv that cannot matter.
run_measure case15a "RUNNER_NAME=${CALIBRATION_BOX_NAME}"
if [[ "${rc}" -ne 0 ]]; then
  fail "case 15a: the measure script exited ${rc} with RUNNER_NAME set; output: $(cat "${WORK}/case15a.out")"
elif [[ ! -f "${WORK}/case15a.argv" ]]; then
  fail "case 15a: benchd was never dispatched; output: $(cat "${WORK}/case15a.out")"
elif grep -qx -- '--box' "${WORK}/case15a.argv"; then
  fail "case 15a: --box was passed while RUNNER_NAME is set; benchd lets RUNNER_NAME win, so the flag is dead argv"
fi

# Case 15b: on a hand run there is no RUNNER_NAME, so the flag carries the box
# name the calibration file itself records.
run_measure case15b
if [[ "${rc}" -ne 0 ]]; then
  fail "case 15b: the measure script exited ${rc} with RUNNER_NAME unset; output: $(cat "${WORK}/case15b.out")"
elif [[ ! -f "${WORK}/case15b.argv" ]]; then
  fail "case 15b: benchd was never dispatched; output: $(cat "${WORK}/case15b.out")"
elif ! argv_pair "${WORK}/case15b.argv" --box "${CALIBRATION_BOX_NAME}"; then
  fail "case 15b: argv does not carry --box ${CALIBRATION_BOX_NAME} from the calibration file: $(tr '\n' ' ' < "${WORK}/case15b.argv")"
fi

# ---------------------------------------------------------------------------
# Cases 16-19: the live_goldens list.
# ---------------------------------------------------------------------------
MEASURE_CONTRACT="${RES_ROOT}/fixtures/nemotron3_5_30b_a3b_mlx_v1_track.json"

# argv_values FILE FLAG -- every value that follows FLAG, one per line, in argv
# order.
argv_values() {
  python3 - "$1" "$2" <<'PYEOF'
import sys
path, flag = sys.argv[1], sys.argv[2]
args = open(path, encoding="utf-8").read().splitlines()
for i, arg in enumerate(args):
    if arg == flag and i + 1 < len(args):
        print(args[i + 1])
PYEOF
}

# pool_pins FIELD -- the timed_prompt_pool FIELD of each live golden, in
# live_goldens order.
pool_pins() {
  jq -r --arg f "$1" '. as $c | .live_goldens[] as $n
    | first($c.timed_prompt_pool[] | select(.r2_path | endswith("/" + $n + ".golden.json")))[$f]
    | tostring' "${MEASURE_CONTRACT}"
}

# Case 16: every live golden reaches benchd as repeated flags, in fixture order.
run_measure case16
if [[ "${rc}" -ne 0 ]]; then
  fail "case 16: the measure script exited ${rc}; output: $(cat "${WORK}/case16.out")"
elif [[ ! -f "${WORK}/case16.argv" ]]; then
  fail "case 16: benchd was never dispatched; output: $(cat "${WORK}/case16.out")"
else
  want_paths="$(printf "${GOLDEN_DIR}/%s.golden.json\n" "${LIVE_GOLDEN_NAMES[@]}")"
  want_shas="$(pool_pins sha256)"
  want_bytes="$(pool_pins bytes)"
  [[ "$(argv_values "${WORK}/case16.argv" --golden)" == "${want_paths}" ]] \
    || fail "case 16: the --golden values are not the live goldens in fixture order: $(argv_values "${WORK}/case16.argv" --golden | tr '\n' ' ')"
  [[ "$(argv_values "${WORK}/case16.argv" --golden-sha256)" == "${want_shas}" ]] \
    || fail "case 16: the --golden-sha256 values are not the live goldens' pins in fixture order"
  [[ "$(argv_values "${WORK}/case16.argv" --golden-bytes)" == "${want_bytes}" ]] \
    || fail "case 16: the --golden-bytes values are not the live goldens' pins in fixture order"
  [[ "$(argv_values "${WORK}/case16.argv" --control-golden)" == "${want_paths}" ]] \
    || fail "case 16: the --control-golden values are not the live goldens in fixture order: $(argv_values "${WORK}/case16.argv" --control-golden | tr '\n' ' ')"
  [[ "$(argv_values "${WORK}/case16.argv" --control-golden-sha256)" == "${want_shas}" ]] \
    || fail "case 16: the --control-golden-sha256 values are not the live goldens' pins in fixture order"
  [[ "$(argv_values "${WORK}/case16.argv" --control-golden-bytes)" == "${want_bytes}" ]] \
    || fail "case 16: the --control-golden-bytes values are not the live goldens' pins in fixture order"
  warmup_path="${GOLDEN_DIR}/$(jq -r '.warmup_golden.r2_path | split("/")[-1]' "${MEASURE_CONTRACT}")"
  argv_pair "${WORK}/case16.argv" --warmup-golden "${warmup_path}" \
    || fail "case 16: argv carries no pinned --warmup-golden"
  argv_pair "${WORK}/case16.argv" --warmup-golden-sha256 "$(jq -r '.warmup_golden.sha256' "${MEASURE_CONTRACT}")" \
    || fail "case 16: argv carries no --warmup-golden-sha256 pin"
  argv_pair "${WORK}/case16.argv" --warmup-golden-bytes "$(jq -r '.warmup_golden.bytes' "${MEASURE_CONTRACT}")" \
    || fail "case 16: argv carries no --warmup-golden-bytes pin"
fi

# run_measure_with_contract CASE FILTER [ENV=VAL...] -- run_measure against a
# copy of the fixture changed by the jq FILTER, then restore the fixture.
run_measure_with_contract() {
  local case_name="$1" filter="$2"
  shift 2
  cp "${MEASURE_CONTRACT}" "${WORK}/measure-contract.saved"
  jq "${filter}" "${WORK}/measure-contract.saved" > "${MEASURE_CONTRACT}"
  run_measure "${case_name}" "$@"
  cp "${WORK}/measure-contract.saved" "${MEASURE_CONTRACT}"
}

# expect_measure_refusal CASE NEEDLE -- the run refused, named NEEDLE, and never
# dispatched benchd.
expect_measure_refusal() {
  local case_name="$1" needle="$2"
  if [[ "${rc}" -eq 0 ]]; then
    fail "${case_name}: the measure script ran; it must refuse"
  elif ! grep -qF -- "${needle}" "${WORK}/${case_name}.out"; then
    fail "${case_name}: the refusal does not name '${needle}'; got: $(cat "${WORK}/${case_name}.out")"
  fi
  if [[ -f "${WORK}/${case_name}.argv" ]]; then
    fail "${case_name}: benchd was dispatched despite the refusal"
  fi
}

# Case 17: an empty list is refused by name.
run_measure_with_contract case17 '.live_goldens = []'
expect_measure_refusal case17 "fixture declares no live_goldens"

# Case 18: a live golden with no pool entry is refused by name.
run_measure_with_contract case18 '.live_goldens += ["no-such-prompt"]'
expect_measure_refusal case18 "live golden 'no-such-prompt' has no timed_prompt_pool entry"

# Case 19: a declared depth with no oracle for one live golden is refused by
# name. The last live golden loses its mtp3 oracle.
SPEC_MANIFEST="${WORK}/mtp3.manifest.json"
printf '{"version":1,"source":"pinned","spec":{"enabled":true,"num_speculative_tokens":3}}\n' > "${SPEC_MANIFEST}"
LAST_LIVE="${LIVE_GOLDEN_NAMES[${#LIVE_GOLDEN_NAMES[@]}-1]}"
run_measure_with_contract case19 'del(.live_golden_speculative.mtp3[.live_goldens[-1]])' \
  "SPEC_DECLARATION_MANIFEST=${SPEC_MANIFEST}"
expect_measure_refusal case19 "declared spec 'mtp3' has no live_golden_speculative entry for live golden '${LAST_LIVE}'"

# Case 20: with no RESIDENT_UP_LOG_DIR, benchd gets one outside both trees.
mkdir -p "${WORK}/case20-tmp"
run_measure case20 "RESIDENT_UP_LOG_DIR=" "RUNNER_TEMP=" "TMPDIR=${WORK}/case20-tmp"
case20_dir="$(cat "${WORK}/case20.argv.resident-log-dir" 2>/dev/null)"
if [[ "${rc}" -ne 0 ]]; then
  fail "case 20: the measure script exited ${rc} with RESIDENT_UP_LOG_DIR unset; output: $(cat "${WORK}/case20.out")"
elif [[ "${case20_dir}" != "${WORK}/case20-tmp/resident-up."* || ! -d "${case20_dir}" ]]; then
  fail "case 20: benchd got RESIDENT_UP_LOG_DIR '${case20_dir}', not a new directory in the job's temporary directory"
fi

# Case 21: a staged warm-up whose bytes do not match the fixture is refused
# before benchd gets a run argv.
WARMUP_FILE="${GOLDEN_DIR}/$(jq -r '.warmup_golden.r2_path | split("/")[-1]' "${MEASURE_CONTRACT}")"
cp "${WARMUP_FILE}" "${WORK}/warmup.saved"
printf 'mismatch\n' >> "${WARMUP_FILE}"
run_measure case21
if [[ "${rc}" -eq 0 ]]; then
  fail "case 21: a warm-up golden with the wrong byte count was accepted"
elif ! grep -q "warm-up golden byte-count mismatch" "${WORK}/case21.out"; then
  fail "case 21: the pin refusal did not name the warm-up byte count: $(cat "${WORK}/case21.out")"
elif [[ -f "${WORK}/case21.argv" ]]; then
  fail "case 21: benchd received a run argv after the warm-up pin mismatch"
fi
mv "${WORK}/warmup.saved" "${WARMUP_FILE}"

# Case 22: an old benchd that lacks --warmup-golden is refused. Such a build
# would silently ignore the fixture's new guard keys.
OLD_BENCHD="${WORK}/old-benchd"
cat > "${OLD_BENCHD}" <<'OLDEOF'
#!/usr/bin/env bash
if [[ "${1:-}" == "iterate" && "${2:-}" == "--help" ]]; then
  echo 'usage: benchd iterate --engine PATH'
  exit 0
fi
exit 0
OLDEOF
chmod +x "${OLD_BENCHD}"
run_measure case22 "BENCHD=${OLD_BENCHD}"
if [[ "${rc}" -eq 0 ]]; then
  fail "case 22: an old benchd without --warmup-golden was accepted"
elif ! grep -q "old benchd would ignore the new anti-cheat guards" "${WORK}/case22.out"; then
  fail "case 22: the refusal did not name the old benchd guard gap: $(cat "${WORK}/case22.out")"
elif [[ -f "${WORK}/case22.argv" ]]; then
  fail "case 22: the normal benchd stub got a run argv after the old-benchd refusal"
fi

# Case 23: the real benchd parses the stripped local contract, then stops at
# the deliberately missing engine. This cannot load a model or touch the GPU.
REAL_BENCHD="${REAL_BENCHD:-${REPO_ROOT}/benchd-bin/benchd}"
real_smoke="skipped (no executable at ${REAL_BENCHD})"
if [[ -x "${REAL_BENCHD}" ]]; then
  mkdir -p "${WORK}/case23-weights"
  env -u MLXFAST_QWEN_MTP_TRACK_ID \
    BENCHD="${REAL_BENCHD}" \
    MLXFAST_ENGINE_BIN="${WORK}/case23-missing-engine" \
    MLXFAST_WEIGHTS_PATH="${WORK}/case23-weights" \
    MLXFAST_CORRECTNESS_GOLDEN_PATH="${REPO_ROOT}/correctness_prompts/public_longcopy_gate_english_1024_256.json" \
    MLXFAST_SCORE_PATH="${WORK}/case23.score.json" \
    MLXFAST_INTEGRITY_PATH="${WORK}/case23.integrity.json" \
    TMPDIR="${WORK}" \
    "${FACADE}" --local-iterate > "${WORK}/case23.out" 2>&1
  rc=$?
  if [[ "${rc}" -eq 0 ]]; then
    fail "case 23: real benchd accepted the deliberately missing engine"
  elif grep -Eq 'WARMUP-GOLDEN-REQUIRED|decode_golden_offset' "${WORK}/case23.out"; then
    fail "case 23: real benchd refused a paired-only local guard: $(cat "${WORK}/case23.out")"
  elif ! grep -q 'failed to spawn engine' "${WORK}/case23.out"; then
    fail "case 23: real benchd did not reach the later missing-engine check: $(cat "${WORK}/case23.out")"
  else
    real_smoke="passed"
  fi
else
  echo "SKIP: case 23 real-binary smoke (${real_smoke})"
fi

if [[ "${failures}" -eq 0 ]]; then
  echo "test-benchmark-facade-env.sh: all checks passed; real-binary smoke ${real_smoke} (trackId=${EXPECTED})"
  exit 0
fi
echo "test-benchmark-facade-env.sh: ${failures} case(s) failed" >&2
exit 1
