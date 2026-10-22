#!/usr/bin/env bash
# record-goldens.sh -- record this track's PER-DEPTH MTP ORACLES on the ranked
# box, for live goldens whose serial tapes are already pinned.
#
# Ported from the Bonsai 9B MLX track. This port names the Nemotron contract,
# uses stop token 11 (<|im_end|>) from this checkpoint's tokenizer, reads
# depths 1 to 7 from mtp_head.permitted_draft_depths, and removes the dflash
# path because this track has no dflash_drafter block.
#
# ORGANIZER TOOL, box only. The candidate leg at MTP depth D replays the
# engine's free run at depth D, and benchd checks it token for token against
# the oracle that live_golden_speculative[mtpD][<name>] pins. A depth oracle
# can equal the serial tape or differ from it. Each depth still has its own
# contract key and is recorded through the same free-run path the candidate
# leg replays (author-with-what-you-replay). The serial-control leg stays
# checked against the serial tape (--control-golden).
#
# THE CUDA TRACKS' tools/record-goldens.sh IS THE MODEL. One resident per depth,
# booted by THIS TREE's own tools/resident-up.sh --boot (its engine, its
# weights, its MTP head), every live golden recorded under it, and each
# oracle carries the PINNED serial cases[] spliced in, so every per-depth golden
# has the same token-fidelity cases as the serial tape. The one difference: the
# MLX bench-worker takes the decoder per request, so the free run is requested
# at depth D with `record-correctness-golden --benchmark-spec mtp:<D>` (benchd
# PR #404); a CUDA serve-up fixes the spec at boot instead.
#
# THE RECORDING TREE IS THIS CHECKOUT, NOT THE REFERENCE TREE. The reference
# tree at baseline_reference_commit runs the serial-control leg only. Each
# permitted depth needs the engine that serves it, so the organizer runs this
# tool from a built, clean checkout of the engine commit the oracles are pinned
# for. Its tracked files must match its HEAD.
#
# THE TOOL TAKES THE GPU LOCK ITSELF, for the whole window, the way
# tools/calibrate-box.sh does, and the resident boots check that it is held.
#
# Usage:
#   tools/record-goldens.sh --prompts <DIR> --out <DIR>
#                           [--depths 1,2,...] [--apply]
#     <DIR>/<name>.tokens.json for every live golden; <out>/<name>.golden.json
#     must already be the PINNED serial golden of every live golden (stage it
#     with tools/fetch-goldens.sh --all --out <out>, or copy the staged one).
#     Writes <out>/<name>.mtp<D>.golden.json for every live golden and every
#     depth (default: mtp_head.permitted_draft_depths), validates each with
#     benchd validate-golden, and prints the {sha256, bytes} pins. --apply writes
#     live_golden_speculative[mtpD][<name>] into the fixture, at the R2 key
#     <dir>/<name>.mtp<D>.golden.json, where <dir> is the folder of the pinned
#     serial tape's r2_path: each oracle sits beside its serial tape, so a new
#     checkpoint's tapes and oracles share one new folder of the append-only
#     bucket. Upload is the organizer's own step; this tool holds no credential.
#
# Env:
#   BENCHD_BIN_DIR              the benchd pair (default <repo>/benchd-bin); its
#                               record-correctness-golden must know
#                               --benchmark-spec.
#   RECORD_GOLDENS_STOP_TOKEN   the end-of-turn id the serial tapes were
#                               recorded with (default 11, <|im_end|>).
#   RESIDENT_UP_LOCK_PATH       the GPU lock (default /tmp/mtplx-gpu-exclusive.lock).
#   MLXFAST_GPU_LOCK_TIMEOUT_S  ceiling on the wait for it (default 1800).
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
FIXTURE="${ROOT}/fixtures/nemotron3_5_30b_a3b_mlx_v1_track.json"
log() { printf 'record-goldens: %s\n' "$*" >&2; }
die() { printf 'record-goldens: %s\n' "$*" >&2; exit 1; }

PROMPTS="" OUT="" APPLY=0 DEPTHS_ARG="" IN_LOCK=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompts) PROMPTS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --depths) DEPTHS_ARG="$2"; shift 2 ;;
    --apply) APPLY=1; shift ;;
    --in-lock) IN_LOCK=1; shift ;;
    *) die "unknown argument '$1'" ;;
  esac
done
[[ -d "${PROMPTS}" && -d "${OUT}" ]] || die "usage: --prompts <DIR> --out <DIR> [--depths 1,2,...] [--apply]"
OUT="$(cd -- "${OUT}" && pwd -P)"
PROMPTS="$(cd -- "${PROMPTS}" && pwd -P)"
command -v jq >/dev/null 2>&1 || die "jq is required"
TREE="${ROOT}"
git -C "${TREE}" diff --quiet HEAD -- 2>/dev/null \
  || die "the recording tree ${TREE} has tracked changes against its HEAD; record from a clean checkout"
[[ -x "${TREE}/.build/release/bench-worker" && -d "${TREE}/weights" ]] \
  || die "the recording tree ${TREE} is not built (no .build/release/bench-worker or weights/)"
BIN="${BENCHD_BIN_DIR:-${ROOT}/benchd-bin}"
RECORDER="${BIN}/record-correctness-golden"
BENCHD="${BIN}/benchd"
[[ -x "${RECORDER}" && -x "${BENCHD}" ]] || die "no benchd pair in ${BIN} (run tools/fetch-benchd.sh)"
# --help exits non-zero, so read its text first; under pipefail a pipe into
# grep would report the recorder's exit status, not the match.
RECORDER_HELP="$("${RECORDER}" --help 2>&1 || true)"
grep -q -- '--benchmark-spec' <<<"${RECORDER_HELP}" \
  || die "${RECORDER} has no --benchmark-spec; it would record every depth serial"
grep -q -- 'mtp:<D>' <<<"${RECORDER_HELP}" \
  || die "${RECORDER} does not record mtp:<D>; fetch the channel benchd (tools/fetch-benchd.sh)"

TRACK="$(jq -r .trackId "${ROOT}/benchmark.json")"
REPO="$(jq -r .target.upstream_model_id "${FIXTURE}")"
REV="$(jq -r .target.upstream_revision "${FIXTURE}")"
STOP="${RECORD_GOLDENS_STOP_TOKEN:-11}"
[[ "${STOP}" =~ ^[0-9]+$ ]] || die "RECORD_GOLDENS_STOP_TOKEN must be a token id"
NAMES=()
while IFS= read -r n; do NAMES+=("${n}"); done < <(jq -r '(.live_goldens // [])[]' "${FIXTURE}")
[[ "${#NAMES[@]}" -gt 0 ]] || die "the fixture declares no live_goldens"
PERMITTED=()
while IFS= read -r d; do PERMITTED+=("${d}"); done < <(jq -r '.mtp_head.permitted_draft_depths[]' "${FIXTURE}")
if [[ -n "${DEPTHS_ARG}" ]]; then
  IFS=, read -ra DEPTHS <<<"${DEPTHS_ARG}"
  for d in "${DEPTHS[@]}"; do
    [[ " ${PERMITTED[*]} " == *" ${d} "* ]] || die "--depths ${d}: not a permitted draft depth (${PERMITTED[*]})"
  done
else
  DEPTHS=("${PERMITTED[@]}")
fi

# Every oracle splices the PINNED serial cases[], so the serial tape in <out>
# must be the pinned one, byte for byte.
for n in "${NAMES[@]}"; do
  [[ -f "${PROMPTS}/${n}.tokens.json" ]] || die "no ${PROMPTS}/${n}.tokens.json"
  serial="${OUT}/${n}.golden.json"
  [[ -f "${serial}" ]] || die "the pinned serial golden of ${n} is not in ${OUT}"
  want="$(jq -r --arg f "/${n}.golden.json" 'first(.timed_prompt_pool[] | select(.r2_path | endswith($f))) | "\(.sha256) \(.bytes)"' "${FIXTURE}")"
  have="$(shasum -a 256 "${serial}" | cut -d' ' -f1) $(wc -c < "${serial}" | tr -d ' ')"
  [[ "${have}" == "${want}" ]] || die "${serial} is not the pinned serial golden of ${n} (${want})"
done

# THE LOCK WINDOW. The first invocation takes the lock on an inheritable
# descriptor and re-executes this script inside it.
if [[ "${IN_LOCK}" == "0" ]]; then
  LOCK="${RESIDENT_UP_LOCK_PATH:-/tmp/mtplx-gpu-exclusive.lock}"
  TIMEOUT="${MLXFAST_GPU_LOCK_TIMEOUT_S:-1800}"
  export RESIDENT_UP_LOCK_PATH="${LOCK}"
  log "taking the GPU lock ${LOCK}"
  args=(--prompts "${PROMPTS}" --out "${OUT}" --in-lock)
  [[ -n "${DEPTHS_ARG}" ]] && args+=(--depths "${DEPTHS_ARG}")
  [[ "${APPLY}" == "1" ]] && args+=(--apply)
  exec python3 -c "
import fcntl, os, signal, sys
fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o666)
os.set_inheritable(fd, True)
def on_timeout(*_):
    sys.stderr.write('record-goldens: another owner holds ' + sys.argv[1] + '; nothing has been loaded\n')
    sys.exit(1)
signal.signal(signal.SIGALRM, on_timeout)
signal.alarm(int(sys.argv[2]))
fcntl.flock(fd, fcntl.LOCK_EX)
signal.alarm(0)
os.execv(sys.argv[3], sys.argv[3:])
" "${LOCK}" "${TIMEOUT}" "${BASH}" "${BASH_SOURCE[0]}" "${args[@]}"
fi

# Keep resident logs outside OUT because the arming tool accepts golden files only.
RESIDENT_LOGS="${OUT}.resident-logs"
mkdir -p "${RESIDENT_LOGS}"
SOCKET=""
stop_resident() {
  [[ -z "${SOCKET}" ]] && return 0
  RESIDENT_UP_LOG_DIR="${RESIDENT_LOGS}" "${TREE}/tools/resident-up.sh" --stop --socket "${SOCKET}" >&2
  SOCKET=""
}
stop_resident_on_exit() {
  # A failed normal stop leaves SOCKET set, so the EXIT trap can retry it.
  [[ -z "${SOCKET}" ]] || RESIDENT_UP_LOG_DIR="${RESIDENT_LOGS}" "${TREE}/tools/resident-up.sh" --stop --socket "${SOCKET}" >&2 || true
}
trap stop_resident_on_exit EXIT

for d in "${DEPTHS[@]}"; do
  log "mtp${d}: booting this tree's resident"
  RESIDENT_UP_LOG_DIR="${RESIDENT_LOGS}" "${TREE}/tools/resident-up.sh" \
    --boot --spec mtp --draft-len "${d}" --socket-out "${RESIDENT_LOGS}/socket" >&2
  SOCKET="$(head -1 "${RESIDENT_LOGS}/socket")"
  for n in "${NAMES[@]}"; do
    raw="${OUT}/${n}.mtp${d}.raw.json"
    (cd "${TREE}" && BENCH_WORKER_RESIDENT_SOCKET="${SOCKET}" MLXFAST_QWEN_MTP_TRACK_ID="${TRACK}" "${RECORDER}" \
       --worker-bin "${TREE}/.build/release/bench-worker" --weights "${TREE}/weights" \
       --prompt-tokens "${PROMPTS}/${n}.tokens.json" --case-name "${n}" \
       --steps 1024 --stop-token "${STOP}" \
       --benchmark-free-run --benchmark-spec "mtp:${d}" \
       --track "${TRACK}" --contract "${FIXTURE}" \
       --model-provenance-repo "${REPO}" --model-provenance-rev "${REV}" --out "${raw}") >&2
    jq --slurpfile serial "${OUT}/${n}.golden.json" '.cases = $serial[0].cases' "${raw}" > "${OUT}/${n}.mtp${d}.golden.json"
    rm -f "${raw}"
    "${BENCHD}" validate-golden --golden "${OUT}/${n}.mtp${d}.golden.json" --track "${TRACK}" --contract "${FIXTURE}" >&2 \
      || die "benchd refused ${OUT}/${n}.mtp${d}.golden.json"
  done
  stop_resident
done

pins="{}"
for d in "${DEPTHS[@]}"; do
  for n in "${NAMES[@]}"; do
    f="${OUT}/${n}.mtp${d}.golden.json"
    serial_key="$(jq -r --arg f "/${n}.golden.json" 'first(.timed_prompt_pool[] | select(.r2_path | endswith($f))) | .r2_path' "${FIXTURE}")"
    pins="$(jq -c --arg k "mtp${d}" --arg n "${n}" \
      --arg r "${serial_key%/*}/${n}.mtp${d}.golden.json" \
      --arg s "$(shasum -a 256 "${f}" | cut -d' ' -f1)" --argjson b "$(wc -c < "${f}" | tr -d ' ')" \
      '.[$k][$n] = {r2_path: $r, sha256: $s, bytes: $b}' <<<"${pins}")"
  done
done
jq . <<<"${pins}"
log "upload each file to R2 at its r2_path; the bucket is append-only, so a key that exists with other bytes is a stop"

if [[ "${APPLY}" == "1" ]]; then
  jq --argjson p "${pins}" '.live_golden_speculative = ((.live_golden_speculative // {}) * $p)' "${FIXTURE}" > "${FIXTURE}.tmp"
  mv "${FIXTURE}.tmp" "${FIXTURE}"
  log "pinned ${FIXTURE}; review the diff, then commit"
fi
