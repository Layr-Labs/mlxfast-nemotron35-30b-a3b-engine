#!/usr/bin/env bash
# Hermetic test for tools/nemotron35-30b-a3b-yukon-run.sh. No GPU, no benchd.
#
# The script copies into a temporary tree. Stubs replace the ranked script,
# ./benchmark.sh and the trusted CLI, and each stub records how it was called.
# tools/spec-declaration.sh and the track fixture are the real files.
#
#   1. GITHUB_ACTIONS=true goes to the ranked script with all arguments.
#   2. each ranked input alone goes to the ranked script.
#   3. no ranked input, weights/ absent, depth 3 declared: the transform runs,
#      then ./benchmark.sh --local-submit --mtp-depth 3 on the 1024 capture.
#   4. no ranked input, weights/ present, serial: no transform, no --mtp-depth.
#   5. the local path refuses an argument.
#   6. a declaration outside the envelope stops the run before benchmark.sh.
#
# Usage: tools/test-yukon-run.sh
# Exit:  0 all cases pass, 1 a case failed (printed with a FAIL prefix)
set -uo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
fails=0
fail() { echo "FAIL $*" >&2; fails=$((fails + 1)); }
pass() { echo "ok   $*"; }

# make_tree <dir> <manifest-json>
make_tree() {
  local t="$1"
  mkdir -p "${t}/tools" "${t}/fixtures" "${t}/.build/release"
  cp "${SRC}/tools/nemotron35-30b-a3b-yukon-run.sh" "${SRC}/tools/spec-declaration.sh" "${t}/tools/"
  cp "${SRC}/fixtures/nemotron3_5_30b_a3b_mlx_v1_track.json" "${t}/fixtures/"
  printf '%s' "$2" > "${t}/mtp-head.manifest.json"
  cat > "${t}/tools/nemotron35-30b-a3b-measure-and-score.sh" <<'EOF'
#!/usr/bin/env bash
echo "ranked $*" >> "$(dirname "$0")/../calls"
EOF
  cat > "${t}/benchmark.sh" <<'EOF'
#!/usr/bin/env bash
echo "local golden=${MLXFAST_CORRECTNESS_GOLDEN_PATH} score=${MLXFAST_SCORE_PATH:-} engine=${MLXFAST_ENGINE_BIN} $*" >> "$(dirname "$0")/calls"
EOF
  cat > "${t}/.build/release/mlxfast-swift" <<'EOF'
#!/usr/bin/env bash
echo "cli $*" >> "$(dirname "$0")/../../calls"
mkdir -p weights && : > weights/config.json
EOF
  chmod +x "${t}/tools/"*.sh "${t}/benchmark.sh" "${t}/.build/release/mlxfast-swift"
}

clean_env=(env -u GITHUB_ACTIONS -u MLXFAST_QWEN38_GOLDEN_DIR -u MLXFAST_BASELINE_WORKSPACE -u MLXFAST_BASELINE_CALIBRATION -u MLXFAST_REFERENCE_DIR)
MTP3='{"version":1,"source":"pinned","spec":{"enabled":true,"num_speculative_tokens":3}}'
SERIAL='{"version":1,"source":"pinned"}'
BAD='{"version":1,"source":"pinned","spec":{"enabled":true,"num_speculative_tokens":9}}'

# 1
t="${WORK}/c1"; make_tree "${t}" "${MTP3}"
"${clean_env[@]}" GITHUB_ACTIONS=true "${t}/tools/nemotron35-30b-a3b-yukon-run.sh" --preflight-only
if [[ "$(cat "${t}/calls")" == "ranked --preflight-only" ]]; then pass "case 1: GITHUB_ACTIONS=true is ranked"; else fail "case 1: $(cat "${t}/calls")"; fi

# 2
for var in MLXFAST_QWEN38_GOLDEN_DIR MLXFAST_BASELINE_WORKSPACE MLXFAST_BASELINE_CALIBRATION; do
  t="${WORK}/c2-${var}"; make_tree "${t}" "${MTP3}"
  "${clean_env[@]}" "${var}=/nonexistent" "${t}/tools/nemotron35-30b-a3b-yukon-run.sh"
  if [[ "$(cat "${t}/calls")" == "ranked " ]]; then pass "case 2: ${var} alone is ranked"; else fail "case 2 ${var}: $(cat "${t}/calls")"; fi
done

# 3
t="${WORK}/c3"; make_tree "${t}" "${MTP3}"
"${clean_env[@]}" MLXFAST_SCORE_PATH=score.json "${t}/tools/nemotron35-30b-a3b-yukon-run.sh" 2>/dev/null
want="cli transform --reference reference_weights/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit-mtp --output weights
local golden=correctness_prompts/public_longcopy_gate_english_1024_1024.json score=score.json engine=.build/release/bench-worker --local-submit --mtp-depth 3"
if [[ "$(cat "${t}/calls")" == "${want}" ]]; then pass "case 3: local path transforms, then runs --local-submit at depth 3"; else fail "case 3: $(cat "${t}/calls")"; fi

# 4
t="${WORK}/c4"; make_tree "${t}" "${SERIAL}"; mkdir -p "${t}/weights"; : > "${t}/weights/config.json"
"${clean_env[@]}" "${t}/tools/nemotron35-30b-a3b-yukon-run.sh" 2>/dev/null
want="local golden=correctness_prompts/public_longcopy_gate_english_1024_1024.json score= engine=.build/release/bench-worker --local-submit"
if [[ "$(cat "${t}/calls")" == "${want}" ]]; then pass "case 4: weights present and serial: no transform, no --mtp-depth"; else fail "case 4: $(cat "${t}/calls")"; fi

# 5
t="${WORK}/c5"; make_tree "${t}" "${SERIAL}"
"${clean_env[@]}" "${t}/tools/nemotron35-30b-a3b-yukon-run.sh" --preflight-only 2>/dev/null; rc=$?
if [[ "${rc}" == "2" && ! -e "${t}/calls" ]]; then pass "case 5: the local path refuses an argument"; else fail "case 5: rc=${rc}"; fi

# 6
t="${WORK}/c6"; make_tree "${t}" "${BAD}"; mkdir -p "${t}/weights"; : > "${t}/weights/config.json"
"${clean_env[@]}" "${t}/tools/nemotron35-30b-a3b-yukon-run.sh" 2>/dev/null; rc=$?
if [[ "${rc}" != "0" && ! -e "${t}/calls" ]]; then pass "case 6: an out-of-envelope declaration stops before benchmark.sh"; else fail "case 6: rc=${rc}"; fi

[[ "${fails}" == "0" ]] || exit 1
echo "all cases passed"
