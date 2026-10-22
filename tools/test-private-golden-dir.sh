#!/usr/bin/env bash
# Prove the ranked path exports a private golden root and rejects escapes.
set -uo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
TEST_ROOT="${WORK}/root"
mkdir -p "${TEST_ROOT}/tools" "${TEST_ROOT}/fixtures"
cp "${ROOT_DIR}/tools/nemotron35-30b-a3b-measure-and-score.sh" "${TEST_ROOT}/tools/"
cp "${ROOT_DIR}/tools/spec-declaration.sh" "${TEST_ROOT}/tools/"
cp "${ROOT_DIR}/benchmark.json" "${TEST_ROOT}/"
cp "${ROOT_DIR}/fixtures/nemotron3_5_30b_a3b_mlx_v1_track.json" "${TEST_ROOT}/fixtures/"
chmod +x "${TEST_ROOT}/tools/"*.sh
TARGET="${TEST_ROOT}/tools/nemotron35-30b-a3b-measure-and-score.sh"
fails=0
pass() { printf 'test-private-golden-dir: PASS -- %s\n' "$*"; }
fail() { printf 'test-private-golden-dir: FAIL -- %s\n' "$*" >&2; fails=$((fails + 1)); }

mkdir -p "${WORK}/bin" "${WORK}/private/goldens" "${WORK}/outside" \
  "${WORK}/priv" "${WORK}/priv2"
CANONICAL_GOLDEN_DIR="$(cd "${WORK}/private/goldens" && pwd -P)"
cat > "${WORK}/bin/benchd" <<'EOF'
#!/usr/bin/env bash
if [[ "${1:-}" == "iterate" && "${2:-}" == "--help" ]]; then
  echo '    --warmup-golden PATH'
  exit 0
fi
printf '%s\n' "${MLXFAST_PRIVATE_DIR:-}" >> "${PRIVATE_DIR_RECORD}"
exit 0
EOF
chmod +x "${WORK}/bin/benchd"

warmup_body='{"synthetic_warmup":true}'
for dir in "${WORK}/private/goldens" "${WORK}/outside" "${WORK}/priv2"; do
  printf '%s\n' "${warmup_body}" > "${dir}/warmup.golden.json"
done
warmup_sha="$(shasum -a 256 "${WORK}/private/goldens/warmup.golden.json" | awk '{print $1}')"
warmup_bytes="$(wc -c < "${WORK}/private/goldens/warmup.golden.json" | tr -d '[:space:]')"
jq --arg path 'correctness_prompts/test/warmup.golden.json' --arg sha "${warmup_sha}" \
  --argjson bytes "${warmup_bytes}" \
  '.warmup_golden = {r2_path: $path, sha256: $sha, bytes: $bytes}' \
  "${TEST_ROOT}/fixtures/nemotron3_5_30b_a3b_mlx_v1_track.json" > "${WORK}/fixture.tmp"
mv "${WORK}/fixture.tmp" "${TEST_ROOT}/fixtures/nemotron3_5_30b_a3b_mlx_v1_track.json"

while IFS= read -r name; do
  for dir in "${WORK}/private/goldens" "${WORK}/outside" "${WORK}/priv2"; do
    : > "${dir}/${name}.golden.json"
  done
done < <(jq -r '.live_goldens[]' "${TEST_ROOT}/fixtures/nemotron3_5_30b_a3b_mlx_v1_track.json")

SPEC_MANIFEST="${WORK}/mtp1-manifest.json"
jq '.spec = {enabled: true, num_speculative_tokens: 1}' \
  "${ROOT_DIR}/mtp-head.manifest.json" > "${SPEC_MANIFEST}"
while IFS= read -r path; do
  : > "${WORK}/private/goldens/${path##*/}"
done < <(jq -r '.live_golden_speculative.mtp1[].r2_path' \
  "${TEST_ROOT}/fixtures/nemotron3_5_30b_a3b_mlx_v1_track.json")

run_preflight() {
  local golden_dir="$1" private_dir="$2" record="$3" spec_manifest="${4:-}" target="${5:-${TARGET}}"
  local run_env=(BENCHD="${WORK}/bin/benchd" PRIVATE_DIR_RECORD="${record}"
    MLXFAST_QWEN38_GOLDEN_DIR="${golden_dir}"
    SPEC_DECLARATION_MANIFEST="${spec_manifest}")
  if [[ "${private_dir}" == unset ]]; then
    env -u MLXFAST_PRIVATE_DIR "${run_env[@]}" \
      "${target}" --preflight-only
  else
    env "${run_env[@]}" MLXFAST_PRIVATE_DIR="${private_dir}" \
      "${target}" --preflight-only
  fi
}

record="${WORK}/default.record"
if run_preflight "${WORK}/private/goldens" unset "${record}" >/dev/null 2>&1 \
  && [[ -s "${record}" ]] \
  && ! grep -vxF "${CANONICAL_GOLDEN_DIR}" "${record}" >/dev/null; then
  pass "unset MLXFAST_PRIVATE_DIR defaults to and exports the golden directory"
else
  fail "default private directory was not exported to benchd"
fi

record="${WORK}/inside.record"
if run_preflight "${WORK}/private/goldens" "${WORK}/private" "${record}" >/dev/null 2>&1; then
  pass "goldens inside MLXFAST_PRIVATE_DIR are accepted"
else
  fail "inside golden paths were refused"
fi

record="${WORK}/empty.record"
if run_preflight "${WORK}/private/goldens" "" "${record}" >/dev/null 2>&1 \
  && [[ -s "${record}" ]] \
  && ! grep -vxF "${CANONICAL_GOLDEN_DIR}" "${record}" >/dev/null; then
  pass "empty MLXFAST_PRIVATE_DIR defaults to and exports the golden directory"
else
  fail "empty private directory did not use the golden directory"
fi

record="${WORK}/outside.record"
out="$(run_preflight "${WORK}/outside" "${WORK}/private" "${record}" 2>&1)"; rc=$?
if [[ "${rc}" -eq 1 && "${out}" == *"outside canonical MLXFAST_PRIVATE_DIR"* && ! -e "${record}" ]]; then
  pass "goldens outside MLXFAST_PRIVATE_DIR are refused before benchd"
else
  fail "outside golden path was not refused before benchd (rc ${rc})"
fi

record="${WORK}/sibling.record"
out="$(run_preflight "${WORK}/priv2" "${WORK}/priv" "${record}" 2>&1)"; rc=$?
if [[ "${rc}" -eq 1 && "${out}" == *"outside canonical MLXFAST_PRIVATE_DIR"* && ! -e "${record}" ]]; then
  pass "a sibling prefix is not treated as inside MLXFAST_PRIVATE_DIR"
else
  fail "sibling-prefix golden path was not refused before benchd (rc ${rc})"
fi

first_live="$(jq -r '.live_goldens[0]' "${ROOT_DIR}/fixtures/nemotron3_5_30b_a3b_mlx_v1_track.json")"
serial_path="${WORK}/private/goldens/${first_live}.golden.json"
outside_target="${WORK}/outside/final-file.golden.json"
: > "${outside_target}"
outside_target_real="$(cd "$(dirname "${outside_target}")" && pwd -P)/$(basename "${outside_target}")"
rm "${serial_path}"
ln -s "${outside_target}" "${serial_path}"

record="${WORK}/golden-symlink.record"
out="$(run_preflight "${WORK}/private/goldens" "${WORK}/private" "${record}" 2>&1)"; rc=$?
if [[ "${rc}" -eq 1 && "${out}" == *"--golden path ${outside_target_real} is outside canonical MLXFAST_PRIVATE_DIR"* && ! -e "${record}" ]]; then
  pass "a --golden final-file symlink cannot escape MLXFAST_PRIVATE_DIR"
else
  fail "an escaping --golden final-file symlink was not refused before benchd (rc ${rc}): ${out}"
fi
rm "${serial_path}"
: > "${serial_path}"

# The warm-up path is checked with the same whole-path rule.
warmup_path="${WORK}/private/goldens/warmup.golden.json"
mv "${warmup_path}" "${WORK}/warmup.inside"
ln -s "${WORK}/outside/warmup.golden.json" "${warmup_path}"
record="${WORK}/warmup-symlink.record"
out="$(run_preflight "${WORK}/private/goldens" "${WORK}/private" "${record}" 2>&1)"; rc=$?
if [[ "${rc}" -eq 1 && "${out}" == *"--warmup-golden path"* && "${out}" == *"outside canonical MLXFAST_PRIVATE_DIR"* && ! -e "${record}" ]]; then
  pass "an escaping --warmup-golden final-file symlink is refused before benchd"
else
  fail "an escaping --warmup-golden final-file symlink was not refused before benchd (rc ${rc}): ${out}"
fi
rm "${warmup_path}"
mv "${WORK}/warmup.inside" "${warmup_path}"

# Restore the serial escape for the --control-golden case below.
rm "${serial_path}"
ln -s "${outside_target}" "${serial_path}"

# Give the candidate tape a distinct basename so the first loop passes and the
# serial tape reaches the --control-golden check.
CONTROL_ROOT="${WORK}/control-root"
mkdir -p "${CONTROL_ROOT}/tools" "${CONTROL_ROOT}/fixtures"
cp "${TARGET}" "${CONTROL_ROOT}/tools/"
cp "${ROOT_DIR}/tools/spec-declaration.sh" "${CONTROL_ROOT}/tools/"
cp "${ROOT_DIR}/benchmark.json" "${CONTROL_ROOT}/"
jq '(.live_golden_speculative.mtp1) |= with_entries(.value.r2_path |= sub("\\.golden\\.json$"; ".mtp1.golden.json"))' \
  "${TEST_ROOT}/fixtures/nemotron3_5_30b_a3b_mlx_v1_track.json" \
  > "${CONTROL_ROOT}/fixtures/nemotron3_5_30b_a3b_mlx_v1_track.json"
chmod +x "${CONTROL_ROOT}/tools/nemotron35-30b-a3b-measure-and-score.sh" \
  "${CONTROL_ROOT}/tools/spec-declaration.sh"

record="${WORK}/control-golden-symlink.record"
out="$(run_preflight "${WORK}/private/goldens" "${WORK}/private" "${record}" "${SPEC_MANIFEST}" \
  "${CONTROL_ROOT}/tools/nemotron35-30b-a3b-measure-and-score.sh" 2>&1)"; rc=$?
if [[ "${rc}" -eq 1 && "${out}" == *"--control-golden path ${outside_target_real} is outside canonical MLXFAST_PRIVATE_DIR"* && ! -e "${record}" ]]; then
  pass "a --control-golden final-file symlink cannot escape MLXFAST_PRIVATE_DIR"
else
  fail "an escaping --control-golden final-file symlink was not refused before benchd (rc ${rc}): ${out}"
fi

if [[ "${fails}" -eq 0 ]]; then
  printf 'test-private-golden-dir: all cases pass\n'
  exit 0
fi
printf 'test-private-golden-dir: %s failed\n' "${fails}" >&2
exit 1
