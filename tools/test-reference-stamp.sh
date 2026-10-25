#!/usr/bin/env bash
# Native tests for the setup.sh reference checkpoint verification stamp.
# No model, GPU, network, root, or sudo is used.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
SETUP="${ROOT_DIR}/setup.sh"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/test-reference-stamp.XXXXXX")"
FUNCTIONS="${WORK}/setup-functions.sh"
fails=0

cleanup() {
  rm -rf "${WORK}"
}
trap cleanup EXIT

pass() { printf 'test-reference-stamp: PASS %s\n' "$1"; }
fail() { printf 'test-reference-stamp: FAIL %s\n' "$1" >&2; fails=$((fails + 1)); }

# setup.sh performs provisioning at top level, so source only the production
# functions this hermetic test exercises and their direct helpers.
{
  sed -n '/^reference_manifest_entry() {$/,/^}$/p' "${SETUP}"
  sed -n '/^stat_value() {$/,/^}$/p' "${SETUP}"
  sed -n '/^file_metadata_signature() {$/,/^}$/p' "${SETUP}"
  sed -n '/^reference_cache_lock_matches_manifest_snapshot() {$/,/^}$/p' "${SETUP}"
} > "${FUNCTIONS}"
# shellcheck source=/dev/null
source "${FUNCTIONS}"

REFERENCE_MODEL_REPO="test/reference-model"
REFERENCE_REVISION="0123456789abcdef"

signature_field() {
  local signature="$1"
  local field="$2"
  printf '%s\n' "${signature}" | cut -d: -f "${field}"
}

different_number() {
  if [[ "$1" == "0" ]]; then
    printf '1\n'
  else
    printf '0\n'
  fi
}

signature_with_field() {
  local signature="$1"
  local field="$2"
  local replacement="$3"
  local device inode size mtime ctime generation
  IFS=: read -r device inode size mtime ctime generation <<< "${signature}"
  case "${field}" in
    1) device="${replacement}" ;;
    2) inode="${replacement}" ;;
    3) size="${replacement}" ;;
    4) mtime="${replacement}" ;;
    5) ctime="${replacement}" ;;
    6) generation="${replacement}" ;;
    *) return 1 ;;
  esac
  printf '%s:%s:%s:%s:%s:%s\n' \
    "${device}" "${inode}" "${size}" "${mtime}" "${ctime}" "${generation}"
}

new_fixture() {
  local name="$1"
  local first_hash second_hash first_size second_size
  local first_signature second_signature

  CASE_DIR="${WORK}/${name}"
  REFERENCE_DIR="${CASE_DIR}/reference"
  MANIFEST="${CASE_DIR}/manifest.sha256"
  LOCK="${REFERENCE_DIR}/.mlxfast-reference-cache.lock"
  mkdir -p "${REFERENCE_DIR}"
  printf 'alpha\n' > "${REFERENCE_DIR}/first.txt"
  printf 'bravo!\n' > "${REFERENCE_DIR}/second.txt"

  first_hash="$(shasum -a 256 "${REFERENCE_DIR}/first.txt" | awk '{print $1}')"
  second_hash="$(shasum -a 256 "${REFERENCE_DIR}/second.txt" | awk '{print $1}')"
  first_size="$(wc -c < "${REFERENCE_DIR}/first.txt" | tr -d ' ')"
  second_size="$(wc -c < "${REFERENCE_DIR}/second.txt" | tr -d ' ')"
  first_signature="$(file_metadata_signature "${REFERENCE_DIR}/first.txt")"
  second_signature="$(file_metadata_signature "${REFERENCE_DIR}/second.txt")"

  printf '%s %s %s\n' "${first_hash}" "${first_size}" first.txt > "${MANIFEST}"
  printf '%s %s %s\n' "${second_hash}" "${second_size}" second.txt >> "${MANIFEST}"
  MANIFEST_HASH="$(shasum -a 256 "${MANIFEST}" | awk '{print $1}')"
  FILE_COUNT=2
  BYTE_COUNT=$((first_size + second_size))

  {
    printf 'version=3\n'
    printf 'model_repo=%s\n' "${REFERENCE_MODEL_REPO}"
    printf 'revision=%s\n' "${REFERENCE_REVISION}"
    printf 'manifest_sha256=%s\n' "${MANIFEST_HASH}"
    printf 'file_count=%s\n' "${FILE_COUNT}"
    printf 'byte_count=%s\n' "${BYTE_COUNT}"
    printf 'verified_at=2026-10-25T00:00:00Z\n'
    printf '%s\n' '--files--'
    printf 'first.txt\t%s\t%s\n' "${first_size}" "${first_signature}"
    printf 'second.txt\t%s\t%s\n' "${second_size}" "${second_signature}"
  } > "${LOCK}"
}

stamp_signature() {
  awk -F '\t' '$1 == "first.txt" { print $3; exit }' "${LOCK}"
}

replace_stamp_signature() {
  local signature="$1"
  local replacement="${LOCK}.replacement"
  awk -F '\t' -v OFS='\t' -v signature="${signature}" \
    '$1 == "first.txt" { $3 = signature } { print }' "${LOCK}" > "${replacement}"
  mv "${replacement}" "${LOCK}"
}

stamp_matches() {
  reference_cache_lock_matches_manifest_snapshot \
    "${REFERENCE_DIR}" "${LOCK}" "${MANIFEST}" "${MANIFEST_HASH}" \
    "${FILE_COUNT}" "${BYTE_COUNT}"
}

expect_accept() {
  local label="$1"
  if stamp_matches; then
    pass "${label}"
  else
    fail "${label}"
  fi
}

expect_reject() {
  local label="$1"
  if stamp_matches; then
    fail "${label}"
  else
    pass "${label}"
  fi
}

mutate_signature_field() {
  local field="$1"
  local signature current replacement
  signature="$(stamp_signature)"
  current="$(signature_field "${signature}" "${field}")"
  replacement="$(different_number "${current}")"
  replace_stamp_signature "$(signature_with_field "${signature}" "${field}" "${replacement}")"
}

mutate_signature_fields() {
  local signature field current replacement
  signature="$(stamp_signature)"
  for field in "$@"; do
    current="$(signature_field "${signature}" "${field}")"
    replacement="$(different_number "${current}")"
    signature="$(signature_with_field "${signature}" "${field}" "${replacement}")"
  done
  replace_stamp_signature "${signature}"
}

new_fixture exact
expect_accept '1 exact stamp is accepted'

new_fixture device
mutate_signature_field 1
expect_accept '2 device-only difference is accepted'

new_fixture inode
mutate_signature_field 2
expect_reject '3 inode difference is rejected'

new_fixture size
mutate_signature_field 3
expect_reject '4 signature size difference is rejected'

new_fixture mtime
mutate_signature_field 4
expect_reject '5 mtime difference is rejected'

new_fixture ctime
mutate_signature_field 5
expect_reject '6 ctime difference is rejected'

new_fixture generation
mutate_signature_field 6
expect_reject '7 generation difference is rejected'

new_fixture invalid-device
signature="$(stamp_signature)"
replace_stamp_signature "$(signature_with_field "${signature}" 1 '')"
empty_rejected=0
stamp_matches || empty_rejected=1
new_fixture invalid-device-nondigits
signature="$(stamp_signature)"
replace_stamp_signature "$(signature_with_field "${signature}" 1 invalid)"
nondigits_rejected=0
stamp_matches || nondigits_rejected=1
if [[ "${empty_rejected}" == "1" && "${nondigits_rejected}" == "1" ]]; then
  pass '8 empty and non-digit stamp devices are rejected'
else
  fail '8 empty and non-digit stamp devices are rejected'
fi

new_fixture device-inode
mutate_signature_fields 1 2
expect_reject '9 device and inode differences are rejected'

new_fixture device-size
mutate_signature_fields 1 3
expect_reject '10 device and signature size differences are rejected'

new_fixture device-mtime
mutate_signature_fields 1 4
expect_reject '11 device and mtime differences are rejected'

new_fixture device-ctime
mutate_signature_fields 1 5
expect_reject '12 device and ctime differences are rejected'

new_fixture device-generation
mutate_signature_fields 1 6
expect_reject '13 device and generation differences are rejected'

new_fixture content-change
before_signature="$(stamp_signature)"
before_ctime="$(signature_field "${before_signature}" 5)"
before_second="$(date +%s)"
while [[ "$(date +%s)" == "${before_second}" ]]; do
  sleep 0.05
done
printf 'omega\n' > "${REFERENCE_DIR}/first.txt"
after_signature="$(file_metadata_signature "${REFERENCE_DIR}/first.txt")"
after_ctime="$(signature_field "${after_signature}" 5)"
if [[ "$(wc -c < "${REFERENCE_DIR}/first.txt" | tr -d ' ')" != "6" \
    || "${after_ctime}" == "${before_ctime}" ]]; then
  fail '14 fixture rewrite preserves size and changes ctime'
elif stamp_matches; then
  fail '14 file content rewrite is rejected'
else
  pass '14 file content rewrite is rejected'
fi

new_fixture invalid-live-device
LIVE_DEVICE_OVERRIDE=''
file_metadata_signature() {
  local file_path="$1"
  local real_signature
  real_signature="$(stat_value '%d:%i:%z:%Fm:%Fc:%v' '%d:%i:%s:%Y:%Z:0' "${file_path}")" \
    || return 1
  printf '%s:%s\n' "${LIVE_DEVICE_OVERRIDE}" "${real_signature#*:}"
}
empty_live_rejected=0
stamp_matches || empty_live_rejected=1
LIVE_DEVICE_OVERRIDE=invalid
nondigits_live_rejected=0
stamp_matches || nondigits_live_rejected=1
if [[ "${empty_live_rejected}" == "1" && "${nondigits_live_rejected}" == "1" ]]; then
  pass '15 empty and non-digit live devices are rejected'
else
  fail '15 empty and non-digit live devices are rejected'
fi

if [[ "${fails}" -ne 0 ]]; then
  printf 'test-reference-stamp: %d case(s) failed\n' "${fails}" >&2
  exit 1
fi

echo 'test-reference-stamp: all 15 cases passed'
