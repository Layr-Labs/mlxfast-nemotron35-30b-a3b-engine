#!/usr/bin/env bash
#
# spec-declaration.sh -- the SINGLE trusted source that reads the participant's
# speculative-decode DECLARATION and derives the draft depth the ranked run
# requests from it. tools/nemotron35-30b-a3b-measure-and-score.sh resolves the
# depth THROUGH this one script, so the fail-closed validation and the
# derivation can never drift between the arm gate and the benchd request.
#
# NOT AN EDITABLE PATH. The DECLARATION (mtp-head.manifest.json) is editable --
# a submission EXPRESSES a value there -- but this DERIVATION is trusted, so a
# submission cannot rewrite how the value is interpreted or relax the envelope.
#
# HOW THE DEPTH REACHES THE ENGINE ON THIS TRACK. The runtime worker is
# in-process, so the draft depth rides the Engine Protocol wire per request:
# benchd sends the candidate spec {"mode":"mtp","mtp":{"depth":N}} on
# free_decode_begin (`benchd iterate --mtp-depth N`), and the engine echoes
# `effective_spec`, which the sealed score carries. With no declaration, or
# with `enabled: false`, benchd sends no spec and the engine runs SERIAL
# (depth 0) -- the baseline validation's leg. The same declaration shape as
# the CUDA sibling track (tools/spec-declaration.sh there), so a participant
# moves between the two tracks with one file.
#
# DECLARATION SHAPE (mtp-head.manifest.json, optional key):
#
#   "spec": { "enabled": true, "num_speculative_tokens": N }
#
#   enabled=false or N=0  => serial (no spec on the wire)
#   enabled=true, N in the contract's mtp_head.permitted_draft_depths => depth N
#   anything else          => REFUSED here, before any engine spawns
#
# VERBS
#   speculative  prints 1 when a depth is requested, else 0
#   draft-len    prints the requested depth (0 when serial)
#   describe     prints "serial" or "mtpN"
#   validate     exits 0 when the declaration is well-formed and in-envelope
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
MANIFEST="${SPEC_DECLARATION_MANIFEST:-${REPO_ROOT}/mtp-head.manifest.json}"
CONTRACT="${SPEC_DECLARATION_CONTRACT:-${REPO_ROOT}/fixtures/nemotron3_5_30b_a3b_mlx_v1_track.json}"

# The structural sanity ceiling for a declared draft length (a8/David pin: the
# knob accepts a declared integer 0..8, not a hardcoded value). The contract's
# permitted_draft_depths is the tighter, AUTHORITATIVE set enforced below when
# the value is enabled; this is the outer type/range guard around it.
SPEC_MAX_TOKENS=8

fail() {
  echo "spec-declaration.sh: REFUSING -- $*" >&2
  exit 1
}

command -v jq >/dev/null 2>&1 || fail "jq is required to read the declaration"

# --- read the declaration ---------------------------------------------------
# An absent manifest OR an absent `spec` block is SERIAL: the pure no-op. It is
# resolved WITHOUT requiring the file to exist, so the stock-repo derivation
# never depends on a spec block being present.
enabled="false"
raw_tokens="0"
if [[ -f "${MANIFEST}" ]]; then
  jq -e . >/dev/null 2>&1 < "${MANIFEST}" || fail "mtp-head.manifest.json is not valid JSON"
  if [[ "$(jq -r 'has("spec")' "${MANIFEST}")" == "true" ]]; then
    [[ "$(jq -r '.spec | type' "${MANIFEST}")" == "object" ]] \
      || fail "the \"spec\" declaration must be an object with keys {enabled, num_speculative_tokens}"
    # A typo'd key must not silently read as its default: config drift is a
    # refusal, matching the sibling's rejectUnknownSpecKeys.
    unknown="$(jq -r '.spec | keys[] | select(. != "enabled" and . != "num_speculative_tokens")' "${MANIFEST}")"
    [[ -z "${unknown}" ]] \
      || fail "the \"spec\" declaration carries unknown key(s): $(printf '%s' "${unknown}" | tr '\n' ' '); allowed keys are enabled, num_speculative_tokens"
    if [[ "$(jq -r '.spec | has("enabled")' "${MANIFEST}")" == "true" ]]; then
      [[ "$(jq -r '.spec.enabled | type' "${MANIFEST}")" == "boolean" ]] \
        || fail "spec.enabled must be a boolean (got $(jq -r '.spec.enabled | type' "${MANIFEST}"))"
      enabled="$(jq -r '.spec.enabled' "${MANIFEST}")"
    fi
    if [[ "$(jq -r '.spec | has("num_speculative_tokens")' "${MANIFEST}")" == "true" ]]; then
      [[ "$(jq -r '.spec.num_speculative_tokens | type' "${MANIFEST}")" == "number" ]] \
        || fail "spec.num_speculative_tokens must be an integer (got $(jq -r '.spec.num_speculative_tokens | type' "${MANIFEST}"))"
      raw_tokens="$(jq -r '.spec.num_speculative_tokens' "${MANIFEST}")"
    fi
  fi
fi

# integer + structural 0..SPEC_MAX_TOKENS ceiling
printf '%s' "${raw_tokens}" | grep -Eq '^-?[0-9]+$' \
  || fail "spec.num_speculative_tokens must be an integer (got '${raw_tokens}')"
if (( raw_tokens < 0 || raw_tokens > SPEC_MAX_TOKENS )); then
  fail "spec.num_speculative_tokens=${raw_tokens} is outside the permitted range 0..${SPEC_MAX_TOKENS}"
fi

# --- derive the effective serve spec ----------------------------------------
# enabled:false OR num_speculative_tokens 0 => serial (the no-op). enabled:true
# with N>0 must be a CONTRACT-permitted depth.
spec="0"
draft="0"
if [[ "${enabled}" == "true" && "${raw_tokens}" -gt 0 ]]; then
  # permitted_draft_depths is the fixture's authority (the same envelope
  # harness/protocol-adapter/src/ds4_backend.rs mirrors as MTP_MIN_DEPTH..=
  # MTP_MAX_DEPTH). Enforce membership so the serve can never boot a depth the
  # scored envelope would refuse. When the contract declares no such set, the
  # structural 0..SPEC_MAX_TOKENS ceiling above stands alone.
  if jq -e '.mtp_head.permitted_draft_depths | arrays' >/dev/null 2>&1 < "${CONTRACT}"; then
    if [[ "$(jq -r --argjson n "${raw_tokens}" '(.mtp_head.permitted_draft_depths | index($n)) != null' "${CONTRACT}")" != "true" ]]; then
      permitted="$(jq -r '.mtp_head.permitted_draft_depths | map(tostring) | join(", ")' "${CONTRACT}")"
      fail "spec.num_speculative_tokens=${raw_tokens} is not a contract-permitted draft depth (permitted_draft_depths: ${permitted})"
    fi
  fi
  spec="1"
  draft="${raw_tokens}"
fi

case "${1:-}" in
  speculative) echo "${spec}" ;;
  draft-len)   echo "${draft}" ;;
  describe)    if [[ "${spec}" == "1" ]]; then echo "mtp${draft}"; else echo "serial"; fi ;;
  validate)    : ;;  # validation already ran above; a clean exit means valid
  *)
    echo "spec-declaration.sh: usage: spec-declaration.sh {speculative|draft-len|describe|validate}" >&2
    exit 2
    ;;
esac
