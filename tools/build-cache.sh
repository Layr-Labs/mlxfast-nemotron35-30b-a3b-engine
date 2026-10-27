#!/usr/bin/env bash
# build-cache.sh -- keep the ranked job's worker build between dispatches.
#
# WHY. actions/checkout cleans the workspace, so every ranked dispatch rebuilt
# bench-worker and (with Xcode) mlx.metallib from cold: minutes
# of the job that measure nothing. The build is a pure function of the tracked
# sources (the vendored engine fork included), the resolved packages, the
# vendored Metal sources
# and the toolchain, so it is cached under a content key and restored when the
# key matches. A miss builds as before; a hit skips the build and the metallib
# and lets setup.sh verify the restored pair.
#
# Usage:
#   tools/build-cache.sh key             print the content key
#   tools/build-cache.sh restore         restore the artefacts for the key; exit 0 on a hit, 1 on a miss
#   tools/build-cache.sh save            save the artefacts under the key (after a build)
#
# Environment:
#   MLXFAST_BUILD_CACHE_DIR   cache root (default ~/.cache/mlxfast-engine-build); the box owns it
#   MLXFAST_BUILD_CACHE_KEEP  complete key dirs to keep (default 8; integer values below 1 clamp to 1)
#
# The artefacts are verified by sha256 on restore against the MANIFEST the save
# wrote; any mismatch is a miss, never a partial restore.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
cd "${ROOT_DIR}"
log() { printf 'build-cache.sh: %s\n' "$*"; }
die() { printf 'build-cache.sh: %s\n' "$*" >&2; exit 2; }
CACHE_ROOT="${MLXFAST_BUILD_CACHE_DIR:-${HOME}/.cache/mlxfast-engine-build}"
KEEP="${MLXFAST_BUILD_CACHE_KEEP:-8}"
if [[ ! "${KEEP}" =~ ^-?[0-9]+$ ]]; then
  KEEP=8
elif [[ "${KEEP}" == -* ]]; then
  KEEP=1
else
  KEEP="$(printf '%s\n' "${KEEP}" | sed 's/^0*//')"
  [[ -n "${KEEP}" ]] || KEEP=1
fi
ARTEFACTS=(
  ".build-worker/release/bench-worker"
  ".build-worker/release/mlx.metallib"
  ".build-worker/release/mlx.metallib.fingerprint"
)
sha256_of_file() { shasum -a 256 -- "$1" | cut -d ' ' -f 1; }
sha256_of_stdin() { shasum -a 256 | cut -d ' ' -f 1; }
mtime_of_dir() {
  perl -e 'my @s = stat shift; exit 1 unless @s; print $s[9]' -- "$1" 2>/dev/null
}
prune_cache() {
  local entry base mtime i oldest oldest_mtime active remove_count failed_count keep_others kept
  local -a paths mtimes

  if ! find -H "${CACHE_ROOT}" -mindepth 1 -maxdepth 1 -type d -name '*.tmp.*' -mmin +1440 -exec rm -rf -- {} +; then
    log "prune warning: could not remove one or more stale temporary dirs; ignored"
  fi

  paths=(); mtimes=()
  for entry in "${CACHE_ROOT}"/*; do
    [[ -d "${entry}" ]] || continue
    base="${entry##*/}"
    [[ "${base}" =~ ^[0-9a-f]{64}$ ]] || continue
    [[ "${base}" == "${key}" ]] && continue
    if ! mtime="$(mtime_of_dir "${entry}")" || [[ ! "${mtime}" =~ ^[0-9]+$ ]]; then
      log "prune warning: could not read mtime for ${base}; ignored"
      continue
    fi
    paths+=("${entry}")
    mtimes+=("${mtime}")
  done

  active=${#paths[@]}
  if [[ ${#KEEP} -gt 9 ]]; then
    keep_others=${active}
  else
    keep_others=$((KEEP - 1))
  fi
  remove_count=0
  failed_count=0
  while [[ ${active} -gt ${keep_others} ]]; do
    oldest=-1; oldest_mtime=0
    for ((i = 0; i < ${#paths[@]}; i++)); do
      [[ -n "${paths[i]:-}" ]] || continue
      if [[ ${oldest} -lt 0 || ${mtimes[i]} -lt ${oldest_mtime} ]]; then
        oldest=${i}; oldest_mtime=${mtimes[i]}
      fi
    done
    [[ ${oldest} -ge 0 ]] || break
    if rm -rf -- "${paths[oldest]}"; then
      remove_count=$((remove_count + 1))
    else
      log "prune warning: could not remove ${paths[oldest]##*/}; ignored"
      failed_count=$((failed_count + 1))
    fi
    paths[oldest]=''
    active=$((active - 1))
  done
  kept=$((1 + active + failed_count))
  log "pruned ${remove_count} old cache entries (kept ${kept})"
}
content_key() {
  local paths rel blob
  # Vendor/mlx-swift-lm is a VENDORED TREE, so the fork's sources are hashed
  # here like every other tracked source. It used to be a gitlink, which the key
  # carried as one commit id.
  paths=('Sources/*' 'Vendor/mlx-swift/*' 'Vendor/mlx-swift-lm/*' 'Package.swift' 'Package.resolved' 'tools/build-mlx-metallib.sh' 'tools/stage-bench-worker.sh')
  # Hash each complete NUL-delimited index record: mode, blob id, stage and
  # full path. A path modified in the working tree is re-hashed so a dirty
  # tree never keys as its committed state.
  {
    printf 'index\0'
    git ls-files -s -z -- "${paths[@]}"
    printf 'dirty\0'
    while IFS= read -r -d '' rel; do
      [[ -f "${rel}" ]] || die "tracked source ${rel} is missing from the working tree; refusing to key a build on it"
      blob="$(git hash-object -- "${rel}")"
      printf '%s\0%s\0' "${rel}" "${blob}"
    done < <(git diff --name-only -z -- "${paths[@]}")
    printf 'metal\0%s\0' "$(tools/build-mlx-metallib.sh --print-fingerprint 2>/dev/null || echo none)"
    printf 'swift\0%s\0' "$(swift --version 2>/dev/null | head -1)"
    printf 'host\0%s\0' "$(uname -m)"
    # The products embed their build directory (SwiftPM resource bundles resolve
    # mlx.metallib from an absolute path baked in at build time), so a product
    # built under one root does not run under another: run 34030992573 restored
    # a worker built elsewhere and the resident died with "Failed to load the
    # default metallib". The root is therefore part of the key.
    printf 'root\0%s\0' "${ROOT_DIR}"
  } | sha256_of_stdin
}
case "${1:-}" in
  key) content_key ;;
  restore)
    key="$(content_key)"; dir="${CACHE_ROOT}/${key}"
    [[ -f "${dir}/MANIFEST" ]] || { log "miss: no cache for key ${key:0:12}"; exit 1; }
    for a in "${ARTEFACTS[@]}"; do
      want="$(awk -v a="${a}" '$1==a{print $2}' "${dir}/MANIFEST")"
      [[ -n "${want}" && -f "${dir}/${a}" ]] || { log "miss: ${a} absent from the cache for key ${key:0:12}"; exit 1; }
      [[ "$(sha256_of_file "${dir}/${a}")" == "${want}" ]] || { log "miss: ${a} in the cache does not match its manifest; refusing a partial restore"; exit 1; }
    done
    for a in "${ARTEFACTS[@]}"; do mkdir -p "$(dirname "${a}")"; cp -p "${dir}/${a}" "${a}"; done
    touch "${dir}"
    log "hit: restored ${#ARTEFACTS[@]} artefacts for key ${key:0:12}"
    ;;
  save)
    key="$(content_key)"; dir="${CACHE_ROOT}/${key}"; tmp="${dir}.tmp.$$"
    for a in "${ARTEFACTS[@]}"; do [[ -f "${a}" ]] || die "cannot save: ${a} is missing (build first)"; done
    rm -rf "${tmp}"; mkdir -p "${tmp}"
    : > "${tmp}/MANIFEST"
    for a in "${ARTEFACTS[@]}"; do mkdir -p "${tmp}/$(dirname "${a}")"; cp -p "${a}" "${tmp}/${a}"; printf '%s %s\n' "${a}" "$(sha256_of_file "${a}")" >> "${tmp}/MANIFEST"; done
    rm -rf "${dir}"; mv "${tmp}" "${dir}"
    log "saved ${#ARTEFACTS[@]} artefacts under key ${key:0:12} in ${CACHE_ROOT}"
    prune_cache || log "prune warning: unexpected error; ignored"
    ;;
  *) die "usage: tools/build-cache.sh key|restore|save" ;;
esac
