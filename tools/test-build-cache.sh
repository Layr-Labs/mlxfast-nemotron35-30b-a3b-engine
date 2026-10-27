#!/usr/bin/env bash
# test-build-cache.sh -- prove tools/build-cache.sh without a toolchain build.
#   1. the key is stable across two calls and changes when tracked package or source files change
#      including a full path with spaces and an index mode change
#   2. restore misses on an empty cache; save then restore hits and restores byte-identical artefacts
#   3. a corrupted cached artefact is a miss, and nothing is restored
#   4. save prunes old complete keys and stale temporary dirs without touching other entries
#   5. restore refreshes the key dir mtime, and odd entries cannot fail a save
#   6. GNU stat behavior, a symlinked cache root, and KEEP=0 are safe
# Hermetic: runs in a scratch git clone of this repository with stub artefacts.
set -uo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
WORK="$(mktemp -d)"; trap 'rm -rf "${WORK}"' EXIT
fails=0; pass() { printf 'test-build-cache: PASS -- %s\n' "$*"; }; fail() { printf 'test-build-cache: FAIL -- %s\n' "$*" >&2; fails=$((fails + 1)); }
git clone -q --no-hardlinks "${ROOT_DIR}" "${WORK}/repo" || { fail "cannot clone"; exit 1; }
cp "${ROOT_DIR}/tools/build-cache.sh" "${WORK}/repo/tools/build-cache.sh"
cd "${WORK}/repo" || exit 1
export MLXFAST_BUILD_CACHE_DIR="${WORK}/cache"
mtime_of_path() { perl -e 'my @s = stat shift; exit 1 unless @s; print $s[9]' -- "$1"; }
k1="$(tools/build-cache.sh key)"; k2="$(tools/build-cache.sh key)"
if [[ -n "${k1}" && "${k1}" == "${k2}" ]]; then pass "the key is stable (${k1:0:12})"; else fail "unstable key: ${k1} vs ${k2}"; fi
printf '\n// touched\n' >> Package.swift; k3="$(tools/build-cache.sh key)"; git checkout -q -- Package.swift
if [[ "${k3}" != "${k1}" ]]; then pass "a tracked source change changes the key"; else fail "the key ignored a source change"; fi
printf '\n// touched\n' >> Sources/MLXFastTransform/Transform.swift; k4="$(tools/build-cache.sh key)"; git checkout -q -- Sources/MLXFastTransform/Transform.swift
if [[ "${k4}" != "${k1}" ]]; then pass "a tracked transform source change still changes the key"; else fail "the key ignored a tracked transform source change"; fi
printf 'same bytes\n' > "Sources/cache name one.swift"
git add "Sources/cache name one.swift"
k_space_one="$(tools/build-cache.sh key)"
git mv "Sources/cache name one.swift" "Sources/cache name two.swift"
k_space_two="$(tools/build-cache.sh key)"
if [[ "${k_space_one}" != "${k_space_two}" ]]; then pass "full tracked paths with spaces affect the key"; else fail "the key truncated a tracked path at its first space"; fi
git reset -q -- "Sources/cache name two.swift"
rm -f "Sources/cache name two.swift"
printf 'same bytes\n' > Sources/cache-mode.swift
git add Sources/cache-mode.swift
git update-index --chmod=-x Sources/cache-mode.swift
k_mode_plain="$(tools/build-cache.sh key)"
git update-index --chmod=+x Sources/cache-mode.swift
k_mode_exec="$(tools/build-cache.sh key)"
if [[ "${k_mode_plain}" != "${k_mode_exec}" ]]; then pass "the tracked file mode affects the key"; else fail "the key ignored a tracked file mode change"; fi
git reset -q -- Sources/cache-mode.swift
rm -f Sources/cache-mode.swift
out="$(tools/build-cache.sh restore 2>&1)"; rc=$?
if [[ "${rc}" -eq 1 && "${out}" == *"miss"* ]]; then pass "an empty cache is a miss"; else fail "empty cache did not miss (rc ${rc}): ${out}"; fi
mkdir -p .build-worker/release
printf 'worker' > .build-worker/release/bench-worker; printf 'lib' > .build-worker/release/mlx.metallib; printf 'mlxfast-metallib-fingerprint-v1 abc\n' > .build-worker/release/mlx.metallib.fingerprint
tools/build-cache.sh save >/dev/null || fail "save failed"
rm -rf .build-worker .build
out="$(tools/build-cache.sh restore 2>&1)"; rc=$?
if [[ "${rc}" -eq 0 && "$(cat .build-worker/release/bench-worker)" == "worker" && -f .build-worker/release/mlx.metallib.fingerprint && ! -e .build/release/mlxfast-swift ]]; then pass "save then restore hits and restores only worker artefacts"; else fail "restore did not reproduce the artefacts (rc ${rc}): ${out}"; fi
printf 'tampered' > "${MLXFAST_BUILD_CACHE_DIR}/${k1}/.build-worker/release/bench-worker"; rm -rf .build-worker .build
out="$(tools/build-cache.sh restore 2>&1)"; rc=$?
if [[ "${rc}" -eq 1 && "${out}" == *"does not match"* && ! -e .build-worker/release/bench-worker ]]; then pass "a corrupted cached artefact is a miss and nothing is restored"; else fail "corruption not refused (rc ${rc}): ${out}"; fi
old1="$(printf '1%.0s' {1..64})"; old2="$(printf '2%.0s' {1..64})"; old3="$(printf '3%.0s' {1..64})"; newest="$(printf '4%.0s' {1..64})"
for fake in "${old1}" "${old2}" "${old3}" "${newest}"; do mkdir -p "${MLXFAST_BUILD_CACHE_DIR}/${fake}"; done
touch -t 202001010001 "${MLXFAST_BUILD_CACHE_DIR}/${old1}"
touch -t 202001010002 "${MLXFAST_BUILD_CACHE_DIR}/${old2}"
touch -t 202001010003 "${MLXFAST_BUILD_CACHE_DIR}/${old3}"
touch -t 202001010004 "${MLXFAST_BUILD_CACHE_DIR}/${newest}"
mkdir -p .build-worker/release
printf 'worker' > .build-worker/release/bench-worker; printf 'lib' > .build-worker/release/mlx.metallib; printf 'mlxfast-metallib-fingerprint-v1 abc\n' > .build-worker/release/mlx.metallib.fingerprint
mkdir -p "${WORK}/gnu-bin"
printf '%s\n' '#!/usr/bin/env bash' "if [[ \"\${1:-}\" == \"-f\" ]]; then printf \"gnu stat file-system output\\\\n\"; exit 0; fi" "exec /usr/bin/stat \"\$@\"" > "${WORK}/gnu-bin/stat"
chmod +x "${WORK}/gnu-bin/stat"
out="$(PATH="${WORK}/gnu-bin:${PATH}" MLXFAST_BUILD_CACHE_KEEP=2 tools/build-cache.sh save 2>&1)"; rc=$?
if [[ "${rc}" -eq 0 && -d "${MLXFAST_BUILD_CACHE_DIR}/${k1}" && -d "${MLXFAST_BUILD_CACHE_DIR}/${newest}" && ! -e "${MLXFAST_BUILD_CACHE_DIR}/${old1}" && ! -e "${MLXFAST_BUILD_CACHE_DIR}/${old2}" && ! -e "${MLXFAST_BUILD_CACHE_DIR}/${old3}" && "${out}" == *"pruned 3 old cache entries (kept 2)"* ]]; then pass "KEEP=2 keeps the current key and newest other key"; else fail "bounded prune kept the wrong keys (rc ${rc}): ${out}"; fi
if [[ "${rc}" -eq 0 && "${out}" != *"gnu stat file-system output"* ]]; then pass "prune orders entries by mtime with GNU stat behavior on PATH"; else fail "GNU stat behavior broke prune ordering (rc ${rc}): ${out}"; fi
fresh_tmp="${MLXFAST_BUILD_CACHE_DIR}/${k1}.tmp.fresh"; stale_tmp="${MLXFAST_BUILD_CACHE_DIR}/${k1}.tmp.stale"
mkdir -p "${fresh_tmp}" "${stale_tmp}"; touch -t 202001010000 "${stale_tmp}"
mkdir -p "${MLXFAST_BUILD_CACHE_DIR}/not-a-key"; printf 'keep\n' > "${MLXFAST_BUILD_CACHE_DIR}/not-a-key/value"
printf 'keep\n' > "${MLXFAST_BUILD_CACHE_DIR}/ordinary-file"
MLXFAST_BUILD_CACHE_KEEP=2 tools/build-cache.sh save >/dev/null 2>&1; rc=$?
if [[ "${rc}" -eq 0 && -d "${fresh_tmp}" && ! -e "${stale_tmp}" ]]; then pass "save keeps fresh temporary dirs and removes stale ones"; else fail "temporary dir pruning was wrong (rc ${rc})"; fi
if [[ "$(cat "${MLXFAST_BUILD_CACHE_DIR}/not-a-key/value" 2>/dev/null)" == "keep" && "$(cat "${MLXFAST_BUILD_CACHE_DIR}/ordinary-file" 2>/dev/null)" == "keep" ]]; then pass "save does not touch non-key dirs or files"; else fail "save changed a non-key entry"; fi
symlink_target="${WORK}/symlink-cache-target"; symlink_root="${WORK}/symlink-cache"
stale_symlink_tmp="${symlink_target}/${k1}.tmp.stale"
mkdir -p "${stale_symlink_tmp}"; touch -t 202001010000 "${stale_symlink_tmp}"; ln -s "${symlink_target}" "${symlink_root}"
MLXFAST_BUILD_CACHE_DIR="${symlink_root}" MLXFAST_BUILD_CACHE_KEEP=1 tools/build-cache.sh save >/dev/null 2>&1; rc=$?
if [[ "${rc}" -eq 0 && ! -e "${stale_symlink_tmp}" && -d "${symlink_target}/${k1}" ]]; then pass "a symlinked cache root removes a stale temporary dir"; else fail "symlinked cache root pruning failed (rc ${rc})"; fi
touch -t 202001010000 "${MLXFAST_BUILD_CACHE_DIR}/${k1}"; before="$(mtime_of_path "${MLXFAST_BUILD_CACHE_DIR}/${k1}")"
tools/build-cache.sh restore >/dev/null 2>&1; rc=$?; after="$(mtime_of_path "${MLXFAST_BUILD_CACHE_DIR}/${k1}")"
if [[ "${rc}" -eq 0 && "${after}" -gt "${before}" ]]; then pass "restore hit refreshes the cache dir mtime"; else fail "restore did not refresh mtime (rc ${rc}): ${before} -> ${after}"; fi
odd="$(printf '5%.0s' {1..64})"; mkdir -p "${MLXFAST_BUILD_CACHE_DIR}/${odd}"; chmod 000 "${MLXFAST_BUILD_CACHE_DIR}/${odd}"
MLXFAST_BUILD_CACHE_KEEP=invalid tools/build-cache.sh save >/dev/null 2>&1; rc=$?
chmod 700 "${MLXFAST_BUILD_CACHE_DIR}/${odd}" 2>/dev/null || true
if [[ "${rc}" -eq 0 ]]; then pass "an odd entry and invalid KEEP do not fail save"; else fail "an odd entry made save fail (rc ${rc})"; fi
MLXFAST_BUILD_CACHE_KEEP=0 tools/build-cache.sh save >/dev/null 2>&1; rc=$?
if [[ "${rc}" -eq 0 && -d "${MLXFAST_BUILD_CACHE_DIR}/${k1}" && ! -e "${MLXFAST_BUILD_CACHE_DIR}/${newest}" && ! -e "${MLXFAST_BUILD_CACHE_DIR}/${odd}" ]]; then pass "KEEP=0 clamps to 1 and keeps only the current key"; else fail "KEEP=0 did not keep only the current key (rc ${rc})"; fi
kA="$(tools/build-cache.sh key)"; cp -R "${WORK}/repo" "${WORK}/repo2"; kB="$(cd "${WORK}/repo2" && tools/build-cache.sh key)"
if [[ -n "${kA}" && "${kA}" != "${kB}" ]]; then pass "the same tree under another root keys differently (products embed their build path)"; else fail "two roots share a key: ${kA} ${kB}"; fi
if [[ "${fails}" -eq 0 ]]; then printf 'test-build-cache: all cases pass\n'; exit 0; fi
printf 'test-build-cache: %s failed\n' "${fails}" >&2
exit 1
