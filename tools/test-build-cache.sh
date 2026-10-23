#!/usr/bin/env bash
# test-build-cache.sh -- prove tools/build-cache.sh without a toolchain build.
#   1. the key is stable across two calls and changes when tracked package or source files change
#      including a full path with spaces and an index mode change
#   2. restore misses on an empty cache; save then restore hits and restores byte-identical artefacts
#   3. a corrupted cached artefact is a miss, and nothing is restored
# Hermetic: runs in a scratch git clone of this repository with stub artefacts.
set -uo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null && pwd -P)"
WORK="$(mktemp -d)"; trap 'rm -rf "${WORK}"' EXIT
fails=0; pass() { printf 'test-build-cache: PASS -- %s\n' "$*"; }; fail() { printf 'test-build-cache: FAIL -- %s\n' "$*" >&2; fails=$((fails + 1)); }
git clone -q --no-hardlinks "${ROOT_DIR}" "${WORK}/repo" || { fail "cannot clone"; exit 1; }
cp "${ROOT_DIR}/tools/build-cache.sh" "${WORK}/repo/tools/build-cache.sh"
cd "${WORK}/repo" || exit 1
export MLXFAST_BUILD_CACHE_DIR="${WORK}/cache"
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
kA="$(tools/build-cache.sh key)"; cp -R "${WORK}/repo" "${WORK}/repo2"; kB="$(cd "${WORK}/repo2" && tools/build-cache.sh key)"
if [[ -n "${kA}" && "${kA}" != "${kB}" ]]; then pass "the same tree under another root keys differently (products embed their build path)"; else fail "two roots share a key: ${kA} ${kB}"; fi
if [[ "${fails}" -eq 0 ]]; then printf 'test-build-cache: all cases pass\n'; exit 0; fi
printf 'test-build-cache: %s failed\n' "${fails}" >&2
exit 1
