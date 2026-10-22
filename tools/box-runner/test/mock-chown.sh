#!/usr/bin/env bash
# Hold the privileged test after the real chown and before supervisor uses su.
set -u
: "${REAL_CHOWN:?}"
: "${MOCK_JIT_PATH_FILE:?}"
: "${MOCK_CHOWN_RELEASE_FILE:?}"

"${REAL_CHOWN}" "$@"
printf '%s\n' "${2:?}" > "${MOCK_JIT_PATH_FILE}"
waited=0
while [ ! -e "${MOCK_CHOWN_RELEASE_FILE}" ] && [ "${waited}" -lt 300 ]; do
  sleep 0.1
  waited=$((waited + 1))
done
