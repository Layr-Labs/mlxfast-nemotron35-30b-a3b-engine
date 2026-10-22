#!/usr/bin/env bash
# Unit test for tools/author-prompts.py. Offline and hermetic: a tiny WordLevel
# tokenizer, a tiny chat template, and a synthetic fixture and manifest.
#
# Usage: tools/test-author-prompts.sh
# Exit: 0 all cases pass (or dependencies are absent), 1 a case failed
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TOOL="${SCRIPT_DIR}/tools/author-prompts.py"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

missing="$(
  python3 - <<'PY'
import importlib.util

print(" ".join(name for name in ("tokenizers", "jinja2")
               if importlib.util.find_spec(name) is None))
PY
)"
if [[ -n "${missing}" ]]; then
  echo "SKIP: tokenizers/jinja2 not installed"
  exit 0
fi
python3 -c 'import tokenizers, jinja2'

SNAPSHOT="${WORK}/snapshot"
FIXTURE="${WORK}/fixture.json"
MANIFEST="${WORK}/manifest.sha256"
mkdir -p "${SNAPSHOT}" "${WORK}/out"

python3 - "${SNAPSHOT}" "${FIXTURE}" "${MANIFEST}" <<'PY'
import hashlib
import json
import os
import sys

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

snapshot, fixture_path, manifest_path = sys.argv[1:]
tok = Tokenizer(WordLevel(
    vocab={"[UNK]": 0, "one": 1, "two": 2, "three": 3, "four": 4},
    unk_token="[UNK]",
))
tok.pre_tokenizer = Whitespace()
tok.save(os.path.join(snapshot, "tokenizer.json"))
template = (
    "{% if add_generation_prompt and enable_thinking is sameas false %}"
    "{{ messages[0]['content'] }}"
    "{% else %}four{% endif %}"
)
open(os.path.join(snapshot, "chat_template.jinja"), "w").write(template)

with open(manifest_path, "w") as manifest:
    for name in ("tokenizer.json", "chat_template.jinja"):
        data = open(os.path.join(snapshot, name), "rb").read()
        manifest.write(f"{hashlib.sha256(data).hexdigest()} {len(data)} {name}\n")

fixture = {
    "seed_tokens": 3,
    "target": {
        "vocab_size": 5,
        "manifest_path": manifest_path,
    },
}
json.dump(fixture, open(fixture_path, "w"))
PY

printf 'one two three four\n' > "${WORK}/raw.txt"
printf 'one two\n' > "${WORK}/short.txt"
printf 'one two three' > "${WORK}/chat-good.txt"
printf 'one two' > "${WORK}/chat-short.txt"
printf 'one two three four' > "${WORK}/chat-long.txt"

python3 "${TOOL}" --snapshot "${SNAPSHOT}" --fixture "${FIXTURE}" \
  --out "${WORK}/out" raw="${WORK}/raw.txt" > "${WORK}/raw.log"
python3 - "${WORK}/out/raw.tokens.json" <<'PY'
import json
import sys

assert json.load(open(sys.argv[1])) == [1, 2, 3]
PY
echo "ok: raw text writes exactly the seed ids"

if python3 "${TOOL}" --snapshot "${SNAPSHOT}" --fixture "${FIXTURE}" \
  --out "${WORK}/out" short="${WORK}/short.txt" > /dev/null 2> "${WORK}/short.err"; then
  echo "FAIL: short raw text was accepted" >&2
  exit 1
elif grep -q "2 tokens; the seed needs 3" "${WORK}/short.err"; then
  echo "ok: short raw text is refused"
else
  echo "FAIL: short raw text was refused for the wrong reason" >&2
  cat "${WORK}/short.err" >&2
  exit 1
fi

python3 "${TOOL}" --snapshot "${SNAPSHOT}" --fixture "${FIXTURE}" \
  --out "${WORK}/out" --chat chat="${WORK}/chat-good.txt" > "${WORK}/chat.log"
python3 - "${WORK}/out/chat.tokens.json" <<'PY'
import json
import sys

assert json.load(open(sys.argv[1])) == [1, 2, 3]
PY
echo "ok: exact-length chat rendering writes the seed ids"

if python3 "${TOOL}" --snapshot "${SNAPSHOT}" --fixture "${FIXTURE}" \
  --out "${WORK}/out" --chat bad-chat="${WORK}/chat-short.txt" \
  > /dev/null 2> "${WORK}/chat-short.err"; then
  echo "FAIL: wrong-length chat rendering was accepted" >&2
  exit 1
elif grep -q "the chat rendering is 2 tokens; the seed is exactly 3" "${WORK}/chat-short.err"; then
  echo "ok: wrong-length chat rendering is refused"
else
  echo "FAIL: wrong-length chat rendering was refused for the wrong reason" >&2
  cat "${WORK}/chat-short.err" >&2
  exit 1
fi

if python3 "${TOOL}" --snapshot "${SNAPSHOT}" --fixture "${FIXTURE}" \
  --out "${WORK}/out" --chat long-chat="${WORK}/chat-long.txt" \
  > /dev/null 2> "${WORK}/chat-long.err"; then
  echo "FAIL: an overlong chat rendering was accepted (trimmed?)" >&2
  exit 1
elif grep -q "the chat rendering is 4 tokens; the seed is exactly 3" "${WORK}/chat-long.err"; then
  echo "ok: overlong chat rendering is refused, never trimmed"
else
  echo "FAIL: overlong chat rendering was refused for the wrong reason" >&2
  cat "${WORK}/chat-long.err" >&2
  exit 1
fi

cp "${MANIFEST}" "${WORK}/bad-sha.sha256"
python3 - "${WORK}/bad-sha.sha256" <<'PY'
import sys

path = sys.argv[1]
lines = open(path).readlines()
parts = lines[0].split()
parts[0] = "0" * 64
lines[0] = " ".join(parts) + "\n"
open(path, "w").writelines(lines)
PY
python3 - "${FIXTURE}" "${WORK}/bad-sha-fixture.json" "${WORK}/bad-sha.sha256" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1]))
data["target"]["manifest_path"] = sys.argv[3]
json.dump(data, open(sys.argv[2], "w"))
PY
if python3 "${TOOL}" --snapshot "${SNAPSHOT}" --fixture "${WORK}/bad-sha-fixture.json" \
  --out "${WORK}/out" bad-sha="${WORK}/raw.txt" > /dev/null 2> "${WORK}/bad-sha.err"; then
  echo "FAIL: tokenizer with a wrong sha256 was accepted" >&2
  exit 1
elif grep -q "tokenizer.json is sha256" "${WORK}/bad-sha.err"; then
  echo "ok: tokenizer with a wrong sha256 is refused"
else
  echo "FAIL: tokenizer with a wrong sha256 was refused for the wrong reason" >&2
  cat "${WORK}/bad-sha.err" >&2
  exit 1
fi

cp "${MANIFEST}" "${WORK}/bad-bytes.sha256"
python3 - "${WORK}/bad-bytes.sha256" <<'PY'
import sys

path = sys.argv[1]
lines = open(path).readlines()
parts = lines[0].split()
parts[1] = str(int(parts[1]) + 1)
lines[0] = " ".join(parts) + "\n"
open(path, "w").writelines(lines)
PY
python3 - "${FIXTURE}" "${WORK}/bad-bytes-fixture.json" "${WORK}/bad-bytes.sha256" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1]))
data["target"]["manifest_path"] = sys.argv[3]
json.dump(data, open(sys.argv[2], "w"))
PY
if python3 "${TOOL}" --snapshot "${SNAPSHOT}" --fixture "${WORK}/bad-bytes-fixture.json" \
  --out "${WORK}/out" bad-bytes="${WORK}/raw.txt" > /dev/null 2> "${WORK}/bad-bytes.err"; then
  echo "FAIL: tokenizer with a wrong byte count was accepted" >&2
  exit 1
elif grep -q "tokenizer.json is .* bytes, the manifest pins" "${WORK}/bad-bytes.err"; then
  echo "ok: tokenizer with a wrong byte count is refused"
else
  echo "FAIL: tokenizer with a wrong byte count was refused for the wrong reason" >&2
  cat "${WORK}/bad-bytes.err" >&2
  exit 1
fi

echo
echo "test-author-prompts.sh: all cases passed"
