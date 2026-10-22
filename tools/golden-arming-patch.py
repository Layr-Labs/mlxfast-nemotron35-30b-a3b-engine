#!/usr/bin/env python3
"""Write the arming patch for fixtures/nemotron3_5_30b_a3b_mlx_v1_track.json.

The organizer records the track's goldens on the track's box. This tool reads
a directory of those recorded files and writes the fixture values that arm the
track. It computes every {sha256, bytes} pin itself. It reads the files; it
never prints their contents.

The directory holds these files, and nothing else:

  <name>.golden.json              the timed-pool tapes, recorded serial, at
                                  least one and all distinct. Each --live
                                  names one of them.
  <live>.mtpN.golden.json         one per-depth oracle for every live golden
                                  and every depth in
                                  mtp_head.permitted_draft_depths.
  public_longcopy_gate_english_1024_256.json
  public_longcopy_gate_english_1024_1024.json
                                  the two public captures for local-iterate and
                                  local-submit. They are not uploads: they ship
                                  in git under correctness_prompts/. The tool
                                  checks them and prints where to copy them.

THE PER-DEPTH SHAPE IS DECIDED BY THE BYTES. A per-depth oracle can differ from
the serial tape (a speculative round verifies at M > 1, and MLX dispatches a
different kernel there), and it can equal it. Every live_golden_speculative key
gets its own pin, but keys whose bytes are identical share ONE R2 object: a key
whose oracle equals the serial live golden points at the live golden's pool
key, and a key whose oracle equals an earlier key's points at that key's
object. The organizer uploads each distinct object once. The upload list goes
to stderr, followed by one `ship:` line per public capture naming the
repository path to copy it to.

Usage:
  python3 tools/golden-arming-patch.py --dir DIR --live NAME [--live NAME ...]
  python3 tools/golden-arming-patch.py --dir DIR --live NAME [--live NAME ...] --apply

Without --apply the tool prints the patch (JSON, values only) on stdout. With
--apply it writes the values into the fixture in place.
Exit: 0 on success, 1 on any refusal (printed with a REFUSE prefix).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACT = os.path.join(ROOT, "fixtures", "nemotron3_5_30b_a3b_mlx_v1_track.json")
PUBLIC_CAPTURES = {
    "local_iterate": (
        "public_longcopy_gate_english_1024_256.json",
        "correctness_prompts/public_longcopy_gate_english_1024_256.json",
    ),
    "local_submit": (
        "public_longcopy_gate_english_1024_1024.json",
        "correctness_prompts/public_longcopy_gate_english_1024_1024.json",
    ),
}
FORBIDDEN_BENCHMARK_KEYS = (
    "baseline_prefill_seconds_per_token",
    "baseline_decode_seconds_per_token",
)


class Refusal(Exception):
    pass


def pin(path: str) -> dict:
    """The {sha256, bytes} pin of one recorded file, after a shape check."""
    with open(path, "rb") as fh:
        data = fh.read()
    try:
        document = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Refusal(f"{os.path.basename(path)} is not JSON: {exc.__class__.__name__}")
    benchmark = document.get("benchmark") if isinstance(document, dict) else None
    if isinstance(benchmark, dict):
        stored = [key for key in FORBIDDEN_BENCHMARK_KEYS if key in benchmark]
        if stored:
            raise Refusal(
                f"{os.path.basename(path)} stores a baseline pair ({', '.join(stored)}); "
                "the pair is measured live on the box and never stored"
            )
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def case_token_fingerprints(path: str) -> dict[str, set[str]]:
    """Canonical hashes of the parsed token fields, without exposing tokens."""
    with open(path, encoding="utf-8") as fh:
        document = json.load(fh)
    fingerprints = {"prompt_tokens": set(), "expected_tokens": set()}
    cases = document.get("cases", []) if isinstance(document, dict) else []
    for case in cases:
        if not isinstance(case, dict):
            continue
        for field in fingerprints:
            if field in case:
                canonical = json.dumps(
                    case[field], sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ).encode("utf-8")
                fingerprints[field].add(hashlib.sha256(canonical).hexdigest())
    return fingerprints


def build_patch(contract: dict, directory: str, lives: list[str]) -> tuple[dict, list[str], list[str]]:
    track = contract["track_id"]
    prefix = f"correctness_prompts/{track}/"
    public_names = {name for name, _ in PUBLIC_CAPTURES.values()}
    names = sorted(
        n for n in os.listdir(directory)
        if n.endswith(".golden.json") or n in public_names
    )
    others = sorted(
        n for n in os.listdir(directory)
        if not n.endswith(".golden.json") and n not in public_names
    )
    if others:
        raise Refusal(
            "the directory holds files that are not *.golden.json or a public capture: "
            f"{', '.join(others)}"
        )

    public_roles = {name: role for role, (name, _) in PUBLIC_CAPTURES.items()}
    depth_keys = [f"mtp{d}" for d in contract["mtp_head"]["permitted_draft_depths"]]
    depth_re = re.compile(r"^(?P<name>.+)\.(?P<key>mtp[0-9]+)\.golden\.json$")

    pool_names, depth_files, public_files = [], {}, {}
    for name in names:
        match = depth_re.match(name)
        if name in public_roles:
            public_files[public_roles[name]] = name
        elif match:
            if match["name"] not in lives:
                raise Refusal(f"{name} is a per-depth oracle for {match['name']}, which is not a live golden")
            if match["key"] not in depth_keys:
                raise Refusal(f"{name} is for {match['key']}, which no decoder declares")
            depth_files[(match["key"], match["name"])] = name
        else:
            pool_names.append(name[: -len(".golden.json")])

    if not pool_names:
        raise Refusal("found no timed-pool tape")
    absent = [live for live in lives if live not in pool_names]
    if absent:
        raise Refusal(f"--live {', '.join(absent)} is not one of the timed-pool tapes")
    if len(set(lives)) != len(lives):
        raise Refusal("a live golden is named twice")
    official_pairs = contract["official_pairs"]
    if official_pairs % len(lives) != 0:
        raise Refusal(
            f"official_pairs {official_pairs} is not divisible by the "
            f"{len(lives)} live goldens; every job must measure every live golden"
        )
    missing = [f"{key} ({live})" for key in depth_keys for live in lives if (key, live) not in depth_files]
    if missing:
        raise Refusal(f"no per-depth oracle for: {', '.join(missing)} (the measure script refuses a declared depth without one)")
    missing = [role for role in PUBLIC_CAPTURES if role not in public_files]
    if missing:
        raise Refusal(f"no public capture for: {', '.join(missing)}")

    pool = []
    live_entries = {}
    for name in pool_names:
        entry = {"r2_path": f"{prefix}{name}.golden.json", **pin(os.path.join(directory, f"{name}.golden.json"))}
        pool.append(entry)
        if name in lives:
            live_entries[name] = entry
    if len({entry["sha256"] for entry in pool}) != len(pool):
        raise Refusal("two timed-pool tapes have the same bytes; pool entries must be distinct")

    # Content addressing: identical bytes share one R2 object.
    object_by_sha = {live_entries[live]["sha256"]: live_entries[live]["r2_path"] for live in lives}
    speculative = {}
    for key in depth_keys:
        speculative[key] = {}
        for live in lives:
            own = pin(os.path.join(directory, depth_files[(key, live)]))
            r2_path = object_by_sha.setdefault(own["sha256"], f"{prefix}{depth_files[(key, live)]}")
            speculative[key][live] = {"r2_path": r2_path, **own}
    oracle_entries = [entry for per_live in speculative.values() for entry in per_live.values()]

    hidden_shas = {entry["sha256"] for entry in pool} | {entry["sha256"] for entry in oracle_entries}
    hidden_tokens = {"prompt_tokens": set(), "expected_tokens": set()}
    hidden_files = [f"{name}.golden.json" for name in pool_names] + list(depth_files.values())
    for name in hidden_files:
        fingerprints = case_token_fingerprints(os.path.join(directory, name))
        for field in hidden_tokens:
            hidden_tokens[field].update(fingerprints[field])
    ships = []
    for role, (_, repository_path) in PUBLIC_CAPTURES.items():
        public_path = os.path.join(directory, public_files[role])
        own = pin(public_path)
        if own["sha256"] in hidden_shas:
            raise Refusal(f"the public capture {role} has the same bytes as a hidden tape or oracle; shipping it would publish that tape")
        public_tokens = case_token_fingerprints(public_path)
        for field in hidden_tokens:
            if public_tokens[field] & hidden_tokens[field]:
                raise Refusal(
                    f"the public capture {role} has the same parsed {field} as "
                    "a hidden tape or oracle; shipping it would publish that tape"
                )
        ships.append(
            f"{public_files[role]} -> {repository_path} "
            f"({own['sha256'][:12]}..., {own['bytes']} bytes)"
        )

    first = live_entries[lives[0]]
    patch = {
        "official_scoring_enabled": True,
        "live_goldens": list(lives),
        "timed_prompt_pool": pool,
        "hidden_correctness_golden": {"sha256": first["sha256"], "bytes": first["bytes"]},
        "live_golden_speculative": speculative,
    }

    uploads = []
    local_by_r2 = {entry["r2_path"]: f"{os.path.basename(entry['r2_path'])}" for entry in pool}
    for key in depth_keys:
        for live in lives:
            local_by_r2.setdefault(speculative[key][live]["r2_path"], depth_files[(key, live)])
    pins_by_r2 = {e["r2_path"]: e for e in pool + oracle_entries}
    for r2_path in sorted(local_by_r2):
        entry = pins_by_r2[r2_path]
        uploads.append(f"{local_by_r2[r2_path]} -> {r2_path} ({entry['sha256'][:12]}..., {entry['bytes']} bytes)")
    total = len(depth_keys) * len(lives)
    shared = sum(
        1 for key in depth_keys for live in lives
        if speculative[key][live]["r2_path"] != f"{prefix}{depth_files[(key, live)]}"
    )
    uploads.append(f"{len(local_by_r2)} distinct object(s); {shared} of {total} per-depth key(s) share an earlier object")
    return patch, uploads, ships


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True, help="the directory of recorded golden files")
    ap.add_argument("--live", required=True, action="append", help="a pool tape the track scores over (its name without .golden.json); repeat for each live golden, the first is the hidden correctness golden")
    ap.add_argument("--apply", action="store_true", help="write the values into the fixture in place")
    ap.add_argument("--contract", default=CONTRACT, help=argparse.SUPPRESS)
    args = ap.parse_args()

    with open(args.contract, encoding="utf-8") as fh:
        contract = json.load(fh)
    try:
        patch, uploads, ships = build_patch(contract, args.dir, args.live)
    except Refusal as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 1

    for line in uploads:
        print(f"upload: {line}", file=sys.stderr)
    for line in ships:
        print(f"ship: {line}", file=sys.stderr)
    if args.apply:
        contract.update(patch)
        with open(args.contract, "w", encoding="utf-8") as fh:
            json.dump(contract, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        print(f"applied: {os.path.relpath(args.contract, ROOT)}", file=sys.stderr)
    else:
        json.dump(patch, sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
