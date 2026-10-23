#!/usr/bin/env python3
"""Write the Seatbelt profile that confines participant code on the ranked box.

This is the one source of the profile rules. Three trusted callers use it:

  tools/sandboxed-cli.sh         the transform and the other CLI verbs
  tools/resident-up.sh           the resident bench-worker
  tools/sandboxed-build.sh       the compile steps of the build (--build)

The profile prints on stdout. The rules:

  * deny all network, except the Unix sockets that --unix-socket names;
  * deny scheduling-priority changes in every profile;
  * deny process-fork, and deny process-exec except the paths --exec names
    (sandbox-exec itself execs the first program after it applies the
    profile, so that program must be named). With --build, fork and exec are
    allowed, because a build starts compilers and linkers;
  * deny the DNS resolver mach services;
  * deny every file write, then allow writes to the device nodes, to every
    --write-subpath tree, to every path that starts with a --write-prefix, and
    to every --unix-socket path;
  * in runtime profiles, deny file flag and ACL changes after those allowances;
  * last, so that they win over every allow: deny every read and write of the
    evaluator-only paths that the environment names.

The evaluator-only paths come from the environment, not from arguments, so
that every caller denies the same set: MLXFAST_QWEN38_GOLDEN_DIR,
MLXFAST_CORRECTNESS_GOLDEN_PATH, MLXFAST_PRIVATE_DIR,
MLXFAST_BASELINE_WORKSPACE, MLXFAST_BASELINE_CALIBRATION, BENCHD_BIN_DIR,
BENCHD_PRIVATE_AUDIT_PATH. The rule resolves the audit file and its parent. If
the parent is / or contains any protected root, it denies the audit file only.
Otherwise it denies the parent directory, including a dedicated private
directory inside /private/tmp. The protected roots are /, HOME, --tree,
RUNNER_TEMP, TMPDIR, /tmp, /private/tmp, /var/tmp and /private/var/tmp. The
stable resident log root is also denied
(MLXFAST_RESIDENT_LOG_ROOT, falling back to RESIDENT_UP_LOG_DIR and then
~/.mlxfast-resident-logs), the build
cache root (MLXFAST_BUILD_CACHE_DIR, else
~/.cache/mlxfast-engine-build), the runner registration files two levels
above RUNNER_WORKSPACE, and the registration files in every actions-runner*
directory directly under HOME. The reference workspace is not denied when
--tree is inside it, because then the confined program is that workspace's
own.

With --official, the four paths that an official run must name
(MLXFAST_QWEN38_GOLDEN_DIR, MLXFAST_BASELINE_WORKSPACE,
MLXFAST_BASELINE_CALIBRATION, BENCHD_BIN_DIR) are required. The script exits 2
and names the unset ones when one is missing, because the deny list would
otherwise be incomplete.

Seatbelt matches resolved paths. Each path is resolved at its deepest existing
ancestor (/tmp becomes /private/tmp) and the rest is kept, because an output
tree may not exist yet.
"""

import argparse
import os
import sys

OFFICIAL_REQUIRED = (
    "MLXFAST_QWEN38_GOLDEN_DIR",
    "MLXFAST_BASELINE_WORKSPACE",
    "MLXFAST_BASELINE_CALIBRATION",
    "BENCHD_BIN_DIR",
)
OTHER_EVALUATOR_PATHS = (
    "MLXFAST_CORRECTNESS_GOLDEN_PATH",
    "MLXFAST_PRIVATE_DIR",
    "BENCHD_PRIVATE_AUDIT_PATH",
)
RUNNER_REGISTRATION_FILES = (".credentials", ".credentials_rsaparams", ".runner")


def resolved(path):
    path = os.path.abspath(path)
    rest = []
    while not os.path.lexists(path) and path != "/":
        path, tail = os.path.split(path)
        rest.insert(0, tail)
    return os.path.join(os.path.realpath(path), *rest)


def resolved_prefix(prefix):
    # A prefix ends in a partial file name, so only its directory is resolved.
    directory, partial = os.path.split(os.path.abspath(prefix))
    return os.path.join(resolved(directory), partial)


def quoted(path):
    return '"' + path.replace("\\", "\\\\").replace('"', '\\"') + '"'


def regex_escaped(value):
    special = "\\^$.|?*+()[]{}\""
    return "".join("\\" + c if c in special else c for c in value)


def inside(path, root):
    return path == root or path.startswith(root.rstrip("/") + "/")


def evaluator_paths(env, tree):
    paths = [env.get(name, "") for name in OFFICIAL_REQUIRED + OTHER_EVALUATOR_PATHS
             if name != "BENCHD_PRIVATE_AUDIT_PATH"]
    home = env.get("HOME") or os.path.expanduser("~")
    paths.append(env.get("MLXFAST_RESIDENT_LOG_ROOT")
                 or env.get("RESIDENT_UP_LOG_DIR")
                 or os.path.join(home, ".mlxfast-resident-logs"))
    paths.append(env.get("MLXFAST_BUILD_CACHE_DIR") or os.path.join(home, ".cache/mlxfast-engine-build"))
    audit = env.get("BENCHD_PRIVATE_AUDIT_PATH", "")
    if audit:
        audit_path = resolved(audit)
        audit_parent = os.path.dirname(audit_path)
        protected = ["/", home, tree, env.get("RUNNER_TEMP", ""),
                     env.get("TMPDIR", ""), "/tmp", "/private/tmp",
                     "/var/tmp", "/private/var/tmp"]
        protected = [resolved(root) for root in protected if root]
        # A broad parent must not hide a protected tree. A directory below a
        # protected tree is dedicated and can be denied as a whole.
        if audit_parent == "/" or any(inside(root, audit_parent)
                                       for root in protected):
            paths.append(audit_path)
        else:
            paths.append(audit_parent)
    baseline = env.get("MLXFAST_BASELINE_WORKSPACE", "")
    if baseline and tree and inside(resolved(tree), resolved(baseline)):
        paths.remove(baseline)
    runner_workspace = env.get("RUNNER_WORKSPACE", "")
    if runner_workspace:
        runner_root = os.path.dirname(os.path.dirname(os.path.abspath(runner_workspace)))
        paths += [os.path.join(runner_root, name) for name in RUNNER_REGISTRATION_FILES]
    try:
        home_entries = sorted(os.listdir(home))
    except OSError:
        home_entries = []
    for entry in home_entries:
        runner_root = os.path.join(home, entry)
        # Another runner under the same account carries credentials that a
        # compiler could print, even when RUNNER_WORKSPACE names this runner.
        if entry.startswith("actions-runner") and os.path.isdir(runner_root):
            paths += [os.path.join(runner_root, name) for name in RUNNER_REGISTRATION_FILES]
    return [resolved(path) for path in paths if path]


def main():
    parser = argparse.ArgumentParser(description="Print the ranked-box Seatbelt profile.")
    parser.add_argument("--exec", dest="execs", action="append", default=[],
                        help="a program the confined process may exec (repeat)")
    parser.add_argument("--build", action="store_true",
                        help="allow fork and every exec, for a build")
    parser.add_argument("--write-subpath", action="append", default=[],
                        help="a tree the confined process may write (repeat)")
    parser.add_argument("--write-prefix", action="append", default=[],
                        help="a path prefix the confined process may write (repeat)")
    parser.add_argument("--unix-socket", action="append", default=[],
                        help="a Unix socket the confined process may bind and use (repeat)")
    parser.add_argument("--tree", default="",
                        help="the checkout the confined program comes from")
    parser.add_argument("--official", action="store_true",
                        help="require the evaluator paths that an official run must name")
    args = parser.parse_args()
    if not args.build and not args.execs:
        parser.error("--exec is required without --build")

    env = os.environ
    if args.official:
        # RUNNER_WORKSPACE gives the path of the runner registration files.
        missing = [name for name in OFFICIAL_REQUIRED + ("RUNNER_WORKSPACE",) if not env.get(name)]
        if missing:
            sys.stderr.write(
                "seatbelt-profile.py: an official run needs the runner environment to name "
                "the evaluator-only paths the profile denies; unset: %s\n" % ", ".join(missing))
            return 2

    lines = ["(version 1)", "(allow default)", "(deny network*)"]
    sockets = []
    for sock in args.unix_socket:
        for path in sorted({os.path.abspath(sock), resolved(sock)}):
            if path not in sockets:
                sockets.append(path)
    for sock in sockets:
        lines.append("(allow network-bind network-inbound network-outbound (local unix-socket (path-literal %s)))" % quoted(sock))
        lines.append("(allow network-outbound (remote unix-socket (path-literal %s)))" % quoted(sock))
    lines.append("(deny system-sched)")
    if not args.build:
        # Signals: deny all, then allow the process itself (raise, abort, thread signals).
        # "(deny signal (target others))" alone is not enforced on macOS 27 (measured:
        # a sandboxed kill -TERM still reached a sibling process); this pair is.
        lines += ["(deny process-fork)", "(deny process-exec*)", "(deny signal)", "(allow signal (target self))"]
        for program in args.execs:
            lines.append("(allow process-exec (literal %s))" % quoted(resolved(program)))
        # No state that outlives the process: shared memory, SysV IPC and the
        # preferences daemon would let one job keep answers for the next.
        lines += [
            "(deny ipc-posix-shm*)",
            "(deny ipc-sysv*)",
            "(deny ipc-posix-sem*)",
            "(deny user-preference-write)",
            "(deny mach-lookup)",
            '(allow mach-lookup (global-name "com.apple.MTLCompilerService") (global-name "com.apple.system.opendirectoryd.libinfo") (global-name "com.apple.system.opendirectoryd.membership") (global-name "com.apple.logd") (global-name "com.apple.diagnosticd"))',
        ]
    lines += [
        '(deny mach-lookup (global-name "com.apple.mDNSResponder"))',
        '(deny mach-lookup (global-name "com.apple.system.mDNSResponder"))',
        '(deny mach-lookup (global-name-prefix "com.apple.mDNSResponder"))',
        "(deny file-write*)",
        '(allow file-write* (literal "/dev/null") (literal "/dev/zero") (literal "/dev/dtracehelper"))',
    ]
    for path in args.write_subpath:
        lines.append("(allow file-write* (subpath %s))" % quoted(resolved(path)))
    for prefix in args.write_prefix:
        lines.append("(allow file-write* (regex #\"^%s\"))" % regex_escaped(resolved_prefix(prefix)))
    for sock in sockets:
        lines.append("(allow file-write* (literal %s))" % quoted(sock))
    if not args.build:
        lines += ["(deny file-write-flags)", "(deny file-write-acl)"]
    # Last, so that these rules win over every allow above.
    for path in evaluator_paths(env, args.tree):
        lines.append("(deny file-read* file-write* (subpath %s))" % quoted(path))
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
