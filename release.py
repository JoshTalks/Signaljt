#!/usr/bin/env python3
"""Cut a new signaljt release.

Bumps the version (relative to the current one), keeps pyproject.toml and
signaljt/__init__.py in sync, commits, tags, and pushes. Optionally builds and
uploads to PyPI.

Usage:
    python release.py                  # patch bump  (0.1.0 -> 0.1.1)
    python release.py minor            #             (0.1.0 -> 0.2.0)
    python release.py major            #             (0.1.0 -> 1.0.0)
    python release.py 1.4.2            # set an explicit version
    python release.py patch --no-push  # commit + tag locally, don't push
    python release.py patch --publish  # also `uv build && uv publish` to PyPI
    python release.py patch --dry-run  # print the plan, change nothing

The PyPI token (for --publish) is read by `uv publish` from the UV_PUBLISH_TOKEN
environment variable or an interactive prompt — it is never stored in the repo.
"""
import argparse
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
PYPROJECT = ROOT / "pyproject.toml"
INIT = ROOT / "signaljt" / "__init__.py"

PYPROJECT_RE = re.compile(r'^version\s*=\s*"(\d+)\.(\d+)\.(\d+)"', re.M)
INIT_RE = re.compile(r'^__version__\s*=\s*"(\d+)\.(\d+)\.(\d+)"', re.M)


def sh(*args, capture=False, check=True):
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=capture, check=check)


def git_out(*args):
    return sh("git", *args, capture=True).stdout.strip()


def current_version():
    m = PYPROJECT_RE.search(PYPROJECT.read_text())
    if not m:
        sys.exit('Could not find `version = "X.Y.Z"` in pyproject.toml')
    if not INIT_RE.search(INIT.read_text()):
        sys.exit('Could not find `__version__ = "X.Y.Z"` in signaljt/__init__.py')
    return tuple(int(x) for x in m.groups())


def next_version(cur, spec):
    if re.fullmatch(r"\d+\.\d+\.\d+", spec):
        return tuple(int(x) for x in spec.split("."))
    major, minor, patch = cur
    return {
        "major": (major + 1, 0, 0),
        "minor": (major, minor + 1, 0),
        "patch": (major, minor, patch + 1),
    }.get(spec) or sys.exit("bump must be major | minor | patch | X.Y.Z")


def write_version(ver):
    PYPROJECT.write_text(PYPROJECT_RE.sub('version = "%s"' % ver, PYPROJECT.read_text(), count=1))
    INIT.write_text(INIT_RE.sub('__version__ = "%s"' % ver, INIT.read_text(), count=1))


def main():
    ap = argparse.ArgumentParser(description="Cut a new signaljt release.")
    ap.add_argument("bump", nargs="?", default="patch",
                    help="major | minor | patch | explicit X.Y.Z (default: patch)")
    ap.add_argument("--no-push", action="store_true", help="commit and tag locally, do not push")
    ap.add_argument("--publish", action="store_true", help="also build and upload to PyPI")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")
    args = ap.parse_args()

    if sh("git", "rev-parse", "--is-inside-work-tree", capture=True, check=False).returncode != 0:
        sys.exit("Not a git repository. Run `git init` first.")
    if git_out("status", "--porcelain"):
        sys.exit("Working tree is dirty. Commit or stash your changes first.")

    cur = current_version()
    new = next_version(cur, args.bump)
    if new <= cur:
        sys.exit("New version %s is not greater than current %s"
                 % (".".join(map(str, new)), ".".join(map(str, cur))))
    ver = "%d.%d.%d" % new
    tag = "v" + ver
    branch = git_out("rev-parse", "--abbrev-ref", "HEAD")

    if git_out("tag", "-l", tag):
        sys.exit("Tag %s already exists." % tag)

    print("Release: %s -> %s   (branch %s, tag %s)"
          % (".".join(map(str, cur)), ver, branch, tag))

    if args.dry_run:
        actions = ["update pyproject.toml + __init__.py", "commit", "tag " + tag]
        if not args.no_push:
            actions.append("push %s + %s" % (branch, tag))
        if args.publish:
            actions.append("build + publish to PyPI")
        print("[dry-run] would: " + ", ".join(actions))
        return

    write_version(ver)
    sh("git", "add", "pyproject.toml", "signaljt/__init__.py")
    sh("git", "commit", "-m", "Release " + tag)
    sh("git", "tag", "-a", tag, "-m", "signaljt " + ver)
    print("Committed and tagged %s." % tag)

    if not args.no_push:
        sh("git", "push", "origin", branch)
        sh("git", "push", "origin", tag)
        print("Pushed %s and %s." % (branch, tag))

    if args.publish:
        print("Building and publishing to PyPI ...")
        shutil.rmtree(ROOT / "dist", ignore_errors=True)
        sh("uv", "build")
        sh("uv", "publish")
        print("Published %s to PyPI." % ver)

    print("Done.")


if __name__ == "__main__":
    main()