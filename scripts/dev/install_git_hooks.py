#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Point this clone's Git hooks at repository .githooks/ (pre-push smoke gate)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_HOOKS_REL = ".githooks"
_HOOKS_DIR = _REPO / _HOOKS_REL


def _run_git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=_REPO,
        check=False,
        capture_output=True,
        text=True,
    )


def _repo_root() -> Path:
    result = _run_git("rev-parse", "--show-toplevel")
    if result.returncode != 0:
        print(result.stderr or "not a git repository", file=sys.stderr)
        raise SystemExit(result.returncode)
    return Path(result.stdout.strip())


def _hooks_path() -> str | None:
    result = _run_git("config", "--get", "core.hooksPath")
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def install() -> int:
    root = _repo_root()
    if root != _REPO.resolve():
        print(f"Expected repository root {_REPO}, got {root}", file=sys.stderr)
        return 1
    if not _HOOKS_DIR.is_dir():
        print(f"Missing hooks directory: {_HOOKS_DIR}", file=sys.stderr)
        return 1
    pre_push = _HOOKS_DIR / "pre-push"
    if not pre_push.is_file():
        print(f"Missing pre-push hook: {pre_push}", file=sys.stderr)
        return 1

    result = _run_git("config", "core.hooksPath", _HOOKS_REL)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return result.returncode

    if os.name != "nt":
        os.chmod(pre_push, pre_push.stat().st_mode | 0o111)

    print(f"[OK] core.hooksPath = {_HOOKS_REL}")
    print("Pre-push runs: uv run python scripts/ci/run_push_gate.py")
    print("Docs: docs/project/technical/guides/LOCAL_PUSH_GATE.md")
    return 0


def uninstall() -> int:
    current = _hooks_path()
    if current != _HOOKS_REL:
        print(f"[INFO] core.hooksPath is '{current or '(default)'}'; nothing to uninstall.")
        return 0
    result = _run_git("config", "--unset", "core.hooksPath")
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return result.returncode
    print("[OK] core.hooksPath unset (Git default hooks directory).")
    return 0


def status() -> int:
    current = _hooks_path()
    active = current == _HOOKS_REL
    print(f"core.hooksPath: {current or '(default)'}")
    print(f"intergrax pre-push gate: {'active' if active else 'inactive'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install Intergrax .githooks for this clone.")
    parser.add_argument("--uninstall", action="store_true", help="Unset core.hooksPath for this clone.")
    parser.add_argument("--status", action="store_true", help="Show hook configuration.")
    args = parser.parse_args(argv)
    if args.uninstall:
        return uninstall()
    if args.status:
        return status()
    return install()


if __name__ == "__main__":
    raise SystemExit(main())
