#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Ruff check + format --check on changed Python paths only (pre-push / pre-commit)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_CI_DIR = Path(__file__).resolve().parent
if str(_CI_DIR) not in sys.path:
    sys.path.insert(0, str(_CI_DIR))

from git_changed_paths import push_paths, staged_paths  # noqa: E402


def _python_paths(paths: frozenset[str]) -> list[str]:
    return sorted(path for path in paths if path.endswith(".py") and (_REPO / path).is_file())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ruff on changed Python files only.")
    parser.add_argument("--mode", choices=("push", "staged"), default="push")
    args = parser.parse_args(list(argv) if argv is not None else None)
    paths = _python_paths(push_paths() if args.mode == "push" else staged_paths())
    if not paths:
        print("[INFO] ruff changed-paths: no Python files in scope; skip.")
        return 0
    print(f"[INFO] ruff changed-paths: {len(paths)} file(s)", flush=True)
    check = subprocess.run(["uv", "run", "ruff", "check", *paths], cwd=_REPO, check=False)
    if check.returncode != 0:
        return check.returncode
    return subprocess.run(["uv", "run", "ruff", "format", "--check", *paths], cwd=_REPO, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
