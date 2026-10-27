#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Pre-commit gate: ruff on staged Python files + diff-scoped maintenance scripts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_CI_DIR = Path(__file__).resolve().parent
if str(_CI_DIR) not in sys.path:
    sys.path.insert(0, str(_CI_DIR))

from git_changed_paths import staged_paths  # noqa: E402
from push_gate_config import load_push_gate_config  # noqa: E402


def _truthy_env(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _staged_python_files() -> list[str]:
    return sorted(path for path in staged_paths() if path.endswith(".py"))


def _run_ruff(paths: list[str]) -> int:
    if not paths:
        print("[INFO] pre-commit ruff: no staged Python files.")
        return 0
    print(f"[INFO] pre-commit ruff: {len(paths)} file(s)", flush=True)
    check = subprocess.run(
        ["uv", "run", "ruff", "check", *paths],
        cwd=_REPO,
        check=False,
    )
    if check.returncode != 0:
        return check.returncode
    return subprocess.run(
        ["uv", "run", "ruff", "format", "--check", *paths],
        cwd=_REPO,
        check=False,
    ).returncode


def main() -> int:
    config = load_push_gate_config()
    if _truthy_env(config.skip_env_pre_commit):
        print(
            f"[WARN] Pre-commit gate skipped ({config.skip_env_pre_commit}).",
            file=sys.stderr,
        )
        return 0
    if not config.pre_commit_ruff and not config.diff_scoped_gates:
        return 0

    exit_code = 0
    if config.pre_commit_ruff:
        exit_code = _run_ruff(_staged_python_files()) or exit_code
    if config.diff_scoped_gates and exit_code == 0:
        diff = subprocess.run(
            [sys.executable, str(_CI_DIR / "diff_scoped_gates.py"), "--mode", "staged"],
            cwd=_REPO,
            check=False,
        )
        exit_code = diff.returncode
    if exit_code != 0:
        print("[ERROR] Pre-commit gate failed.", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
