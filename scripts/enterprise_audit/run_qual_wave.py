#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""L1 → L2 → L3 qualification orchestrator for ENT-AUDIT-X (C051)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_STAGES: tuple[tuple[str, str], ...] = (
    ("l1", "ent_l1"),
    ("l2", "ent_l2"),
    ("l3", "ent_l3_heavy"),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_pytest(marker: str) -> int:
    root = _repo_root()
    cmd = [
        "uv",
        "run",
        "pytest",
        "tests/qualification/ent_audit_x",
        "-m",
        marker,
        "-q",
        "--tb=line",
    ]
    completed = subprocess.run(cmd, cwd=root, check=False)
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ENT-AUDIT-X qualification wave (L1/L2/L3).")
    parser.add_argument("--dry-run", action="store_true", help="Print stages only; do not run pytest.")
    parser.add_argument(
        "--inject-fail-after",
        choices=[name for name, _ in _STAGES],
        help="Test hook: exit non-zero after named stage (orchestrator self-test).",
    )
    args = parser.parse_args()
    if args.dry_run:
        for name, marker in _STAGES:
            print(f"stage={name} marker={marker}")
        if args.inject_fail_after:
            return 1
        return 0
    for name, marker in _STAGES:
        code = _run_pytest(marker)
        if code != 0:
            print(f"stage {name} failed with exit {code}", file=sys.stderr)
            return code
        if args.inject_fail_after == name:
            print(f"injected fail after stage {name}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
