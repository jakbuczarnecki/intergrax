#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-20.2 — policy change impact visualization CLI gate."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable


def main() -> int:
    cli = REPO_ROOT / "scripts" / "policy_change_impact_cli.py"
    for cmd in (
        ["uv", "run", "python", str(cli), "--top", "3"],
        [PYTHON, str(cli), "--top", "3"],
    ):
        completed = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
        if completed.returncode == 0 and "Policy change impact" in completed.stdout:
            print("OK: policy change impact CLI")
            return 0
    print("policy change impact CLI failed", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
