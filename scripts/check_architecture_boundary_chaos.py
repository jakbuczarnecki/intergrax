#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-26.1 — architecture-boundary chaos job for weekly CI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

_BOUNDARY_SCRIPTS: tuple[str, ...] = (
    "check_intergrax_no_applications_imports.py",
    "check_agents_no_tier3_imports.py",
    "check_agents_no_vendor_sdk_imports.py",
    "check_agent_registry_bypass.py",
    "check_agents_no_inline_prompts.py",
)


def _run(script: str) -> int:
    script_path = str(REPO_ROOT / "scripts" / script)
    for cmd in (
        ["uv", "run", "python", script_path],
        [PYTHON, script_path],
    ):
        completed = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        if completed.returncode == 0:
            return 0
    print(f"FAILED: {script}", file=sys.stderr)
    return 1


def main() -> int:
    exit_code = 0
    for script in _BOUNDARY_SCRIPTS:
        exit_code = exit_code or _run(script)
    if exit_code == 0:
        print(f"OK: architecture boundary chaos ({len(_BOUNDARY_SCRIPTS)} checks)")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
