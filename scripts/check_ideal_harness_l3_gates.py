#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""IDEAL-26.1 — umbrella gate for Ideal Harness L3 depth checks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _run(script: str, *extra: str) -> int:
    script_path = str(REPO_ROOT / "scripts" / script)
    for cmd in (
        ["uv", "run", "python", script_path, *extra],
        [PYTHON, script_path, *extra],
    ):
        completed = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        if completed.returncode == 0:
            return 0
    print(f"FAILED: {script}", file=sys.stderr)
    return 1


def main() -> int:
    scripts = [
        ("check_agents_no_inline_prompts.py", ()),
        ("check_agents_no_vendor_sdk_imports.py", ()),
        ("check_harness_prompt_golden_catalog.py", ()),
        ("check_agents_lifecycle_metadata.py", ()),
        ("harness_maturity_report.py", ("--enforce-l3-critical",)),
        ("phase_v_capability_graph_guard.py", ("--enforce",)),
    ]
    exit_code = 0
    for script, extra in scripts:
        code = _run(script, *extra)
        exit_code = exit_code or code
    if exit_code == 0:
        print("OK: Ideal Harness L3 gate checks passed")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
