#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-25.3 — product release eval gate (context golden + baseline)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def main() -> int:
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile

    profile = ApplicationEnvironmentProfile.product_defaults()
    if not profile.evaluation_profile.require_baseline_for_release:
        print("product_defaults must set require_baseline_for_release=True", file=sys.stderr)
        return 1
    scripts = ("check_context_golden.py", "check_eval_scenario_library.py")
    for script in scripts:
        path = ROOT / "scripts" / script
        completed = subprocess.run([PYTHON, str(path)], cwd=ROOT, check=False)
        if completed.returncode != 0:
            print(f"FAILED: {script}", file=sys.stderr)
            return 1
    print("OK: product release eval gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
