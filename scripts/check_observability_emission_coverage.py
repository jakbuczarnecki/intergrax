#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Regression gate: OBS-BUS-3 emission coverage for key catalog events."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/unit/runtime/events/test_observability_emission_coverage.py",
        "-q",
    ]
    result = subprocess.run(cmd, cwd=repo_root, check=False)
    if result.returncode == 0:
        print("observability emission coverage audit: OK")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
