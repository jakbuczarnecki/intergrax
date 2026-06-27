#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Regression gate: OBS-BUS-5 observability persistence conformance."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/unit/runtime/events/test_observability_persistence_conformance.py",
        "tests/integration/runtime/test_nexus_loop_runtime_event_persistence.py",
        "-q",
    ]
    result = subprocess.run(cmd, cwd=repo_root, check=False)
    if result.returncode == 0:
        print("observability persistence conformance audit: OK")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
