#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Regression gate: observability extension SDK and scaffold tracing templates (OBS-BUS-4)."""

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
        "tests/unit/runtime/observability/test_extension_sdk.py",
        "tests/unit/scaffold/test_scaffold_tracing_extension.py",
        "-q",
    ]
    result = subprocess.run(cmd, cwd=repo_root, check=False)
    if result.returncode == 0:
        print("payload schema registry / extension SDK audit: OK")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
