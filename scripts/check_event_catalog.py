#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Regression gate: EventCatalog SSOT, kind registry, and sampling (OBS-EVOL-9.6)."""

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
        "tests/unit/runtime/events/test_event_catalog.py",
        "tests/unit/runtime/events/test_event_kind_registry.py",
        "tests/unit/runtime/events/test_event_bus_sampling.py",
        "tests/unit/runtime/events/test_event_bus_taxonomy_subscribe.py",
        "-q",
    ]
    result = subprocess.run(cmd, cwd=repo_root, check=False)
    if result.returncode == 0:
        print("event catalog audit (OBS-EVOL-9.6): OK")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
