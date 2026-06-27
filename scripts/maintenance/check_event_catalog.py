#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Regression gate: EventCatalog SSOT, kind registry, sampling, spine budget (OBS-EVOL-9.6/9.7)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from intergrax.runtime.events.spine_consolidation import assert_publication_spine_budget

    assert_publication_spine_budget()

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/unit/runtime/events/test_event_catalog.py",
        "tests/unit/runtime/events/test_event_kind_registry.py",
        "tests/unit/runtime/events/test_event_bus_sampling.py",
        "tests/unit/runtime/events/test_event_bus_taxonomy_subscribe.py",
        "tests/unit/runtime/events/test_spine_consolidation.py",
        "tests/unit/runtime/events/test_w3c_trace_context.py",
        "-q",
    ]
    result = subprocess.run(cmd, cwd=repo_root, check=False)
    if result.returncode == 0:
        print("event catalog audit (OBS-EVOL-9.6/9.7): OK")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
