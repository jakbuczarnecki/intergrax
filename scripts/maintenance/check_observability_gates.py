#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Regression gate: OBS-BUS CI umbrella — emission, schema registry, persistence, L4 §21."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
_CI_DIR = REPO_ROOT / "scripts" / "ci"
if str(_CI_DIR) not in sys.path:
    sys.path.insert(0, str(_CI_DIR))
from script_paths import resolve_script  # noqa: E402

_AUDIT_SCRIPTS = (
    "check_trace_bridge_event_catalog.py",
    "check_observability_emission_coverage.py",
    "check_payload_schema_registry.py",
    "check_observability_persistence_conformance.py",
    "check_rag_otel_span_registry.py",
    "check_event_catalog.py",
    "check_llm_catalog_miss_observability.py",
)


def _run(script: str) -> int:
    path = resolve_script(script)
    for cmd in (
        ["uv", "run", "python", str(path)],
        [PYTHON, str(path)],
    ):
        completed = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        if completed.returncode == 0:
            return 0
    return 1


def main() -> int:
    for name in _AUDIT_SCRIPTS:
        if _run(name) != 0:
            return 1

    l4_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/unit/runtime/events/test_observability_layer_depth_gate.py",
        "-q",
    ]
    l4_result = subprocess.run(l4_cmd, cwd=REPO_ROOT, check=False)
    if l4_result.returncode != 0:
        return l4_result.returncode

    print("observability gates audit (OBS-BUS-7): OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
