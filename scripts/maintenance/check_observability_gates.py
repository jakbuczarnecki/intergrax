#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Regression gate: OBS-BUS CI umbrella — emission, schema registry, persistence, L4 §21."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_AUDIT_SCRIPTS = (
    "check_trace_bridge_event_catalog.py",
    "check_observability_emission_coverage.py",
    "check_payload_schema_registry.py",
    "check_observability_persistence_conformance.py",
    "check_rag_otel_span_registry.py",
    "check_event_catalog.py",
    "check_llm_catalog_miss_observability.py",
)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    scripts_dir = repo_root / "scripts"
    for name in _AUDIT_SCRIPTS:
        script = scripts_dir / name
        result = subprocess.run([sys.executable, str(script)], cwd=repo_root, check=False)
        if result.returncode != 0:
            return result.returncode

    l4_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/unit/runtime/events/test_observability_layer_depth_gate.py",
        "-q",
    ]
    l4_result = subprocess.run(l4_cmd, cwd=repo_root, check=False)
    if l4_result.returncode != 0:
        return l4_result.returncode

    print("observability gates audit (OBS-BUS-7): OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
