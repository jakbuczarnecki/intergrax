#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Regression gate: catalog miss observability spine (M-LLM-X.15.4 · LLM-MAINT-05)."""

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
        "tests/unit/llm_adapters/test_catalog_miss_trace.py",
        "tests/unit/llm_adapters/test_catalog_miss_metrics.py",
        "tests/acceptance/llm_routing/test_catalog_miss_trace_e2e.py",
        "-m",
        "gate and not no_ci",
        "-q",
    ]
    result = subprocess.run(cmd, cwd=repo_root, check=False)
    if result.returncode != 0:
        return result.returncode

    required_snippets = (
        ("intergrax/llm_adapters/registry/catalog_miss_diag.py", "CatalogResolutionTier"),
        ("intergrax/llm_adapters/tracking/metrics.py", "intergrax_llm_catalog_miss_total"),
        (
            "intergrax/runtime/nexus/engine/runtime_state.py",
            "wire_catalog_miss_trace_sink(self.trace_event)",
        ),
        ("intergrax/runtime/events/trace_bridge.py", "_CORE_LLM_CATALOG_MISS_SCHEMA"),
    )
    for rel_path, needle in required_snippets:
        text = (repo_root / rel_path).read_text(encoding="utf-8")
        if needle not in text:
            print(f"check_llm_catalog_miss_observability: missing {needle!r} in {rel_path}")
            return 1

    print("check_llm_catalog_miss_observability: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
