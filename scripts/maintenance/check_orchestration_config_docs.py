#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""ORCH-CONFIG.9 — CFG IDs referenced in harness tests must appear in ORCHESTRATION canon."""

from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_ARCH = _REPO / "docs" / "architecture" / "ORCHESTRATION.md"
_TEST_GLOBS = (
    "tests/integration/runtime/test_orchestration_cfg_simulation.py",
    "tests/unit/applications/test_graph_spec_to_plan.py",
    "tests/unit/runtime/nexus/test_intent_routing.py",
    "tests/unit/runtime/nexus/test_orchestration_capabilities.py",
)

_CFG_RE = re.compile(r"CFG-\d{2}")
_ORCH_CONFIG_RE = re.compile(r"ORCH-CONFIG\.\d+")


def _collect_ids(path: Path, pattern: re.Pattern[str]) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return set(pattern.findall(text))


def main() -> int:
    if not _ARCH.is_file():
        print(f"missing architecture canon: {_ARCH}", file=sys.stderr)
        return 1

    arch_text = _ARCH.read_text(encoding="utf-8")
    arch_cfgs = set(_CFG_RE.findall(arch_text))
    arch_orch = set(_ORCH_CONFIG_RE.findall(arch_text))

    errors: list[str] = []
    for rel in _TEST_GLOBS:
        test_path = _REPO / rel
        if not test_path.is_file():
            errors.append(f"missing harness test file: {rel}")
            continue
        test_cfgs = _collect_ids(test_path, _CFG_RE)
        for cfg in sorted(test_cfgs):
            if cfg not in arch_text:
                errors.append(f"{rel} references {cfg} but canon omits it")

    plan_path = _REPO / "docs" / "plan" / "ORCHESTRATION.md"
    if plan_path.is_file():
        plan_text = plan_path.read_text(encoding="utf-8")
        for orch_id in sorted(arch_orch):
            if orch_id not in plan_text:
                errors.append(f"architecture lists {orch_id} but plan/ORCHESTRATION.md omits it")

    if errors:
        for line in errors:
            print(line, file=sys.stderr)
        return 1

    print(
        f"orchestration config docs (ORCH-CONFIG.9): OK "
        f"({len(arch_cfgs)} CFG ids, {len(arch_orch)} ORCH-CONFIG ids)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
