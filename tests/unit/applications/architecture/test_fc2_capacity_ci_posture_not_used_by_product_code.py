# © Artur Czarnecki. All rights reserved.

"""T-IGX-043-B (PT-FC2) GATE-1: the CI-only capacity posture is never selected by product code."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_DEFINING_MODULE = Path("intergrax/runtime/capacity/production_adapters.py")
_OPT_IN = re.compile(r"CapacityBackendPosture\s*\.\s*CI_DETERMINISTIC\b|\bCI_DETERMINISTIC\b(?!_ONLY)")


def test_no_product_module_uses_capacity_ci_posture() -> None:
    offenders: list[str] = []
    for root_name in ("intergrax", "agents", "applications"):
        for path in sorted((_REPO / root_name).rglob("*.py")):
            relative = path.relative_to(_REPO)
            parts = relative.parts
            if relative == _DEFINING_MODULE or "tests" in parts or "__pycache__" in parts:
                continue
            if _OPT_IN.search(path.read_text(encoding="utf-8", errors="replace")):
                offenders.append(relative.as_posix())
    assert offenders == [], f"CI_DETERMINISTIC capacity posture selected outside tests: {offenders}"
