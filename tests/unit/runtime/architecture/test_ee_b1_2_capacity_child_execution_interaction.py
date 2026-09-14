# © Artur Czarnecki. All rights reserved.

"""EE-B1.2 — child execution must not double-acquire root capacity."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_CHILD_RUNNER = _REPO / "intergrax" / "runtime" / "execution" / "child.py"


def test_ee_b1_2_child_runner_does_not_import_root_capacity_port() -> None:
    text = _CHILD_RUNNER.read_text(encoding="utf-8")
    forbidden = (
        "ExecutionCapacityAdmissionPort",
        "LocalExecutionCapacityAdmission",
        "execution_capacity_admission",
    )
    hits = [sym for sym in forbidden if sym in text]
    assert hits == [], f"child runner must not own root capacity: {hits}"


def test_ee_b1_2_nested_root_slot_deadlock_documented() -> None:
    doc = (
        _REPO
        / "docs"
        / "project"
        / "maintainers"
        / "architecture"
        / "EXECUTION_ENGINE_CAPACITY_AND_BACKPRESSURE_MODEL.md"
    )
    text = doc.read_text(encoding="utf-8")
    assert "ChildExecutionRunner" in text
    assert (
        "no root-slot nested deadlock" in text.lower()
        or "does **not** acquire root" in text
    )
