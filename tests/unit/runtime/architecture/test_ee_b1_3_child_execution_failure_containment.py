# © Artur Czarnecki. All rights reserved.

"""EE-B1.3 — child / fan-out failure containment compatibility gates."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_CHILD = _REPO / "intergrax" / "runtime" / "execution" / "child.py"
_MODEL = (
    _REPO
    / "docs"
    / "project"
    / "maintainers"
    / "architecture"
    / "EXECUTION_ENGINE_WORKER_ISOLATION_AND_FAILURE_CONTAINMENT_MODEL.md"
)
_NPSC5E_R3 = (
    _REPO
    / "tests"
    / "unit"
    / "runtime"
    / "architecture"
    / "test_npsc5e_r3_child_fanout_partial_recovery.py"
)


def test_ee_b1_3_child_runner_no_concurrent_work_pool_owner() -> None:
    text = _CHILD.read_text(encoding="utf-8")
    assert "execute_concurrent_execution_work" not in text
    assert "WorkerRuntime" not in text


def test_ee_b1_3_child_failure_documented_with_recovery_plane() -> None:
    doc = _MODEL.read_text(encoding="utf-8")
    assert "ChildExecutionRunner" in doc
    assert "NPSC-5E" in doc


def test_ee_b1_3_fan_out_regression_module_present() -> None:
    assert _NPSC5E_R3.is_file()
