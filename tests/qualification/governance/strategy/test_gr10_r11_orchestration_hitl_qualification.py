# © Artur Czarnecki. All rights reserved.

"""GR-10-R11 — ORCHESTRATION HITL enterprise qualification."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.qualification.governance.strategy.catalog import (
    GR10_ORCHESTRATION_CAPABILITY_SEMANTICS,
    GR10_ORCHESTRATION_HITL_INVENTORY,
    GR10_R11_NEXT_REMEDIATION,
    Gr10CoverageStatus,
    gr10_matrix_orchestration_status,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_GATE = _REPO_ROOT / "intergrax" / "runtime" / "policy" / "mse_hitl_effect_gate.py"
_INVOKER = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "tools" / "invoker.py"
_SLOT = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "nexus"
    / "orchestration"
    / "governed_consequential_operation.py"
)
_BRIDGE = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "nexus"
    / "tools"
    / "mse_governed_continuation_hitl_bridge.py"
)


def test_gr10_r11_orchestration_hitl_qualified() -> None:
    row = next(row for row in GR10_ORCHESTRATION_CAPABILITY_SEMANTICS if row.capability == "HITL")
    assert row.coverage is Gr10CoverageStatus.QUALIFIED
    assert gr10_matrix_orchestration_status("HITL") is Gr10CoverageStatus.QUALIFIED


def test_gr10_r11_hitl_inventory_has_no_gap_rows() -> None:
    assert GR10_ORCHESTRATION_HITL_INVENTORY
    for row in GR10_ORCHESTRATION_HITL_INVENTORY:
        assert row.coverage in {
            "QUALIFIED",
            "N/A",
            "delegated to canonical owner",
            "delegated to another canonical owner",
        }


def test_gr10_r11_continuation_row_remains_partial() -> None:
    assert gr10_matrix_orchestration_status("Continuation") is Gr10CoverageStatus.PARTIAL


def test_gr10_r11_next_remediation_is_continuation() -> None:
    assert GR10_R11_NEXT_REMEDIATION.capability == "Continuation"
    assert "GR-10-R12" in GR10_R11_NEXT_REMEDIATION.task_name


def test_gr10_r11_mse_hitl_gate_module_exists() -> None:
    source = _GATE.read_text(encoding="utf-8-sig")
    assert "def evaluate_mse_hitl_effect_gate" in source
    assert "MseHitlEffectGateDisposition" in source
    assert "matches_current_requirement" in source
    tree = ast.parse(source, filename=str(_GATE))
    assert tree is not None


def test_gr10_r11_invoker_wires_hitl_gate_ast() -> None:
    source = _INVOKER.read_text(encoding="utf-8-sig")
    assert "evaluate_mse_hitl_effect_gate" in source
    assert "governed_continuation_request=" in source
    assert "peek_governed_execution_task" in source


def test_gr10_r11_slot_executor_wires_hitl_gate_ast() -> None:
    source = _SLOT.read_text(encoding="utf-8-sig")
    assert "evaluate_mse_hitl_effect_gate" in source
    assert "governed_continuation_request" in source


def test_gr10_r11_mse_hitl_bridge_exists() -> None:
    source = _BRIDGE.read_text(encoding="utf-8-sig")
    assert "raise_mse_governed_continuation_hitl_pause" in source
    assert "GovernedContinuationHitlPauseRequired" in source
    assert "apply_governed_continuation_pause" in source
