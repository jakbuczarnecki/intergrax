# © Artur Czarnecki. All rights reserved.

"""GR-10-R10 — ORCHESTRATION decision-bound effect qualification."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.qualification.governance.strategy.catalog import (
    GR10_ORCHESTRATION_CAPABILITY_SEMANTICS,
    GR10_ORCHESTRATION_DECISION_BOUND_EFFECT_INVENTORY,
    GR10_R10_NEXT_REMEDIATION,
    Gr10CoverageStatus,
    gr10_matrix_orchestration_status,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_WIRING = (
    _REPO_ROOT
    / "intergrax"
    / "applications"
    / "_shared"
    / "harness_meaningful_side_effect_authorization_wiring.py"
)
_COMPOSITION = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "governance"
    / "orchestration_decision_bound_effect_composition.py"
)
_MSE_BOUNDARY = (
    _REPO_ROOT / "intergrax" / "runtime" / "policy" / "meaningful_side_effect_authorization.py"
)


def test_gr10_r10_orchestration_decision_bound_qualified() -> None:
    row = next(
        row for row in GR10_ORCHESTRATION_CAPABILITY_SEMANTICS if row.capability == "Decision-bound effect"
    )
    assert row.coverage is Gr10CoverageStatus.QUALIFIED
    assert gr10_matrix_orchestration_status("Decision-bound effect") is Gr10CoverageStatus.QUALIFIED


def test_gr10_r10_decision_bound_inventory_has_no_gap_rows() -> None:
    assert GR10_ORCHESTRATION_DECISION_BOUND_EFFECT_INVENTORY
    for row in GR10_ORCHESTRATION_DECISION_BOUND_EFFECT_INVENTORY:
        assert row.coverage in {
            "QUALIFIED",
            "N/A",
            "delegated to another canonical owner",
        }


def test_gr10_r10_next_remediation_is_hitl() -> None:
    assert GR10_R10_NEXT_REMEDIATION.capability == "HITL"
    assert "GR-10-R11" in GR10_R10_NEXT_REMEDIATION.task_name


def test_gr10_r10_harness_wires_production_orchestration_decision_policy() -> None:
    source = _WIRING.read_text(encoding="utf-8-sig")
    assert "build_production_orchestration_meaningful_side_effect_authorization_boundary" in source
    assert "decision_requirement_policy=" in source
    tree = ast.parse(source, filename=str(_WIRING))
    assert tree is not None


def test_gr10_r10_composition_module_exports_production_builder() -> None:
    source = _COMPOSITION.read_text(encoding="utf-8-sig")
    assert "def build_production_orchestration_meaningful_side_effect_authorization_boundary" in source
    assert "resolve_orchestration_decision_requirement_policy" in source


def test_gr10_r10_mse_boundary_enforces_decision_before_gate_ast() -> None:
    source = _MSE_BOUNDARY.read_text(encoding="utf-8-sig")
    assert "_enforce_decision_requirement" in source
    assert "classify_decision_requirement" in source
    assert "decision provenance required but absent" in source
