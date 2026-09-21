# © Artur Czarnecki. All rights reserved.

"""GR-10-R14 — ORCHESTRATION Governance Evidence enterprise qualification."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.qualification.governance.strategy.catalog import (
    GR10_ORCHESTRATION_CAPABILITY_SEMANTICS,
    GR10_ORCHESTRATION_GOVERNANCE_EVIDENCE_INVENTORY,
    GR10_R13_NEXT_REMEDIATION,
    GR10_R14_NEXT_REMEDIATION,
    Gr10CoverageStatus,
    gr10_matrix_orchestration_status,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MSE_AUTH = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "policy"
    / "meaningful_side_effect_authorization.py"
)
_ORCH_EVIDENCE_COMPOSITION = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "governance"
    / "orchestration_governance_evidence_composition.py"
)
_PRODUCTION_MSE = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "governance"
    / "orchestration_decision_bound_effect_composition.py"
)


def test_gr10_r14_orchestration_governance_evidence_qualified() -> None:
    row = next(
        row
        for row in GR10_ORCHESTRATION_CAPABILITY_SEMANTICS
        if row.capability == "Governance Evidence"
    )
    assert row.coverage is Gr10CoverageStatus.QUALIFIED
    assert gr10_matrix_orchestration_status("Governance Evidence") is Gr10CoverageStatus.QUALIFIED


def test_gr10_r14_next_remediation_points_to_agentic_closure() -> None:
    assert GR10_R14_NEXT_REMEDIATION.strategy == "AGENTIC"
    assert "GR-10" in GR10_R14_NEXT_REMEDIATION.task_name


def test_gr10_r14_r13_remediation_was_governance_evidence() -> None:
    assert "GR-10-R14" in GR10_R13_NEXT_REMEDIATION.task_name


def test_gr10_r14_production_mse_requires_governance_evidence_persistence_ast() -> None:
    source = _PRODUCTION_MSE.read_text(encoding="utf-8-sig")
    assert "governance_evidence_persistence" in source
    assert "OrchestrationGovernanceEvidenceCompositionError" in source


def test_gr10_r14_mse_records_via_canonical_builder_ast() -> None:
    source = _MSE_AUTH.read_text(encoding="utf-8-sig")
    assert "build_governance_fact_from_policy_decision" in source
    assert "PolicyAction.ESCALATE" in source
    tree = ast.parse(source, filename=str(_MSE_AUTH))
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(getattr(node.func, "id", None), str)
        and node.func.id == "dict"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    )


def test_gr10_r14_orchestration_evidence_composition_is_pluginable_ast() -> None:
    source = _ORCH_EVIDENCE_COMPOSITION.read_text(encoding="utf-8-sig")
    assert "build_orchestration_governance_evidence_recorder" in source
    assert "InMemoryGovernanceEvidencePersistence" not in source


def test_gr10_r14_r1_orchestration_governance_evidence_inventory_zero_gap() -> None:
    gaps = [
        row.path
        for row in GR10_ORCHESTRATION_GOVERNANCE_EVIDENCE_INVENTORY
        if row.status not in ("QUALIFIED", "NOT_APPLICABLE")
    ]
    assert gaps == []
