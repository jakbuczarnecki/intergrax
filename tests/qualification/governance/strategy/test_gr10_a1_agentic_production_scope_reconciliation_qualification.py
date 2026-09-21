# © Artur Czarnecki. All rights reserved.

"""GR-10-A1 — AGENTIC production scope reconciliation gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.qualification.governance.strategy.catalog import (
    GR10_AGENTIC_CAPABILITY_SEMANTICS,
    GR10_AGENTIC_GEP_SEMANTICS,
    GR10_AGENTIC_LEGAL_PRODUCTION_PATHS,
    GR10_A1_NEXT_REMEDIATION,
    GR10_A1_R1_NEXT_REMEDIATION,
    GR10_FINAL_CAPABILITY_MATRIX,
    GR10_GEP_COVERAGE_INVENTORY,
    Gr10Applicability,
    Gr10CoverageStatus,
    gr10_agentic_gep_semantics,
    gr10_matrix_agentic_status,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_HOST_TASK = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "host_task.py"
_AGENT_ENGINE = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "agents" / "agent_engine.py"
_RUNTIME_CONTEXT = (
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "engine" / "runtime_context.py"
)
_STEP_KERNEL = _REPO_ROOT / "intergrax" / "runtime" / "kernel" / "step_kernel.py"


def _inventory_row(gep: str) -> object:
    return next(row for row in GR10_GEP_COVERAGE_INVENTORY if row.gep == gep)


def test_gr10_a1_agentic_gep_semantics_inventory_aligned() -> None:
    for sem in GR10_AGENTIC_GEP_SEMANTICS:
        inv = _inventory_row(sem.gep)
        expected_applicable = sem.applicability is Gr10Applicability.APPLICABLE
        assert inv.agentic_applicable is expected_applicable, sem.gep
        assert inv.agentic_coverage is sem.coverage, sem.gep


def test_gr10_a1_agentic_no_applicable_gep_partial() -> None:
    partial = [
        row.gep
        for row in GR10_GEP_COVERAGE_INVENTORY
        if row.agentic_applicable
        and row.agentic_coverage
        in (
            Gr10CoverageStatus.PARTIAL,
            Gr10CoverageStatus.GAP,
            Gr10CoverageStatus.WIRED_NOT_QUALIFIED,
        )
    ]
    assert partial == []


def test_gr10_a1_agentic_capability_partial_excludes_only_root_and_decision_bound() -> None:
    partial = {
        row.capability
        for row in GR10_AGENTIC_CAPABILITY_SEMANTICS
        if row.coverage is Gr10CoverageStatus.PARTIAL
    }
    assert partial == {
        "Inner Governance",
        "Policy evaluation",
        "MSE",
        "HITL",
        "Continuation",
        "Reliability",
        "Governance Evidence",
    }


def test_gr10_a1_legal_production_paths_no_vague_other_delegates() -> None:
    qualified = [row for row in GR10_AGENTIC_LEGAL_PRODUCTION_PATHS if row.status == "QUALIFIED"]
    assert len(qualified) == 1
    assert qualified[0].path_id == "P-UAEP"
    assert "TaskBoundAgenticDelegate" in qualified[0].legal_entry
    acp = next(row for row in GR10_AGENTIC_LEGAL_PRODUCTION_PATHS if row.path_id == "P-ACP-SESSION")
    assert acp.status == "WIRED_NOT_QUALIFIED"
    assert "LEGACY_NO_PRODUCTION_USER" not in acp.status


def test_gr10_a1_next_remediation_points_to_acp_governance_closure() -> None:
    assert GR10_A1_NEXT_REMEDIATION.strategy == "AGENTIC"
    assert "ACP" in GR10_A1_NEXT_REMEDIATION.task_name
    assert GR10_A1_NEXT_REMEDIATION is GR10_A1_R1_NEXT_REMEDIATION


def test_gr10_a1_host_task_single_agentic_delegate_ast() -> None:
    source = _HOST_TASK.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(_HOST_TASK))
    delegate_classes = [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name.endswith("AgenticDelegate")
    ]
    assert delegate_classes == ["TaskBoundAgenticDelegate"]


def test_gr10_a1_strict_production_requires_governance_service_ast() -> None:
    source = _RUNTIME_CONTEXT.read_text(encoding="utf-8-sig")
    assert "production_mode=True" in source.replace(" ", "")
    assert "GovernanceService is required when production_mode=True" in source


def test_gr10_a1_step_kernel_terminal_pre_output_ast() -> None:
    source = _STEP_KERNEL.read_text(encoding="utf-8-sig")
    assert "evaluate_pre_output" in source


def test_gr10_a1_agent_decision_owner_documented() -> None:
    sem = gr10_agentic_gep_semantics("AGENT_DECISION")
    assert sem.coverage is Gr10CoverageStatus.QUALIFIED
    assert "ExecutionInterruptHandler" in sem.canonical_owner
    assert "other agent delegates" in sem.reason.lower() or "ssot drift" in sem.reason.lower()


def test_gr10_a1_matrix_matches_semantics() -> None:
    for row in GR10_AGENTIC_CAPABILITY_SEMANTICS:
        assert gr10_matrix_agentic_status(row.capability) is row.coverage
    for row in GR10_FINAL_CAPABILITY_MATRIX:
        sem = next(s for s in GR10_AGENTIC_CAPABILITY_SEMANTICS if s.capability == row.capability)
        assert row.agentic is sem.coverage, row.capability


def test_gr10_a1_acp_session_branch_is_not_canonical_production_owner() -> None:
    source = _AGENT_ENGINE.read_text(encoding="utf-8-sig")
    assert "acp_session_enabled" in source
    assert "governance, None" in source.replace(" ", "") or "None, dict" in source
