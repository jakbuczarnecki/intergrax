# © Artur Czarnecki. All rights reserved.

"""GR-10-R15 — ORCHESTRATION GEP SSOT reconciliation and strategy closure gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.qualification.governance.strategy.catalog import (
    GR10_FINAL_CAPABILITY_MATRIX,
    GR10_GEP_COVERAGE_INVENTORY,
    GR10_ORCHESTRATION_CAPABILITY_SEMANTICS,
    GR10_ORCHESTRATION_GEP_SEMANTICS,
    GR10_ORCHESTRATION_GOVERNANCE_EVIDENCE_INVENTORY,
    GR10_R14_NEXT_REMEDIATION,
    GR10_R15_NEXT_REMEDIATION,
    Gr10Applicability,
    Gr10CoverageStatus,
    gr10_matrix_orchestration_status,
    gr10_orchestration_gep_semantics,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_NEXUS_LOOP = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "nexus_loop.py"
_INVOKER = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "tools" / "invoker.py"
_RUNTIME_CONTEXT = (
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "engine" / "runtime_context.py"
)
_GRAPH_RUNNER = (
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "orchestration" / "graph_runner.py"
)


def _inventory_row(gep: str) -> object:
    return next(row for row in GR10_GEP_COVERAGE_INVENTORY if row.gep == gep)


def test_gr10_r15_orchestration_gep_semantics_inventory_aligned() -> None:
    for sem in GR10_ORCHESTRATION_GEP_SEMANTICS:
        inv = _inventory_row(sem.gep)
        expected_applicable = sem.applicability is Gr10Applicability.APPLICABLE
        assert inv.orchestration_applicable is expected_applicable, sem.gep
        assert inv.orchestration_coverage is sem.coverage, sem.gep


def test_gr10_r15_orchestration_no_applicable_gep_partial() -> None:
    partial = [
        row.gep
        for row in GR10_GEP_COVERAGE_INVENTORY
        if row.orchestration_applicable
        and row.orchestration_coverage
        in (
            Gr10CoverageStatus.PARTIAL,
            Gr10CoverageStatus.GAP,
            Gr10CoverageStatus.WIRED_NOT_QUALIFIED,
        )
    ]
    assert partial == []


def test_gr10_r15_orchestration_capability_matrix_all_qualified() -> None:
    for row in GR10_ORCHESTRATION_CAPABILITY_SEMANTICS:
        assert row.applicability is Gr10Applicability.APPLICABLE
        assert row.coverage is Gr10CoverageStatus.QUALIFIED, row.capability
    for row in GR10_FINAL_CAPABILITY_MATRIX:
        assert row.orchestration is Gr10CoverageStatus.QUALIFIED, row.capability


def test_gr10_r15_orchestration_gep_evidence_inventory_no_conflict() -> None:
    """Deferred GEP rows use DEFERRED_TO_GR13 in inventory — not NOT_APPLICABLE (GR-10-R15-R1)."""
    evidence_by_path = {row.path: row for row in GR10_ORCHESTRATION_GOVERNANCE_EVIDENCE_INVENTORY}
    deferred_gep_to_path = {
        "PRE_OUTPUT": "PRE_OUTPUT evaluation point",
        "POST_RUN": "POST_RUN evaluation point",
    }
    for gep, path in deferred_gep_to_path.items():
        sem = gr10_orchestration_gep_semantics(gep)
        assert sem.gr8_evidence_applicability is Gr10Applicability.APPLICABLE, gep
        assert evidence_by_path[path].status == "DEFERRED_TO_GR13", gep


def test_gr10_r15_four_residuals_final_status() -> None:
    assert gr10_orchestration_gep_semantics("INTERRUPT").coverage is Gr10CoverageStatus.NOT_APPLICABLE
    assert gr10_orchestration_gep_semantics("TOOL_INVOCATION_POLICY").coverage is Gr10CoverageStatus.QUALIFIED
    assert gr10_orchestration_gep_semantics("PRE_OUTPUT").coverage is Gr10CoverageStatus.QUALIFIED
    assert gr10_orchestration_gep_semantics("POST_RUN").coverage is Gr10CoverageStatus.QUALIFIED


def test_gr10_r15_next_remediation_agentic_closure() -> None:
    assert GR10_R15_NEXT_REMEDIATION.strategy == "AGENTIC"
    assert "Recertification" in GR10_R15_NEXT_REMEDIATION.task_name or "Closure" in GR10_R15_NEXT_REMEDIATION.task_name
    assert GR10_R14_NEXT_REMEDIATION.task_name.startswith("GR-10-R15")


def test_gr10_r15_nexus_finish_pre_output_and_post_run_ast() -> None:
    source = _NEXUS_LOOP.read_text(encoding="utf-8-sig")
    assert "apply_pre_output_policy" in source
    assert "invoke_post_run_governance" in source
    tree = ast.parse(source, filename=str(_NEXUS_LOOP))
    finish_methods = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_finish_task"
    ]
    assert finish_methods == ["_finish_task"]


def test_gr10_r15_runtime_tool_invoker_declarative_policy_ast() -> None:
    source = _INVOKER.read_text(encoding="utf-8-sig")
    assert "evaluate_tool_invocation" in source
    assert "resolve_declarative_policy_enforcer" in source


def test_gr10_r15_strict_production_requires_governance_service_ast() -> None:
    source = _RUNTIME_CONTEXT.read_text(encoding="utf-8-sig")
    assert "production_mode and governance_service is None" in source.replace("\n", " ")
    assert "GovernanceService is required when production_mode=True" in source


def test_gr10_r15_graph_runner_cancellation_not_interrupt_gep_ast() -> None:
    source = _GRAPH_RUNNER.read_text(encoding="utf-8-sig")
    assert "CancellationCoordinator" in source
    assert "evaluate_interrupt" not in source


def test_gr10_r15_governance_evidence_capability_qualified() -> None:
    assert gr10_matrix_orchestration_status("Governance Evidence") is Gr10CoverageStatus.QUALIFIED
