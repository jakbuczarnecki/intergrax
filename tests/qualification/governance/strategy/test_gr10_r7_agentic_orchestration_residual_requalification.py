# © Artur Czarnecki. All rights reserved.

"""GR-10-R7 — AGENTIC & ORCHESTRATION residual strategy matrix requalification."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.qualification.governance.strategy.catalog import (
    GR10_AGENTIC_CAPABILITY_SEMANTICS,
    GR10_FINAL_CAPABILITY_MATRIX,
    GR10_GEP_COVERAGE_INVENTORY,
    GR10_ORCHESTRATION_CAPABILITY_SEMANTICS,
    GR10_R7_NEXT_REMEDIATION,
    GR10_SCENARIO_CATALOG,
    Gr10Applicability,
    Gr10CoverageStatus,
    gr10_matrix_agentic_status,
    gr10_matrix_inference_status,
    gr10_matrix_orchestration_status,
)
from tests.qualification.governance.strategy.gr10_inference_current_doc_ssot import (
    gr10_architecture_remaining_gaps_slice,
    gr10_maintainer_roadmap_slice,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_NEXUS_LOOP = _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "nexus_loop.py"
_HOST_TASK = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "host_task.py"


def _matrix_column(
    semantics: tuple[object, ...],
    getter: object,
    column: str,
) -> None:
    matrix_by_cap = {row.capability: row for row in GR10_FINAL_CAPABILITY_MATRIX}
    assert semantics  # type: ignore[truthy-function]
    for row in semantics:  # type: ignore[union-attr]
        cap = row.capability
        expected = getter(cap)  # type: ignore[operator]
        cell = getattr(matrix_by_cap[cap], column)
        assert cell is expected, (cap, cell, expected)


def test_gr10_r7_matrix_agentic_matches_semantics_ssot() -> None:
    _matrix_column(GR10_AGENTIC_CAPABILITY_SEMANTICS, gr10_matrix_agentic_status, "agentic")


def test_gr10_r7_matrix_orchestration_matches_semantics_ssot() -> None:
    _matrix_column(
        GR10_ORCHESTRATION_CAPABILITY_SEMANTICS,
        gr10_matrix_orchestration_status,
        "orchestration",
    )


def test_gr10_r7_inference_rows_unchanged_no_residual_blocker() -> None:
    for row in GR10_FINAL_CAPABILITY_MATRIX:
        assert row.inference is gr10_matrix_inference_status(row.capability)
        if row.inference is Gr10CoverageStatus.GAP:
            pytest.fail(f"INFERENCE must not show GAP: {row.capability}")


def test_gr10_r7_no_historical_r3_defect_wording_in_catalog() -> None:
    from tests.qualification.governance.strategy import catalog as catalog_mod

    source = Path(catalog_mod.__file__).read_text(encoding="utf-8-sig")
    assert "historical R3 defect" in source or "GR-10-R3 closed" in source
    assert "PolicyDecision.DENY" not in source or "GR-10-R3" in source
    assert "kernel policy_pre DENY" in source


def test_gr10_r7_agentic_partial_rows_have_precise_reasons() -> None:
    partial = [
        row
        for row in GR10_AGENTIC_CAPABILITY_SEMANTICS
        if row.coverage is Gr10CoverageStatus.PARTIAL
    ]
    assert {row.capability for row in partial} == {
        "Inner Governance",
        "Policy evaluation",
        "MSE",
        "Governance Evidence",
    }
    for row in partial:
        assert len(row.reason) > 40
        lowered = row.reason.lower()
        assert (
            "not all" in lowered
            or "residual" in lowered
            or "optional" in lowered
            or "not enterprise-adopted for all" in lowered
        )


def test_gr10_r7_orchestration_partial_inventory() -> None:
    partial_caps = {
        row.capability
        for row in GR10_ORCHESTRATION_CAPABILITY_SEMANTICS
        if row.coverage is Gr10CoverageStatus.PARTIAL
    }
    assert partial_caps == {
        "Continuation",
        "Reliability",
        "Governance Evidence",
    }


def test_gr10_r7_gep_inventory_covers_required_geps() -> None:
    geps = {row.gep for row in GR10_GEP_COVERAGE_INVENTORY}
    required = {
        "ROOT_EXECUTION_ADMISSION",
        "AGENT_DECISION",
        "INTERRUPT",
        "PRE_MODEL",
        "TOOL_PLAN_OR_ACCESS",
        "TOOL_INVOCATION_AUTHORIZATION",
        "TOOL_INVOCATION_POLICY",
        "MEANINGFUL_SIDE_EFFECT",
        "PRE_OUTPUT",
        "POST_RUN",
    }
    assert required <= geps


def test_gr10_r7_next_remediation_targets_orchestration_inner() -> None:
    assert GR10_R7_NEXT_REMEDIATION.strategy == "ORCHESTRATION"
    assert GR10_R7_NEXT_REMEDIATION.capability == "Inner Governance"
    assert "GR-10-R8" in GR10_R7_NEXT_REMEDIATION.task_name


def test_gr10_r7_nexus_finish_task_post_run_uses_active_run_id_ast() -> None:
    source = _NEXUS_LOOP.read_text(encoding="utf-8-sig")
    assert "invoke_post_run_governance" in source
    assert "require_active_execution_identity" in source
    assert "run_id=active_run_id" in source.replace(" ", "")


def test_gr10_r7_host_task_does_not_wire_inference_at_root() -> None:
    source = _HOST_TASK.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(_HOST_TASK))
    assert "inference_executor" not in source


def test_gr10_r7_scenario_catalog_agentic_qualified_slices() -> None:
    by_id = {entry.scenario_id: entry for entry in GR10_SCENARIO_CATALOG}
    assert by_id["AGT-ROOT"].expected_status is Gr10CoverageStatus.QUALIFIED
    assert by_id["AGT-MSE-HITL"].expected_status is Gr10CoverageStatus.QUALIFIED
    assert by_id["ORCH-INNER"].expected_status is Gr10CoverageStatus.QUALIFIED
    assert by_id["ORCH-HITL"].expected_status is Gr10CoverageStatus.QUALIFIED


def test_gr10_r7_docs_record_gr10_r7_residual_requalification() -> None:
    arch = gr10_architecture_remaining_gaps_slice()
    plan = gr10_maintainer_roadmap_slice()
    assert "GR-10-R7" in plan or "GR-10-R7" in arch
    assert "INFERENCE" in arch and "NOT_APPLICABLE" in arch or "N/A" in arch
    assert "remaining INFERENCE blocker" not in arch.lower()


def test_gr10_r7_agentic_evidence_downgrade_from_overclaim() -> None:
    status = gr10_matrix_agentic_status("Governance Evidence")
    assert status is Gr10CoverageStatus.PARTIAL


def test_gr10_r7_not_applicable_not_used_to_hide_gaps() -> None:
    for semantics in (GR10_AGENTIC_CAPABILITY_SEMANTICS, GR10_ORCHESTRATION_CAPABILITY_SEMANTICS):
        for row in semantics:
            if row.applicability is Gr10Applicability.NOT_APPLICABLE:
                assert row.coverage is None
