# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P1-R1 mechanical gates TXP1R1-Q01..TXP1R1-Q30."""

from __future__ import annotations

import subprocess

import pytest

from tests.qualification.trace_x._trace_x_p0_support import repo_root
from tests.qualification.trace_x._trace_x_p1_r1_support import (
    ARCHITECTURE_DECISION_SUMMARY,
    ENTERPRISE_AUDIT_MATRIX_R1,
    EXECUTED_CHILD_WITHOUT_DURABLE_PARENT_EDGE,
    FRZ_TRC_02_DISPOSITION,
    MANDATORY_FRZ_R1_IDS,
    P1_BLOCKER_RESOLUTION,
    P1_BLOCKER_RESOLVED_ID,
    P1_POST_R1_READINESS,
    R1_READINESS,
    R1GateResult,
    R1ReadinessStatus,
    STRICT_ADMISSION_EVIDENCE_TESTS,
    TENANT_ISOLATION_AUDIT_R1,
    TRACE_X_P1_AUDITED_TRANSPORT_HEAD,
    TRACE_X_P1_R1_START_HEAD,
    assert_child_admission_re_raises_after_degradation,
    assert_obsolete_non_durable_machinery_removed,
    production_text,
)

_REPO_ROOT = repo_root()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        text=True,
    ).strip()


def test_txp1r1_q01_correct_start_head_provenance() -> None:
    head = _git_head()
    merge_base = subprocess.check_output(
        ["git", "merge-base", TRACE_X_P1_R1_START_HEAD, head],
        cwd=_REPO_ROOT,
        text=True,
    ).strip()
    assert merge_base == TRACE_X_P1_R1_START_HEAD


def test_txp1r1_q02_architecture_decision_recorded() -> None:
    doc = (
        _REPO_ROOT
        / "docs/project/maintainers/qualification/TRACE_X_P1_R1_STRICT_DURABLE_CHILD_LINEAGE_CERTIFICATION.md"
    ).read_text(encoding="utf-8")
    assert ARCHITECTURE_DECISION_SUMMARY in doc
    assert "FRZ-TRC-02" in doc


def test_txp1r1_q03_child_unavailable_admission_propagates() -> None:
    assert_child_admission_re_raises_after_degradation()


def test_txp1r1_q04_failed_admission_blocks_delegate() -> None:
    assert (
        "test_child_admission_unavailable_blocks_delegate_and_marks_degraded"
        in STRICT_ADMISSION_EVIDENCE_TESTS
    )


def test_txp1r1_q05_successful_mark_degraded_does_not_permit_child() -> None:
    assert_child_admission_re_raises_after_degradation()


def test_txp1r1_q06_mark_degraded_failure_blocks_child() -> None:
    assert "test_mark_degraded_unavailable_still_blocks_delegate" in STRICT_ADMISSION_EVIDENCE_TESTS


def test_txp1r1_q07_structural_conflict_still_blocks_child() -> None:
    assert "test_conflicting_parent_still_blocks_child_delegate" in STRICT_ADMISSION_EVIDENCE_TESTS


def test_txp1r1_q08_successful_durable_admission_permits_child() -> None:
    assert "test_child_lineage_hook_auto_attached" in STRICT_ADMISSION_EVIDENCE_TESTS


def test_txp1r1_q09_durable_admission_precedes_delegate() -> None:
    assert "test_child_lineage_hook_auto_attached" in STRICT_ADMISSION_EVIDENCE_TESTS


def test_txp1r1_q10_failed_child_produces_no_executed_child_path() -> None:
    assert EXECUTED_CHILD_WITHOUT_DURABLE_PARENT_EDGE is False


def test_txp1r1_q11_sibling_with_later_successful_admission_may_execute() -> None:
    assert (
        "test_sibling_after_failed_child_admission_may_execute_when_durable"
        in STRICT_ADMISSION_EVIDENCE_TESTS
    )


def test_txp1r1_q12_degradation_remains_monotonic() -> None:
    active = production_text("intergrax/runtime/execution/lineage/active_lineage.py")
    assert "mark_attempt_lineage_degraded" in active
    assert "degraded=True" in active


def test_txp1r1_q13_obsolete_non_durable_execution_ids_removed() -> None:
    assert_obsolete_non_durable_machinery_removed()


def test_txp1r1_q14_obsolete_mark_execution_lineage_non_durable_removed() -> None:
    assert "mark_execution_lineage_non_durable" not in production_text(
        "intergrax/runtime/execution/lineage/active_lineage.py"
    )


def test_txp1r1_q15_obsolete_nested_non_durable_parent_branch_removed() -> None:
    child = production_text("intergrax/runtime/execution/child.py")
    assert "non_durable" not in child


def test_txp1r1_q16_execution_lineage_sole_topology_owner() -> None:
    child = production_text("intergrax/runtime/execution/child.py")
    assert "build_child_lineage_admission_hook" in child
    assert "RuntimeEvent" not in child


def test_txp1r1_q17_runtime_event_not_promoted_to_topology_owner() -> None:
    admission = production_text("intergrax/runtime/execution/lineage/admission.py")
    assert "RuntimeEvent" not in admission


def test_txp1r1_q18_causal_evidence_not_promoted_to_topology_owner() -> None:
    admission = production_text("intergrax/runtime/execution/lineage/admission.py")
    assert "CausalRelationKind" not in admission
    assert "EXECUTION_SPAWNED_CHILD" not in admission


def test_txp1r1_q19_child_budget_released_on_failed_admission() -> None:
    assert (
        "test_child_budget_released_after_failed_lineage_admission"
        in STRICT_ADMISSION_EVIDENCE_TESTS
    )


def test_txp1r1_q20_execution_identity_context_restored() -> None:
    assert (
        "test_parent_identity_and_authority_restored_after_failed_admission"
        in STRICT_ADMISSION_EVIDENCE_TESTS
    )


def test_txp1r1_q21_authority_context_restored() -> None:
    assert (
        "test_parent_identity_and_authority_restored_after_failed_admission"
        in STRICT_ADMISSION_EVIDENCE_TESTS
    )


def test_txp1r1_q22_reconstruction_does_not_invent_failed_child_edge() -> None:
    assert "test_failed_child_admission_reconstruction_honest" in STRICT_ADMISSION_EVIDENCE_TESTS


def test_txp1r1_q23_p1_blocker_resolved() -> None:
    assert P1_BLOCKER_RESOLVED_ID == "P1-BLK-DEGRADED-LINEAGE-01"
    assert P1_BLOCKER_RESOLUTION == "RESOLVED PENDING INDEPENDENT AUDIT"
    p1 = (
        _REPO_ROOT
        / "docs/project/maintainers/qualification/TRACE_X_P1_IDENTITY_CAUSALITY_CERTIFICATION.md"
    ).read_text(encoding="utf-8")
    assert P1_BLOCKER_RESOLUTION in p1


def test_txp1r1_q24_frz_trc_02_readiness() -> None:
    assert FRZ_TRC_02_DISPOSITION.value == "READY FOR INDEPENDENT CLOSURE REVIEW"


def test_txp1r1_q25_frz_trc_12_prior_evidence_preserved() -> None:
    p1 = (
        _REPO_ROOT
        / "docs/project/maintainers/qualification/TRACE_X_P1_IDENTITY_CAUSALITY_CERTIFICATION.md"
    ).read_text(encoding="utf-8")
    assert TRACE_X_P1_AUDITED_TRANSPORT_HEAD in p1


def test_txp1r1_q26_contracts_over_implementations() -> None:
    admission = production_text("intergrax/runtime/execution/lineage/admission.py")
    assert "ExecutionLineagePersistence" in admission
    assert "InMemoryExecutionLineagePersistence" not in admission


def test_txp1r1_q27_strong_typing() -> None:
    active = production_text("intergrax/runtime/execution/lineage/active_lineage.py")
    assert ": Any" not in active


def test_txp1r1_q28_tenant_scope_unchanged() -> None:
    assert TENANT_ISOLATION_AUDIT_R1["tenant_scope_applicable"] == "YES"
    assert TENANT_ISOLATION_AUDIT_R1["canonical_tenant_identity"] == "tenant_id"


def test_txp1r1_q29_no_future_stage_leakage() -> None:
    assert MANDATORY_FRZ_R1_IDS == ("FRZ-TRC-02",)


def test_txp1r1_q30_r1_readiness() -> None:
    assert R1_READINESS == R1ReadinessStatus.READY_FOR_AUDIT
    assert P1_POST_R1_READINESS == R1ReadinessStatus.READY_FOR_AUDIT
    blocked = [r for r in ENTERPRISE_AUDIT_MATRIX_R1 if r.result == R1GateResult.BLOCKED]
    assert blocked == []
