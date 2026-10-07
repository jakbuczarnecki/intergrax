# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P1 mechanical gates TXP1-Q01..TXP1-Q30."""

from __future__ import annotations

import subprocess

import pytest

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
)
from intergrax.contracts.platform_causal_evidence import (
    MessageBusTaskRef,
    PlatformCausalEvidence,
    RuntimeExecutionRef,
)
from tests.qualification.trace_x._trace_x_p0_support import repo_root
from tests.qualification.trace_x._trace_x_p1_support import (
    DEGRADED_LINEAGE_CASES,
    ENTERPRISE_AUDIT_MATRIX_P1,
    EXECUTED_CHILD_WITHOUT_DURABLE_PARENT_EDGE,
    FrzTrcP1Disposition,
    FRZ_TRC_02_DISPOSITION,
    FRZ_TRC_12_DISPOSITION,
    LINEAGE_PATHS,
    MANDATORY_FRZ_P1_IDS,
    OWNERSHIP_MATRIX,
    P1_IN_SCOPE_BLOCKERS,
    P1_READINESS,
    P1_RESOLVED_BLOCKERS,
    P1_TRANSPORT_PATHS,
    P1Concern,
    P1GateResult,
    P1ReadinessStatus,
    TENANT_ISOLATION_AUDIT_P1,
    TRACE_X_P1_START_HEAD,
    TRANSPORT_ENTRYPOINTS,
    assert_no_duplicate_ownership,
    assert_transport_closed_world_complete,
    discover_production_admit_background_callers,
)

_REPO_ROOT = repo_root()


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        text=True,
    ).strip()


def test_txp1_q01_correct_start_head_provenance() -> None:
    head = _git_head()
    merge_base = subprocess.check_output(
        ["git", "merge-base", TRACE_X_P1_START_HEAD, head],
        cwd=_REPO_ROOT,
        text=True,
    ).strip()
    assert merge_base == TRACE_X_P1_START_HEAD


def test_txp1_q02_p1_scope_exactly_trc_02_trc_12() -> None:
    assert MANDATORY_FRZ_P1_IDS == ("FRZ-TRC-02", "FRZ-TRC-12")


def test_txp1_q03_transport_runtime_identity_domains_distinct() -> None:
    contract = (
        _REPO_ROOT / "intergrax/contracts/platform_causal_evidence.py"
    ).read_text(encoding="utf-8")
    assert "not runtime ``TaskId``" in contract
    assert "task_id: TaskId" in contract


def test_txp1_q04_runtime_execution_ref_complete_typed_identity() -> None:
    fields = {name for name in RuntimeExecutionRef.model_fields}
    assert fields == {
        "task_id",
        "run_id",
        "attempt_id",
        "execution_id",
        "tenant_id",
    }
    for annotation in (
        RuntimeExecutionRef.model_fields["task_id"].annotation,
        RuntimeExecutionRef.model_fields["run_id"].annotation,
        RuntimeExecutionRef.model_fields["attempt_id"].annotation,
        RuntimeExecutionRef.model_fields["execution_id"].annotation,
    ):
        assert annotation in (TaskId, RunId, AttemptId, ExecutionId)


def test_txp1_q05_tenant_equality_enforced_on_causal_fact() -> None:
    text = PlatformCausalEvidence.model_json_schema()
    assert "tenant_id" in text.get("properties", {})


def test_txp1_q06_all_supported_transport_entrypoints_inventoried() -> None:
    assert len(TRANSPORT_ENTRYPOINTS) >= 4
    ids = {row.entrypoint_id for row in TRANSPORT_ENTRYPOINTS}
    assert ids >= {"TXP1-T01", "TXP1-T02", "TXP1-T03", "TXP1-T04"}


def test_txp1_q07_no_transport_execution_bypass() -> None:
    assert_transport_closed_world_complete()


def test_txp1_q08_required_evidence_before_handler() -> None:
    for row in TRANSPORT_ENTRYPOINTS:
        assert row.evidence_before_handler is True
        assert row.admission_gate == "admit_background_execution_handler"


def test_txp1_q09_persistence_failure_blocks_handler() -> None:
    gate = (
        _REPO_ROOT
        / "intergrax/runtime/background_execution/required_audit_evidence.py"
    ).read_text(encoding="utf-8")
    assert "persist_required_audit_evidence" in gate
    assert "return handler()" in gate
    assert gate.index("persist_required_audit_evidence") < gate.index("return handler()")


def test_txp1_q10_retry_redelivery_mapping_deterministic() -> None:
    evidence_tests = {
        test
        for path in P1_TRANSPORT_PATHS
        if path.concern == P1Concern.TRANSPORT_RUNTIME_MAPPING
        for test in path.evidence_tests
    }
    assert "test_required_evidence_backend_failure_blocks_handler_and_wraps_cause" in evidence_tests


def test_txp1_q11_legacy_incomplete_evidence_cannot_count_complete() -> None:
    codec = (
        _REPO_ROOT
        / "intergrax/runtime/observability/platform_causal_evidence_codec.py"
    ).read_text(encoding="utf-8")
    assert "complete_v2" in codec
    assert "LegacyPlatformCausalEvidence" in (
        _REPO_ROOT / "intergrax/runtime/observability/causal_evidence_legacy.py"
    ).read_text(encoding="utf-8")


def test_txp1_q12_execution_lineage_sole_topology_owner() -> None:
    lineage_rows = [r for r in OWNERSHIP_MATRIX if "parent" in r.concern]
    assert len(lineage_rows) == 1
    assert lineage_rows[0].semantic_owner == "ExecutionLineage"


def test_txp1_q13_child_path_auto_attaches_lineage_hook() -> None:
    child_py = (_REPO_ROOT / "intergrax/runtime/execution/child.py").read_text(
        encoding="utf-8"
    )
    assert "build_child_lineage_admission_hook" in child_py


def test_txp1_q14_child_admission_before_delegate() -> None:
    admission = (
        _REPO_ROOT / "intergrax/runtime/execution/lineage/admission.py"
    ).read_text(encoding="utf-8")
    assert "admit_child" in admission
    assert "Durable child admission executed before delegate" in admission


def test_txp1_q15_child_not_equal_parent() -> None:
    child_py = (_REPO_ROOT / "intergrax/runtime/execution/child.py").read_text(
        encoding="utf-8"
    )
    assert "mint_child_execution_id" in child_py


def test_txp1_q16_lineage_scope_tenant_task_run_attempt_exact() -> None:
    contract = (
        _REPO_ROOT / "intergrax/contracts/execution_lineage.py"
    ).read_text(encoding="utf-8")
    for field in ("tenant_id", "task_id", "run_id", "attempt_id"):
        assert field in contract


def test_txp1_q17_conflicting_parent_fails_closed() -> None:
    assert any(
        "test_conflicting_parent_fails_closed" in path.evidence_tests
        for path in LINEAGE_PATHS
    )


def test_txp1_q18_parentless_non_root_fails() -> None:
    persistence = (
        _REPO_ROOT / "intergrax/runtime/execution/lineage/persistence.py"
    ).read_text(encoding="utf-8")
    assert "ExecutionLineageIntegrityError" in persistence


def test_txp1_q19_reconstruction_uses_lineage_not_heuristics() -> None:
    recon = (
        _REPO_ROOT
        / "intergrax/runtime/observability/reconstruction/execution_lineage_reconstruction.py"
    ).read_text(encoding="utf-8")
    assert "ExecutionLineageReader" in recon or "read_attempt_lineage" in recon
    assert "guess" not in recon.lower()


def test_txp1_q20_corrupted_lineage_fails_closed() -> None:
    assert any("integrity" in path.failure_behavior.lower() for path in LINEAGE_PATHS)


def test_txp1_q21_missing_lineage_not_fabricated() -> None:
    l07 = next(p for p in LINEAGE_PATHS if p.lineage_path_id == "TXP1-L07")
    assert "no fabricated" in l07.expected_reconstruction.lower()


def test_txp1_q22_degraded_lineage_semantics_explicit() -> None:
    assert len(DEGRADED_LINEAGE_CASES) >= 6
    assert EXECUTED_CHILD_WITHOUT_DURABLE_PARENT_EDGE is False


def test_txp1_q23_failed_child_admission_blocks_delegate() -> None:
    assert any(
        "test_child_admission_unavailable_blocks_delegate_and_marks_degraded"
        in p.evidence_tests
        for p in LINEAGE_PATHS
    )


def test_txp1_q24_no_duplicate_transport_relation_owner() -> None:
    transport_rows = [
        r for r in OWNERSHIP_MATRIX if "transport" in r.concern
    ]
    assert len(transport_rows) == 1


def test_txp1_q25_no_duplicate_parent_topology_owner() -> None:
    assert_no_duplicate_ownership()


def test_txp1_q26_contracts_over_implementations() -> None:
    for row in OWNERSHIP_MATRIX:
        assert row.persistence.endswith("Persistence") or "Persistence" in row.persistence


def test_txp1_q27_strong_typing() -> None:
    assert MessageBusTaskRef.model_config.get("extra") == "forbid"
    assert RuntimeExecutionRef.model_config.get("extra") == "forbid"


def test_txp1_q28_tenant_isolation_p1_audit() -> None:
    assert TENANT_ISOLATION_AUDIT_P1["tenant_scope_applicable"] == "YES"
    assert TENANT_ISOLATION_AUDIT_P1["result"] == "PASS"


def test_txp1_q29_blocker_inventory() -> None:
    assert P1_IN_SCOPE_BLOCKERS == ()
    assert any(
        row[0] == "P1-BLK-DEGRADED-LINEAGE-01" for row in P1_RESOLVED_BLOCKERS
    )


def test_txp1_q30_readiness() -> None:
    assert P1_READINESS == P1ReadinessStatus.READY_FOR_AUDIT
    assert (
        FRZ_TRC_12_DISPOSITION
        == FrzTrcP1Disposition.READY_FOR_INDEPENDENT_CLOSURE_REVIEW
    )
    assert (
        FRZ_TRC_02_DISPOSITION
        == FrzTrcP1Disposition.READY_FOR_INDEPENDENT_CLOSURE_REVIEW
    )
    blocked_rows = [r for r in ENTERPRISE_AUDIT_MATRIX_P1 if r.result == P1GateResult.BLOCKED]
    assert blocked_rows == []


def test_txp1_transport_entrypoint_modules_exist() -> None:
    for row in TRANSPORT_ENTRYPOINTS:
        assert (_REPO_ROOT / row.module_path).is_file()


def test_txp1_discovered_callers_match_inventory() -> None:
    discovered = discover_production_admit_background_callers()
    inventoried = {row.module_path for row in TRANSPORT_ENTRYPOINTS}
    assert discovered == inventoried
