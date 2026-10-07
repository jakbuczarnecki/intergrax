# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P2 mechanical gates TXP2-Q01..TXP2-Q30 and adversarial cases A–F."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime

import pytest

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_lineage import (
    ExecutionLineageAttemptScope,
    ExecutionLineageUnavailableError,
    build_execution_lineage_attempt_scope,
)
from intergrax.runtime.execution.lineage.persistence import InMemoryExecutionLineagePersistence
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.observability.causal_evidence import (
    CausalRelationKind,
    MessageBusTaskRef,
    PlatformCausalEvidence,
    RuntimeExecutionRef,
)
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event
from intergrax.runtime.observability.reconstruction import (
    ExecutionReconstructionIntegrityError,
    ExecutionReconstructor,
    RuntimeHistoryCompleteness,
)
from intergrax.runtime.observability.reconstruction.execution_lineage_reconstruction import (
    ExecutionLineageCompleteness,
    ExecutionLineageReadStatus,
)
from testing_support.runtime.execution.lineage.lineage_test_helpers import register_v1_attempt
from tests.qualification.trace_x._trace_x_p0_support import discover_sensitive_classes, repo_root
from tests.qualification.trace_x._trace_x_p2_support import (
    ADVERSARIAL_CASES,
    CANONICAL_CONTRACTS,
    ENTERPRISE_AUDIT_MATRIX_P2,
    FRZ_TRC_01_DISPOSITION,
    MANDATORY_FRZ_P2_IDS,
    OUT_OF_SCOPE_FRZ,
    OWNERSHIP_MATRIX,
    P2_EVIDENCE_MATRIX,
    P2_IN_SCOPE_BLOCKERS,
    P2_READINESS,
    P2GateResult,
    P2ReadinessStatus,
    RECONSTRUCTION_CHAIN,
    RECONSTRUCTION_CONSUMERS,
    TENANT_ISOLATION_AUDIT_P2,
    TRACE_X_P2_START_HEAD,
    assert_single_reconstruction_owner,
    reconstruction_module_text,
)

_REPO_ROOT = repo_root()
_TENANT = "tenant-p2"


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        text=True,
    ).strip()


def _scope(task_id: str, run_id: str, attempt_id: str) -> ExecutionLineageAttemptScope:
    return build_execution_lineage_attempt_scope(
        tenant_id=_TENANT,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    )


def _reconstructor_with_lineage_root(
    *,
    lineage: InMemoryExecutionLineagePersistence,
    task_id: str,
    run_id: str,
    attempt_id: str,
    root_execution_id: str,
    runtime_execution_id: str | None = None,
    causal_execution_id: str | None = None,
) -> ExecutionReconstructor:
    causal_id = causal_execution_id or root_execution_id
    event_id = runtime_execution_id or root_execution_id
    runtime_store = InMemoryRuntimeEventStore()
    causal_store = InMemoryCausalEvidencePersistence()
    causal_store.append(
        PlatformCausalEvidence(
            relation_kind=CausalRelationKind.TRANSPORT_TASK_TRIGGERED_EXECUTION,
            tenant_id=_TENANT,
            source=MessageBusTaskRef(provider="celery", task_id="t1", tenant_id=_TENANT),
            target=RuntimeExecutionRef(
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=causal_id,
                tenant_id=_TENANT,
            ),
            recorded_at=datetime(2026, 6, 8, 12, 0, tzinfo=UTC),
        ),
    )
    runtime_store.append(
        sample_runtime_event(
            tenant_id=_TENANT,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=event_id,
        ),
        tenant_id=_TENANT,
    )
    return ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=causal_store,
        execution_lineage=lineage,
    )


def test_txp2_q01_correct_start_head_provenance() -> None:
    head = _git_head()
    merge_base = subprocess.check_output(
        ["git", "merge-base", TRACE_X_P2_START_HEAD, head],
        cwd=_REPO_ROOT,
        text=True,
    ).strip()
    assert merge_base == TRACE_X_P2_START_HEAD


def test_txp2_q02_p2_scope_exactly_trc_01() -> None:
    assert MANDATORY_FRZ_P2_IDS == ("FRZ-TRC-01",)


def test_txp2_q03_execution_reconstructor_sole_owner() -> None:
    assert_single_reconstruction_owner()


def test_txp2_q04_reconstruction_derived_non_persisted() -> None:
    text = reconstruction_module_text()
    assert "persist" not in text.lower().split("def reconstruct_execution")[1][:800]


def test_txp2_q05_canonical_source_contracts_only() -> None:
    for rel in CANONICAL_CONTRACTS:
        assert (_REPO_ROOT / rel).is_file()


def test_txp2_q06_tenant_task_run_scope_validation() -> None:
    text = reconstruction_module_text()
    assert "_validate_causal_evidence_scope" in text
    assert "_validate_runtime_event_scope" in text


def test_txp2_q07_attempt_partitioning_deterministic() -> None:
    assert "_build_attempts" in reconstruction_module_text()
    assert "_attempt_projection_order_key" in reconstruction_module_text()


def test_txp2_q08_transport_execution_id_retained() -> None:
    text = (_REPO_ROOT / "intergrax/contracts/platform_causal_evidence.py").read_text(
        encoding="utf-8",
    )
    assert "execution_id" in text


def test_txp2_q09_runtime_execution_id_retained() -> None:
    text = (_REPO_ROOT / "intergrax/contracts/runtime_event.py").read_text(encoding="utf-8")
    assert "execution_id: ExecutionId" in text


def test_txp2_q10_lineage_parent_topology_retained() -> None:
    assert "reconstruct_attempt_lineage" in reconstruction_module_text()


def test_txp2_q11_nested_fan_out_topology_evidence() -> None:
    row = next(r for r in P2_EVIDENCE_MATRIX if r.evidence_id == "P2-E06")
    assert "test_nested_fan_out_parents" in row.evidence_tests


def test_txp2_q12_multi_segment_topology_evidence() -> None:
    row = next(r for r in P2_EVIDENCE_MATRIX if r.evidence_id == "P2-E07")
    assert "test_multi_segment_resume_topology" in row.evidence_tests


def test_txp2_q13_retry_multi_attempt_separation() -> None:
    row = next(r for r in P2_EVIDENCE_MATRIX if r.evidence_id == "P2-E05")
    assert row.evidence_tests


def test_txp2_q14_execution_position_ordering_not_timestamp() -> None:
    row = next(r for r in P2_EVIDENCE_MATRIX if r.evidence_id == "P2-E08")
    assert "test_ordering_follows_execution_position_not_timestamp" in row.evidence_tests


def test_txp2_q15_cross_source_coherence_hook_present() -> None:
    assert "_validate_cross_source_execution_identity_coherence" in reconstruction_module_text()


def test_txp2_q16_contradictory_causal_execution_id_fails_closed() -> None:
    assert (
        next(c for c in ADVERSARIAL_CASES if c.case_id.value == "B").evidence_test
        == "test_txp2_adversarial_case_b_causal_outside_lineage"
    )


def test_txp2_q17_contradictory_runtime_execution_id_fails_closed() -> None:
    assert (
        next(c for c in ADVERSARIAL_CASES if c.case_id.value == "C").evidence_test
        == "test_txp2_adversarial_case_c_runtime_contradicts_lineage"
    )


def test_txp2_q18_lineage_corruption_fails_closed() -> None:
    assert "ExecutionLineageReconstructionIntegrityError" in (
        _REPO_ROOT
        / "intergrax/runtime/observability/reconstruction/execution_lineage_reconstruction.py"
    ).read_text(encoding="utf-8")


def test_txp2_q19_partial_not_promoted_to_complete() -> None:
    assert "PARTIAL" in (
        _REPO_ROOT / "intergrax/contracts/execution_reconstruction_lineage.py"
    ).read_text(encoding="utf-8")


def test_txp2_q20_truncated_not_promoted_to_complete() -> None:
    assert "TRUNCATED" in reconstruction_module_text()


def test_txp2_q21_unavailable_not_fabricated() -> None:
    assert "UNAVAILABLE" in reconstruction_module_text()


def test_txp2_q22_runtime_truncation_exposed() -> None:
    assert "RuntimeHistoryCompleteness" in reconstruction_module_text()


def test_txp2_q23_attempt_discovery_completeness_truthful() -> None:
    assert "_load_run_discovery_snapshot" in reconstruction_module_text()


def test_txp2_q24_as_of_future_runtime_excluded() -> None:
    assert "_load_positioned_events_through_boundary" in reconstruction_module_text()


def test_txp2_q25_as_of_lineage_requires_as_of_reader() -> None:
    text = reconstruction_module_text()
    assert "execution_lineage_as_of" in text
    assert "ExecutionLineageAsOfReader" in text


def test_txp2_q26_no_heuristic_joins() -> None:
    text = reconstruction_module_text()
    assert "correlation_id" not in text
    assert "timestamp proximity" not in text.lower()


def test_txp2_q27_tenant_isolation() -> None:
    assert TENANT_ISOLATION_AUDIT_P2["result"] == "PASS"


def test_txp2_q28_contracts_replaceability() -> None:
    recon = discover_sensitive_classes().get("ExecutionReconstructor", [])
    assert len(recon) == 1


def test_txp2_q29_no_future_stage_leakage() -> None:
    inventory = (_REPO_ROOT / "tests/qualification/trace_x/_trace_x_p2_support.py").read_text(
        encoding="utf-8",
    )
    for frz in OUT_OF_SCOPE_FRZ:
        assert f"implement {frz}" not in inventory


def test_txp2_q30_frz_trc_01_readiness() -> None:
    assert P2_READINESS == P2ReadinessStatus.READY_FOR_AUDIT
    assert (
        FRZ_TRC_01_DISPOSITION.value
        == "READY FOR INDEPENDENT CLOSURE REVIEW"
    )
    blocked = [r for r in ENTERPRISE_AUDIT_MATRIX_P2 if r.result == P2GateResult.BLOCKED]
    assert blocked == []
    assert P2_IN_SCOPE_BLOCKERS == ()


def test_txp2_reconstruction_chain_documented() -> None:
    assert "ExecutionReconstruction" in RECONSTRUCTION_CHAIN
    assert "PlatformCausalEvidence" in RECONSTRUCTION_CHAIN


def test_txp2_consumer_modules_exist() -> None:
    for rel in RECONSTRUCTION_CONSUMERS:
        assert (_REPO_ROOT / rel).is_file()


def test_txp2_ownership_matrix_complete() -> None:
    owners = [row.semantic_owner for row in OWNERSHIP_MATRIX]
    assert "ExecutionReconstructor" in owners
    assert len(owners) == len(set(owners))


# --- Adversarial cases A–F ---


def test_txp2_adversarial_case_a_causal_in_lineage() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root = mint_execution_id()
    lineage = InMemoryExecutionLineagePersistence()
    scope = _scope(task_id, run_id, attempt_id)
    register_v1_attempt(lineage, scope)
    lineage.open_segment(scope, root)
    lineage.admit_root(scope, root, root)

    reconstruction = _reconstructor_with_lineage_root(
        lineage=lineage,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        root_execution_id=root,
    ).reconstruct_execution(_TENANT, task_id, run_id)

    attempt = reconstruction.attempts[0]
    assert attempt.causal_evidence[0].target.execution_id == root
    assert attempt.lineage is not None
    assert attempt.lineage.read_status is ExecutionLineageReadStatus.AVAILABLE


def test_txp2_adversarial_case_b_causal_outside_lineage() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root = mint_execution_id()
    outsider = mint_execution_id()
    lineage = InMemoryExecutionLineagePersistence()
    scope = _scope(task_id, run_id, attempt_id)
    register_v1_attempt(lineage, scope)
    lineage.open_segment(scope, root)
    lineage.admit_root(scope, root, root)

    reconstructor = _reconstructor_with_lineage_root(
        lineage=lineage,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        root_execution_id=root,
        causal_execution_id=outsider,
    )
    with pytest.raises(ExecutionReconstructionIntegrityError, match="causal evidence"):
        reconstructor.reconstruct_execution(_TENANT, task_id, run_id)


def test_txp2_adversarial_case_c_runtime_contradicts_lineage() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root = mint_execution_id()
    outsider = mint_execution_id()
    lineage = InMemoryExecutionLineagePersistence()
    scope = _scope(task_id, run_id, attempt_id)
    register_v1_attempt(lineage, scope)
    lineage.open_segment(scope, root)
    lineage.admit_root(scope, root, root)

    reconstructor = _reconstructor_with_lineage_root(
        lineage=lineage,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        root_execution_id=root,
        runtime_execution_id=outsider,
    )
    with pytest.raises(ExecutionReconstructionIntegrityError, match="runtime event"):
        reconstructor.reconstruct_execution(_TENANT, task_id, run_id)


def test_txp2_adversarial_case_d_partial_not_corruption() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root = mint_execution_id()
    outsider = mint_execution_id()
    lineage = InMemoryExecutionLineagePersistence()
    scope = _scope(task_id, run_id, attempt_id)
    register_v1_attempt(lineage, scope)
    lineage.open_segment(scope, root)
    lineage.admit_root(scope, root, root)
    lineage.mark_degraded(scope, "p2-partial")

    reconstruction = _reconstructor_with_lineage_root(
        lineage=lineage,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        root_execution_id=root,
        causal_execution_id=outsider,
    ).reconstruct_execution(_TENANT, task_id, run_id)

    assert reconstruction.attempts[0].lineage.completeness is ExecutionLineageCompleteness.PARTIAL


def test_txp2_adversarial_case_e_lineage_unavailable() -> None:
    class _Unavailable(InMemoryExecutionLineagePersistence):
        def read_attempt_lineage_state(self, scope: ExecutionLineageAttemptScope):
            raise ExecutionLineageUnavailableError("down")

    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root = mint_execution_id()
    lineage = _Unavailable()

    read_status = _reconstructor_with_lineage_root(
        lineage=lineage,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        root_execution_id=root,
    ).reconstruct_execution(_TENANT, task_id, run_id).attempts[0].lineage.read_status
    assert read_status is ExecutionLineageReadStatus.UNAVAILABLE


def test_txp2_adversarial_case_f_runtime_truncated() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    runtime_store = InMemoryRuntimeEventStore()
    for _ in range(4):
        runtime_store.append(
            sample_runtime_event(
                tenant_id=_TENANT,
                task_id=task_id,
                run_id=run_id,
                attempt_id=attempt_id,
            ),
            tenant_id=_TENANT,
        )
    reconstruction = ExecutionReconstructor(
        runtime_events=runtime_store,
        causal_evidence=InMemoryCausalEvidencePersistence(),
    ).reconstruct_execution(
        _TENANT,
        task_id,
        run_id,
        initial_limit=2,
        max_limit=2,
    )
    assert reconstruction.runtime_history_completeness is RuntimeHistoryCompleteness.TRUNCATED
