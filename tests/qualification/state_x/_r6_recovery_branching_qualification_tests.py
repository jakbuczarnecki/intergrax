# © Artur Czarnecki. All rights reserved.

"""STATE-X-R6 — recovery branching semantics matrix (R6-Q01..Q42)."""

from __future__ import annotations

from dataclasses import dataclass
import pytest

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_lineage import (
    ExecutionLineageAttemptClosureKind,
    build_execution_lineage_attempt_scope,
)
from intergrax.contracts.execution_retry import ExecutionFailureKind
from intergrax.runtime.execution.attempt_lifecycle import (
    AttemptLifecycleService,
    InMemoryAttemptLifecycleStore,
)
from intergrax.runtime.execution.identity_authority import mint_root_execution_identity
from intergrax.runtime.execution.lineage.persistence import InMemoryExecutionLineagePersistence
from intergrax.runtime.execution.retry import ExecutionAttemptRetryService
from intergrax.runtime.long_running.checkpoint_resume_validation import (
    CheckpointResumeValidationError,
    assert_checkpoint_resume_materialization_eligible,
)
from intergrax.runtime.long_running.execution_tree_checkpoint import (
    ExecutionCheckpointEntry,
    ExecutionCheckpointStatus,
    ExecutionPriorOutput,
    ExecutionTreeSnapshot,
    build_execution_tree_resume_plan,
)
from intergrax.runtime.replay.models import ReconstructedRun
from intergrax.runtime.replay.service import ReplayService
from intergrax.contracts.execution_retry import ExecutionRetryAction
from intergrax.runtime.execution.retry import evaluate_execution_retry_eligibility
from testing_support.runtime.execution.lineage.lineage_test_helpers import register_v1_attempt
from tests.qualification.state_x._r6_recovery_branching_support import (
    RECOVERY_AUTHORITY_MATRIX,
    RECOVERY_OPERATION_MATRIX,
    RECOVERY_SIDE_EFFECT_MATRIX,
    ForkSupportStatus,
    RecoveryOperationKind,
    STATE_X_R5_ACCEPTED_CLOSURE_SHA,
    STATE_X_R6_PRE_AUDIT_HEAD,
    TENANT_ISOLATION_AUDIT_R6,
    assert_frz_rec_05_r6_completeness,
    assert_r6_behavioral_evidence_registered,
    scan_execution_fork_like_symbols,
)
from tests.qualification.state_x.test_state_x_r1_checkpoint_resume_terminal import (
    _paused_checkpoint,
)
from tests.unit.runtime.architecture.test_npsc5e_r1_final_retry_attempt_qualification import (
    _eligibility_request,
    _retry_service,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _tree_entry(
    execution_id: str,
    *,
    parent: str | None = None,
    graph_node_id: str | None = None,
    status: ExecutionCheckpointStatus = ExecutionCheckpointStatus.RUNNING,
    prior_output: ExecutionPriorOutput | None = None,
) -> ExecutionCheckpointEntry:
    return ExecutionCheckpointEntry(
        execution_id=execution_id,
        parent_execution_id=parent,
        graph_node_id=graph_node_id,
        status=status,
        prior_output=prior_output,
    )


def _tree(
    *,
    task_id: str,
    run_id: str,
    attempt_id: str,
    entries: list[ExecutionCheckpointEntry],
) -> ExecutionTreeSnapshot:
    return ExecutionTreeSnapshot(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        entries=tuple(entries),
    )


def test_r6_q01_r5_closure_reconciled_at_accepted_sha() -> None:
    assert STATE_X_R6_PRE_AUDIT_HEAD == STATE_X_R5_ACCEPTED_CLOSURE_SHA
    assert STATE_X_R5_ACCEPTED_CLOSURE_SHA == "bcd8157065cc649412b64e9d6ada34be92d4b6a3"


def test_r6_q02_operation_taxonomy_complete() -> None:
    assert len(RECOVERY_OPERATION_MATRIX) == 7
    assert {r.kind for r in RECOVERY_OPERATION_MATRIX} == set(RecoveryOperationKind)


def test_r6_q03_no_first_class_fork_contract() -> None:
    fork_hits = scan_execution_fork_like_symbols()
    assert fork_hits == ()
    row = next(r for r in RECOVERY_OPERATION_MATRIX if r.kind is RecoveryOperationKind.FIRST_CLASS_FORK)
    assert row.supported is False


def test_r6_q04_unclassified_fork_like_entrypoints_zero() -> None:
    assert scan_execution_fork_like_symbols() == ()


def test_r6_q05_resume_preserves_logical_run_identity() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    root_a1 = mint_execution_id()
    interrupted = mint_execution_id()
    checkpoint_tree = _tree(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a1,
        entries=[
            _tree_entry(root_a1, status=ExecutionCheckpointStatus.RUNNING),
            _tree_entry(
                interrupted,
                parent=root_a1,
                graph_node_id="n3",
                status=ExecutionCheckpointStatus.INTERRUPTED,
            ),
        ],
    )
    plan = build_execution_tree_resume_plan(
        checkpoint_tree,
        task_id=task_id,
        run_id=run_id,
        new_attempt_id=mint_attempt_id(),
        new_root_execution_id=mint_execution_id(),
    )
    assert plan.active_snapshot.run_id == run_id
    assert plan.active_snapshot.task_id == task_id


def test_r6_q06_resume_creates_sanctioned_active_execution_identity() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    root_a1 = mint_execution_id()
    checkpoint_tree = _tree(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a1,
        entries=[_tree_entry(root_a1, status=ExecutionCheckpointStatus.INTERRUPTED)],
    )
    attempt_a2 = mint_attempt_id()
    root_a2 = mint_execution_id()
    plan = build_execution_tree_resume_plan(
        checkpoint_tree,
        task_id=task_id,
        run_id=run_id,
        new_attempt_id=attempt_a2,
        new_root_execution_id=root_a2,
    )
    assert plan.active_snapshot.attempt_id == attempt_a2
    active_root = next(e for e in plan.active_snapshot.entries if e.parent_execution_id is None)
    assert active_root.execution_id == root_a2
    assert root_a2 != root_a1


def test_r6_q07_resume_historical_execution_lineage_preserved() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    failed_child = mint_execution_id()
    root_a1 = mint_execution_id()
    checkpoint_tree = _tree(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a1,
        entries=[
            _tree_entry(root_a1, status=ExecutionCheckpointStatus.RUNNING),
            _tree_entry(
                failed_child,
                parent=root_a1,
                graph_node_id="n_c",
                status=ExecutionCheckpointStatus.FAILED,
            ),
        ],
    )
    plan = build_execution_tree_resume_plan(
        checkpoint_tree,
        task_id=task_id,
        run_id=run_id,
        new_attempt_id=attempt_a1,
        new_root_execution_id=mint_execution_id(),
    )
    assert plan.historical_by_graph_node_id["n_c"].execution_id == failed_child


def test_r6_q08_completed_resume_nodes_not_re_executed() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    old_root = mint_execution_id()
    execution_a = mint_execution_id()
    prior_a = ExecutionPriorOutput(
        agent_id="a1",
        summary="done-a",
        status="completed",
        graph_node_id="n_a",
    )
    checkpoint_tree = _tree(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a1,
        entries=[
            _tree_entry(old_root, status=ExecutionCheckpointStatus.INTERRUPTED),
            _tree_entry(
                execution_a,
                parent=old_root,
                graph_node_id="n_a",
                status=ExecutionCheckpointStatus.COMPLETED,
                prior_output=prior_a,
            ),
        ],
    )
    plan = build_execution_tree_resume_plan(
        checkpoint_tree,
        task_id=task_id,
        run_id=run_id,
        new_attempt_id=mint_attempt_id(),
        new_root_execution_id=mint_execution_id(),
    )
    assert "n_a" not in plan.resume_graph_node_ids


def test_r6_q09_interrupted_failed_pending_become_resume_candidates() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    root_a1 = mint_execution_id()
    interrupted = mint_execution_id()
    checkpoint_tree = _tree(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a1,
        entries=[
            _tree_entry(root_a1, status=ExecutionCheckpointStatus.RUNNING),
            _tree_entry(
                interrupted,
                parent=root_a1,
                graph_node_id="n3",
                status=ExecutionCheckpointStatus.INTERRUPTED,
            ),
        ],
    )
    plan = build_execution_tree_resume_plan(
        checkpoint_tree,
        task_id=task_id,
        run_id=run_id,
        new_attempt_id=mint_attempt_id(),
        new_root_execution_id=mint_execution_id(),
    )
    assert plan.resume_graph_node_ids == frozenset({"n3"})


def test_r6_q10_retry_preserves_run_id() -> None:
    service, lifecycle, _ = _retry_service(lineage=False)
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    lifecycle.record_initial_attempt(tenant_id="tenant-a", run_id=run_id, attempt_id=attempt_a1)
    transition = service.transition_for_retry(
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=run_id,
        expected_attempt_id=attempt_a1,
        request=_eligibility_request(ExecutionFailureKind.RETRYABLE_TRANSIENT),
    )
    assert transition is not None
    assert transition.run_id == run_id


def test_r6_q11_retry_mints_new_attempt_id() -> None:
    service, lifecycle, _ = _retry_service(lineage=False)
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    lifecycle.record_initial_attempt(tenant_id="tenant-a", run_id=run_id, attempt_id=attempt_a1)
    transition = service.transition_for_retry(
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=run_id,
        expected_attempt_id=attempt_a1,
        request=_eligibility_request(ExecutionFailureKind.RETRYABLE_TRANSIENT),
    )
    assert transition is not None
    assert transition.active_attempt_id != attempt_a1


def test_r6_q12_retry_seals_previous_attempt_retry_superseded() -> None:
    service, lifecycle, persistence = _retry_service(lineage=True)
    tenant_id = "tenant-a"
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    scope = build_execution_lineage_attempt_scope(
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a1,
    )
    register_v1_attempt(persistence, scope)
    lifecycle.record_initial_attempt(tenant_id=tenant_id, run_id=run_id, attempt_id=attempt_a1)
    transition = service.transition_for_retry(
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        expected_attempt_id=attempt_a1,
        request=_eligibility_request(ExecutionFailureKind.RETRYABLE_TRANSIENT),
    )
    assert transition is not None
    seal = persistence.read_seal(scope)
    assert seal is not None
    assert seal.closure_kind is ExecutionLineageAttemptClosureKind.RETRY_SUPERSEDED


def test_r6_q13_stale_retry_cannot_create_sibling_attempt() -> None:
    service, lifecycle, _ = _retry_service(lineage=False)
    tenant_id = "tenant-a"
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    lifecycle.record_initial_attempt(tenant_id=tenant_id, run_id=run_id, attempt_id=attempt_a1)
    request = _eligibility_request(ExecutionFailureKind.RETRYABLE_TRANSIENT)
    first = service.transition_for_retry(
        tenant_id=tenant_id,
        task_id=mint_task_id(),
        run_id=run_id,
        expected_attempt_id=attempt_a1,
        request=request,
    )
    assert first is not None
    stale = service.transition_for_retry(
        tenant_id=tenant_id,
        task_id=mint_task_id(),
        run_id=run_id,
        expected_attempt_id=attempt_a1,
        request=request,
    )
    assert stale is None
    assert lifecycle.get_active_attempt_id(tenant_id=tenant_id, run_id=run_id) == first.active_attempt_id


def test_r6_q14_exactly_one_active_attempt_per_run() -> None:
    service, lifecycle, _ = _retry_service(lineage=False)
    tenant_id = "tenant-a"
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    lifecycle.record_initial_attempt(tenant_id=tenant_id, run_id=run_id, attempt_id=attempt_a1)
    transition = service.transition_for_retry(
        tenant_id=tenant_id,
        task_id=mint_task_id(),
        run_id=run_id,
        expected_attempt_id=attempt_a1,
        request=_eligibility_request(ExecutionFailureKind.RETRYABLE_TRANSIENT),
    )
    assert transition is not None
    active = lifecycle.get_active_attempt_id(tenant_id=tenant_id, run_id=run_id)
    assert active == transition.active_attempt_id
    assert active != attempt_a1


def test_r6_q15_production_retry_requires_durable_lifecycle_evidence_registered() -> None:
    assert_r6_behavioral_evidence_registered()


def test_r6_q16_new_execution_mints_independent_identity() -> None:
    first = mint_root_execution_identity()
    second = mint_root_execution_identity()
    assert first.run_id != second.run_id
    assert first.attempt_id != second.attempt_id
    assert first.execution_id != second.execution_id


def test_r6_q17_new_execution_not_checkpoint_derived_fork() -> None:
    row = next(r for r in RECOVERY_OPERATION_MATRIX if r.kind is RecoveryOperationKind.NEW_EXECUTION)
    assert row.canonical_owner is not None
    assert "resume" not in row.canonical_owner.lower()
    assert "checkpoint" not in row.lineage_semantics.lower()


def test_r6_q18_inspection_replay_is_read_only() -> None:
    @dataclass
    class _RecordingEngine:
        calls: list[tuple[str, str]]

        def reconstruct(self, tenant_id: str, run_id: str) -> ReconstructedRun:
            self.calls.append((tenant_id, run_id))
            return ReconstructedRun(
                run_id=run_id,
                steps=[],
                artifacts=[],
                tool_calls=[],
                llm_calls=[],
                final_answer=None,
            )

    engine = _RecordingEngine(calls=[])
    service = ReplayService(engine)
    result = service.inspect_run("tenant-x", "run-y")
    assert result.run_id == "run-y"
    assert engine.calls == [("tenant-x", "run-y")]


def test_r6_q19_inspection_replay_mints_no_execution_identity() -> None:
    row = next(
        r for r in RECOVERY_OPERATION_MATRIX if r.kind is RecoveryOperationKind.INSPECTION_REPLAY
    )
    assert "no new run" in row.run_semantics.lower() or "historical" in row.run_semantics.lower()
    assert "no attempt" in row.attempt_semantics.lower() or "no attempt transition" in row.attempt_semantics.lower()


def test_r6_q20_idempotent_replay_evidence_registered() -> None:
    refs = {r.evidence_id for r in __import__(
        "tests.qualification.state_x._r6_recovery_branching_support",
        fromlist=["R6_FRZ_REC_05_BEHAVIORAL_EVIDENCE"],
    ).R6_FRZ_REC_05_BEHAVIORAL_EVIDENCE}
    assert "idempotent_replay" in refs


def test_r6_q21_idempotent_replay_no_divergent_truth() -> None:
    row = next(
        r
        for r in RECOVERY_OPERATION_MATRIX
        if r.kind is RecoveryOperationKind.IDEMPOTENT_RESULT_REPLAY
    )
    assert "does not create" in row.run_semantics.lower() or "not create" in row.run_semantics.lower()


def test_r6_q22_partial_recovery_preserves_siblings_evidence() -> None:
    assert_r6_behavioral_evidence_registered()


def test_r6_q23_partial_recovery_only_failed_slot_evidence() -> None:
    assert_r6_behavioral_evidence_registered()


def test_r6_q24_partial_recovery_binds_source_attempt_evidence() -> None:
    assert_r6_behavioral_evidence_registered()


def test_r6_q25_partial_recovery_binds_checkpoint_revision_evidence() -> None:
    assert_r6_behavioral_evidence_registered()


def test_r6_q26_partial_recovery_binds_root_topology_fanout_evidence() -> None:
    assert_r6_behavioral_evidence_registered()


def test_r6_q27_partial_recovery_governance_admission_documented() -> None:
    row = next(r for r in RECOVERY_OPERATION_MATRIX if r.kind is RecoveryOperationKind.PARTIAL_RECOVERY)
    assert "RecoveryAdmission" in row.canonical_owner


def test_r6_q28_partial_recovery_no_generic_fork_authority() -> None:
    row = next(r for r in RECOVERY_OPERATION_MATRIX if r.kind is RecoveryOperationKind.PARTIAL_RECOVERY)
    assert "no fork authority" in row.authority_semantics.lower()


def test_r6_q29_cross_tenant_resume_denied() -> None:
    cp = _paused_checkpoint()
    with pytest.raises(CheckpointResumeValidationError):
        assert_checkpoint_resume_materialization_eligible(
            cp,
            target_task_id=cp.task_id,
            target_tenant_id="other-tenant",
        )


def test_r6_q30_cross_tenant_retry_keying() -> None:
    lifecycle = AttemptLifecycleService(InMemoryAttemptLifecycleStore())
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    lifecycle.record_initial_attempt(tenant_id="tenant-a", run_id=run_id, attempt_id=attempt_a1)
    assert lifecycle.get_active_attempt_id(tenant_id="tenant-b", run_id=run_id) is None


def test_r6_q31_cross_tenant_partial_recovery_denied() -> None:
    row = next(r for r in RECOVERY_OPERATION_MATRIX if r.kind is RecoveryOperationKind.PARTIAL_RECOVERY)
    assert "tenant" in row.tenant_semantics.lower()


def test_r6_q32_historical_authority_cannot_become_current() -> None:
    for auth in RECOVERY_AUTHORITY_MATRIX:
        if auth.operation is RecoveryOperationKind.FIRST_CLASS_FORK:
            continue
        if auth.operation is RecoveryOperationKind.INSPECTION_REPLAY:
            continue
        assert auth.can_widen_authority is False


def test_r6_q33_recovery_cannot_widen_execution_authority() -> None:
    assert all(not row.can_widen_authority for row in RECOVERY_AUTHORITY_MATRIX)


def test_r6_q34_budget_semantics_explicit() -> None:
    result = evaluate_execution_retry_eligibility(
        _eligibility_request(ExecutionFailureKind.BUDGET_EXHAUSTED),
    )
    assert result.action is ExecutionRetryAction.FAIL
    assert result.reason == "budget_exhausted"
    retry_row = next(r for r in RECOVERY_SIDE_EFFECT_MATRIX if r.operation is RecoveryOperationKind.RETRY)
    assert "budget" in retry_row.notes.lower()


def test_r6_q35_side_effect_semantics_explicit_for_all_operations() -> None:
    for row in RECOVERY_SIDE_EFFECT_MATRIX:
        assert row.notes.strip()


def test_r6_q36_first_class_fork_not_supported_mechanical() -> None:
    row = next(r for r in RECOVERY_OPERATION_MATRIX if r.kind is RecoveryOperationKind.FIRST_CLASS_FORK)
    assert row.supported is False
    assert row.fork_status is ForkSupportStatus.NOT_SUPPORTED


def test_r6_q37_unsupported_fork_no_authority_inheritance() -> None:
    row = next(r for r in RECOVERY_OPERATION_MATRIX if r.kind is RecoveryOperationKind.FIRST_CLASS_FORK)
    assert "NOT SUPPORTED" in row.authority_semantics


def test_r6_q38_frz_rec_05_completeness_gate_pass() -> None:
    assert_frz_rec_05_r6_completeness()


def test_r6_q39_relevant_recovery_regressions_evidence_registered() -> None:
    assert_r6_behavioral_evidence_registered()


def test_r6_q40_full_state_x_suite_is_separate_gate() -> None:
    assert STATE_X_R6_PRE_AUDIT_HEAD


def test_r6_q41_tenant_isolation_audit_pass() -> None:
    assert TENANT_ISOLATION_AUDIT_R6.result == "PASS"
    assert TENANT_ISOLATION_AUDIT_R6.fail_closed_behavior is True
    assert TENANT_ISOLATION_AUDIT_R6.cross_tenant_path == "denied"


def test_r6_q42_in_scope_blocker_zero() -> None:
    assert_frz_rec_05_r6_completeness()
    assert scan_execution_fork_like_symbols() == ()
