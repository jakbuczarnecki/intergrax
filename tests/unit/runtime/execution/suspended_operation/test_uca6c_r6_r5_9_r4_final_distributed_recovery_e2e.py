# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.9-R4 — final distributed recovery E2E (restart + multi-host + W4)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.contracts.execution.crash_injection import (
    ExecutionSuspendedWorkReentryCrashCheckpoint,
    SimulatedHostProcessLostError,
)
from intergrax.contracts.execution.execution_terminal_outcome_by_execution_id import (
    ExecutionTerminalOutcomeByExecutionIdDisposition,
)
from intergrax.contracts.execution.suspended_operation.claim import (
    SuspendedOperationMutationOutcome,
)
from intergrax.contracts.execution.suspended_operation.claim_authority import (
    SuspendedOperationClaimAuthority,
)
from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedOperationMaterializationState,
)
from intergrax.contracts.execution.suspended_operation.reentry import (
    ExecutionSuspendedWorkReentryDisposition,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.runtime.execution.suspended_operation.crash_injection import (
    DeterministicExecutionSuspendedWorkReentryCrashInjection,
)
from intergrax.runtime.governance.active_execution_governance_identity import (
    reset_active_execution_governance_identity,
)
from intergrax.runtime.governance.active_governed_execution_task import (
    reset_governed_execution_task,
)
from intergrax.runtime.long_running.resume_planner import build_checkpoint_resume_task
from intergrax.runtime.task.task import Task
from tests.unit.autonomous_work.test_uca6c_r4_real_codecraft_execution import (
    _TASK_ID,
    _TENANT,
)
from tests.unit.runtime.execution.suspended_operation.test_uca6c_r6_r5_9_r2_multi_host_fencing import (
    OWNER_HOST_A,
    OWNER_HOST_B,
    advance_lease_clock,
    reclaim_as,
)
from tests.unit.runtime.execution.suspended_operation.test_uca6c_r6_r5_9_r2_r1_canonical_reentry_fencing import (
    _ReclaimBridge,
    _advance_to_gen3_blocked,
    _claim_authority_for_coordinator,
    _claim_host_a_short_lease,
    _reenter,
    _resume_gen3_without_reentry,
    _sync_host_stores,
)
from tests.unit.runtime.execution.suspended_operation.test_uca6c_r6_r5_9_r3_crash_windows import (
    _identity_context,
)
from tests.unit.runtime.execution.suspended_operation.uca6c_r59_r3_crash_harness import (
    build_r59_r4_durable_dual_host_fixture,
    seal_host_a_and_rebuild_host_b_process_equivalent,
    terminal_disposition,
)

pytestmark = pytest.mark.unit


@dataclass
class DistributedRecoveryProofCounters:
    host_a_claims: int = 0
    host_b_reclaims: int = 0
    stale_host_mutations_accepted: int = 0
    host_a_toolruntime_attempts: int = 0
    host_b_toolruntime_attempts: int = 0


def _load_task_from_durable_checkpoint(
    checkpoint_store: object,
) -> Task:
    checkpoint = checkpoint_store.get_latest(_TASK_ID, _TENANT)
    assert checkpoint is not None
    return build_checkpoint_resume_task(checkpoint)


def test_r59_r4_primary_w4_distributed_failover_and_stale_host_blocked(
    tmp_path,
) -> None:
    host_a_crash = DeterministicExecutionSuspendedWorkReentryCrashInjection(
        scheduled=ExecutionSuspendedWorkReentryCrashCheckpoint.AFTER_EFFECT_COMMIT_BEFORE_CONSUME,
    )
    fixture, crash_counters, _idem, kit = build_r59_r4_durable_dual_host_fixture(
        tmp_path,
        host_a_reentry_crash=host_a_crash,
    )
    proof = DistributedRecoveryProofCounters()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    task_a = Task(
        tenant_id=_TENANT,
        user_id="u1",
        message="r59r4-primary",
        task_id=_TASK_ID,
    )
    tokens = _identity_context(task_a, run_id, attempt_id, execution_id)
    fence_a = 0
    fence_b = 0
    revision_a = 0
    revision_b = 0
    try:
        c3, d3 = _advance_to_gen3_blocked(
            fixture,
            task_a,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        )
        assert str(d3.identity.run_id) == str(run_id)
        assert str(d3.identity.attempt_id) == str(attempt_id)
        assert str(d3.identity.execution_id) == str(execution_id)

        fence_a, revision_a, _pause_gen = _claim_host_a_short_lease(fixture, d3)
        proof.host_a_claims = 1

        _resume_gen3_without_reentry(
            fixture,
            task_a,
            continuation_id=c3,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            hitl=fixture.hitl_a,
        )

        co_a = fixture.composition_a.suspended_work_reentry_coordinator
        assert co_a is not None
        with pytest.raises(SimulatedHostProcessLostError):
            _reenter(
                co_a,
                continuation_id=c3,
                identity=d3.identity,
                task=task_a,
                counters=fixture.counters,
                host="a",
            )
        proof.host_a_toolruntime_attempts = crash_counters.toolruntime_physical_attempts
        assert crash_counters.backend_logical_effects == 1
        assert terminal_disposition(fixture.terminal_store, execution_id) is None
        loaded_after_crash = co_a.store.load(d3.suspended_operation_id)
        assert loaded_after_crash is not None
        assert (
            loaded_after_crash.materialization_state
            is SuspendedOperationMaterializationState.CLAIMED
        )
        assert loaded_after_crash.claim_ownership is not None
        assert loaded_after_crash.claim_ownership.fence == fence_a

        seal_host_a_and_rebuild_host_b_process_equivalent(fixture, kit)

        expired_now = datetime.now(UTC) + timedelta(hours=2)
        bridge = _ReclaimBridge(
            descriptor=d3,
            store_a=fixture.composition_a.suspended_work_reentry_coordinator.store,
            store_b=fixture.composition_b.suspended_work_reentry_coordinator.store,
        )
        with advance_lease_clock(expired_now):
            reclaimed = reclaim_as(
                bridge,
                "b",
                expected_revision=revision_a,
                owner_id=OWNER_HOST_B,
                expected_fence=fence_a,
                lease_at=expired_now + timedelta(minutes=5),
            )
        assert reclaimed.outcome is SuspendedOperationMutationOutcome.APPLIED
        proof.host_b_reclaims = 1
        assert reclaimed.descriptor is not None
        assert reclaimed.descriptor.claim_ownership is not None
        fence_b = reclaimed.descriptor.claim_ownership.fence
        revision_b = reclaimed.descriptor.materialization_revision
        assert fence_b > fence_a
        assert revision_b > revision_a

        _sync_host_stores(fixture)

        task_b = _load_task_from_durable_checkpoint(fixture.checkpoint_store)
        task_tokens = _identity_context(task_b, run_id, attempt_id, execution_id)
        co_b = fixture.composition_b.suspended_work_reentry_coordinator
        assert co_b is not None

        backend_before_b = crash_counters.backend_physical_attempts
        completed = _reenter(
            co_b,
            continuation_id=c3,
            identity=d3.identity,
            task=task_b,
            counters=fixture.counters,
            host="b",
            claim_authority=_claim_authority_for_coordinator(co_b, c3),
        )
        assert (
            completed.disposition is ExecutionSuspendedWorkReentryDisposition.COMPLETED
        )
        proof.host_b_toolruntime_attempts = (
            crash_counters.toolruntime_physical_attempts
            - proof.host_a_toolruntime_attempts
        )
        assert crash_counters.backend_physical_attempts == backend_before_b
        assert crash_counters.backend_logical_effects == 1
        assert proof.host_b_toolruntime_attempts == 0

        consumed = co_b.store.load(d3.suspended_operation_id)
        assert consumed is not None
        assert (
            consumed.materialization_state
            is SuspendedOperationMaterializationState.CONSUMED
        )
        assert (
            terminal_disposition(fixture.terminal_store, execution_id)
            is ExecutionTerminalOutcomeByExecutionIdDisposition.SUCCEEDED
        )
        assert fixture.counters.terminal_writes == 1

        stale_consume = co_a.store.mark_consumed(
            suspended_operation_id=d3.suspended_operation_id,
            expected_materialization_revision=consumed.materialization_revision,
            owner_id=OWNER_HOST_A,
            fence=fence_a,
        )
        assert stale_consume.outcome is SuspendedOperationMutationOutcome.STALE_CLAIM
        stale_authority = SuspendedOperationClaimAuthority(
            owner_id=OWNER_HOST_A,
            fence=fence_a,
            materialization_revision=consumed.materialization_revision,
            pause_generation=d3.pause_generation,
        )
        stale_reentry = _reenter(
            co_a,
            continuation_id=c3,
            identity=d3.identity,
            task=task_a,
            counters=fixture.counters,
            host="a",
            claim_authority=stale_authority,
        )
        assert stale_reentry.disposition in {
            ExecutionSuspendedWorkReentryDisposition.FAILED,
            ExecutionSuspendedWorkReentryDisposition.REJECTED,
            ExecutionSuspendedWorkReentryDisposition.NOT_READY,
        }
        assert stale_reentry.reason_detail in {
            "stale_claim_owner",
            "stale_materialization_revision",
            "execution_already_terminal",
        }
        assert crash_counters.backend_physical_attempts == 1
        assert proof.stale_host_mutations_accepted == 0
        assert crash_counters.backend_logical_effects == 1

        repeat = _reenter(
            co_b,
            continuation_id=c3,
            identity=d3.identity,
            task=None,
            counters=fixture.counters,
            host="b",
        )
        assert repeat.disposition is ExecutionSuspendedWorkReentryDisposition.NOT_READY
        assert repeat.reason_detail == "execution_already_terminal"
        assert crash_counters.backend_logical_effects == 1
    finally:
        reset_governed_execution_task(tokens[2])
        reset_active_execution_governance_identity(tokens[1])
        reset_active_execution_identity(tokens[0])
        if "task_tokens" in locals():
            reset_governed_execution_task(task_tokens[2])
            reset_active_execution_governance_identity(task_tokens[1])
            reset_active_execution_identity(task_tokens[0])
