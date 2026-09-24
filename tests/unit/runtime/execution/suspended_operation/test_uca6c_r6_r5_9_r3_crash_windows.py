# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.9-R3 — crash windows W1–W6 on canonical re-entry path."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.contracts.execution.crash_injection import (
    ExecutionSuspendedWorkReentryCrashCheckpoint,
    SimulatedHostProcessLostError,
    ToolRuntimeEffectCrashCheckpoint,
)
from intergrax.contracts.execution.execution_terminal_outcome_by_execution_id import (
    ExecutionTerminalOutcomeByExecutionIdDisposition,
)
from intergrax.contracts.execution.suspended_operation.claim import (
    SuspendedOperationMutationOutcome,
)
from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedOperationMaterializationState,
)
from intergrax.contracts.execution.suspended_operation.reentry import (
    ExecutionSuspendedWorkReentryDisposition,
)
from intergrax.contracts.execution_continuation import (
    ExecutionContinuationLifecycleState,
    ExecutionContinuationLookup,
)
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.contracts.idempotency_store import InvocationUncertaintyError
from intergrax.runtime.execution.suspended_operation.crash_injection import (
    DeterministicExecutionSuspendedWorkReentryCrashInjection,
)
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from intergrax.runtime.governance.active_governed_execution_task import (
    bind_governed_execution_task,
    reset_governed_execution_task,
)
from intergrax.runtime.nexus.tools.tool_runtime_effect_crash_injection import (
    DeterministicToolRuntimeEffectCrashInjection,
)
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.task.task import Task
from tests.unit.autonomous_work.test_uca6c_r4_real_codecraft_execution import (
    _TASK_ID,
    _TENANT,
)
from tests.unit.runtime.execution.suspended_operation.test_uca6c_r6_r5_9_r2_multi_host_fencing import (
    OWNER_HOST_B,
    advance_lease_clock,
    reclaim_as,
)
from tests.unit.runtime.execution.suspended_operation.test_uca6c_r6_r5_9_r2_r1_canonical_reentry_fencing import (
    DualHostReentryFixture,
    _advance_to_gen3_blocked,
    _claim_authority_for_coordinator,
    _claim_host_a_short_lease,
    _multi_host_fixture,
    _reenter,
    _resume_gen3_without_reentry,
    _sync_host_stores,
    _ReclaimBridge,
)
from tests.unit.runtime.execution.suspended_operation.uca6c_r59_r3_crash_harness import (
    build_crash_dual_host_fixture,
    sync_hosts_after_restart,
    terminal_disposition,
)

pytestmark = pytest.mark.unit


def _identity_context(task: Task, run_id, attempt_id, execution_id):
    id_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    gov_token = bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(
            tenant_id=_TENANT,
            workspace_id="workspace-uca6c",
            principal_id="principal-uca6c",
        ),
    )
    task_token = bind_governed_execution_task(task)
    return id_token, gov_token, task_token


def _align_task_continuation_projection(
    task: Task,
    *,
    continuation_id: str,
    coordinator: object,
) -> None:
    pending = coordinator.continuation_port.get_pending(
        ExecutionContinuationLookup(continuation_id=continuation_id),
    )
    HumanPauseCoordinator.project_continuation(task, pending)


def _attach_reentry_crash(
    fixture: DualHostReentryFixture,
    crash: DeterministicExecutionSuspendedWorkReentryCrashInjection,
    *,
    host: str = "a",
) -> None:
    if host == "a":
        co = fixture.composition_a.suspended_work_reentry_coordinator
        assert co is not None
        new_co = replace(co, crash_injection=crash)
        fixture.composition_a = replace(
            fixture.composition_a,
            suspended_work_reentry_coordinator=new_co,
        )
        fixture.hitl_a = replace(
            fixture.hitl_a,
            suspended_work_reentry_coordinator=new_co,
        )
        return
    co = fixture.composition_b.suspended_work_reentry_coordinator
    assert co is not None
    new_co = replace(co, crash_injection=crash)
    fixture.composition_b = replace(
        fixture.composition_b,
        suspended_work_reentry_coordinator=new_co,
    )
    fixture.hitl_b = replace(
        fixture.hitl_b,
        suspended_work_reentry_coordinator=new_co,
    )


def _maybe_resume_gen3(
    fixture: DualHostReentryFixture,
    task: Task,
    *,
    continuation_id: str,
    run_id,
    attempt_id,
    execution_id,
) -> None:
    co = fixture.composition_a.suspended_work_reentry_coordinator
    assert co is not None
    pending = co.continuation_port.get_pending(
        ExecutionContinuationLookup(continuation_id=continuation_id),
    )
    if pending.lifecycle_state is not ExecutionContinuationLifecycleState.RESUMED:
        _resume_gen3_without_reentry(
            fixture,
            task,
            continuation_id=continuation_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            hitl=fixture.hitl_a,
        )


def _complete_on_host_b(
    fixture: DualHostReentryFixture,
    task: Task,
    *,
    continuation_id: str,
    identity,
    run_id,
    attempt_id,
    execution_id,
    resume_before_reenter: bool = False,
) -> ExecutionSuspendedWorkReentryDisposition:
    sync_hosts_after_restart(fixture)
    if resume_before_reenter:
        _resume_gen3_without_reentry(
            fixture,
            task,
            continuation_id=continuation_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            hitl=fixture.hitl_a,
        )
    co_b = fixture.composition_b.suspended_work_reentry_coordinator
    assert co_b is not None
    result = _reenter(
        co_b,
        continuation_id=continuation_id,
        identity=identity,
        task=task,
        counters=fixture.counters,
        host="b",
        claim_authority=_claim_authority_for_coordinator(co_b, continuation_id),
    )
    return result.disposition


def test_w1_crash_after_claim_before_toolruntime(tmp_path) -> None:
    host_a_crash = DeterministicExecutionSuspendedWorkReentryCrashInjection(
        scheduled=ExecutionSuspendedWorkReentryCrashCheckpoint.AFTER_CLAIM_BEFORE_TOOL_RUNTIME,
    )
    with _multi_host_fixture(tmp_path) as fixture:
        run_id = mint_run_id()
        attempt_id = mint_attempt_id()
        execution_id = mint_execution_id()
        task = Task(tenant_id=_TENANT, user_id="u1", message="w1", task_id=_TASK_ID)
        tokens = _identity_context(task, run_id, attempt_id, execution_id)
        try:
            c3, d3 = _advance_to_gen3_blocked(
                fixture,
                task,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            )
            fence_a, revision_a, _ = _claim_host_a_short_lease(fixture, d3)
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
            _sync_host_stores(fixture)
            _resume_gen3_without_reentry(
                fixture,
                task,
                continuation_id=c3,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                hitl=fixture.hitl_a,
            )
            _attach_reentry_crash(fixture, host_a_crash, host="b")
            reentry_b = fixture.composition_b.suspended_work_reentry_coordinator
            assert reentry_b is not None
            with pytest.raises(SimulatedHostProcessLostError):
                _reenter(
                    reentry_b,
                    continuation_id=c3,
                    identity=d3.identity,
                    task=task,
                    counters=fixture.counters,
                    host="b",
                )
            assert fixture.backend_a.calls == 0
            assert fixture.backend_b.calls == 0
            loaded = reentry_b.store.load(d3.suspended_operation_id)
            assert loaded is not None
            assert (
                loaded.materialization_state
                is SuspendedOperationMaterializationState.CLAIMED
            )
            assert fixture.terminal_store.get_recorded_disposition(execution_id) is None

            host_a_crash.fired = False
            host_a_crash.scheduled = None
            completed = _reenter(
                reentry_b,
                continuation_id=c3,
                identity=d3.identity,
                task=task,
                counters=fixture.counters,
                host="b",
                claim_authority=_claim_authority_for_coordinator(reentry_b, c3),
            )
            assert (
                completed.disposition
                is ExecutionSuspendedWorkReentryDisposition.COMPLETED
            )
            assert fixture.backend_b.calls == 1
            assert fixture.counters.terminal_writes == 1
        finally:
            reset_governed_execution_task(tokens[2])
            reset_active_execution_governance_identity(tokens[1])
            reset_active_execution_identity(tokens[0])


def test_w2_crash_after_admission_before_backend(tmp_path) -> None:
    tool_crash = DeterministicToolRuntimeEffectCrashInjection(scheduled=None)
    with _multi_host_fixture(tmp_path) as fixture:
        run_id = mint_run_id()
        attempt_id = mint_attempt_id()
        execution_id = mint_execution_id()
        task = Task(tenant_id=_TENANT, user_id="u1", message="w2", task_id=_TASK_ID)
        tokens = _identity_context(task, run_id, attempt_id, execution_id)
        try:
            c3, d3 = _advance_to_gen3_blocked(
                fixture,
                task,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            )
            fence_a, revision_a, _ = _claim_host_a_short_lease(fixture, d3)
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
            _sync_host_stores(fixture)
            _resume_gen3_without_reentry(
                fixture,
                task,
                continuation_id=c3,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                hitl=fixture.hitl_a,
            )
            invoker_b = fixture.composition_b.invoker
            invoker_b.tool_invoker._effect_crash_injection = tool_crash
            tool_crash.scheduled = (
                ToolRuntimeEffectCrashCheckpoint.AFTER_TOOL_RUNTIME_ADMISSION_BEFORE_BACKEND
            )
            reentry_b = fixture.composition_b.suspended_work_reentry_coordinator
            assert reentry_b is not None
            with pytest.raises(SimulatedHostProcessLostError):
                _reenter(
                    reentry_b,
                    continuation_id=c3,
                    identity=d3.identity,
                    task=task,
                    counters=fixture.counters,
                    host="b",
                )
            assert fixture.backend_a.calls == 0
            assert fixture.backend_b.calls == 0
            tool_crash.scheduled = None
            result = _reenter(
                reentry_b,
                continuation_id=c3,
                identity=d3.identity,
                task=task,
                counters=fixture.counters,
                host="b",
                claim_authority=_claim_authority_for_coordinator(reentry_b, c3),
            )
            assert result.disposition is ExecutionSuspendedWorkReentryDisposition.COMPLETED
            assert fixture.backend_b.calls == 1
            assert fixture.backend_a.calls == 0
        finally:
            reset_governed_execution_task(tokens[2])
            reset_active_execution_governance_identity(tokens[1])
            reset_active_execution_identity(tokens[0])


def test_w3_crash_after_backend_before_effect_accounting(tmp_path) -> None:
    tool_crash = DeterministicToolRuntimeEffectCrashInjection(scheduled=None)
    with _multi_host_fixture(tmp_path) as fixture:
        run_id = mint_run_id()
        attempt_id = mint_attempt_id()
        execution_id = mint_execution_id()
        task = Task(tenant_id=_TENANT, user_id="u1", message="w3", task_id=_TASK_ID)
        tokens = _identity_context(task, run_id, attempt_id, execution_id)
        try:
            c3, d3 = _advance_to_gen3_blocked(
                fixture,
                task,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            )
            fence_a, revision_a, _ = _claim_host_a_short_lease(fixture, d3)
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
            _sync_host_stores(fixture)
            _resume_gen3_without_reentry(
                fixture,
                task,
                continuation_id=c3,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                hitl=fixture.hitl_a,
            )
            invoker_b = fixture.composition_b.invoker
            invoker_b.tool_invoker._effect_crash_injection = tool_crash
            tool_crash.scheduled = (
                ToolRuntimeEffectCrashCheckpoint.AFTER_BACKEND_BEFORE_EFFECT_COMMIT
            )
            reentry_b = fixture.composition_b.suspended_work_reentry_coordinator
            assert reentry_b is not None
            with pytest.raises(SimulatedHostProcessLostError):
                _reenter(
                    reentry_b,
                    continuation_id=c3,
                    identity=d3.identity,
                    task=task,
                    counters=fixture.counters,
                    host="b",
                )
            assert fixture.backend_b.calls == 1
            assert fixture.terminal_store.get_recorded_disposition(execution_id) in {
                None,
                ExecutionTerminalOutcomeByExecutionIdDisposition.FAILED,
            }
            loaded = reentry_b.store.load(d3.suspended_operation_id)
            assert loaded is not None
            assert (
                loaded.materialization_state
                is SuspendedOperationMaterializationState.CLAIMED
            )
            with pytest.raises(InvocationUncertaintyError):
                _reenter(
                    reentry_b,
                    continuation_id=c3,
                    identity=d3.identity,
                    task=task,
                    counters=fixture.counters,
                    host="b",
                    claim_authority=_claim_authority_for_coordinator(reentry_b, c3),
                )
            assert fixture.backend_b.calls == 1
        finally:
            reset_governed_execution_task(tokens[2])
            reset_active_execution_governance_identity(tokens[1])
            reset_active_execution_identity(tokens[0])


def test_w4_crash_after_effect_before_consume(tmp_path) -> None:
    fixture, counters, idem = build_crash_dual_host_fixture(tmp_path)
    host_a_crash = DeterministicExecutionSuspendedWorkReentryCrashInjection(
        scheduled=ExecutionSuspendedWorkReentryCrashCheckpoint.AFTER_EFFECT_COMMIT_BEFORE_CONSUME,
    )
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    task = Task(tenant_id=_TENANT, user_id="u1", message="w4", task_id=_TASK_ID)
    tokens = _identity_context(task, run_id, attempt_id, execution_id)
    try:
        c3, d3 = _advance_to_gen3_blocked(
            fixture,
            task,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        )
        _resume_gen3_without_reentry(
            fixture,
            task,
            continuation_id=c3,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            hitl=fixture.hitl_a,
        )
        _attach_reentry_crash(fixture, host_a_crash)
        co_a = fixture.composition_a.suspended_work_reentry_coordinator
        assert co_a is not None
        with pytest.raises(SimulatedHostProcessLostError):
            _reenter(
                co_a,
                continuation_id=c3,
                identity=d3.identity,
                task=task,
                counters=fixture.counters,
                host="a",
            )
        assert counters.backend_logical_effects == 1
        loaded = co_a.store.load(d3.suspended_operation_id)
        assert loaded is not None
        assert loaded.materialization_state is SuspendedOperationMaterializationState.CLAIMED
        host_a_crash.fired = False
        host_a_crash.scheduled = None
        result = _reenter(
            co_a,
            continuation_id=c3,
            identity=d3.identity,
            task=task,
            counters=fixture.counters,
            host="a",
        )
        assert result.disposition is ExecutionSuspendedWorkReentryDisposition.COMPLETED
        assert counters.backend_logical_effects == 1
        del idem
    finally:
        reset_governed_execution_task(tokens[2])
        reset_active_execution_governance_identity(tokens[1])
        reset_active_execution_identity(tokens[0])


def test_w5_crash_after_consume_before_terminal(tmp_path) -> None:
    fixture, counters, _idem = build_crash_dual_host_fixture(tmp_path)
    host_a_crash = DeterministicExecutionSuspendedWorkReentryCrashInjection(
        scheduled=ExecutionSuspendedWorkReentryCrashCheckpoint.AFTER_CONSUME_BEFORE_TERMINAL,
    )
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    task = Task(tenant_id=_TENANT, user_id="u1", message="w5", task_id=_TASK_ID)
    tokens = _identity_context(task, run_id, attempt_id, execution_id)
    try:
        c3, d3 = _advance_to_gen3_blocked(
            fixture,
            task,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        )
        _resume_gen3_without_reentry(
            fixture,
            task,
            continuation_id=c3,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            hitl=fixture.hitl_a,
        )
        _attach_reentry_crash(fixture, host_a_crash)
        co_a = fixture.composition_a.suspended_work_reentry_coordinator
        assert co_a is not None
        with pytest.raises(SimulatedHostProcessLostError):
            _reenter(
                co_a,
                continuation_id=c3,
                identity=d3.identity,
                task=task,
                counters=fixture.counters,
                host="a",
            )
        loaded = co_a.store.load(d3.suspended_operation_id)
        assert loaded is not None
        assert loaded.materialization_state is SuspendedOperationMaterializationState.CONSUMED
        assert terminal_disposition(fixture.terminal_store, execution_id) is None
        assert counters.backend_logical_effects == 1
        sync_hosts_after_restart(fixture)
        co_b = fixture.composition_b.suspended_work_reentry_coordinator
        assert co_b is not None
        result = _reenter(
            co_b,
            continuation_id=c3,
            identity=d3.identity,
            task=None,
            counters=fixture.counters,
            host="b",
            claim_authority=_claim_authority_for_coordinator(co_b, c3),
        )
        assert result.disposition is ExecutionSuspendedWorkReentryDisposition.COMPLETED
        assert result.reason_detail == "consumed_terminal_reconciled"
        assert counters.backend_logical_effects == 1
        assert (
            terminal_disposition(fixture.terminal_store, execution_id)
            is ExecutionTerminalOutcomeByExecutionIdDisposition.SUCCEEDED
        )
    finally:
        reset_governed_execution_task(tokens[2])
        reset_active_execution_governance_identity(tokens[1])
        reset_active_execution_identity(tokens[0])


def test_w6_crash_after_terminal_before_return(tmp_path) -> None:
    fixture, counters, _idem = build_crash_dual_host_fixture(tmp_path)
    host_a_crash = DeterministicExecutionSuspendedWorkReentryCrashInjection(
        scheduled=ExecutionSuspendedWorkReentryCrashCheckpoint.AFTER_TERMINAL_BEFORE_RETURN,
    )
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    task = Task(tenant_id=_TENANT, user_id="u1", message="w6", task_id=_TASK_ID)
    tokens = _identity_context(task, run_id, attempt_id, execution_id)
    try:
        c3, d3 = _advance_to_gen3_blocked(
            fixture,
            task,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        )
        _resume_gen3_without_reentry(
            fixture,
            task,
            continuation_id=c3,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            hitl=fixture.hitl_a,
        )
        _attach_reentry_crash(fixture, host_a_crash)
        co_a = fixture.composition_a.suspended_work_reentry_coordinator
        assert co_a is not None
        with pytest.raises(SimulatedHostProcessLostError):
            _reenter(
                co_a,
                continuation_id=c3,
                identity=d3.identity,
                task=task,
                counters=fixture.counters,
                host="a",
            )
        assert (
            terminal_disposition(fixture.terminal_store, execution_id)
            is ExecutionTerminalOutcomeByExecutionIdDisposition.SUCCEEDED
        )
        before = counters.backend_logical_effects
        sync_hosts_after_restart(fixture)
        co_b = fixture.composition_b.suspended_work_reentry_coordinator
        assert co_b is not None
        result = _reenter(
            co_b,
            continuation_id=c3,
            identity=d3.identity,
            task=None,
            counters=fixture.counters,
            host="b",
        )
        assert result.disposition is ExecutionSuspendedWorkReentryDisposition.NOT_READY
        assert result.reason_detail == "execution_already_terminal"
        assert counters.backend_logical_effects == before
        assert (
            terminal_disposition(fixture.terminal_store, execution_id)
            is ExecutionTerminalOutcomeByExecutionIdDisposition.SUCCEEDED
        )
    finally:
        reset_governed_execution_task(tokens[2])
        reset_active_execution_governance_identity(tokens[1])
        reset_active_execution_identity(tokens[0])
