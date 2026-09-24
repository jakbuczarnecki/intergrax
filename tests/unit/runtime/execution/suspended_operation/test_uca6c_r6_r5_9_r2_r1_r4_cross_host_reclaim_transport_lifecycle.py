# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-R5.9-R2-R1-R4 — cross-host reclaim and authority transport lifecycle."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from intergrax.contracts.execution.execution_terminal_outcome_by_execution_id import (
    ExecutionTerminalOutcomeByExecutionIdDisposition,
)
from intergrax.contracts.execution.suspended_operation.claim_authority import (
    SuspendedOperationClaimAuthority,
)
from intergrax.contracts.execution.suspended_operation.resume_authority_context import (
    ExecutionSuspendedWorkResumeAuthorityContext,
)
from intergrax.contracts.execution_identity import (
    ActiveExecutionIdentity,
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.runtime.execution.suspended_operation.claim_lifecycle_wiring import (
    claim_lifecycle_from_hitl_continuation,
)
from intergrax.runtime.execution.suspended_operation.resume_authority_transport import (
    ExecutionSuspendedWorkResumeAuthorityTransport,
    ExecutionSuspendedWorkResumeAuthorityTransportConflictError,
    ProductionSuspendedWorkAuthorityTelemetry,
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
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.nexus.orchestration.intake_runner import NexusIntakeRunner
from intergrax.runtime.task.task import Task, TaskState
from intergrax.runtime.task.task_lifecycle import TaskLifecycle
from intergrax.runtime.task.task_trace import TaskTraceEmitter
from testing_support.runtime_event_metric_scope_for_tests import (
    open_runtime_event_metric_scope_for_tests,
)
from tests.unit.autonomous_work.test_uca6c_r4_real_codecraft_execution import (
    _TASK_ID,
    _TENANT,
)
from tests.unit.runtime.execution.suspended_operation.test_uca6c_r6_r5_9_r2_multi_host_fencing import (
    OWNER_HOST_A,
    OWNER_HOST_B,
    advance_lease_clock,
)
from tests.unit.runtime.execution.suspended_operation.test_uca6c_r6_r5_9_r2_r1_canonical_reentry_fencing import (
    DualHostReentryFixture,
    _advance_to_gen3_blocked,
    _build_dual_host_fixture,
    _claim_host_a_short_lease,
    _sync_host_stores,
)

pytestmark = pytest.mark.unit


@contextmanager
def _multi_host_fixture(tmp_path: Path) -> Iterator[DualHostReentryFixture]:
    yield _build_dual_host_fixture(tmp_path)


def _authority(
    *,
    owner_id: str = OWNER_HOST_B,
    fence: int = 1,
    revision: int = 1,
    pause_generation: int = 3,
) -> SuspendedOperationClaimAuthority:
    return SuspendedOperationClaimAuthority(
        owner_id=owner_id,
        fence=fence,
        materialization_revision=revision,
        pause_generation=pause_generation,
    )


def _context(
    continuation_id: str, authority: SuspendedOperationClaimAuthority
) -> ExecutionSuspendedWorkResumeAuthorityContext:
    return ExecutionSuspendedWorkResumeAuthorityContext(
        continuation_id=continuation_id,
        claim_authority=authority,
    )


def test_lifecycle_cross_host_reclaim_after_a_lease_expires(tmp_path: Path) -> None:
    with _multi_host_fixture(tmp_path) as fixture:
        run_id = mint_run_id()
        attempt_id = mint_attempt_id()
        execution_id = mint_execution_id()
        task = Task(
            tenant_id=_TENANT, user_id="u1", message="r59r2r1r4-xh", task_id=_TASK_ID
        )
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
        try:
            c3, d3 = _advance_to_gen3_blocked(
                fixture,
                task,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            )
            fence_a, revision_a, pause_gen = _claim_host_a_short_lease(fixture, d3)
            _sync_host_stores(fixture)
            lifecycle_b = claim_lifecycle_from_hitl_continuation(fixture.hitl_b)
            assert lifecycle_b is not None
            store_b = fixture.composition_b.suspended_work_reentry_coordinator.store
            loaded = store_b.load_active_for_continuation(c3)
            assert loaded is not None
            assert loaded.claim_ownership is not None
            assert loaded.claim_ownership.owner_id == OWNER_HOST_A
            expired_now = datetime.now(UTC) + timedelta(hours=2)
            lease_at = expired_now + timedelta(minutes=5)
            with advance_lease_clock(expired_now):
                context = lifecycle_b.reclaim_expired_lease(
                    loaded,
                    lease_expires_at=lease_at,
                    expected_fence=fence_a,
                )
            assert context is not None
            assert context.claim_authority.owner_id == OWNER_HOST_B
            assert context.claim_authority.fence > fence_a
            assert context.claim_authority.pause_generation == pause_gen
            assert context.continuation_id == c3
        finally:
            reset_governed_execution_task(task_token)
            reset_active_execution_governance_identity(gov_token)
            reset_active_execution_identity(id_token)


def test_transport_wrong_continuation_take_and_discard() -> None:
    transport = ExecutionSuspendedWorkResumeAuthorityTransport()
    auth = _authority()
    transport.deliver(_context("c1", auth))
    assert transport.take_for_continuation("c2") is None
    assert transport.peek() is not None
    assert transport.discard_for_continuation("c2") is False
    assert transport.peek() is not None
    assert transport.discard_for_continuation("c1") is True
    assert transport.peek() is None


def test_transport_conflicting_deliver_fail_closed() -> None:
    transport = ExecutionSuspendedWorkResumeAuthorityTransport()
    transport.deliver(_context("c1", _authority(fence=1)))
    with pytest.raises(ExecutionSuspendedWorkResumeAuthorityTransportConflictError):
        transport.deliver(_context("c2", _authority(fence=2)))


def test_transport_same_continuation_deliver_idempotent() -> None:
    transport = ExecutionSuspendedWorkResumeAuthorityTransport()
    ctx = _context("c1", _authority(fence=3))
    transport.deliver(ctx)
    transport.deliver(ctx)
    assert transport.peek() == ctx


def _configure_task_verdict(task: Task, verdict: HumanResponseVerdict) -> None:
    pause_record = task.runtime.governance.pause_record
    human_request = task.runtime.governance.human_request
    assert pause_record is not None and human_request is not None
    task.options.human.verdict = verdict.value
    task.options.human.pause_id = pause_record.pause_id
    task.options.human.human_request_id = pause_record.human_request_id
    task.options.human.approver = local_development_approver_evidence(
        tenant_id=task.tenant_id,
    )
    task.state = TaskState.WAITING_FOR_HUMAN
    task.sync_metadata()


def _build_intake_runner(
    fixture: DualHostReentryFixture,
    *,
    hitl,
    transport: ExecutionSuspendedWorkResumeAuthorityTransport,
    telemetry: ProductionSuspendedWorkAuthorityTelemetry,
    execution_identity: ActiveExecutionIdentity,
) -> NexusIntakeRunner:
    lifecycle = claim_lifecycle_from_hitl_continuation(hitl)
    return NexusIntakeRunner(
        hitl=MagicMock(
            persist_human_decision=MagicMock(),
            handle_human_rejection=AsyncMock(return_value=MagicMock()),
            handle_human_escalation=AsyncMock(return_value=MagicMock()),
        ),
        human_hooks=MagicMock(after_response=AsyncMock()),
        publish=AsyncMock(),
        restore_long_running=AsyncMock(),
        execution_identity=execution_identity,
        hitl_continuation=hitl,
        task_checkpoint_store=fixture.checkpoint_store,
        suspended_work_claim_lifecycle=lifecycle,
        suspended_work_resume_authority_transport=transport,
        suspended_work_authority_telemetry=telemetry,
    )


async def _run_intake(
    runner: NexusIntakeRunner,
    task: Task,
    *,
    run_id,
    attempt_id,
) -> None:
    metric_scope = open_runtime_event_metric_scope_for_tests(
        task_id=_TASK_ID,
        run_id=str(run_id),
    )
    try:
        await runner.run(
            task,
            lifecycle=TaskLifecycle(),
            trace_emitter=TaskTraceEmitter(
                run_id=str(run_id),
                attempt_id=str(attempt_id),
            ),
            runtime_event_metric_scope=metric_scope,
        )
    finally:
        metric_scope.close()


@pytest.mark.asyncio
async def test_intake_reject_discards_delivered_authority(tmp_path: Path) -> None:
    transport = ExecutionSuspendedWorkResumeAuthorityTransport()
    with _multi_host_fixture(tmp_path) as fixture:
        run_id = mint_run_id()
        attempt_id = mint_attempt_id()
        execution_id = mint_execution_id()
        task = Task(
            tenant_id=_TENANT, user_id="u1", message="r59r2r1r4-rej", task_id=_TASK_ID
        )
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
        execution_identity = ActiveExecutionIdentity()
        try:
            c3, _ = _advance_to_gen3_blocked(
                fixture,
                task,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            )
            transport.deliver(_context(c3, _authority()))
            _configure_task_verdict(task, HumanResponseVerdict.REJECT)
            runner = _build_intake_runner(
                fixture,
                hitl=fixture.hitl_b,
                transport=transport,
                telemetry=ProductionSuspendedWorkAuthorityTelemetry(),
                execution_identity=execution_identity,
            )
            await _run_intake(runner, task, run_id=run_id, attempt_id=attempt_id)
            assert transport.peek() is None
        finally:
            reset_governed_execution_task(task_token)
            reset_active_execution_governance_identity(gov_token)
            reset_active_execution_identity(id_token)


@pytest.mark.asyncio
async def test_intake_escalate_discards_delivered_authority(tmp_path: Path) -> None:
    transport = ExecutionSuspendedWorkResumeAuthorityTransport()
    with _multi_host_fixture(tmp_path) as fixture:
        run_id = mint_run_id()
        attempt_id = mint_attempt_id()
        execution_id = mint_execution_id()
        task = Task(
            tenant_id=_TENANT,
            user_id="u1",
            message="r59r2r1r4-esc",
            task_id=_TASK_ID,
        )
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
        execution_identity = ActiveExecutionIdentity()
        try:
            c3, _ = _advance_to_gen3_blocked(
                fixture,
                task,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            )
            transport.deliver(_context(c3, _authority()))
            _configure_task_verdict(task, HumanResponseVerdict.ESCALATE)
            runner = _build_intake_runner(
                fixture,
                hitl=fixture.hitl_b,
                transport=transport,
                telemetry=ProductionSuspendedWorkAuthorityTelemetry(),
                execution_identity=execution_identity,
            )
            await _run_intake(runner, task, run_id=run_id, attempt_id=attempt_id)
            assert transport.peek() is None
        finally:
            reset_governed_execution_task(task_token)
            reset_active_execution_governance_identity(gov_token)
            reset_active_execution_identity(id_token)


def test_c1_discard_does_not_block_c2_deliver() -> None:
    transport = ExecutionSuspendedWorkResumeAuthorityTransport()
    transport.deliver(_context("c1", _authority(fence=1)))
    assert transport.discard_for_continuation("c1") is True
    transport.deliver(_context("c2", _authority(fence=2)))
    taken = transport.take_for_continuation("c2")
    assert taken is not None
    assert taken.fence == 2


@pytest.mark.asyncio
async def test_production_cross_host_reclaim_resume_completes(tmp_path: Path) -> None:
    transport = ExecutionSuspendedWorkResumeAuthorityTransport()
    telemetry = ProductionSuspendedWorkAuthorityTelemetry()
    with _multi_host_fixture(tmp_path) as fixture:
        run_id = mint_run_id()
        attempt_id = mint_attempt_id()
        execution_id = mint_execution_id()
        task = Task(
            tenant_id=_TENANT,
            user_id="u1",
            message="r59r2r1r4-prod",
            task_id=_TASK_ID,
        )
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
        execution_identity = ActiveExecutionIdentity()
        try:
            c3, d3 = _advance_to_gen3_blocked(
                fixture,
                task,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            )
            fence_a, revision_a, _ = _claim_host_a_short_lease(fixture, d3)
            _sync_host_stores(fixture)
            _configure_task_verdict(task, HumanResponseVerdict.APPROVE)
            runner = _build_intake_runner(
                fixture,
                hitl=fixture.hitl_b,
                transport=transport,
                telemetry=telemetry,
                execution_identity=execution_identity,
            )
            expired_now = datetime.now(UTC) + timedelta(hours=2)
            with advance_lease_clock(expired_now):
                await _run_intake(runner, task, run_id=run_id, attempt_id=attempt_id)
            assert telemetry.reclaim_successes == 1
            assert fixture.backend_b.calls == 1
            assert telemetry.terminal_writes == 1
            terminal = fixture.terminal_store.get_recorded_disposition(execution_id)
            assert (
                terminal is ExecutionTerminalOutcomeByExecutionIdDisposition.SUCCEEDED
            )
            assert fence_a > 0
        finally:
            reset_governed_execution_task(task_token)
            reset_active_execution_governance_identity(gov_token)
            reset_active_execution_identity(id_token)


@pytest.mark.asyncio
async def test_stale_host_a_authority_rejected_after_b_reclaim(
    tmp_path: Path,
) -> None:
    transport = ExecutionSuspendedWorkResumeAuthorityTransport()
    telemetry = ProductionSuspendedWorkAuthorityTelemetry()
    with _multi_host_fixture(tmp_path) as fixture:
        run_id = mint_run_id()
        attempt_id = mint_attempt_id()
        execution_id = mint_execution_id()
        task = Task(
            tenant_id=_TENANT,
            user_id="u1",
            message="r59r2r1r4-stale-a",
            task_id=_TASK_ID,
        )
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
        execution_identity = ActiveExecutionIdentity()
        lifecycle_b = claim_lifecycle_from_hitl_continuation(fixture.hitl_b)
        assert lifecycle_b is not None
        try:
            c3, d3 = _advance_to_gen3_blocked(
                fixture,
                task,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
            )
            fence_a, revision_a, pause_gen = _claim_host_a_short_lease(fixture, d3)
            _sync_host_stores(fixture)
            store_b = fixture.composition_b.suspended_work_reentry_coordinator.store
            loaded = store_b.load_active_for_continuation(c3)
            assert loaded is not None
            expired_now = datetime.now(UTC) + timedelta(hours=2)
            lease_at = expired_now + timedelta(minutes=5)
            with advance_lease_clock(expired_now):
                reloaded = store_b.load_active_for_continuation(c3)
                assert reloaded is not None
                reclaimed = lifecycle_b.reclaim_expired_lease(
                    reloaded,
                    lease_expires_at=lease_at,
                    expected_fence=fence_a,
                )
            assert reclaimed is not None
            _sync_host_stores(fixture)
            stale_a = SuspendedOperationClaimAuthority(
                owner_id=OWNER_HOST_A,
                fence=fence_a,
                materialization_revision=revision_a,
                pause_generation=pause_gen,
            )
            transport.deliver(_context(c3, stale_a))
            _configure_task_verdict(task, HumanResponseVerdict.APPROVE)
            runner = _build_intake_runner(
                fixture,
                hitl=fixture.hitl_b,
                transport=transport,
                telemetry=telemetry,
                execution_identity=execution_identity,
            )
            await _run_intake(runner, task, run_id=run_id, attempt_id=attempt_id)
            assert fixture.backend_b.calls == 0
            assert telemetry.terminal_writes == 0
        finally:
            reset_governed_execution_task(task_token)
            reset_active_execution_governance_identity(gov_token)
            reset_active_execution_identity(id_token)
