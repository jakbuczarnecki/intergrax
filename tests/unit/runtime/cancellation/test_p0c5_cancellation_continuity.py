# © Artur Czarnecki. All rights reserved.

"""P0C-5 cancellation / timeout continuity proofs."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.contracts.execution_terminal import ExecutionTerminalOutcome, ExecutionTerminalRecord
from intergrax.runtime.cancellation.coordinator import CancellationCoordinator
from intergrax.runtime.cancellation.resume_admission import (
    TERMINALLY_CANCELLED_RESUME_MSG,
    CheckpointNotResumableError,
    assert_checkpoint_resumable,
    is_checkpoint_resumable,
)
from intergrax.runtime.execution.execution_terminal import (
    ExecutionTerminalService,
    InMemoryExecutionTerminalStore,
)
from intergrax.runtime.execution.execution_terminal.persistence import (
    CheckpointStoreExecutionTerminalStore,
)
from intergrax.runtime.long_running.coordinator import LongRunningCoordinator
from intergrax.runtime.long_running.execution_tree_checkpoint import minimal_runtime_checkpoint
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.scheduler import LongRunningScheduler, UnifiedTaskResumeExecutor
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.nexus.orchestration.graph_runner import NexusGraphRunner
from intergrax.runtime.nexus.response.final_response_composer import FinalResponseComposer
from intergrax.runtime.execution.attempt_lifecycle import AttemptLifecycleService, InMemoryAttemptLifecycleStore
from intergrax.runtime.task.task import Task, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_TENANT = "tenant-p0c5"


def _paused_checkpoint(*, task_id: str | None = None) -> TaskCheckpoint:
    resolved_task_id = task_id or str(mint_task_id())
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    task = Task(
        task_id=resolved_task_id,
        tenant_id=_TENANT,
        user_id="user",
        message="paused",
        state=TaskState.WAITING_FOR_HUMAN,
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True, resume_token="rt-p0c5"),
        ),
    )
    return TaskCheckpoint(
        task_id=resolved_task_id,
        tenant_id=_TENANT,
        resume_token="rt-p0c5",
        task_state=TaskState.WAITING_FOR_HUMAN,
        task_snapshot=task.model_dump(mode="json"),
        runtime=minimal_runtime_checkpoint(
            task_id=resolved_task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            root_execution_id=mint_execution_id(),
        ),
    )


def test_terminal_cancellation_blocks_stale_checkpoint_resume() -> None:
    checkpoint = _paused_checkpoint()
    terminal = ExecutionTerminalService(InMemoryExecutionTerminalStore())
    terminal.record_cancellation(
        tenant_id=checkpoint.tenant_id,
        task_id=checkpoint.task_id,
        reason="operator_cancel",
    )
    assert is_checkpoint_resumable(checkpoint, execution_terminal=terminal) is False
    with pytest.raises(CheckpointNotResumableError, match=TERMINALLY_CANCELLED_RESUME_MSG):
        assert_checkpoint_resumable(checkpoint, execution_terminal=terminal)


def test_terminal_cancellation_survives_process_restart(tmp_path) -> None:
    db_path = tmp_path / "ckpt.db"
    store = SQLiteTaskCheckpointStore(db_path=db_path)
    task = Task(
        tenant_id=_TENANT,
        user_id="user",
        message="paused",
        options=TaskExecutionOptions(long_running=TaskLongRunningOptions(enabled=True)),
    )
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=mint_execution_id(),
    )
    try:
        checkpoint = LongRunningCoordinator.persist_checkpoint(
            task,
            store,
            run_id=run_id,
            attempt_id=attempt_id,
            progress_message="awaiting human",
        )
    finally:
        reset_active_execution_identity(token)

    terminal_a = ExecutionTerminalService(CheckpointStoreExecutionTerminalStore(store))
    terminal_a.record_cancellation(
        tenant_id=task.tenant_id,
        task_id=task.task_id,
        run_id=run_id,
        reason="operator_cancel",
    )

    restarted_store = SQLiteTaskCheckpointStore(db_path=db_path)
    terminal_b = ExecutionTerminalService(CheckpointStoreExecutionTerminalStore(restarted_store))
    loaded = restarted_store.get_by_token(task.task_id, _TENANT, checkpoint.resume_token)
    assert loaded is not None
    assert is_checkpoint_resumable(loaded, execution_terminal=terminal_b) is False


def test_idempotent_terminal_cancellation() -> None:
    task_id = str(mint_task_id())
    terminal = ExecutionTerminalService(InMemoryExecutionTerminalStore())
    first = terminal.record_cancellation(tenant_id=_TENANT, task_id=task_id, reason="one")
    second = terminal.record_cancellation(tenant_id=_TENANT, task_id=task_id, reason="two")
    assert first.outcome is ExecutionTerminalOutcome.CANCELLED
    assert second.outcome is ExecutionTerminalOutcome.CANCELLED
    assert second.reason == first.reason


def test_tenant_isolation_for_terminal_cancellation() -> None:
    task_id = str(mint_task_id())
    checkpoint_a = _paused_checkpoint(task_id=task_id)
    checkpoint_b = TaskCheckpoint(
        task_id=task_id,
        tenant_id="tenant-other",
        resume_token="rt-other",
        task_state=TaskState.WAITING_FOR_HUMAN,
        task_snapshot=checkpoint_a.task_snapshot,
    )
    terminal = ExecutionTerminalService(InMemoryExecutionTerminalStore())
    terminal.record_cancellation(tenant_id=_TENANT, task_id=task_id)
    assert is_checkpoint_resumable(checkpoint_a, execution_terminal=terminal) is False
    assert is_checkpoint_resumable(checkpoint_b, execution_terminal=terminal) is True


def test_corrupt_terminal_record_fails_closed() -> None:
    checkpoint = _paused_checkpoint()

    class _CorruptTerminalStore(InMemoryExecutionTerminalStore):
        def load_record(self, *, tenant_id: str, task_id: str) -> ExecutionTerminalRecord | None:
            return ExecutionTerminalRecord(
                tenant_id=tenant_id,
                task_id=task_id,
                outcome=ExecutionTerminalOutcome.CANCELLED,
                reason="",
                recorded_at_utc="",
            )

    terminal = ExecutionTerminalService(_CorruptTerminalStore())
    assert is_checkpoint_resumable(checkpoint, execution_terminal=terminal) is False


@pytest.mark.asyncio
async def test_graph_runner_persists_terminal_cancellation_before_cleanup() -> None:
    task = Task(task_id=str(mint_task_id()), tenant_id=_TENANT, user_id="user", message="cancel me")
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=mint_execution_id(),
    )
    terminal = ExecutionTerminalService(InMemoryExecutionTerminalStore())
    runner = NexusGraphRunner(
        registry=MagicMock(),
        graph_executor=MagicMock(),
        validation_engine=MagicMock(),
        composer=FinalResponseComposer(),
        hitl=MagicMock(),
        events=MagicMock(),
        finish_task=AsyncMock(),
        finalize_trace=AsyncMock(),
        maybe_checkpoint=AsyncMock(),
        attempt_lifecycle=AttemptLifecycleService(InMemoryAttemptLifecycleStore()),
        execution_terminal=terminal,
    )
    runner.events.publish_from_task_state = AsyncMock()
    CancellationCoordinator.request(task, reason="operator_cancel")
    try:
        await runner._handle_cancellation(
            task,
            plan=MagicMock(),
            graph=MagicMock(),
            executions=[],
            retry_records=[],
            lifecycle=MagicMock(),
            trace_emitter=MagicMock(),
        )
    finally:
        reset_active_execution_identity(token)

    record = terminal.get_terminal_record(tenant_id=_TENANT, task_id=task.task_id)
    assert record is not None
    assert record.outcome is ExecutionTerminalOutcome.CANCELLED
    assert not CancellationCoordinator.is_requested(task.metadata)


@pytest.mark.asyncio
async def test_scheduler_skips_resume_after_terminal_cancellation(tmp_path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "ckpt.db")
    checkpoint = _paused_checkpoint()
    store.save(checkpoint)
    terminal = ExecutionTerminalService(CheckpointStoreExecutionTerminalStore(store))
    terminal.record_cancellation(
        tenant_id=checkpoint.tenant_id,
        task_id=checkpoint.task_id,
        reason="operator_cancel",
    )
    runner = AsyncMock()
    scheduler = LongRunningScheduler(
        store,
        UnifiedTaskResumeExecutor(runner),
        schedule_store=store,
        ledger=store,
        execution_terminal=terminal,
    )
    from intergrax.runtime.long_running.scheduled_resume import ScheduledResume

    entry = ScheduledResume(
        task_id=checkpoint.task_id,
        tenant_id=checkpoint.tenant_id,
        resume_token=checkpoint.resume_token,
        run_at_utc="2000-01-01T00:00:00+00:00",
    )
    store.schedule(entry)
    processed = await scheduler.tick(now=__import__("datetime").datetime(2026, 1, 1, tzinfo=__import__("datetime").timezone.utc))
    assert processed == 0
    runner.run_task.assert_not_called()


@pytest.mark.asyncio
async def test_governed_resume_denies_cancelled_checkpoint() -> None:
    from dataclasses import dataclass, field

    from intergrax.applications._shared.task_control import governed_resume_checkpoint_task
    from intergrax.contracts.agent_run import RequestIdentity
    from intergrax.contracts.agent_run_enums import PrincipalType
    from intergrax.contracts.control_plane_mutation import ControlPlaneMutationRequest
    from intergrax.contracts.runtime_policy import EnforcementLevel, PolicyAction, PolicyDecision
    from intergrax.runtime.governance.control_plane_mutation_authorization import (
        ControlPlaneMutationAuthorizationBoundary,
    )

    checkpoint = _paused_checkpoint()
    terminal = ExecutionTerminalService(InMemoryExecutionTerminalStore())
    terminal.record_cancellation(
        tenant_id=checkpoint.tenant_id,
        task_id=checkpoint.task_id,
        reason="operator_cancel",
    )

    class _StaticCheckpointStore:
        def get_by_token(self, task_id: str, tenant_id: str, resume_token: str):
            if resume_token == checkpoint.resume_token:
                return checkpoint
            return None

    @dataclass
    class _AllowEvaluator:
        decision: PolicyDecision = field(
            default_factory=lambda: PolicyDecision(
                action=PolicyAction.ALLOW,
                reason="test_allow",
                enforcement_level=EnforcementLevel.MANDATORY,
                policy_rule_id="task_control.test_allow",
                decision_id="dec-allow",
            ),
        )

        def evaluate(self, request: ControlPlaneMutationRequest) -> PolicyDecision:
            return self.decision

    boundary = ControlPlaneMutationAuthorizationBoundary(evaluator=_AllowEvaluator())
    nexus_loop = MagicMock()
    nexus_loop.execution_terminal = terminal
    runner = MagicMock()
    runner.nexus_loop = nexus_loop

    with patch(
        "intergrax.applications._shared.task_control._resume_task_with_token",
        new_callable=AsyncMock,
    ) as resume_call:
        outcome = await governed_resume_checkpoint_task(
            runner,
            task_id=checkpoint.task_id,
            tenant_id=checkpoint.tenant_id,
            resume_token=checkpoint.resume_token,
            mutation_id="mut-1",
            principal=RequestIdentity(
                tenant_id=_TENANT,
                user_id="operator-1",
                principal_type=PrincipalType.USER,
                auth_subject="operator-1",
            ),
            mutation_boundary=boundary,
            checkpoint_store=_StaticCheckpointStore(),
        )
    assert outcome.accepted is False
    assert outcome.blocked is not None
    assert outcome.blocked.detail == "execution_terminally_cancelled"
    resume_call.assert_not_called()
