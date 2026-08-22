# © Artur Czarnecki. All rights reserved.

"""PCM-CHECKPOINT-SCHEDULER-INTEGRITY scheduler claim tests (PCM-05)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from intergrax.contracts.agent_decision import AgentDecisionType, HumanRequest
from intergrax.contracts.lease_claim import StaleClaimError
from intergrax.runtime.human.request_contract import HumanTimeoutCoordinator
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.notification import NotificationAdapter
from intergrax.runtime.notifications.models import NotificationMessage
from intergrax.runtime.long_running.persistence_contract import SchedulerLedger
from intergrax.runtime.long_running.scheduler import LongRunningScheduler
from intergrax.runtime.long_running.scheduler_claim import ScheduledResumeCancellationError
from intergrax.runtime.long_running.scheduled_resume import ScheduledResume, ScheduledResumeStatus
from intergrax.runtime.long_running.scheduled_resume import ScheduledResumePersistence
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions
from intergrax.utils.time_provider import SystemTimeProvider

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


class _CountingResumeExecutor:
    def __init__(self) -> None:
        self.calls = 0

    async def resume_task(self, task: Task, *, checkpoint: TaskCheckpoint) -> TaskResult:
        self.calls += 1
        return _ok_result(task.task_id)


class _CountingNotificationAdapter(NotificationAdapter):
    def __init__(self) -> None:
        self.count = 0

    async def notify(self, message: NotificationMessage) -> None:
        self.count += 1


def _ok_result(task_id: str = "task-1") -> TaskResult:
    return TaskResult(task_id=task_id, state=TaskState.FAILED, success=True)


def _scheduled_pause_checkpoint(task_id: str = "task-1") -> TaskCheckpoint:
    """Paused task without expired human timeout (for scheduled-resume races)."""
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="paused for schedule",
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True),
        ),
    )
    task.state = TaskState.WAITING_FOR_RESOURCES
    return TaskCheckpoint(
        task_id=task_id,
        tenant_id="t1",
        resume_token="token-1",
        task_state=TaskState.WAITING_FOR_RESOURCES,
        task_snapshot=task.model_dump(mode="json"),
        progress_message="awaiting resources",
        created_at_utc=SystemTimeProvider.utc_now().isoformat(),
    )


def _paused_checkpoint(task_id: str = "task-1") -> TaskCheckpoint:
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="paused",
        context=TaskContext(capability="hitl.timeout_fail"),
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True),
        ),
    )
    task.state = TaskState.WAITING_FOR_HUMAN
    HumanTimeoutCoordinator.attach_to_task(
        task,
        HumanRequest(
            request_id="hr_timeout",
            prompt="Approve?",
            options=["approve", "reject"],
            timeout_seconds=30,
            default_on_timeout=AgentDecisionType.FAIL,
        ),
    )
    task.runtime.governance.human_request_expires_at = (
        datetime.now(timezone.utc) - timedelta(seconds=60)
    ).isoformat()
    task.sync_metadata()
    return TaskCheckpoint(
        task_id=task_id,
        tenant_id="t1",
        resume_token="token-1",
        task_state=TaskState.WAITING_FOR_HUMAN,
        task_snapshot=task.model_dump(mode="json"),
        progress_message="awaiting",
        created_at_utc=SystemTimeProvider.utc_now().isoformat(),
    )


def _build_scheduler(
    tmp_path,
    resume_executor: _CountingResumeExecutor | AsyncMock | None = None,
    notification: _CountingNotificationAdapter | None = None,
    owner_id: str = "scheduler-a",
    lease_seconds: int = 300,
) -> tuple[LongRunningScheduler, SQLiteTaskCheckpointStore, _CountingResumeExecutor | AsyncMock, _CountingNotificationAdapter]:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "sched.db")
    executor = resume_executor or _CountingResumeExecutor()
    notifier = notification or _CountingNotificationAdapter()
    scheduler = LongRunningScheduler(
        store,
        executor,
        schedule_store=store,
        ledger=store,
        notification_adapter=notifier,
        owner_id=owner_id,
        lease_seconds=lease_seconds,
    )
    return scheduler, store, executor, notifier


@pytest.mark.asyncio
async def test_atomic_due_claim_exactly_one_winner(tmp_path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "claim.db")
    run_at = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    entry = store.schedule(
        ScheduledResume(
            task_id="task-1",
            tenant_id="t1",
            resume_token="token-1",
            run_at_utc=run_at,
        ),
    )
    before = datetime.now(timezone.utc).isoformat()
    claims_a = store.claim_due(
        before_utc_iso=before,
        owner_id="worker-a",
        lease_seconds=300,
    )
    claims_b = store.claim_due(
        before_utc_iso=before,
        owner_id="worker-b",
        lease_seconds=300,
    )
    assert len(claims_a) == 1
    assert claims_b == []
    assert claims_a[0].schedule_id == entry.schedule_id
    assert claims_a[0].fence == 1


@pytest.mark.asyncio
async def test_two_schedulers_single_resume_call(tmp_path) -> None:
    executor = _CountingResumeExecutor()
    scheduler_a, store, _, _ = _build_scheduler(tmp_path, resume_executor=executor, owner_id="a")
    scheduler_b, _, _, _ = _build_scheduler(
        tmp_path,
        resume_executor=executor,
        owner_id="b",
    )
    checkpoint = _scheduled_pause_checkpoint()
    store.save(checkpoint)
    run_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    scheduler_a.schedule_resume(
        task_id=checkpoint.task_id,
        tenant_id=checkpoint.tenant_id,
        resume_token=checkpoint.resume_token,
        run_at_utc=run_at,
        resume_metadata={"human_approved": True},
    )
    now = datetime.now(timezone.utc)
    processed_a = await scheduler_a.tick(now=now)
    processed_b = await scheduler_b.tick(now=now)
    assert processed_a == 1
    assert processed_b == 0
    assert executor.calls == 1


@pytest.mark.asyncio
async def test_losing_worker_does_not_notify(tmp_path) -> None:
    notifier_a = _CountingNotificationAdapter()
    notifier_b = _CountingNotificationAdapter()
    executor = AsyncMock(return_value=_ok_result())
    scheduler_a, store, _, _ = _build_scheduler(
        tmp_path,
        resume_executor=executor,
        notification=notifier_a,
        owner_id="a",
    )
    scheduler_b, _, _, _ = _build_scheduler(
        tmp_path,
        resume_executor=executor,
        notification=notifier_b,
        owner_id="b",
    )
    checkpoint = _scheduled_pause_checkpoint()
    store.save(checkpoint)
    run_at = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    scheduler_a.schedule_resume(
        task_id=checkpoint.task_id,
        tenant_id=checkpoint.tenant_id,
        resume_token=checkpoint.resume_token,
        run_at_utc=run_at,
        resume_metadata={"human_approved": True},
    )
    now = datetime.now(timezone.utc)
    await scheduler_a.tick(now=now)
    await scheduler_b.tick(now=now)
    assert notifier_a.count == 1
    assert notifier_b.count == 0


@pytest.mark.asyncio
async def test_active_claim_blocks_second_owner(tmp_path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "block.db")
    run_at = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    store.schedule(
        ScheduledResume(
            task_id="task-1",
            tenant_id="t1",
            resume_token="token-1",
            run_at_utc=run_at,
        ),
    )
    before = datetime.now(timezone.utc).isoformat()
    claims = store.claim_due(
        before_utc_iso=before,
        owner_id="worker-a",
        lease_seconds=300,
    )
    assert len(claims) == 1
    blocked = store.claim_due(
        before_utc_iso=before,
        owner_id="worker-b",
        lease_seconds=300,
    )
    assert blocked == []


@pytest.mark.asyncio
async def test_fenced_completion_rejected(tmp_path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "fence.db")
    run_at = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    store.schedule(
        ScheduledResume(
            task_id="task-1",
            tenant_id="t1",
            resume_token="token-1",
            run_at_utc=run_at,
        ),
    )
    before = datetime.now(timezone.utc).isoformat()
    claim_a = store.claim_due(
        before_utc_iso=before,
        owner_id="worker-a",
        lease_seconds=1,
    )[0]
    with patch.object(SystemTimeProvider, "utc_now", return_value=datetime.now(timezone.utc)):
        store.complete_claim(claim_a)
    # Simulate superseded fence by direct status reset is not allowed — use stale claim object
    stale = claim_a
    with pytest.raises(StaleClaimError):
        store.complete_claim(stale)


@pytest.mark.asyncio
async def test_current_owner_completes(tmp_path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "complete.db")
    run_at = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    entry = store.schedule(
        ScheduledResume(
            task_id="task-1",
            tenant_id="t1",
            resume_token="token-1",
            run_at_utc=run_at,
        ),
    )
    claim = store.claim_due(
        before_utc_iso=datetime.now(timezone.utc).isoformat(),
        owner_id="worker-a",
        lease_seconds=300,
    )[0]
    store.complete_claim(claim)
    row = store.list_due(before_utc_iso=datetime.now(timezone.utc).isoformat())
    assert row == []
    with store._connection() as conn:
        status = conn.execute(
            "SELECT status FROM scheduled_resumes WHERE schedule_id = ?",
            (entry.schedule_id,),
        ).fetchone()["status"]
    assert status == ScheduledResumeStatus.COMPLETED.value


@pytest.mark.asyncio
async def test_expired_running_becomes_uncertain_no_second_resume(tmp_path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "uncertain.db")
    run_at = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    entry = store.schedule(
        ScheduledResume(
            task_id="task-1",
            tenant_id="t1",
            resume_token="token-1",
            run_at_utc=run_at,
        ),
    )
    claim = store.claim_due(
        before_utc_iso=datetime.now(timezone.utc).isoformat(),
        owner_id="worker-a",
        lease_seconds=1,
    )[0]
    assert claim.schedule_id == entry.schedule_id
    # Resume observed but scheduler crashed before complete_claim.
    with store._connection() as conn:
        conn.execute(
            """
            UPDATE scheduled_resumes
            SET lease_expires_at_utc = ?
            WHERE schedule_id = ?
            """,
            (
                (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat(),
                entry.schedule_id,
            ),
        )
    after = datetime.now(timezone.utc)
    blocked = store.claim_due(
        before_utc_iso=after.isoformat(),
        owner_id="worker-b",
        lease_seconds=1,
    )
    assert blocked == []
    with store._connection() as conn:
        status = conn.execute(
            "SELECT status FROM scheduled_resumes WHERE schedule_id = ?",
            (entry.schedule_id,),
        ).fetchone()["status"]
    assert status == ScheduledResumeStatus.UNCERTAIN.value


@pytest.mark.asyncio
async def test_cancel_active_running_rejected(tmp_path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "cancel.db")
    run_at = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    entry = store.schedule(
        ScheduledResume(
            task_id="task-1",
            tenant_id="t1",
            resume_token="token-1",
            run_at_utc=run_at,
        ),
    )
    claim = store.claim_due(
        before_utc_iso=datetime.now(timezone.utc).isoformat(),
        owner_id="worker-a",
        lease_seconds=300,
    )[0]
    assert claim.schedule_id == entry.schedule_id
    with pytest.raises(ScheduledResumeCancellationError):
        store.cancel(entry.schedule_id)


@pytest.mark.asyncio
async def test_paused_checkpoint_is_timeout_eligible() -> None:
    task = Task.model_validate(_paused_checkpoint().task_snapshot)
    assert HumanTimeoutCoordinator.is_expired(task)
    assert HumanTimeoutCoordinator.planned_timeout_action(task) is not None


@pytest.mark.asyncio
async def test_timeout_without_ledger_no_side_effects(tmp_path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "no_ledger.db")
    executor = _CountingResumeExecutor()
    notifier = _CountingNotificationAdapter()
    scheduler = LongRunningScheduler(
        store,
        executor,
        schedule_store=None,
        ledger=None,
        notification_adapter=notifier,
        owner_id="solo",
    )
    checkpoint = _paused_checkpoint()
    store.save(checkpoint)
    expired = datetime.now(timezone.utc)
    processed = await scheduler.tick(now=expired)
    assert processed == 0
    assert executor.calls == 0
    assert notifier.count == 0


@pytest.mark.asyncio
async def test_timeout_claim_precedes_notify_and_resume(tmp_path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "claim_order.db")
    executor = _CountingResumeExecutor()
    notifier = _CountingNotificationAdapter()
    call_order: list[str] = []
    original_claim = store.claim_action

    def tracking_claim(*args, **kwargs):
        call_order.append("claim")
        return original_claim(*args, **kwargs)

    original_notify = notifier.notify

    async def tracking_notify(message):
        call_order.append("notify")
        return await original_notify(message)

    original_resume = executor.resume_task

    async def tracking_resume(task, *, checkpoint):
        call_order.append("resume")
        return await original_resume(task, checkpoint=checkpoint)

    executor.resume_task = tracking_resume  # type: ignore[method-assign]
    notifier.notify = tracking_notify  # type: ignore[method-assign]

    scheduler = LongRunningScheduler(
        store,
        executor,
        schedule_store=None,
        ledger=store,
        notification_adapter=notifier,
        owner_id="order-test",
    )
    checkpoint = _paused_checkpoint()
    store.save(checkpoint)
    with patch.object(store, "claim_action", side_effect=tracking_claim):
        expired = datetime.now(timezone.utc)
        assert await scheduler.tick(now=expired) == 1
    assert call_order == ["claim", "notify", "resume"]


def test_scheduled_resume_persistence_has_no_unfenced_mark_completed() -> None:
    assert "mark_completed" not in ScheduledResumePersistence.__dict__


def test_scheduler_ledger_has_no_record_action_terminal_api() -> None:
    assert "record_action" not in SchedulerLedger.__dict__


@pytest.mark.asyncio
async def test_active_ledger_claim_cannot_complete_without_claim(tmp_path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "ledger_no_unfenced.db")
    claim = store.claim_action(
        "timeout:ckpt-unfenced",
        "worker-a",
        300,
        action="human_timeout",
    )
    assert claim is not None
    assert claim.owner_id == "worker-a"
    assert claim.fence == 1
    assert not hasattr(store, "record_action")
    stale = claim.model_copy(update={"owner_id": "worker-b"})
    with pytest.raises(StaleClaimError):
        store.complete_action(stale)


@pytest.mark.asyncio
async def test_valid_ledger_claim_completes(tmp_path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "ledger_complete.db")
    claim = store.claim_action(
        "timeout:ckpt-valid",
        "worker-a",
        300,
        action="human_timeout",
    )
    assert claim is not None
    store.complete_action(claim)
    assert store.has_action("timeout:ckpt-valid")


@pytest.mark.asyncio
async def test_active_running_claim_cannot_complete_without_claim(tmp_path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "no_unfenced.db")
    run_at = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    entry = store.schedule(
        ScheduledResume(
            task_id="task-1",
            tenant_id="t1",
            resume_token="token-1",
            run_at_utc=run_at,
        ),
    )
    claim = store.claim_due(
        before_utc_iso=datetime.now(timezone.utc).isoformat(),
        owner_id="worker-a",
        lease_seconds=300,
    )[0]
    assert claim.schedule_id == entry.schedule_id
    assert not hasattr(store, "mark_completed")
    stale = claim.model_copy(update={"owner_id": "worker-b"})
    with pytest.raises(StaleClaimError):
        store.complete_claim(stale)


@pytest.mark.asyncio
async def test_two_schedulers_one_timeout_resume(tmp_path) -> None:
    executor = _CountingResumeExecutor()
    scheduler_a, store, _, _ = _build_scheduler(tmp_path, resume_executor=executor, owner_id="a")
    scheduler_b, _, _, _ = _build_scheduler(tmp_path, resume_executor=executor, owner_id="b")
    checkpoint = _paused_checkpoint()
    store.save(checkpoint)
    expired = datetime.now(timezone.utc)
    with patch.object(SystemTimeProvider, "utc_now", return_value=expired):
        assert await scheduler_a.tick(now=expired) == 1
        assert await scheduler_b.tick(now=expired) == 0
    assert executor.calls == 1
    assert store.has_action(f"timeout:{checkpoint.checkpoint_id}")


@pytest.mark.asyncio
async def test_timeout_claim_before_notify(tmp_path) -> None:
    notifier_a = _CountingNotificationAdapter()
    notifier_b = _CountingNotificationAdapter()
    executor = AsyncMock(return_value=_ok_result())
    scheduler_a, store, _, _ = _build_scheduler(
        tmp_path,
        resume_executor=executor,
        notification=notifier_a,
        owner_id="a",
    )
    scheduler_b, _, _, _ = _build_scheduler(
        tmp_path,
        resume_executor=executor,
        notification=notifier_b,
        owner_id="b",
    )
    checkpoint = _paused_checkpoint()
    store.save(checkpoint)
    expired = datetime.now(timezone.utc)
    with patch.object(SystemTimeProvider, "utc_now", return_value=expired):
        await scheduler_a.tick(now=expired)
        await scheduler_b.tick(now=expired)
    assert notifier_a.count == 1
    assert notifier_b.count == 0


@pytest.mark.asyncio
async def test_timeout_stale_completion_rejected(tmp_path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "timeout_stale.db")
    claim = store.claim_action(
        "timeout:ckpt-1",
        "worker-a",
        300,
        action="human_timeout",
    )
    assert claim is not None
    store.complete_action(claim)
    with pytest.raises(StaleClaimError):
        store.complete_action(claim)


@pytest.mark.asyncio
async def test_timeout_crash_uncertainty_no_second_resume(tmp_path) -> None:
    store = SQLiteTaskCheckpointStore(db_path=tmp_path / "timeout_crash.db")
    claim = store.claim_action(
        "timeout:ckpt-crash",
        "worker-a",
        1,
        action="human_timeout",
    )
    assert claim is not None
    # Resume observed but scheduler crashed before complete_action.
    with store._connection() as conn:
        conn.execute(
            """
            UPDATE scheduler_ledger
            SET lease_expires_at_utc = ?
            WHERE ledger_key = ?
            """,
            (
                (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat(),
                "timeout:ckpt-crash",
            ),
        )
    blocked = store.claim_action(
        "timeout:ckpt-crash",
        "worker-b",
        1,
        action="human_timeout",
    )
    assert blocked is None
    with store._connection() as conn:
        status = conn.execute(
            "SELECT status FROM scheduler_ledger WHERE ledger_key = ?",
            ("timeout:ckpt-crash",),
        ).fetchone()["status"]
    assert status == "uncertain"
