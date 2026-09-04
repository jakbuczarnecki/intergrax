# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""In-process scheduler for long-running paused tasks (§26, J.4)."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Protocol, runtime_checkable
from uuid import uuid4

from intergrax.contracts.agent_decision import AgentDecisionType
from intergrax.runtime.cancellation.resume_admission import is_checkpoint_resumable
from intergrax.runtime.execution.execution_terminal.service import ExecutionTerminalService
from intergrax.runtime.human.request_contract import HumanTimeoutCoordinator
from intergrax.runtime.long_running.coordinator import LongRunningCoordinator
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.notification import NotificationAdapter
from intergrax.runtime.long_running.persistence_contract import (
    SchedulerLedger,
    TaskCheckpointPersistence,
)
from intergrax.runtime.long_running.resume_planner import (
    build_scheduled_resume_task,
    build_timeout_resume_task,
    timeout_action_to_verdict,
)
from intergrax.runtime.long_running.scheduler_claim import (
    ScheduledResumeClaim,
    SchedulerActionClaim,
)
from intergrax.runtime.long_running.scheduled_resume import (
    ScheduledResume,
    ScheduledResumePersistence,
)
from intergrax.runtime.task.task import Task, TaskResult
from intergrax.utils.time_provider import SystemTimeProvider

logger = logging.getLogger(__name__)

ENV_SCHEDULER_POLL_SECONDS = "INTERGRAX_SCHEDULER_POLL_SECONDS"
DEFAULT_SCHEDULER_POLL_SECONDS = 30.0
DEFAULT_SCHEDULER_LEASE_SECONDS = 300.0
TIMEOUT_LEDGER_PREFIX = "timeout:"


@runtime_checkable
class TaskResumeExecutor(Protocol):
    """Resume paused tasks through the unified execution entry (§41)."""

    async def resume_task(self, task: Task, *, checkpoint: TaskCheckpoint) -> TaskResult: ...


class LongRunningScheduler:
    """
    Polls checkpoint store for delayed resumes and expired human approvals.

    Designed for laboratory / single-process deployments first; worker-queue
    integration can reuse the same tick logic in a Celery beat task later.
    """

    def __init__(
        self,
        checkpoint_store: TaskCheckpointPersistence,
        resume_executor: TaskResumeExecutor,
        *,
        schedule_store: Optional[ScheduledResumePersistence] = None,
        ledger: Optional[SchedulerLedger] = None,
        notification_adapter: Optional[NotificationAdapter] = None,
        poll_interval_seconds: float = DEFAULT_SCHEDULER_POLL_SECONDS,
        lease_seconds: float = DEFAULT_SCHEDULER_LEASE_SECONDS,
        owner_id: Optional[str] = None,
        execution_terminal: ExecutionTerminalService | None = None,
    ) -> None:
        self._checkpoint_store = checkpoint_store
        self._resume_executor = resume_executor
        self._schedule_store = schedule_store
        self._ledger = ledger
        self._notification_adapter = notification_adapter
        self._execution_terminal = execution_terminal
        self._poll_interval_seconds = poll_interval_seconds
        self._lease_seconds = int(lease_seconds)
        self._owner_id = owner_id or f"scheduler-{uuid4().hex[:12]}"
        self._running = False
        self._loop_task: Optional[asyncio.Task] = None

    @property
    def poll_interval_seconds(self) -> float:
        return self._poll_interval_seconds

    @property
    def owner_id(self) -> str:
        return self._owner_id

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._loop_task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self._running = False
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None

    async def tick(self, *, now: Optional[datetime] = None) -> int:
        """Process one scheduler cycle; returns number of resumes triggered."""
        current = _ensure_utc(now or SystemTimeProvider.utc_now())
        processed = 0
        processed += await self._process_due_schedules(current)
        processed += await self._process_expired_human_timeouts(current)
        return processed

    def schedule_resume(
        self,
        *,
        task_id: str,
        tenant_id: str,
        resume_token: str,
        run_at_utc: str,
        resume_metadata: Optional[dict] = None,
    ) -> ScheduledResume:
        if self._schedule_store is None:
            raise RuntimeError("schedule_store is not configured on LongRunningScheduler")
        entry = ScheduledResume(
            task_id=task_id,
            tenant_id=tenant_id,
            resume_token=resume_token,
            run_at_utc=run_at_utc,
            resume_metadata=dict(resume_metadata or {}),
        )
        return self._schedule_store.schedule(entry)

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await self.tick()
            except Exception:
                logger.exception("long-running scheduler tick failed")
            await asyncio.sleep(self._poll_interval_seconds)

    async def _process_due_schedules(self, now: datetime) -> int:
        if self._schedule_store is None:
            return 0

        processed = 0
        claims = self._schedule_store.claim_due(
            before_utc_iso=now.isoformat(),
            owner_id=self._owner_id,
            lease_seconds=self._lease_seconds,
        )
        for claim in claims:
            entry = claim.entry
            checkpoint = self._checkpoint_store.get_by_token(
                entry.task_id,
                entry.tenant_id,
                entry.resume_token,
            )
            if checkpoint is None:
                self._schedule_store.complete_claim(claim)
                continue
            if not self._can_resume_checkpoint(checkpoint):
                self._schedule_store.complete_claim(claim)
                continue

            task = build_scheduled_resume_task(checkpoint, entry)
            await self._execute_resume(
                task,
                checkpoint,
                subject="Scheduled task resume",
                body=f"Delayed resume triggered for task {entry.task_id}",
                ledger_action="scheduled_resume",
                schedule_claim=claim,
            )
            processed += 1
        return processed

    async def _process_expired_human_timeouts(self, now: datetime) -> int:
        processed = 0
        for checkpoint in self._checkpoint_store.list_paused():
            ledger_key = f"{TIMEOUT_LEDGER_PREFIX}{checkpoint.checkpoint_id}"
            task = Task.model_validate(checkpoint.task_snapshot)
            if not HumanTimeoutCoordinator.is_expired(task, now=now):
                continue

            action = HumanTimeoutCoordinator.planned_timeout_action(task)
            if action is None:
                continue

            if self._ledger is None:
                logger.warning(
                    "Skipping expired human timeout for checkpoint %s: "
                    "scheduler ledger is not configured",
                    checkpoint.checkpoint_id,
                )
                continue

            action_claim = self._ledger.claim_action(
                ledger_key,
                self._owner_id,
                self._lease_seconds,
                action="human_timeout",
            )
            if action_claim is None:
                continue

            verdict = timeout_action_to_verdict(action)
            resume_task = build_timeout_resume_task(
                checkpoint,
                verdict=verdict,
                action=action,
            )
            if not self._can_resume_checkpoint(checkpoint):
                self._ledger.complete_action(action_claim)
                continue
            await self._execute_resume(
                resume_task,
                checkpoint,
                subject="Human approval timeout",
                body=(
                    f"Auto-resume after timeout with action={action.value} "
                    f"for task {checkpoint.task_id}"
                ),
                ledger_action="human_timeout",
                action_claim=action_claim,
            )
            processed += 1
        return processed

    def _can_resume_checkpoint(self, checkpoint: TaskCheckpoint) -> bool:
        if not is_checkpoint_resumable(checkpoint, execution_terminal=self._execution_terminal):
            logger.info(
                "Skipping scheduler resume for task %s: checkpoint is not resumable",
                checkpoint.task_id,
            )
            return False
        return True

    async def _execute_resume(
        self,
        task: Task,
        checkpoint: TaskCheckpoint,
        *,
        subject: str,
        body: str,
        ledger_action: str,
        schedule_claim: Optional[ScheduledResumeClaim] = None,
        action_claim: Optional[SchedulerActionClaim] = None,
    ) -> TaskResult:
        await LongRunningCoordinator.notify_progress(
            task,
            subject=subject,
            body=body,
            adapter=self._notification_adapter,
            extra={
                "checkpoint_id": checkpoint.checkpoint_id,
                "resume_token": checkpoint.resume_token,
                "scheduler_action": ledger_action,
            },
        )
        result = await self._resume_executor.resume_task(task, checkpoint=checkpoint)
        if schedule_claim is not None and self._schedule_store is not None:
            self._schedule_store.complete_claim(schedule_claim)
        if action_claim is not None and self._ledger is not None:
            self._ledger.complete_action(action_claim)
        return result


class UnifiedTaskResumeExecutor:
    """Adapter from UnifiedTaskRunner to TaskResumeExecutor."""

    def __init__(self, task_runner) -> None:
        self._task_runner = task_runner

    async def resume_task(self, task: Task, *, checkpoint: TaskCheckpoint) -> TaskResult:
        return await self._task_runner.run_task(task, resume_checkpoint=checkpoint)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value
