# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""In-process scheduler for long-running paused tasks (§26, J.4)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable
from uuid import uuid4

from intergrax.contracts.lease_claim import StaleClaimError
from intergrax.runtime.cancellation.resume_admission import is_checkpoint_resumable
from intergrax.runtime.execution.execution_terminal.service import (
    ExecutionTerminalService,
)
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.human.request_contract import HumanTimeoutCoordinator
from intergrax.runtime.long_running.checkpoint_resume_validation import (
    CheckpointResumeValidationError,
    assert_checkpoint_resume_materialization_eligible,
)
from intergrax.runtime.long_running.coordinator import LongRunningCoordinator
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.notification import NotificationAdapter
from intergrax.runtime.long_running.persistence_contract import (
    SchedulerLedger,
    TaskCheckpointPersistence,
)
from intergrax.runtime.long_running.recurrence import (
    RECURRENCE_RULE_METADATA_KEY,
    RecurrenceMaterializer,
)
from intergrax.runtime.long_running.resume_planner import (
    build_scheduled_resume_task,
    build_timeout_resume_task,
    execution_identity_from_checkpoint,
    timeout_action_to_verdict,
)
from intergrax.runtime.long_running.scheduled_resume import (
    ScheduledResume,
    ScheduledResumePersistence,
)
from intergrax.runtime.long_running.scheduled_resume_metadata import reject_recurrence_metadata
from intergrax.runtime.long_running.scheduler_claim import (
    ScheduledResumeClaim,
    SchedulerActionClaim,
)
from intergrax.runtime.long_running.uncertain_reconciler import UncertainReconciler
from intergrax.runtime.task.task import Task, TaskResult
from intergrax.utils.time_provider import SystemTimeProvider

logger = logging.getLogger(__name__)

ENV_SCHEDULER_POLL_SECONDS = "INTERGRAX_SCHEDULER_POLL_SECONDS"
DEFAULT_SCHEDULER_POLL_SECONDS = 30.0
DEFAULT_SCHEDULER_LEASE_SECONDS = 300.0
DEFAULT_MAX_CLAIM_RELEASE_ATTEMPTS = 5
RELEASE_CAP_REACHED_LOG_CODE = "scheduled_resume_release_cap_reached"
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
        schedule_store: ScheduledResumePersistence | None = None,
        ledger: SchedulerLedger | None = None,
        notification_adapter: NotificationAdapter | None = None,
        poll_interval_seconds: float = DEFAULT_SCHEDULER_POLL_SECONDS,
        lease_seconds: float = DEFAULT_SCHEDULER_LEASE_SECONDS,
        owner_id: str | None = None,
        execution_terminal: ExecutionTerminalService | None = None,
        recurrence: RecurrenceMaterializer | None = None,
        uncertain_reconciler: UncertainReconciler | None = None,
        max_claim_release_attempts: int = DEFAULT_MAX_CLAIM_RELEASE_ATTEMPTS,
    ) -> None:
        if max_claim_release_attempts < 1:
            raise ValueError("max_claim_release_attempts must be at least 1")
        self._recurrence = recurrence
        self._uncertain_reconciler = uncertain_reconciler
        # Consecutive releases per schedule_id (G-20). Process-local: the count is lost on restart.
        self._max_claim_release_attempts = max_claim_release_attempts
        self._release_counts: dict[str, int] = {}
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
        self._loop_task: asyncio.Task | None = None

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

    async def tick(self, *, now: datetime | None = None) -> int:
        """Process one scheduler cycle; returns number of resumes triggered."""
        current = _ensure_utc(now or SystemTimeProvider.utc_now())
        processed = 0
        failures: list[Exception] = []
        # Every phase always runs; a failure of one never skips the others. The first failure is
        # re-raised afterwards so callers (and the poll loop) still see that the tick failed.
        for phase in (
            self._process_due_schedules,
            self._process_expired_human_timeouts,
            self._reconcile_uncertain_schedules,
        ):
            try:
                processed += await phase(current)
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise failures[0]
        return processed

    def schedule_resume(
        self,
        *,
        task_id: str,
        tenant_id: str,
        resume_token: str,
        run_at_utc: str,
        resume_metadata: dict | None = None,
    ) -> ScheduledResume:
        if self._schedule_store is None:
            raise RuntimeError("schedule_store is not configured on LongRunningScheduler")
        # Recurrence chains start only through RecurrenceMaterializer (F-D7-11).
        reject_recurrence_metadata(resume_metadata)
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

    async def _reconcile_uncertain_schedules(self, now: datetime) -> int:
        """Resolve UNCERTAIN schedules through the optional reconciler; it never executes a resume."""
        if self._uncertain_reconciler is not None:
            self._uncertain_reconciler.reconcile_once(now)
        return 0

    async def _process_due_schedules(self, now: datetime) -> int:
        if self._schedule_store is None:
            return 0

        processed = 0
        claims = self._schedule_store.claim_due(
            before_utc_iso=now.isoformat(),
            owner_id=self._owner_id,
            lease_seconds=self._lease_seconds,
        )
        failures: list[Exception] = []
        for claim in claims:
            # Each claim is isolated: one failing resume must not strand its siblings, which
            # would otherwise stay RUNNING until their lease expires and are then lost (F-D7-01).
            # The first failure is re-raised after the whole batch was processed.
            try:
                processed += await self._process_claimed_schedule(claim, now)
            except StaleClaimError:
                logger.warning(
                    "scheduled resume %s lost its lease before completion; the new owner decides the outcome",
                    claim.schedule_id,
                )
            except Exception as exc:
                logger.exception(
                    "scheduled resume %s failed; its claim is left to the lease (at-most-once)",
                    claim.schedule_id,
                )
                failures.append(exc)
        if failures:
            raise failures[0]
        return processed

    async def _process_claimed_schedule(self, claim: ScheduledResumeClaim, now: datetime) -> int:
        entry = claim.entry
        recurring = RECURRENCE_RULE_METADATA_KEY in entry.resume_metadata
        if recurring and self._recurrence is not None:
            # A failing eligibility read must not end the chain (G-02): hand the claim back, retry next tick.
            try:
                checkpoint = self._eligible_target(entry, probe_terminal_authority=True)
            except Exception:
                logger.exception(
                    "eligibility check failed for schedule_id=%s; releasing the claim for retry",
                    claim.schedule_id,
                )
                self._release_claim(claim)
                return 0
        else:
            checkpoint = self._eligible_target(entry)
        if checkpoint is None:
            self._end_unexecuted_claim(claim, recurring=recurring)
            return 0
        if recurring and self._recurrence is not None and not self._materialize_successor(claim, now):
            return 0

        latest = self._checkpoint_store.get_latest(entry.task_id, entry.tenant_id)
        task = build_scheduled_resume_task(
            checkpoint,
            entry,
            latest_checkpoint=latest,
            execution_terminal=self._execution_terminal,
        )
        await self._execute_resume(
            task,
            checkpoint,
            subject="Scheduled task resume",
            body=f"Delayed resume triggered for task {entry.task_id}",
            ledger_action="scheduled_resume",
            schedule_claim=claim,
        )
        return 1

    def _eligible_target(
        self,
        entry: ScheduledResume,
        *,
        probe_terminal_authority: bool = False,
    ) -> TaskCheckpoint | None:
        """The checkpoint a claimed schedule may resume, or None (read-only checks).

        Both resume gates turn an unavailable terminal authority into "not resumable". With
        ``probe_terminal_authority`` the authority is read first, so an ``ExecutionTerminalError``
        reaches the caller as a failure (retry) instead of ending a recurrence chain; a real
        terminal record is still judged by the gates.
        """
        checkpoint = self._checkpoint_store.get_by_token(
            entry.task_id,
            entry.tenant_id,
            entry.resume_token,
        )
        if checkpoint is None:
            return None
        if probe_terminal_authority and self._execution_terminal is not None:
            self._execution_terminal.get_terminal_record(tenant_id=entry.tenant_id, task_id=entry.task_id)
        if not self._can_materialize_checkpoint(
            checkpoint,
            target_task_id=entry.task_id,
            target_tenant_id=entry.tenant_id,
        ):
            return None
        if not self._can_resume_checkpoint(checkpoint):
            return None
        return checkpoint

    def _end_unexecuted_claim(self, claim: ScheduledResumeClaim, *, recurring: bool) -> None:
        """Finish a claim whose target can no longer be resumed (nothing is executed)."""
        assert self._schedule_store is not None
        self._release_counts.pop(claim.schedule_id, None)
        if not recurring:
            self._schedule_store.complete_claim(claim)
            return
        # A recurrence rule targets exactly one checkpoint; once it cannot be resumed the
        # occurrence is recorded as skipped and the chain ends (no successor, F-D7-13).
        logger.warning(
            "recurrence chain ended: target of schedule_id=%s can no longer be resumed",
            claim.schedule_id,
        )
        self._schedule_store.skip_claim(claim)

    def _materialize_successor(self, claim: ScheduledResumeClaim, now: datetime) -> bool:
        """Persist the next occurrence BEFORE this one executes (F-D7-02).

        Returns False when the occurrence must not execute now: either the persisted rule is
        invalid (fail closed, occurrence skipped) or the store failed (claim handed back so the
        next tick retries and the chain is not lost).
        """
        assert self._recurrence is not None and self._schedule_store is not None
        try:
            self._recurrence.materialize_next(claim.entry, now=now)
        except (ValueError, OverflowError):
            logger.exception(
                "invalid recurrence rule for schedule_id=%s; occurrence skipped, nothing executed",
                claim.schedule_id,
            )
            self._release_counts.pop(claim.schedule_id, None)
            self._schedule_store.skip_claim(claim)
            return False
        except Exception:
            logger.exception(
                "recurrence materialization failed for schedule_id=%s; releasing the claim for retry",
                claim.schedule_id,
            )
            self._release_claim(claim)
            return False
        return True

    def _release_claim(self, claim: ScheduledResumeClaim) -> None:
        """Hand a claim back for retry, at most ``max_claim_release_attempts`` times in a row (G-20).

        After the cap the claim is skipped (nothing executed, the recurrence chain ends) with one
        error under ``RELEASE_CAP_REACHED_LOG_CODE`` instead of being retried every tick forever.
        """
        assert self._schedule_store is not None
        released = self._release_counts.get(claim.schedule_id, 0)
        if released >= self._max_claim_release_attempts:
            self._release_counts.pop(claim.schedule_id, None)
            logger.error(
                "%s schedule_id=%s releases=%s; claim skipped, nothing executed",
                RELEASE_CAP_REACHED_LOG_CODE,
                claim.schedule_id,
                released,
            )
            self._schedule_store.skip_claim(claim)
            return
        self._release_counts[claim.schedule_id] = released + 1
        try:
            self._schedule_store.release_claim(claim)
        except NotImplementedError:
            logger.error(
                "schedule store cannot release claims; schedule_id=%s is left to its lease",
                claim.schedule_id,
            )
        except StaleClaimError:
            logger.warning("claim %s was already taken over; nothing to release", claim.schedule_id)

    async def _process_expired_human_timeouts(self, now: datetime) -> int:
        processed = 0
        failures: list[Exception] = []
        for checkpoint in self._checkpoint_store.list_paused():
            # Same isolation as the schedule batch: one failing timeout resume must not stop the rest.
            try:
                processed += await self._process_expired_timeout(checkpoint, now)
            except StaleClaimError:
                logger.warning(
                    "human timeout for checkpoint %s lost its lease before completion",
                    checkpoint.checkpoint_id,
                )
            except Exception as exc:
                logger.exception(
                    "human timeout resume for checkpoint %s failed; its action claim is left to the lease",
                    checkpoint.checkpoint_id,
                )
                failures.append(exc)
        if failures:
            raise failures[0]
        return processed

    async def _process_expired_timeout(self, checkpoint: TaskCheckpoint, now: datetime) -> int:
        ledger_key = f"{TIMEOUT_LEDGER_PREFIX}{checkpoint.checkpoint_id}"
        if not self._can_materialize_checkpoint(
            checkpoint,
            target_task_id=checkpoint.task_id,
            target_tenant_id=checkpoint.tenant_id,
        ):
            return 0
        if not self._can_resume_checkpoint(checkpoint):
            return 0
        task = Task.model_validate(checkpoint.task_snapshot)
        if not HumanTimeoutCoordinator.is_expired(task, now=now):
            return 0

        action = HumanTimeoutCoordinator.planned_timeout_action(task)
        if action is None:
            return 0

        if self._ledger is None:
            logger.warning(
                "Skipping expired human timeout for checkpoint %s: scheduler ledger is not configured",
                checkpoint.checkpoint_id,
            )
            return 0

        action_claim = self._ledger.claim_action(
            ledger_key,
            self._owner_id,
            self._lease_seconds,
            action="human_timeout",
        )
        if action_claim is None:
            return 0

        verdict = timeout_action_to_verdict(action)
        latest = self._checkpoint_store.get_latest(
            checkpoint.task_id,
            checkpoint.tenant_id,
        )
        resume_task = build_timeout_resume_task(
            checkpoint,
            verdict=verdict,
            action=action,
            target_task_id=checkpoint.task_id,
            target_tenant_id=checkpoint.tenant_id,
            latest_checkpoint=latest,
            execution_terminal=self._execution_terminal,
        )
        await self._execute_resume(
            resume_task,
            checkpoint,
            subject="Human approval timeout",
            body=(f"Auto-resume after timeout with action={action.value} for task {checkpoint.task_id}"),
            ledger_action="human_timeout",
            action_claim=action_claim,
        )
        return 1

    def _can_materialize_checkpoint(
        self,
        checkpoint: TaskCheckpoint,
        *,
        target_task_id: str,
        target_tenant_id: str,
    ) -> bool:
        latest = self._checkpoint_store.get_latest(target_task_id, target_tenant_id)
        try:
            assert_checkpoint_resume_materialization_eligible(
                checkpoint,
                target_task_id=target_task_id,
                target_tenant_id=target_tenant_id,
                latest_checkpoint=latest,
                execution_terminal=self._execution_terminal,
            )
        except CheckpointResumeValidationError:
            logger.info(
                "Skipping scheduler resume for task %s: checkpoint failed restore validation",
                checkpoint.task_id,
            )
            return False
        return True

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
        schedule_claim: ScheduledResumeClaim | None = None,
        action_claim: SchedulerActionClaim | None = None,
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
            self._release_counts.pop(schedule_claim.schedule_id, None)
            self._schedule_store.complete_claim(schedule_claim)
        if action_claim is not None and self._ledger is not None:
            self._ledger.complete_action(action_claim)
        return result


class HostTaskResumeExecutor:
    """Adapter from canonical host execution to TaskResumeExecutor."""

    def __init__(
        self,
        host_execution: HostTaskExecutionPort,
        *,
        task_enricher: Callable[[Task], Task] | None = None,
    ) -> None:
        self._host_execution = host_execution
        self._task_enricher = task_enricher

    async def resume_task(self, task: Task, *, checkpoint: TaskCheckpoint) -> TaskResult:
        if self._task_enricher is not None:
            task = self._task_enricher(task)
        run_id, attempt_id = execution_identity_from_checkpoint(checkpoint)
        return await self._host_execution.execute(
            task,
            run_id=run_id,
            attempt_id=attempt_id,
            resume_checkpoint=checkpoint,
        )


class UnifiedTaskResumeExecutor:
    """Adapter from UnifiedTaskRunner to TaskResumeExecutor."""

    def __init__(self, task_runner) -> None:
        self._task_runner = task_runner

    async def resume_task(self, task: Task, *, checkpoint: TaskCheckpoint) -> TaskResult:
        run_id, attempt_id = execution_identity_from_checkpoint(checkpoint)
        return await self._task_runner.run_task(
            task,
            run_id=run_id,
            attempt_id=attempt_id,
            resume_checkpoint=checkpoint,
        )


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
