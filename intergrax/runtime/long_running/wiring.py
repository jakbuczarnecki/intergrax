# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared long-running scheduler wiring for Tier-3 application factories (B.05)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

from intergrax.runtime.long_running.notification import NotificationAdapter
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
from intergrax.runtime.long_running.scheduler import (
    DEFAULT_SCHEDULER_POLL_SECONDS,
    ENV_SCHEDULER_POLL_SECONDS,
    HostTaskResumeExecutor,
    LongRunningScheduler,
    UnifiedTaskResumeExecutor,
)
from intergrax.runtime.execution.execution_terminal.service import ExecutionTerminalService
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner

TaskEnricher = Callable[[Task], Task]


@dataclass(frozen=True)
class LongRunningSchedulerWiring:
    """In-process scheduler bound to checkpoint store + UnifiedTaskRunner."""

    scheduler: LongRunningScheduler


def _poll_interval_seconds(explicit: Optional[float]) -> float:
    if explicit is not None:
        return explicit
    raw = os.getenv(ENV_SCHEDULER_POLL_SECONDS, "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return DEFAULT_SCHEDULER_POLL_SECONDS


def wire_long_running_scheduler(
    *,
    checkpoint_store: TaskCheckpointPersistence,
    task_runner: UnifiedTaskRunner,
    notification_adapter: NotificationAdapter | None = None,
    poll_interval_seconds: Optional[float] = None,
    enabled: bool = True,
) -> LongRunningSchedulerWiring | None:
    """
    Build ``LongRunningScheduler`` for HITL timeout enforcement and delayed resumes.

    Call ``scheduler.start()`` on app startup and ``scheduler.stop()`` on shutdown.
    """
    if not enabled:
        return None
    scheduler = LongRunningScheduler(
        checkpoint_store,
        UnifiedTaskResumeExecutor(task_runner),
        schedule_store=checkpoint_store,
        ledger=checkpoint_store,
        notification_adapter=notification_adapter,
        poll_interval_seconds=_poll_interval_seconds(poll_interval_seconds),
        execution_terminal=task_runner.nexus_loop.execution_terminal,
    )
    return LongRunningSchedulerWiring(scheduler=scheduler)


def wire_long_running_scheduler_with_host_execution(
    *,
    checkpoint_store: TaskCheckpointPersistence,
    host_execution: HostTaskExecutionPort,
    execution_terminal: ExecutionTerminalService | None = None,
    task_enricher: TaskEnricher | None = None,
    notification_adapter: NotificationAdapter | None = None,
    poll_interval_seconds: Optional[float] = None,
    enabled: bool = True,
) -> LongRunningSchedulerWiring | None:
    """Build scheduler wired to canonical host task execution."""
    if not enabled:
        return None
    scheduler = LongRunningScheduler(
        checkpoint_store,
        HostTaskResumeExecutor(host_execution, task_enricher=task_enricher),
        schedule_store=checkpoint_store,
        ledger=checkpoint_store,
        notification_adapter=notification_adapter,
        poll_interval_seconds=_poll_interval_seconds(poll_interval_seconds),
        execution_terminal=execution_terminal,
    )
    return LongRunningSchedulerWiring(scheduler=scheduler)
