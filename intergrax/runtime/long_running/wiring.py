# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared long-running scheduler wiring for Tier-3 application factories (B.05)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from intergrax.runtime.long_running.notification import NotificationAdapter
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
from intergrax.runtime.long_running.scheduler import (
    DEFAULT_SCHEDULER_POLL_SECONDS,
    ENV_SCHEDULER_POLL_SECONDS,
    LongRunningScheduler,
    UnifiedTaskResumeExecutor,
)
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner


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
    )
    return LongRunningSchedulerWiring(scheduler=scheduler)
