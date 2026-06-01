# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tier-3 task intake defaults (Phase M.11 harness).

For graph id and long-running flags use :mod:`intergrax.applications._shared.task_intake`
(Phase Q+-M.2) — not flat ``task.metadata`` keys in new hosts.
"""

from __future__ import annotations

from typing import Callable, Optional

from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_contract import TaskLongRunningOptions
from intergrax.runtime.task.task_metadata_keys import TaskMetadataKey


def apply_default_long_running_notify_channel(
    task: Task,
    *,
    default_channel: str,
) -> Task:
    """
    When long-running is enabled but ``notify_channel`` is unset, apply host default.

    Used by harness lab wiring so HITL escalation routes to PagerDuty without
    repeating ``notify_channel`` on every task payload.
    """
    channel = (default_channel or "").strip()
    if not channel or channel == "log":
        return task

    lr = task.options.long_running
    enabled = lr.enabled or bool(task.metadata.get(TaskMetadataKey.LONG_RUNNING))
    if not enabled:
        return task
    if lr.notify_channel:
        return task

    updated_lr = lr.model_copy(
        update={
            "enabled": True,
            "notify_channel": channel,
        },
        deep=True,
    )
    task.options = task.options.model_copy(update={"long_running": updated_lr}, deep=True)
    task.sync_metadata()
    return task


def make_lab_harness_task_enricher(
    *,
    default_notify_channel: str,
    harness: bool = False,
) -> Optional[Callable[[Task], Task]]:
    """Build a task enricher for harness lab hosts, or ``None`` when not applicable."""
    if not harness:
        return None
    channel = (default_notify_channel or "").strip()
    if not channel or channel == "log":
        return None

    def _enrich(task: Task) -> Task:
        return apply_default_long_running_notify_channel(task, default_channel=channel)

    return _enrich
