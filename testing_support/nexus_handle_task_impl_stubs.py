# © Artur Czarnecki. All rights reserved.

"""Typed helpers for NexusLoop._handle_task_impl test overrides (OBS-R4 contract)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from intergrax.runtime.events.runtime_event_metric_scope import RuntimeEventMetricScope
from intergrax.runtime.task.task import Task, TaskResult


def with_runtime_event_metric_scope(
    impl: Callable[[Task], Awaitable[TaskResult]],
) -> Callable[..., Awaitable[TaskResult]]:
    """Wrap a task-only stub so it satisfies the R4 _handle_task_impl keyword contract."""

    async def _wrapped(
        task: Task,
        *,
        runtime_event_metric_scope: RuntimeEventMetricScope,
    ) -> TaskResult:
        return await impl(task)

    return _wrapped
