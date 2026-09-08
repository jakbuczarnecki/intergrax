# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Task execution boundary for interaction intake (§18)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.task.task import Task, TaskResult

TaskEnricher = Callable[[Task], Task]


@runtime_checkable
class TaskExecutor(Protocol):
    """Execute a normalized platform Task and return TaskResult."""

    async def execute(self, task: Task) -> TaskResult:
        ...


class HostTaskExecutionExecutor:
    """Execute interaction-intake tasks through canonical host task execution."""

    def __init__(
        self,
        host_execution: HostTaskExecutionPort,
        *,
        task_enricher: TaskEnricher | None = None,
    ) -> None:
        self._host_execution = host_execution
        self._task_enricher = task_enricher

    async def execute(self, task: Task) -> TaskResult:
        if self._task_enricher is not None:
            task = self._task_enricher(task)
        return await self._host_execution.execute(task)


__all__ = ["HostTaskExecutionExecutor", "TaskEnricher", "TaskExecutor"]
