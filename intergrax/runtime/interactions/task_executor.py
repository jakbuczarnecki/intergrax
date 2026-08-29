# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Task execution boundary for interaction intake (§18)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.task.task import Task, TaskResult
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner


@runtime_checkable
class TaskExecutor(Protocol):
    """Execute a normalized platform Task and return TaskResult."""

    async def execute(self, task: Task) -> TaskResult:
        ...


class NexusLoopTaskExecutor:
    """Execute tasks through canonical root execution."""

    def __init__(self, nexus_loop: NexusLoop) -> None:
        self._task_runner = UnifiedTaskRunner(nexus_loop)

    async def execute(self, task: Task) -> TaskResult:
        return await self._task_runner.run_task(task)
