# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Task execution boundary for interaction intake (§18)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.execution_identity import mint_run_id
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.task.task import Task, TaskResult


@runtime_checkable
class TaskExecutor(Protocol):
    """Execute a normalized platform Task and return TaskResult."""

    async def execute(self, task: Task) -> TaskResult:
        ...


class NexusLoopTaskExecutor:
    """Backward-compatible executor that delegates directly to NexusLoop."""

    def __init__(self, nexus_loop: NexusLoop) -> None:
        self._nexus_loop = nexus_loop

    async def execute(self, task: Task) -> TaskResult:
        return await self._nexus_loop.handle_task(task, run_id=mint_run_id())
