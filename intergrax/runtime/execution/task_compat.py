# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Transitional Task execution bridge for UE-1B compatibility."""

from __future__ import annotations

from typing import Protocol

from intergrax.runtime.task.task import Task, TaskResult


class TaskRunnerPort(Protocol):
    """Narrow port for existing UnifiedTaskRunner task execution."""

    async def run_task(self, task: Task) -> TaskResult:
        ...


class UnifiedTaskRunnerExecutionDelegate:
    """
    Transitional bridge from :class:`ExecutionBoundary` to existing Task execution.

    Delegates to :class:`~intergrax.runtime.task.unified_task_runner.UnifiedTaskRunner`
    without defining the future neutral ExecutionRequest/ExecutionResult contracts.
    """

    __slots__ = ("_runner",)

    def __init__(self, runner: TaskRunnerPort) -> None:
        self._runner = runner

    async def execute(self, task: Task) -> TaskResult:
        return await self._runner.run_task(task)
