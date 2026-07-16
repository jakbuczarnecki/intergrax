# © Artur Czarnecki. All rights reserved.

"""Shared LKW application execution boundary for HTTP and interaction intake (LKW.6A)."""

from __future__ import annotations

from collections.abc import Callable

from intergrax.runtime.interactions.errors import HostNotAcceptingWorkError
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.task.task import Task, TaskResult
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
from local_workspace_application.host.readiness import LocalWorkspaceReadinessProvider

TaskEnricher = Callable[[Task], Task]


class LocalWorkspaceTaskExecutor:
    """Single LKW execution boundary before NexusLoop."""

    def __init__(
        self,
        nexus_loop: NexusLoop,
        *,
        task_enricher: TaskEnricher | None,
        readiness: LocalWorkspaceReadinessProvider,
    ) -> None:
        self._runner = UnifiedTaskRunner(nexus_loop)
        self._task_enricher = task_enricher
        self._readiness = readiness
        self._nexus_loop = nexus_loop

    @property
    def nexus_loop(self) -> NexusLoop:
        return self._nexus_loop

    def prepare(self, task: Task) -> Task:
        if self._task_enricher is None:
            return task
        return self._task_enricher(task)

    def assert_accepts_new_work(self) -> None:
        snapshot = self._readiness.readiness_snapshot()
        if snapshot.accepts_new_work:
            return
        raise HostNotAcceptingWorkError(
            snapshot.rejection_error_id,
            detail=snapshot.detail,
        )

    async def execute_prepared(self, task: Task) -> TaskResult:
        self.assert_accepts_new_work()
        return await self._runner.run_task(task)

    async def execute(self, task: Task) -> TaskResult:
        return await self.execute_prepared(self.prepare(task))
