# © Artur Czarnecki. All rights reserved.

"""Shared LKW application execution boundary for HTTP and interaction intake (LKW.6A)."""

from __future__ import annotations

from collections.abc import Callable

from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.interactions.errors import HostNotAcceptingWorkError
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.task.task import Task, TaskResult
from local_workspace_application.host.readiness import LocalWorkspaceReadinessProvider

TaskEnricher = Callable[[Task], Task]


class LocalWorkspaceTaskExecutor:
    """Single LKW execution boundary before canonical host task execution."""

    def __init__(
        self,
        host_execution: HostTaskExecutionPort,
        *,
        nexus_loop: NexusLoop,
        task_enricher: TaskEnricher | None,
        readiness: LocalWorkspaceReadinessProvider,
    ) -> None:
        self._host_execution = host_execution
        self._nexus_loop = nexus_loop
        self._task_enricher = task_enricher
        self._readiness = readiness

    @property
    def nexus_loop(self) -> NexusLoop:
        return self._nexus_loop

    @property
    def host_execution(self) -> HostTaskExecutionPort:
        return self._host_execution

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
        return await self._host_execution.execute(task)

    async def execute(self, task: Task) -> TaskResult:
        return await self.execute_prepared(self.prepare(task))
