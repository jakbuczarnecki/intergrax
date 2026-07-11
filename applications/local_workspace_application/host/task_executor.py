# © Artur Czarnecki. All rights reserved.

"""Shared LKW application execution boundary for HTTP and interaction intake (LKW.6A)."""

from __future__ import annotations

from collections.abc import Callable

from intergrax.runtime.interactions.errors import HostNotAcceptingWorkError
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.task.task import Task, TaskResult
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
from local_workspace_application.host.lifecycle import HostLifecycleState, LocalWorkspaceHostLifecycle

TaskEnricher = Callable[[Task], Task]

_NOT_READY_ERROR_IDS: dict[HostLifecycleState, str] = {
    HostLifecycleState.STARTING: "lkw_host_not_ready",
    HostLifecycleState.STOPPING: "lkw_host_stopping",
    HostLifecycleState.STOPPED: "lkw_host_not_ready",
    HostLifecycleState.FAILED: "lkw_host_not_ready",
}


class LocalWorkspaceTaskExecutor:
    """Single LKW execution boundary before NexusLoop."""

    def __init__(
        self,
        nexus_loop: NexusLoop,
        *,
        task_enricher: TaskEnricher | None,
        lifecycle: LocalWorkspaceHostLifecycle,
    ) -> None:
        self._runner = UnifiedTaskRunner(nexus_loop)
        self._task_enricher = task_enricher
        self._lifecycle = lifecycle
        self._nexus_loop = nexus_loop

    @property
    def nexus_loop(self) -> NexusLoop:
        return self._nexus_loop

    def prepare(self, task: Task) -> Task:
        if self._task_enricher is None:
            return task
        return self._task_enricher(task)

    def assert_accepts_new_work(self) -> None:
        if self._lifecycle.accepts_new_work:
            return
        error_id = _NOT_READY_ERROR_IDS.get(
            self._lifecycle.state,
            "lkw_host_not_ready",
        )
        raise HostNotAcceptingWorkError(
            error_id,
            detail=f"host lifecycle state is {self._lifecycle.state.value}",
        )

    async def execute_prepared(self, task: Task) -> TaskResult:
        self.assert_accepts_new_work()
        return await self._runner.run_task(task)

    async def execute(self, task: Task) -> TaskResult:
        return await self.execute_prepared(self.prepare(task))
