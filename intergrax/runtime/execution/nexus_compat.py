# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Transitional Nexus orchestration delegate for UE-3B compatibility."""

from __future__ import annotations

from intergrax.contracts.execution_identity import AttemptId, RunId
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.task.task import Task, TaskResult


class NexusTaskExecutionDelegate:
    """
    Transitional delegate routing a full Task through NexusLoop.handle_task.

    Captures per-invocation run_id and attempt_id resolved by the legacy
    UnifiedTaskRunner entrypoint. Not the final orchestration strategy backend.
    """

    __slots__ = ("_nexus_loop", "_run_id", "_attempt_id")

    def __init__(
        self,
        nexus_loop: NexusLoop,
        *,
        run_id: RunId,
        attempt_id: AttemptId,
    ) -> None:
        self._nexus_loop = nexus_loop
        self._run_id = run_id
        self._attempt_id = attempt_id

    async def execute(self, task: Task) -> TaskResult:
        return await self._nexus_loop.handle_task(
            task,
            run_id=self._run_id,
            attempt_id=self._attempt_id,
        )
