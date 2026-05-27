# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Optional

from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.task import Task, TaskResult
from intergrax.runtime.task.task_run_bridge import (
    new_run_id,
    runtime_request_with_run_id,
    task_from_runtime_request,
)


class UnifiedTaskRunner:
    """
    Single entry point for Task execution via NexusLoop (§41).

    Used by HTTP serving and eval paths; aligns task_id with run_id.
    """

    def __init__(self, nexus_loop: NexusLoop) -> None:
        self._nexus_loop = nexus_loop

    @property
    def nexus_loop(self) -> NexusLoop:
        return self._nexus_loop

    async def run_task(self, task: Task) -> TaskResult:
        return await self._nexus_loop.handle_task(task)

    async def run_runtime_request(
        self,
        request: RuntimeRequest,
        *,
        tenant_id: str,
        user_id: str,
        run_id: Optional[str] = None,
        capability: Optional[str] = None,
    ) -> TaskResult:
        resolved_run_id = run_id or new_run_id()
        task = task_from_runtime_request(
            request,
            tenant_id=tenant_id,
            user_id=user_id,
            run_id=resolved_run_id,
            capability=capability,
        )
        return await self.run_task(task)
