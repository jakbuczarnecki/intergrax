# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from intergrax.contracts.execution_identity import RunId, mint_run_id
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.task import Task, TaskResult
from intergrax.llm_adapters.tracking.context import llm_tenant_scope
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.task_run_bridge import task_from_runtime_request


class UnifiedTaskRunner:
    """
    Single entry point for Task execution via NexusLoop (§41).

    Used by HTTP serving and eval paths.
    """

    def __init__(
        self,
        nexus_loop: NexusLoop,
        *,
        task_enricher: Callable[[Task], Task] | None = None,
    ) -> None:
        self._nexus_loop = nexus_loop
        self._task_enricher = task_enricher

    @property
    def nexus_loop(self) -> NexusLoop:
        return self._nexus_loop

    async def run_task(self, task: Task, *, run_id: Optional[RunId] = None) -> TaskResult:
        if self._task_enricher is not None:
            task = self._task_enricher(task)
        await ActiveTaskRegistry.register(task)
        resolved_run_id = run_id or mint_run_id()
        try:
            with llm_tenant_scope(task.tenant_id):
                return await self._nexus_loop.handle_task(task, run_id=resolved_run_id)
        finally:
            await ActiveTaskRegistry.unregister(task.task_id)

    async def run_runtime_request(
        self,
        request: RuntimeRequest,
        *,
        tenant_id: str,
        user_id: str,
        capability: Optional[str] = None,
    ) -> TaskResult:
        task = task_from_runtime_request(
            request,
            tenant_id=tenant_id,
            user_id=user_id,
            capability=capability,
        )
        return await self.run_task(task, run_id=request.run_id)
