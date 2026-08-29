# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from intergrax.contracts.execution_identity import AttemptId, RunId
from intergrax.llm_adapters.tracking.context import llm_tenant_scope
from intergrax.runtime.execution.orchestration import (
    execute_root_task,
    resolve_root_task_identity,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.task import Task, TaskResult
from intergrax.runtime.task.task_run_bridge import task_from_runtime_request


class UnifiedTaskRunner:
    """
    Thin Task adapter into canonical root execution (§41).

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

    async def run_task(
        self,
        task: Task,
        *,
        run_id: Optional[RunId] = None,
        attempt_id: Optional[AttemptId] = None,
        resume_checkpoint: Optional[TaskCheckpoint] = None,
    ) -> TaskResult:
        if self._task_enricher is not None:
            task = self._task_enricher(task)
        identity = resolve_root_task_identity(
            run_id=run_id,
            attempt_id=attempt_id,
            resume_checkpoint=resume_checkpoint,
        )
        await ActiveTaskRegistry.register(task, identity.run_id)
        try:
            with llm_tenant_scope(task.tenant_id):
                return await execute_root_task(
                    task,
                    nexus_loop=self._nexus_loop,
                    identity=identity,
                    resume_checkpoint=resume_checkpoint,
                )
        finally:
            await ActiveTaskRegistry.unregister(task.task_id, identity.run_id)

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
