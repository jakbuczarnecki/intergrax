# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

from intergrax.contracts.admitted_root_governance_identity import AdmittedRootGovernanceIdentity
from intergrax.contracts.execution_identity import AttemptId, RunId
from intergrax.llm_adapters.tracking.context import llm_tenant_scope
from intergrax.runtime.execution.agent_runtime_io import RuntimeRequest
from intergrax.runtime.execution.harness_task_execution_port import HarnessRootTaskExecutionPort
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.execution.orchestration import resolve_root_task_identity
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.task import Task, TaskResult
from intergrax.runtime.task.task_run_bridge import task_from_runtime_request


class UnifiedTaskRunner:
    """
    Thin Task adapter into canonical root execution (§41).

    HARNESS / SCHEDULING ONLY — not a production Tier-3 execution entry.
    """

    def __init__(
        self,
        execution: HostTaskExecutionPort,
        *,
        task_enricher: Callable[[Task], Task] | None = None,
        admitted_governance_identity_for_task: (
            Callable[[Task], AdmittedRootGovernanceIdentity] | None
        ) = None,
    ) -> None:
        resolved = execution
        if (
            admitted_governance_identity_for_task is not None
            and isinstance(execution, HarnessRootTaskExecutionPort)
        ):
            resolved = execution.with_per_task_governance_admission(
                admitted_governance_identity_for_task,
            )
        self._execution = resolved
        self._task_enricher = task_enricher

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
                return await self._execution.execute(
                    task,
                    run_id=identity.run_id,
                    attempt_id=identity.attempt_id,
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
