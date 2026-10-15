# © Artur Czarnecki. All rights reserved.

"""Tenant authority gate before colocated physical execution admission (P3-A1)."""

from __future__ import annotations

from intergrax.applications._shared.strict_production_serving_execution_tenant_authority import (
    StrictProductionServingExecutionTenantBinding,
    assert_task_tenant_matches_strict_production_serving_authority,
)
from intergrax.contracts.execution_identity import AttemptId, ExecutionId, RunId
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.task.task import Task, TaskResult


class StrictProductionServingTenantAuthorityTaskExecutionPort:
    """Reject unauthorized task tenants before colocated serving-unit admission."""

    def __init__(
        self,
        inner: HostTaskExecutionPort,
        *,
        binding: StrictProductionServingExecutionTenantBinding,
    ) -> None:
        self._inner = inner
        self._binding = binding

    async def execute(
        self,
        task: Task,
        *,
        run_id: RunId | None = None,
        attempt_id: AttemptId | None = None,
        execution_id: ExecutionId | None = None,
        resume_checkpoint: TaskCheckpoint | None = None,
        restore_existing_execution: bool = False,
    ) -> TaskResult:
        assert_task_tenant_matches_strict_production_serving_authority(
            task,
            binding=self._binding,
        )
        return await self._inner.execute(
            task,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            resume_checkpoint=resume_checkpoint,
            restore_existing_execution=restore_existing_execution,
        )


__all__ = ["StrictProductionServingTenantAuthorityTaskExecutionPort"]
