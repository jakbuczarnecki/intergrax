# © Artur Czarnecki. All rights reserved.

"""Wrap host task execution with colocated serving-unit admission accounting (D1b)."""

from __future__ import annotations

from intergrax.applications._shared.serving_runtime_handoff import ServingRuntimeHandoff
from intergrax.contracts.execution_identity import AttemptId, ExecutionId, RunId
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.task.task import Task
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.task.task import TaskResult


class ColocatedServingUnitAdmissionTaskExecutionPort:
    """Admit/release colocated unit execution around each host task invocation."""

    def __init__(
        self,
        inner: HostTaskExecutionPort,
        handoff: ServingRuntimeHandoff,
    ) -> None:
        self._inner = inner
        self._handoff = handoff

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
        admission = self._handoff.colocated_execution_admission
        if admission is None:
            raise RuntimeError(
                "colocated serving runtime handoff lacks execution admission port"
            )
        unit_ref = self._handoff.identity.serving_unit_ref
        admission.admit_execution(unit_ref)
        try:
            return await self._inner.execute(
                task,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                resume_checkpoint=resume_checkpoint,
                restore_existing_execution=restore_existing_execution,
            )
        finally:
            admission.release_execution(unit_ref)


__all__ = ["ColocatedServingUnitAdmissionTaskExecutionPort"]
