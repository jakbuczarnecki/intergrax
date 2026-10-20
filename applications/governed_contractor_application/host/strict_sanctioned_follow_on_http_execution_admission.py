# © Artur Czarnecki. All rights reserved.

"""Strict HTTP execution admission for sanctioned external-work follow-on mutations."""

from __future__ import annotations

from dataclasses import dataclass

from external_contractor_adapter.external_work_adapter import (
    META_PRINCIPAL_ID,
    META_TENANT_ID,
)
from governed_contractor_application.host.stores import GovernedExecutionStore
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    validate_execution_id,
)
from intergrax.contracts.sanctioned_external_work_operation import (
    META_PRIOR_EXECUTION_ID,
    META_SANCTIONED_MUTATION_OPERATION,
    SanctionedExternalWorkMutationOperation,
)
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.task.task import Task, TaskResult


@dataclass(frozen=True, slots=True)
class _SanctionedFollowOnExecutionAdmission:
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId


def _optional_meta_str(metadata: dict[str, object], key: str) -> str | None:
    raw = metadata.get(key)
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def try_resolve_sanctioned_follow_on_execution_admission(
    task: Task,
    *,
    execution_store: GovernedExecutionStore,
) -> _SanctionedFollowOnExecutionAdmission | None:
    """Bind strict HTTP /run to prior CREATE execution when follow-on metadata is present."""
    metadata = dict(task.metadata or {})
    raw_op = metadata.get(META_SANCTIONED_MUTATION_OPERATION)
    if raw_op is None or not str(raw_op).strip():
        return None
    try:
        operation = SanctionedExternalWorkMutationOperation(str(raw_op).strip())
    except ValueError:
        return None
    if operation is SanctionedExternalWorkMutationOperation.CREATE:
        return None
    prior_raw = metadata.get(META_PRIOR_EXECUTION_ID)
    if prior_raw is None or not str(prior_raw).strip():
        return None
    try:
        prior_execution_id = validate_execution_id(str(prior_raw).strip())
    except ValueError:
        return None
    tenant_id = _optional_meta_str(metadata, META_TENANT_ID) or task.tenant_id.strip()
    principal_id = _optional_meta_str(metadata, META_PRINCIPAL_ID)
    if not tenant_id or not principal_id:
        return None
    record = execution_store.get_create_follow_on_record(str(prior_execution_id))
    if record is None:
        return None
    if str(record.execution_id) != str(prior_execution_id):
        return None
    if str(record.tenant_id) != tenant_id:
        return None
    if str(record.principal_id) != principal_id:
        return None
    return _SanctionedFollowOnExecutionAdmission(
        task_id=record.task_id,
        run_id=record.run_id,
        attempt_id=record.attempt_id,
        execution_id=record.execution_id,
    )


class StrictSanctionedExternalWorkFollowOnExecutionAdmissionPort:
    """Reuses authoritative CREATE execution identity for accept/cancel strict HTTP /run."""

    def __init__(
        self,
        inner: HostTaskExecutionPort,
        *,
        execution_store: GovernedExecutionStore,
    ) -> None:
        self._inner = inner
        self._execution_store = execution_store

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
        admitted = try_resolve_sanctioned_follow_on_execution_admission(
            task,
            execution_store=self._execution_store,
        )
        if admitted is None:
            return await self._inner.execute(
                task,
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=execution_id,
                resume_checkpoint=resume_checkpoint,
                restore_existing_execution=restore_existing_execution,
            )
        bound_task = task.model_copy(update={"task_id": admitted.task_id})
        return await self._inner.execute(
            bound_task,
            run_id=admitted.run_id,
            attempt_id=admitted.attempt_id,
            execution_id=admitted.execution_id,
            resume_checkpoint=resume_checkpoint,
            restore_existing_execution=True,
        )


__all__ = [
    "StrictSanctionedExternalWorkFollowOnExecutionAdmissionPort",
    "try_resolve_sanctioned_follow_on_execution_admission",
]
