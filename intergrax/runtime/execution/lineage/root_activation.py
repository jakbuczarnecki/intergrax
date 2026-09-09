# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Root execution lineage activation helpers (DG-001 R1)."""

from __future__ import annotations

from contextvars import Token

from intergrax.contracts.execution_identity import ExecutionId, TaskId, validate_execution_id
from intergrax.contracts.execution_lineage import (
    ExecutionLineageAttemptScope,
    ExecutionLineageConfigurationError,
    ExecutionLineagePersistence,
    build_execution_lineage_attempt_scope,
)
from intergrax.runtime.execution.lineage.active_lineage import (
    ActiveExecutionLineageState,
    bind_active_execution_lineage,
    reset_active_execution_lineage,
)
from intergrax.runtime.execution.lineage.admission import ExecutionLineageRootAdmissionHook
from intergrax.runtime.execution.boundary import ExecutionAdmissionHook


def validate_root_lineage_inputs(
    *,
    tenant_id: str | None,
    task_id: TaskId | None,
    run_id: object,
    attempt_id: object,
    execution_id: ExecutionId,
) -> ExecutionLineageAttemptScope:
    if tenant_id is None or not tenant_id.strip():
        raise ExecutionLineageConfigurationError("lineage requires tenant_id")
    if task_id is None:
        raise ExecutionLineageConfigurationError("lineage requires task_id")
    return build_execution_lineage_attempt_scope(
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    )


def activate_root_execution_lineage(
    *,
    persistence: ExecutionLineagePersistence,
    scope: ExecutionLineageAttemptScope,
    root_execution_id: ExecutionId,
    predecessor_root_execution_id: ExecutionId | None = None,
) -> tuple[ActiveExecutionLineageState, Token]:
    root = validate_execution_id(root_execution_id)
    predecessor = (
        validate_execution_id(predecessor_root_execution_id)
        if predecessor_root_execution_id is not None
        else None
    )
    persistence.open_attempt(scope)
    persistence.open_segment(scope, root, predecessor)
    state = ActiveExecutionLineageState(
        persistence=persistence,
        scope=scope,
        segment_root_execution_id=root,
    )
    token = bind_active_execution_lineage(state)
    return state, token


def build_root_lineage_admission_hook(
    *,
    persistence: ExecutionLineagePersistence,
    scope: ExecutionLineageAttemptScope,
    segment_root_execution_id: ExecutionId,
    execution_id: ExecutionId,
) -> ExecutionAdmissionHook[object]:
    return ExecutionLineageRootAdmissionHook(
        persistence=persistence,
        scope=scope,
        segment_root_execution_id=segment_root_execution_id,
        execution_id=execution_id,
    )


def merge_lineage_root_admission_hooks(
    lineage_hook: ExecutionAdmissionHook[object],
    admission_hooks: tuple[ExecutionAdmissionHook[object], ...],
) -> tuple[ExecutionAdmissionHook[object], ...]:
    return (lineage_hook, *admission_hooks)


def deactivate_root_execution_lineage(token: Token) -> None:
    reset_active_execution_lineage(token)
