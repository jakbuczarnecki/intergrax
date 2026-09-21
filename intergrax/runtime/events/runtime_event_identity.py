# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical RuntimeEvent identity resolution (UE-9B)."""

from __future__ import annotations

from typing import TypedDict

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    require_active_execution_id,
    require_active_execution_identity,
    validate_run_id,
    validate_task_id,
)


class RuntimeEventIdentityKwargs(TypedDict):
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId


def require_bound_runtime_event_identity(
    *,
    task_id: object,
    run_id: object | None = None,
) -> tuple[TaskId, RunId, AttemptId, ExecutionId]:
    """Resolve task/run/attempt/execution from the active execution boundary."""
    active_run_id, attempt_id = require_active_execution_identity()
    execution_id = require_active_execution_id()
    resolved_task_id = validate_task_id(task_id)
    if run_id is None:
        resolved_run_id = active_run_id
    else:
        resolved_run_id = validate_run_id(run_id)
        if resolved_run_id != active_run_id:
            raise RuntimeError("run_id conflicts with active execution identity")
    return resolved_task_id, resolved_run_id, attempt_id, execution_id


def runtime_event_identity_kwargs(
    *,
    task_id: object,
    run_id: object | None = None,
) -> RuntimeEventIdentityKwargs:
    resolved_task_id, resolved_run_id, attempt_id, execution_id = (
        require_bound_runtime_event_identity(task_id=task_id, run_id=run_id)
    )
    return {
        "task_id": resolved_task_id,
        "run_id": resolved_run_id,
        "attempt_id": attempt_id,
        "execution_id": execution_id,
    }
