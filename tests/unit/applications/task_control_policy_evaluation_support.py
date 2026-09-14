# © Artur Czarnecki. All rights reserved.

"""GR-1 execution identity binding for bundle-backed task-control policy tests."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from intergrax.contracts.control_plane_mutation import (
    ControlPlaneMutationAuthorizationResult,
    ControlPlaneMutationRequest,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    reset_active_execution_identity,
)
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.runtime.long_running.models import TaskCheckpoint


@contextmanager
def bound_control_plane_policy_execution_identity(
    *,
    run_id: RunId,
    attempt_id: AttemptId | None = None,
    execution_id: ExecutionId | None = None,
) -> Iterator[None]:
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id or mint_attempt_id(),
        execution_id=execution_id or mint_execution_id(),
    )
    try:
        yield
    finally:
        reset_active_execution_identity(token)


def execution_identity_from_task_checkpoint(
    checkpoint: TaskCheckpoint,
) -> tuple[RunId, AttemptId, ExecutionId]:
    runtime = checkpoint.runtime
    root_execution_id = runtime.execution_tree.entries[0].execution_id
    return runtime.run_id, runtime.attempt_id, root_execution_id


def authorize_bundle_backed_control_plane_mutation(
    boundary: ControlPlaneMutationAuthorizationBoundary,
    request: ControlPlaneMutationRequest,
    *,
    checkpoint: TaskCheckpoint | None = None,
) -> ControlPlaneMutationAuthorizationResult:
    if request.run_id is None:
        raise ValueError(
            "task control mutation request requires run_id for GR-1 policy evaluation",
        )
    if checkpoint is not None:
        run_id, attempt_id, execution_id = execution_identity_from_task_checkpoint(
            checkpoint,
        )
        if run_id != request.run_id:
            raise ValueError("checkpoint run_id must match mutation request run_id")
    else:
        run_id = request.run_id
        attempt_id = None
        execution_id = None
    with bound_control_plane_policy_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        return boundary.authorize(request)
