# © Artur Czarnecki. All rights reserved.

"""Map canonical task checkpoint runtime identity to continuation four-ID binding."""

from __future__ import annotations

from intergrax.contracts.execution_continuation import ExecutionContinuationIdentity
from intergrax.contracts.execution_identity import validate_task_id
from intergrax.runtime.long_running.checkpoint_resume_validation import (
    root_execution_id_from_tree,
)
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.long_running.resume_planner import (
    execution_identity_from_checkpoint,
)


def execution_continuation_identity_from_task_checkpoint(
    checkpoint: TaskCheckpoint,
) -> ExecutionContinuationIdentity:
    """Authoritative four-ID from persisted checkpoint runtime (no minting)."""
    runtime = checkpoint.runtime
    if runtime is None:
        raise ValueError(
            f"checkpoint {checkpoint.checkpoint_id!r} missing canonical runtime identity",
        )
    run_id, attempt_id = execution_identity_from_checkpoint(checkpoint)
    task_id = validate_task_id(checkpoint.task_id)
    execution_id = root_execution_id_from_tree(runtime.execution_tree)
    return ExecutionContinuationIdentity(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )


__all__ = ["execution_continuation_identity_from_task_checkpoint"]
