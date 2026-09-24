# © Artur Czarnecki. All rights reserved.

"""Fail-closed cross-source execution identity correlation after durable restart."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedExecutionOperationDescriptor,
)
from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
    ExecutionContinuationIdentity,
    assert_execution_continuation_identity_match,
)
from intergrax.contracts.execution_continuation_state_store import (
    ExecutionContinuationStateStore,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
)
from intergrax.runtime.long_running.checkpoint_continuation_identity import (
    execution_continuation_identity_from_task_checkpoint,
)
from intergrax.runtime.long_running.models import TaskCheckpoint


@dataclass(frozen=True, slots=True)
class DurableRestartExecutionIdentityCorrelation:
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId
    continuation_identity: ExecutionContinuationIdentity


def correlate_durable_restart_execution_identity(
    *,
    checkpoint: TaskCheckpoint,
    continuation_store: ExecutionContinuationStateStore,
    continuation_id: str,
    suspended_descriptor: SuspendedExecutionOperationDescriptor,
) -> DurableRestartExecutionIdentityCorrelation:
    """Require checkpoint, continuation episode, and suspended descriptor to agree exactly."""
    checkpoint_identity = execution_continuation_identity_from_task_checkpoint(
        checkpoint
    )
    try:
        located = continuation_store.load(continuation_id)
    except ExecutionContinuationError:
        raise
    except Exception as exc:
        raise ExecutionContinuationError(
            "continuation store query failed during restart identity correlation",
            code=ExecutionContinuationErrorCode.STORE_QUERY_FAILED,
        ) from exc
    if located is None:
        raise ExecutionContinuationError(
            "continuation snapshot missing for restart identity correlation",
            code=ExecutionContinuationErrorCode.NOT_FOUND,
        )
    assert_execution_continuation_identity_match(
        checkpoint_identity,
        located.identity,
    )
    assert_execution_continuation_identity_match(
        checkpoint_identity,
        suspended_descriptor.identity,
    )
    return DurableRestartExecutionIdentityCorrelation(
        task_id=checkpoint_identity.task_id,
        run_id=checkpoint_identity.run_id,
        attempt_id=checkpoint_identity.attempt_id,
        execution_id=checkpoint_identity.execution_id,
        continuation_identity=checkpoint_identity,
    )


__all__ = [
    "DurableRestartExecutionIdentityCorrelation",
    "correlate_durable_restart_execution_identity",
]
