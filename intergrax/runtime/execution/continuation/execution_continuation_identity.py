# © Artur Czarnecki. All rights reserved.

"""GR-5-R2-R3 — authoritative four-ID binding for continuation progress enforcement."""

from __future__ import annotations

from typing import TYPE_CHECKING

from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
    ExecutionContinuationIdentity,
)

if TYPE_CHECKING:
    from intergrax.runtime.execution.boundary import ExecutionIdentityBinding


def require_execution_continuation_identity(
    binding: ExecutionIdentityBinding | None,
) -> ExecutionContinuationIdentity:
    """Build mandatory continuation identity from canonical execution binding; fail closed."""
    if binding is None:
        raise ExecutionContinuationError(
            "canonical execution identity required for continuation enforcement",
            code=ExecutionContinuationErrorCode.INCOMPLETE_EXECUTION_IDENTITY,
        )
    if binding.task_id is None:
        raise ExecutionContinuationError(
            "TaskId missing for continuation enforcement",
            code=ExecutionContinuationErrorCode.INCOMPLETE_EXECUTION_IDENTITY,
        )
    return ExecutionContinuationIdentity(
        task_id=binding.task_id,
        run_id=binding.run_id,
        attempt_id=binding.attempt_id,
        execution_id=binding.execution_id,
    )


__all__ = ["require_execution_continuation_identity"]
