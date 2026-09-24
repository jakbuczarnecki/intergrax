# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Map canonical intake failures to worker dispatch semantics (UCA-6C-R6-R5.8-R2)."""

from __future__ import annotations

from intergrax.contracts.execution_intake import CanonicalExecutionInvocationFailed
from intergrax.contracts.execution.qualified_capability_execution_dispatch import (
    QualifiedCapabilityExecutionDispatchDisposition,
    QualifiedCapabilityExecutionDispatchResult,
)
from intergrax.runtime.execution.suspended_operation.pause_required import (
    ExecutionSuspendedWorkPauseRequired,
)


def qualified_dispatch_result_for_invocation_failure(
    exc: CanonicalExecutionInvocationFailed,
    *,
    execution_request_id: str,
) -> QualifiedCapabilityExecutionDispatchResult:
    """Governed pause after root launch still correlates as dispatched execution."""
    if isinstance(exc.cause, ExecutionSuspendedWorkPauseRequired):
        return QualifiedCapabilityExecutionDispatchResult(
            disposition=QualifiedCapabilityExecutionDispatchDisposition.DISPATCHED,
            execution_request_id=execution_request_id,
            run_id=exc.run_id,
            attempt_id=exc.attempt_id,
            execution_id=exc.execution_id,
            reason_detail="execution_suspended_work_pause_required",
        )
    raise exc


__all__ = ["qualified_dispatch_result_for_invocation_failure"]
