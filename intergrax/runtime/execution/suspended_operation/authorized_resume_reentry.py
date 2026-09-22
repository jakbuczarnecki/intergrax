# © Artur Czarnecki. All rights reserved.

"""EE-owned RESUMED → suspended work re-entry (UCA-6C-R6-R2)."""

from __future__ import annotations

from intergrax.contracts.execution_continuation import PendingExecutionContinuation
from intergrax.contracts.execution.suspended_operation.reentry import (
    ExecutionSuspendedWorkReentryPort,
    ExecutionSuspendedWorkReentryRequest,
    ExecutionSuspendedWorkReentryResult,
)
from intergrax.runtime.execution.suspended_operation.reentry_coordinator import (
    ExecutionSuspendedWorkReentryCoordinator,
)
from intergrax.runtime.nexus.orchestration.internal_continuation_orchestration import (
    InternalOrchestrationContinuation,
    canonical_resume_after_authorization,
)
from intergrax.runtime.sandbox.session import SandboxSession
from intergrax.runtime.task.task import Task


def resume_authorized_continuation_with_suspended_work_reentry(
    task: Task,
    authorized: PendingExecutionContinuation,
    *,
    capability: InternalOrchestrationContinuation,
    reentry_coordinator: ExecutionSuspendedWorkReentryCoordinator | None = None,
    reentry_port: ExecutionSuspendedWorkReentryPort | None = None,
    sandbox_session: SandboxSession | None = None,
) -> tuple[PendingExecutionContinuation, ExecutionSuspendedWorkReentryResult | None]:
    """Lifecycle-only resume, then optional Execution-owned tool re-entry."""
    resumed = canonical_resume_after_authorization(
        task,
        authorized,
        capability=capability,
    )
    reentry_result: ExecutionSuspendedWorkReentryResult | None = None
    request = ExecutionSuspendedWorkReentryRequest(
        continuation_id=resumed.continuation_id,
        identity=resumed.identity,
    )
    if reentry_coordinator is not None:
        reentry_result = reentry_coordinator.reenter_after_resume(
            request,
            task=task,
            sandbox_session=sandbox_session,
        )
    elif reentry_port is not None:
        reentry_result = reentry_port.reenter_after_resume(request)
    return resumed, reentry_result


__all__ = ["resume_authorized_continuation_with_suspended_work_reentry"]
