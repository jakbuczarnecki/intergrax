# © Artur Czarnecki. All rights reserved.

"""EE-owned RESUMED → suspended work re-entry (UCA-6C-R6-R2)."""

from __future__ import annotations

from intergrax.contracts.execution_continuation import PendingExecutionContinuation
from intergrax.contracts.execution.suspended_operation.claim_authority import (
    SuspendedOperationClaimAuthority,
)
from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedOperationMaterializationState,
)
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
from intergrax.runtime.task.task import Task


def resume_authorized_continuation_with_suspended_work_reentry(
    task: Task,
    authorized: PendingExecutionContinuation,
    *,
    capability: InternalOrchestrationContinuation,
    reentry_coordinator: ExecutionSuspendedWorkReentryCoordinator | None = None,
    reentry_port: ExecutionSuspendedWorkReentryPort | None = None,
) -> tuple[PendingExecutionContinuation, ExecutionSuspendedWorkReentryResult | None]:
    """Lifecycle-only resume, then optional Execution-owned tool re-entry."""
    resumed = canonical_resume_after_authorization(
        task,
        authorized,
        capability=capability,
    )
    reentry_result: ExecutionSuspendedWorkReentryResult | None = None
    if reentry_coordinator is not None:
        active = reentry_coordinator.store.load_active_for_continuation(
            resumed.continuation_id,
        )
        if active is None:
            claim_authority = SuspendedOperationClaimAuthority(
                owner_id=reentry_coordinator.claim_owner_id,
                fence=0,
                materialization_revision=0,
                pause_generation=1,
            )
        elif (
            active.materialization_state
            is SuspendedOperationMaterializationState.CLAIMED
            and active.claim_ownership is not None
        ):
            claim_authority = SuspendedOperationClaimAuthority.from_claimed_descriptor(
                active,
            )
        else:
            claim_authority = SuspendedOperationClaimAuthority.for_host_pending_claim(
                host_owner_id=reentry_coordinator.claim_owner_id,
                descriptor=active,
            )
        request = ExecutionSuspendedWorkReentryRequest(
            continuation_id=resumed.continuation_id,
            identity=resumed.identity,
            claim_authority=claim_authority,
        )
        reentry_result = reentry_coordinator.reenter_after_resume(
            request,
            task=task,
        )
    elif reentry_port is not None:
        raise RuntimeError(
            "reentry_port requires ExecutionSuspendedWorkReentryRequest.claim_authority; "
            "use reentry_coordinator",
        )
    return resumed, reentry_result


__all__ = ["resume_authorized_continuation_with_suspended_work_reentry"]
