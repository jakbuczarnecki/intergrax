# © Artur Czarnecki. All rights reserved.

"""EE-owned RESUMED → suspended work re-entry (UCA-6C-R6-R2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_continuation import PendingExecutionContinuation
from intergrax.contracts.execution.suspended_operation.claim_authority import (
    SuspendedOperationClaimAuthority,
)
from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedOperationMaterializationState,
)
from intergrax.contracts.execution.suspended_operation.reentry import (
    ExecutionSuspendedWorkReentryDisposition,
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


@dataclass
class SuspendedOperationClaimAuthorityResumeTelemetry:
    """Optional counters for caller-held claim authority at canonical resume."""

    claim_authority_created: int = 0
    claim_authority_passed: int = 0
    authority_refreshes_from_store: int = 0


def resume_authorized_continuation_with_suspended_work_reentry(
    task: Task,
    authorized: PendingExecutionContinuation,
    *,
    capability: InternalOrchestrationContinuation,
    reentry_coordinator: ExecutionSuspendedWorkReentryCoordinator | None = None,
    reentry_port: ExecutionSuspendedWorkReentryPort | None = None,
    claim_authority: SuspendedOperationClaimAuthority | None = None,
    authority_telemetry: SuspendedOperationClaimAuthorityResumeTelemetry | None = None,
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
        telemetry = authority_telemetry
        if active is None:
            if claim_authority is None:
                reentry_result = ExecutionSuspendedWorkReentryResult(
                    disposition=ExecutionSuspendedWorkReentryDisposition.NOT_READY,
                    reason_detail="no_active_suspended_operation",
                )
            else:
                if telemetry is not None:
                    telemetry.claim_authority_passed += 1
                resolved_authority = claim_authority
                request = ExecutionSuspendedWorkReentryRequest(
                    continuation_id=resumed.continuation_id,
                    identity=resumed.identity,
                    claim_authority=resolved_authority,
                )
                reentry_result = reentry_coordinator.reenter_after_resume(
                    request,
                    task=task,
                )
        elif (
            active.materialization_state
            is SuspendedOperationMaterializationState.CLAIMED
            and active.claim_ownership is not None
        ):
            if claim_authority is None:
                reentry_result = ExecutionSuspendedWorkReentryResult(
                    disposition=ExecutionSuspendedWorkReentryDisposition.FAILED,
                    reason_detail="missing_caller_claim_authority",
                )
            else:
                if telemetry is not None:
                    telemetry.claim_authority_passed += 1
                resolved_authority = claim_authority
                request = ExecutionSuspendedWorkReentryRequest(
                    continuation_id=resumed.continuation_id,
                    identity=resumed.identity,
                    claim_authority=resolved_authority,
                )
                reentry_result = reentry_coordinator.reenter_after_resume(
                    request,
                    task=task,
                )
        else:
            if claim_authority is not None:
                if telemetry is not None:
                    telemetry.claim_authority_passed += 1
                resolved_authority = claim_authority
            else:
                resolved_authority = (
                    SuspendedOperationClaimAuthority.for_host_pending_claim(
                        host_owner_id=reentry_coordinator.claim_owner_id,
                        descriptor=active,
                    )
                )
                if telemetry is not None:
                    telemetry.claim_authority_created += 1
            request = ExecutionSuspendedWorkReentryRequest(
                continuation_id=resumed.continuation_id,
                identity=resumed.identity,
                claim_authority=resolved_authority,
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


__all__ = [
    "SuspendedOperationClaimAuthorityResumeTelemetry",
    "resume_authorized_continuation_with_suspended_work_reentry",
]
