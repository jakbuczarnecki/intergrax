# © Artur Czarnecki. All rights reserved.

"""Prepare caller-held claim authority before production Nexus intake resume."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedOperationMaterializationState,
)
from intergrax.runtime.execution.suspended_operation.claim_lifecycle import (
    ExecutionSuspendedWorkClaimLifecycleCoordinator,
)
from intergrax.runtime.execution.suspended_operation.resume_authority_transport import (
    ExecutionSuspendedWorkResumeAuthorityTransport,
    ProductionSuspendedWorkAuthorityTelemetry,
)
from intergrax.runtime.nexus.orchestration.internal_continuation_orchestration import (
    InternalOrchestrationContinuation,
)
from intergrax.runtime.task.task import Task, TaskState


def _continuation_id_for_waiting_hitl(task: Task) -> str | None:
    human_request = task.runtime.governance.human_request
    if human_request is None:
        return None
    governed = human_request.governed_continuation
    if governed is not None:
        return governed.continuation_request_id
    return f"gcr_hr_{human_request.request_id}"


def prepare_suspended_work_caller_authority_for_hitl_intake(
    task: Task,
    *,
    hitl: InternalOrchestrationContinuation | None,
    lifecycle: ExecutionSuspendedWorkClaimLifecycleCoordinator | None,
    transport: ExecutionSuspendedWorkResumeAuthorityTransport | None,
    telemetry: ProductionSuspendedWorkAuthorityTelemetry | None = None,
) -> None:
    """Claim BLOCKED or reclaim expired CLAIMED lease; never mint from passive store read."""
    if task.state is not TaskState.WAITING_FOR_HUMAN:
        return
    if hitl is None or lifecycle is None or transport is None:
        return
    if transport.peek() is not None:
        return
    reentry = hitl.suspended_work_reentry_coordinator
    if reentry is None:
        return
    continuation_id = _continuation_id_for_waiting_hitl(task)
    if continuation_id is None:
        return
    descriptor = reentry.store.load_active_for_continuation(continuation_id)
    if descriptor is None:
        return
    now = datetime.now(timezone.utc)
    state = descriptor.materialization_state
    if state is SuspendedOperationMaterializationState.BLOCKED:
        context = lifecycle.claim_blocked(descriptor)
        if context is None:
            return
        if telemetry is not None:
            telemetry.claim_successes += 1
            telemetry.authority_snapshots_created += 1
        transport.deliver(context)
        if telemetry is not None:
            telemetry.authority_snapshots_transported += 1
        return
    if state is not SuspendedOperationMaterializationState.CLAIMED:
        return
    ownership = descriptor.claim_ownership
    if ownership is None:
        return
    if ownership.lease_expires_at > now:
        return
    lease_at = now + timedelta(seconds=lifecycle.default_lease_seconds)
    context = lifecycle.reclaim_expired_lease(
        descriptor,
        lease_expires_at=lease_at,
        expected_fence=ownership.fence,
    )
    if context is None:
        return
    if telemetry is not None:
        telemetry.reclaim_successes += 1
        telemetry.authority_snapshots_created += 1
    transport.deliver(context)
    if telemetry is not None:
        telemetry.authority_snapshots_transported += 1


__all__ = ["prepare_suspended_work_caller_authority_for_hitl_intake"]
