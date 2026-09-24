# © Artur Czarnecki. All rights reserved.

"""Wiring helpers for EE claim lifecycle from orchestration capability."""

from __future__ import annotations

from intergrax.runtime.execution.suspended_operation.claim_lifecycle import (
    ExecutionSuspendedWorkClaimLifecycleCoordinator,
)
from intergrax.runtime.nexus.orchestration.internal_continuation_orchestration import (
    InternalOrchestrationContinuation,
)


def claim_lifecycle_from_hitl_continuation(
    hitl: InternalOrchestrationContinuation | None,
) -> ExecutionSuspendedWorkClaimLifecycleCoordinator | None:
    if hitl is None:
        return None
    reentry = hitl.suspended_work_reentry_coordinator
    if reentry is None:
        return None
    return ExecutionSuspendedWorkClaimLifecycleCoordinator(
        store=reentry.store,
        claim_owner_id=reentry.claim_owner_id,
        default_lease_seconds=reentry.default_lease_seconds,
    )


__all__ = ["claim_lifecycle_from_hitl_continuation"]
