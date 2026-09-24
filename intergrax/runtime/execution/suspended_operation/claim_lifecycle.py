# © Artur Czarnecki. All rights reserved.

"""EE-owned claim/reclaim and single-point caller-held authority minting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from intergrax.contracts.execution.suspended_operation.claim import (
    SuspendedOperationClaimOutcome,
    SuspendedOperationMutationOutcome,
)
from intergrax.contracts.execution.suspended_operation.claim_authority import (
    SuspendedOperationClaimAuthority,
)
from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedExecutionOperationDescriptor,
    SuspendedOperationMaterializationState,
)
from intergrax.contracts.execution.suspended_operation.resume_authority_context import (
    ExecutionSuspendedWorkResumeAuthorityContext,
)
from intergrax.contracts.execution.suspended_operation.store import (
    SuspendedExecutionOperationStore,
)


@dataclass(frozen=True, slots=True)
class ExecutionSuspendedWorkClaimLifecycleCoordinator:
    """Semantic owner of claim/reclaim → ``from_claimed_descriptor`` for this host."""

    store: SuspendedExecutionOperationStore
    claim_owner_id: str
    default_lease_seconds: int = 120

    def claim_blocked(
        self,
        descriptor: SuspendedExecutionOperationDescriptor,
        *,
        lease_expires_at: datetime | None = None,
    ) -> ExecutionSuspendedWorkResumeAuthorityContext | None:
        if (
            descriptor.materialization_state
            is not SuspendedOperationMaterializationState.BLOCKED
        ):
            return None
        now = datetime.now(timezone.utc)
        lease = lease_expires_at or (
            now + timedelta(seconds=self.default_lease_seconds)
        )
        result = self.store.claim(
            suspended_operation_id=descriptor.suspended_operation_id,
            expected_materialization_revision=descriptor.materialization_revision,
            owner_id=self.claim_owner_id,
            lease_expires_at=lease,
        )
        if result.outcome is not SuspendedOperationClaimOutcome.CLAIMED:
            return None
        if result.descriptor is None:
            return None
        authority = SuspendedOperationClaimAuthority.from_claimed_descriptor(
            result.descriptor,
        )
        return ExecutionSuspendedWorkResumeAuthorityContext(
            continuation_id=descriptor.continuation_id,
            claim_authority=authority,
        )

    def reclaim_expired_lease(
        self,
        descriptor: SuspendedExecutionOperationDescriptor,
        *,
        lease_expires_at: datetime,
        expected_fence: int,
    ) -> ExecutionSuspendedWorkResumeAuthorityContext | None:
        if (
            descriptor.materialization_state
            is not SuspendedOperationMaterializationState.CLAIMED
        ):
            return None
        ownership = descriptor.claim_ownership
        if ownership is None:
            return None
        if ownership.owner_id != self.claim_owner_id:
            return None
        result = self.store.reclaim(
            suspended_operation_id=descriptor.suspended_operation_id,
            expected_materialization_revision=descriptor.materialization_revision,
            owner_id=self.claim_owner_id,
            lease_expires_at=lease_expires_at,
            expected_fence=expected_fence,
        )
        if result.outcome is not SuspendedOperationMutationOutcome.APPLIED:
            return None
        if result.descriptor is None:
            return None
        authority = SuspendedOperationClaimAuthority.from_claimed_descriptor(
            result.descriptor,
        )
        return ExecutionSuspendedWorkResumeAuthorityContext(
            continuation_id=descriptor.continuation_id,
            claim_authority=authority,
        )


__all__ = ["ExecutionSuspendedWorkClaimLifecycleCoordinator"]
