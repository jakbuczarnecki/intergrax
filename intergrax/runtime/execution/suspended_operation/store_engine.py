# © Artur Czarnecki. All rights reserved.

"""Shared suspended-operation store semantics (UCA-6C-R6-R1)."""

from __future__ import annotations

from datetime import datetime, timezone

from intergrax.contracts.execution_continuation import (
    ExecutionContinuationLifecycleState,
    PendingExecutionContinuation,
    assert_execution_continuation_identity_match,
    assert_governed_correlation_matches_continuation,
)
from intergrax.contracts.execution.suspended_operation.claim import (
    SuspendedOperationAbandonReason,
    SuspendedOperationClaimOutcome,
    SuspendedOperationClaimResult,
    SuspendedOperationMutationOutcome,
    SuspendedOperationMutationResult,
)
from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedExecutionOperationDescriptor,
    SuspendedOperationMaterializationState,
)
from intergrax.contracts.execution.suspended_operation.authority_scope import (
    SuspendedOperationAuthorityScope,
)
from intergrax.contracts.governed_continuation_correlation import (
    GovernedContinuationCorrelation,
)
from intergrax.contracts.lease_claim import LeaseOwnership


class SuspendedOperationStoreInvariantError(RuntimeError):
    """Active descriptor uniqueness violated — fail closed."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _validate_lease_expires_at(lease_expires_at: datetime) -> bool:
    if lease_expires_at.tzinfo is None:
        return False
    return lease_expires_at > _utc_now()


class SuspendedOperationBackingStore:
    """In-process descriptor map — backing for memory and document providers."""

    def __init__(self) -> None:
        self._by_id: dict[str, SuspendedExecutionOperationDescriptor] = {}

    def snapshot(self) -> dict[str, SuspendedExecutionOperationDescriptor]:
        return dict(self._by_id)

    def replace_all(
        self,
        records: dict[str, SuspendedExecutionOperationDescriptor],
    ) -> None:
        self._by_id = dict(records)

    def prepare(
        self,
        descriptor: SuspendedExecutionOperationDescriptor,
    ) -> SuspendedOperationMutationResult:
        if descriptor.suspended_operation_id in self._by_id:
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.INVALID_STATE,
            )
        if (
            descriptor.materialization_state
            is not SuspendedOperationMaterializationState.PREPARED
        ):
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.INVALID_STATE,
            )
        self._by_id[descriptor.suspended_operation_id] = descriptor
        return SuspendedOperationMutationResult(
            outcome=SuspendedOperationMutationOutcome.APPLIED,
            descriptor=descriptor,
        )

    def block(
        self,
        *,
        suspended_operation_id: str,
        expected_materialization_revision: int,
        continuation: PendingExecutionContinuation,
        governed_correlation: GovernedContinuationCorrelation,
    ) -> SuspendedOperationMutationResult:
        current = self._by_id.get(suspended_operation_id)
        if current is None:
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.NOT_FOUND,
            )
        if current.materialization_revision != expected_materialization_revision:
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.STALE_REVISION,
            )
        if (
            current.materialization_state
            is not SuspendedOperationMaterializationState.PREPARED
        ):
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.INVALID_STATE,
            )
        if current.continuation_id != continuation.continuation_id:
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.CONTINUATION_MISMATCH,
            )
        try:
            assert_execution_continuation_identity_match(
                current.identity,
                continuation.identity,
            )
        except Exception:
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.IDENTITY_MISMATCH,
            )
        if governed_correlation.operation_id != current.invocation_scope_id:
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.CONTINUATION_MISMATCH,
            )
        try:
            assert_governed_correlation_matches_continuation(
                continuation_id=continuation.continuation_id,
                identity=continuation.identity,
                reason=continuation.reason,
                governed_correlation=governed_correlation,
            )
        except Exception:
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.CONTINUATION_MISMATCH,
            )
        if continuation.lifecycle_state not in {
            ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
            ExecutionContinuationLifecycleState.RESUME_AUTHORIZED,
        }:
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.INVALID_STATE,
            )
        updated = current.model_copy(
            update={
                "materialization_state": SuspendedOperationMaterializationState.BLOCKED,
                "materialization_revision": current.materialization_revision + 1,
                "claim_ownership": None,
            },
        )
        self._by_id[suspended_operation_id] = updated
        return SuspendedOperationMutationResult(
            outcome=SuspendedOperationMutationOutcome.APPLIED,
            descriptor=updated,
        )

    def load(
        self,
        suspended_operation_id: str,
    ) -> SuspendedExecutionOperationDescriptor | None:
        return self._by_id.get(suspended_operation_id)

    def load_active_for_continuation(
        self,
        continuation_id: str,
    ) -> SuspendedExecutionOperationDescriptor | None:
        active = [
            descriptor
            for descriptor in self._by_id.values()
            if descriptor.continuation_id == continuation_id
            and descriptor.materialization_state
            in {
                SuspendedOperationMaterializationState.BLOCKED,
                SuspendedOperationMaterializationState.CLAIMED,
            }
        ]
        if len(active) > 1:
            raise SuspendedOperationStoreInvariantError(
                "multiple active suspended operations for continuation",
            )
        if not active:
            return None
        return active[0]

    def claim(
        self,
        *,
        suspended_operation_id: str,
        expected_materialization_revision: int,
        owner_id: str,
        lease_expires_at: datetime,
    ) -> SuspendedOperationClaimResult:
        if not _validate_lease_expires_at(lease_expires_at):
            return SuspendedOperationClaimResult(
                outcome=SuspendedOperationClaimOutcome.INVALID_STATE,
            )
        current = self._by_id.get(suspended_operation_id)
        if current is None:
            return SuspendedOperationClaimResult(
                outcome=SuspendedOperationClaimOutcome.NOT_FOUND,
            )
        if current.materialization_revision != expected_materialization_revision:
            return SuspendedOperationClaimResult(
                outcome=SuspendedOperationClaimOutcome.STALE_REVISION,
            )
        if current.materialization_state in {
            SuspendedOperationMaterializationState.CONSUMED,
            SuspendedOperationMaterializationState.ABANDONED,
        }:
            return SuspendedOperationClaimResult(
                outcome=SuspendedOperationClaimOutcome.TERMINAL,
            )
        if (
            current.materialization_state
            is SuspendedOperationMaterializationState.CLAIMED
        ):
            return SuspendedOperationClaimResult(
                outcome=SuspendedOperationClaimOutcome.ALREADY_CLAIMED,
            )
        if (
            current.materialization_state
            is not SuspendedOperationMaterializationState.BLOCKED
        ):
            return SuspendedOperationClaimResult(
                outcome=SuspendedOperationClaimOutcome.INVALID_STATE,
            )
        fence = 1
        ownership = LeaseOwnership(
            owner_id=owner_id,
            lease_expires_at=lease_expires_at,
            fence=fence,
        )
        updated = current.model_copy(
            update={
                "materialization_state": SuspendedOperationMaterializationState.CLAIMED,
                "materialization_revision": current.materialization_revision + 1,
                "claim_ownership": ownership,
            },
        )
        self._by_id[suspended_operation_id] = updated
        return SuspendedOperationClaimResult(
            outcome=SuspendedOperationClaimOutcome.CLAIMED,
            descriptor=updated,
        )

    def reclaim(
        self,
        *,
        suspended_operation_id: str,
        expected_materialization_revision: int,
        owner_id: str,
        lease_expires_at: datetime,
        expected_fence: int | None = None,
    ) -> SuspendedOperationMutationResult:
        if not _validate_lease_expires_at(lease_expires_at):
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.INVALID_STATE,
            )
        current = self._by_id.get(suspended_operation_id)
        if current is None:
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.NOT_FOUND,
            )
        if current.materialization_revision != expected_materialization_revision:
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.STALE_REVISION,
            )
        if (
            current.materialization_state
            is not SuspendedOperationMaterializationState.CLAIMED
        ):
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.INVALID_STATE,
            )
        claim = current.claim_ownership
        if claim is None:
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.INVALID_STATE,
            )
        if expected_fence is not None and claim.fence != expected_fence:
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.STALE_CLAIM,
            )
        if claim.lease_expires_at > _utc_now():
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.INVALID_STATE,
            )
        new_fence = claim.fence + 1
        ownership = LeaseOwnership(
            owner_id=owner_id,
            lease_expires_at=lease_expires_at,
            fence=new_fence,
        )
        updated = current.model_copy(
            update={
                "materialization_state": SuspendedOperationMaterializationState.CLAIMED,
                "materialization_revision": current.materialization_revision + 1,
                "claim_ownership": ownership,
            },
        )
        self._by_id[suspended_operation_id] = updated
        return SuspendedOperationMutationResult(
            outcome=SuspendedOperationMutationOutcome.APPLIED,
            descriptor=updated,
        )

    def mark_consumed(
        self,
        *,
        suspended_operation_id: str,
        expected_materialization_revision: int,
        owner_id: str,
        fence: int,
        expected_pause_generation: int | None = None,
    ) -> SuspendedOperationMutationResult:
        current = self._by_id.get(suspended_operation_id)
        if current is None:
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.NOT_FOUND,
            )
        if current.materialization_revision != expected_materialization_revision:
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.STALE_REVISION,
            )
        claim = current.claim_ownership
        if claim is None or claim.owner_id != owner_id or claim.fence != fence:
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.STALE_CLAIM,
            )
        if (
            expected_pause_generation is not None
            and current.pause_generation != expected_pause_generation
        ):
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.STALE_CLAIM,
            )
        if (
            current.materialization_state
            is not SuspendedOperationMaterializationState.CLAIMED
        ):
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.INVALID_STATE,
            )
        updated = current.model_copy(
            update={
                "materialization_state": SuspendedOperationMaterializationState.CONSUMED,
                "materialization_revision": current.materialization_revision + 1,
                "claim_ownership": None,
            },
        )
        self._by_id[suspended_operation_id] = updated
        return SuspendedOperationMutationResult(
            outcome=SuspendedOperationMutationOutcome.APPLIED,
            descriptor=updated,
        )

    def authority_reblock_from_claimed(
        self,
        *,
        suspended_operation_id: str,
        expected_materialization_revision: int,
        expected_pause_generation: int,
        expected_owner_id: str,
        expected_fence: int,
        next_pause_generation: int,
        next_continuation: PendingExecutionContinuation,
        next_governed_correlation: GovernedContinuationCorrelation,
        next_invocation_scope_id: str,
        next_authority_scope: SuspendedOperationAuthorityScope,
    ) -> SuspendedOperationMutationResult:
        if next_pause_generation != expected_pause_generation + 1:
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.INVALID_STATE,
            )
        current = self._by_id.get(suspended_operation_id)
        if current is None:
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.NOT_FOUND,
            )
        if current.materialization_revision != expected_materialization_revision:
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.STALE_REVISION,
            )
        if current.pause_generation != expected_pause_generation:
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.STALE_CLAIM,
            )
        claim = current.claim_ownership
        if claim is None:
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.INVALID_STATE,
            )
        if claim.owner_id != expected_owner_id or claim.fence != expected_fence:
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.STALE_CLAIM,
            )
        if (
            current.materialization_state
            is not SuspendedOperationMaterializationState.CLAIMED
        ):
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.INVALID_STATE,
            )
        if next_continuation.continuation_id != next_governed_correlation.continuation_request_id:
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.CONTINUATION_MISMATCH,
            )
        updated = current.model_copy(
            update={
                "materialization_state": SuspendedOperationMaterializationState.BLOCKED,
                "claim_ownership": None,
                "pause_generation": next_pause_generation,
                "materialization_revision": current.materialization_revision + 1,
                "continuation_id": next_continuation.continuation_id,
                "invocation_scope_id": next_invocation_scope_id.strip(),
            },
        )
        _ = next_authority_scope
        self._by_id[suspended_operation_id] = updated
        return SuspendedOperationMutationResult(
            outcome=SuspendedOperationMutationOutcome.APPLIED,
            descriptor=updated,
        )

    def abandon(
        self,
        *,
        suspended_operation_id: str,
        expected_materialization_revision: int,
        reason: SuspendedOperationAbandonReason,
        owner_id: str | None = None,
        fence: int | None = None,
        expected_pause_generation: int | None = None,
    ) -> SuspendedOperationMutationResult:
        _ = reason
        current = self._by_id.get(suspended_operation_id)
        if current is None:
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.NOT_FOUND,
            )
        if current.materialization_revision != expected_materialization_revision:
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.STALE_REVISION,
            )
        if current.materialization_state in {
            SuspendedOperationMaterializationState.CONSUMED,
            SuspendedOperationMaterializationState.ABANDONED,
        }:
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationMutationOutcome.TERMINAL,
            )
        if (
            current.materialization_state
            is SuspendedOperationMaterializationState.CLAIMED
        ):
            claim = current.claim_ownership
            if claim is None:
                return SuspendedOperationMutationResult(
                    outcome=SuspendedOperationMutationOutcome.INVALID_STATE,
                )
            if owner_id is not None and claim.owner_id != owner_id:
                return SuspendedOperationMutationResult(
                    outcome=SuspendedOperationMutationOutcome.STALE_CLAIM,
                )
            if fence is not None and claim.fence != fence:
                return SuspendedOperationMutationResult(
                    outcome=SuspendedOperationMutationOutcome.STALE_CLAIM,
                )
            if (
                expected_pause_generation is not None
                and current.pause_generation != expected_pause_generation
            ):
                return SuspendedOperationMutationResult(
                    outcome=SuspendedOperationMutationOutcome.STALE_CLAIM,
                )
        updated = current.model_copy(
            update={
                "materialization_state": SuspendedOperationMaterializationState.ABANDONED,
                "materialization_revision": current.materialization_revision + 1,
                "claim_ownership": None,
            },
        )
        self._by_id[suspended_operation_id] = updated
        return SuspendedOperationMutationResult(
            outcome=SuspendedOperationMutationOutcome.APPLIED,
            descriptor=updated,
        )


__all__ = [
    "SuspendedOperationBackingStore",
    "SuspendedOperationStoreInvariantError",
]
