# © Artur Czarnecki. All rights reserved.

"""In-memory suspended execution operation store (UCA-6C-R6)."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from uuid import uuid4

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
    SuspendedOperationMutationResult,
)
from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedExecutionOperationDescriptor,
    SuspendedOperationMaterializationState,
)
from intergrax.contracts.execution.suspended_operation.store import (
    SuspendedExecutionOperationStore,
)
from intergrax.contracts.governed_continuation_correlation import (
    GovernedContinuationCorrelation,
)
from intergrax.contracts.lease_claim import LeaseOwnership, StaleClaimError


class SuspendedOperationStoreInvariantError(RuntimeError):
    """Active descriptor uniqueness violated — fail closed."""


class InMemorySuspendedExecutionOperationStore(SuspendedExecutionOperationStore):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_id: dict[str, SuspendedExecutionOperationDescriptor] = {}

    @property
    def is_durable(self) -> bool:
        return False

    @staticmethod
    def mint_suspended_operation_id() -> str:
        return f"sop_{uuid4().hex}"

    def prepare(
        self,
        descriptor: SuspendedExecutionOperationDescriptor,
    ) -> SuspendedOperationMutationResult:
        with self._lock:
            if descriptor.suspended_operation_id in self._by_id:
                return SuspendedOperationMutationResult(
                    outcome=SuspendedOperationClaimOutcome.INVALID_STATE,
                )
            if (
                descriptor.materialization_state
                is not SuspendedOperationMaterializationState.PREPARED
            ):
                return SuspendedOperationMutationResult(
                    outcome=SuspendedOperationClaimOutcome.INVALID_STATE,
                )
            self._by_id[descriptor.suspended_operation_id] = descriptor
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationClaimOutcome.CLAIMED,
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
        with self._lock:
            current = self._by_id.get(suspended_operation_id)
            if current is None:
                return SuspendedOperationMutationResult(
                    outcome=SuspendedOperationClaimOutcome.NOT_FOUND,
                )
            if current.materialization_revision != expected_materialization_revision:
                return SuspendedOperationMutationResult(
                    outcome=SuspendedOperationClaimOutcome.STALE_REVISION,
                )
            if (
                current.materialization_state
                is not SuspendedOperationMaterializationState.PREPARED
            ):
                return SuspendedOperationMutationResult(
                    outcome=SuspendedOperationClaimOutcome.INVALID_STATE,
                )
            if current.continuation_id != continuation.continuation_id:
                return SuspendedOperationMutationResult(
                    outcome=SuspendedOperationClaimOutcome.CONTINUATION_MISMATCH,
                )
            try:
                assert_execution_continuation_identity_match(
                    current.identity,
                    continuation.identity,
                )
            except Exception:
                return SuspendedOperationMutationResult(
                    outcome=SuspendedOperationClaimOutcome.IDENTITY_MISMATCH,
                )
            if governed_correlation.operation_id != current.invocation_scope_id:
                return SuspendedOperationMutationResult(
                    outcome=SuspendedOperationClaimOutcome.CONTINUATION_MISMATCH,
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
                    outcome=SuspendedOperationClaimOutcome.CONTINUATION_MISMATCH,
                )
            if continuation.lifecycle_state not in {
                ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
                ExecutionContinuationLifecycleState.RESUME_AUTHORIZED,
            }:
                return SuspendedOperationMutationResult(
                    outcome=SuspendedOperationClaimOutcome.INVALID_STATE,
                )
            updated = current.model_copy(
                update={
                    "materialization_state": SuspendedOperationMaterializationState.BLOCKED,
                    "materialization_revision": current.materialization_revision + 1,
                },
            )
            self._by_id[suspended_operation_id] = updated
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationClaimOutcome.CLAIMED,
                descriptor=updated,
            )

    def load(
        self,
        suspended_operation_id: str,
    ) -> SuspendedExecutionOperationDescriptor | None:
        with self._lock:
            return self._by_id.get(suspended_operation_id)

    def load_active_for_continuation(
        self,
        continuation_id: str,
    ) -> SuspendedExecutionOperationDescriptor | None:
        with self._lock:
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
        with self._lock:
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
            if current.materialization_state is SuspendedOperationMaterializationState.CLAIMED:
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
        fence: int,
        lease_expires_at: datetime,
    ) -> SuspendedOperationMutationResult:
        with self._lock:
            current = self._by_id.get(suspended_operation_id)
            if current is None:
                return SuspendedOperationMutationResult(
                    outcome=SuspendedOperationClaimOutcome.NOT_FOUND,
                )
            if current.materialization_revision != expected_materialization_revision:
                return SuspendedOperationMutationResult(
                    outcome=SuspendedOperationClaimOutcome.STALE_REVISION,
                )
            if (
                current.materialization_state
                is not SuspendedOperationMaterializationState.CLAIMED
            ):
                return SuspendedOperationMutationResult(
                    outcome=SuspendedOperationClaimOutcome.INVALID_STATE,
                )
            claim = current.claim_ownership
            if claim is None:
                return SuspendedOperationMutationResult(
                    outcome=SuspendedOperationClaimOutcome.INVALID_STATE,
                )
            now = datetime.now(timezone.utc)
            if claim.lease_expires_at > now:
                return SuspendedOperationMutationResult(
                    outcome=SuspendedOperationClaimOutcome.ALREADY_CLAIMED,
                )
            ownership = LeaseOwnership(
                owner_id=owner_id,
                lease_expires_at=lease_expires_at,
                fence=fence,
            )
            updated = current.model_copy(
                update={
                    "materialization_state": SuspendedOperationMaterializationState.BLOCKED,
                    "materialization_revision": current.materialization_revision + 1,
                    "claim_ownership": ownership,
                },
            )
            self._by_id[suspended_operation_id] = updated
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationClaimOutcome.CLAIMED,
                descriptor=updated,
            )

    def mark_consumed(
        self,
        *,
        suspended_operation_id: str,
        expected_materialization_revision: int,
        owner_id: str,
        fence: int,
    ) -> SuspendedOperationMutationResult:
        with self._lock:
            current = self._by_id.get(suspended_operation_id)
            if current is None:
                return SuspendedOperationMutationResult(
                    outcome=SuspendedOperationClaimOutcome.NOT_FOUND,
                )
            if current.materialization_revision != expected_materialization_revision:
                return SuspendedOperationMutationResult(
                    outcome=SuspendedOperationClaimOutcome.STALE_REVISION,
                )
            claim = current.claim_ownership
            if claim is None or claim.owner_id != owner_id or claim.fence != fence:
                return SuspendedOperationMutationResult(
                    outcome=SuspendedOperationClaimOutcome.STALE_CLAIM,
                )
            if (
                current.materialization_state
                is not SuspendedOperationMaterializationState.CLAIMED
            ):
                return SuspendedOperationMutationResult(
                    outcome=SuspendedOperationClaimOutcome.INVALID_STATE,
                )
            updated = current.model_copy(
                update={
                    "materialization_state": SuspendedOperationMaterializationState.CONSUMED,
                    "materialization_revision": current.materialization_revision + 1,
                },
            )
            self._by_id[suspended_operation_id] = updated
            return SuspendedOperationMutationResult(
                outcome=SuspendedOperationClaimOutcome.CLAIMED,
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
    ) -> SuspendedOperationMutationResult:
        _ = reason
        with self._lock:
            current = self._by_id.get(suspended_operation_id)
            if current is None:
                return SuspendedOperationMutationResult(
                    outcome=SuspendedOperationClaimOutcome.NOT_FOUND,
                )
            if current.materialization_revision != expected_materialization_revision:
                return SuspendedOperationMutationResult(
                    outcome=SuspendedOperationClaimOutcome.STALE_REVISION,
                )
            if current.materialization_state in {
                SuspendedOperationMaterializationState.CONSUMED,
                SuspendedOperationMaterializationState.ABANDONED,
            }:
                return SuspendedOperationMutationResult(
                    outcome=SuspendedOperationClaimOutcome.TERMINAL,
                )
            if current.materialization_state is SuspendedOperationMaterializationState.CLAIMED:
                claim = current.claim_ownership
                if claim is None:
                    return SuspendedOperationMutationResult(
                        outcome=SuspendedOperationClaimOutcome.INVALID_STATE,
                    )
                if owner_id is not None and claim.owner_id != owner_id:
                    return SuspendedOperationMutationResult(
                        outcome=SuspendedOperationClaimOutcome.STALE_CLAIM,
                    )
                if fence is not None and claim.fence != fence:
                    return SuspendedOperationMutationResult(
                        outcome=SuspendedOperationClaimOutcome.STALE_CLAIM,
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
                outcome=SuspendedOperationClaimOutcome.CLAIMED,
                descriptor=updated,
            )


__all__ = [
    "InMemorySuspendedExecutionOperationStore",
    "SuspendedOperationStoreInvariantError",
    "StaleClaimError",
]
