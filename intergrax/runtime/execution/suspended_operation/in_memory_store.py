# © Artur Czarnecki. All rights reserved.

"""In-memory suspended execution operation store (UCA-6C-R6)."""

from __future__ import annotations

import threading

from intergrax.contracts.execution_continuation import PendingExecutionContinuation
from intergrax.contracts.execution.suspended_operation.claim import (
    SuspendedOperationAbandonReason,
    SuspendedOperationClaimResult,
    SuspendedOperationMutationResult,
)
from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedExecutionOperationDescriptor,
)
from intergrax.contracts.execution.suspended_operation.store import (
    SuspendedExecutionOperationStore,
)
from intergrax.contracts.execution.suspended_operation.authority_scope import (
    SuspendedOperationAuthorityScope,
)
from intergrax.contracts.governed_continuation_correlation import (
    GovernedContinuationCorrelation,
)
from intergrax.contracts.lease_claim import StaleClaimError
from intergrax.runtime.execution.suspended_operation.store_engine import (
    SuspendedOperationBackingStore,
    SuspendedOperationStoreInvariantError,
)


class InMemorySuspendedExecutionOperationStore(SuspendedExecutionOperationStore):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._backing = SuspendedOperationBackingStore()

    @property
    def is_durable(self) -> bool:
        return False

    def prepare(
        self,
        descriptor: SuspendedExecutionOperationDescriptor,
    ) -> SuspendedOperationMutationResult:
        with self._lock:
            return self._backing.prepare(descriptor)

    def block(
        self,
        *,
        suspended_operation_id: str,
        expected_materialization_revision: int,
        continuation: PendingExecutionContinuation,
        governed_correlation: GovernedContinuationCorrelation,
    ) -> SuspendedOperationMutationResult:
        with self._lock:
            return self._backing.block(
                suspended_operation_id=suspended_operation_id,
                expected_materialization_revision=expected_materialization_revision,
                continuation=continuation,
                governed_correlation=governed_correlation,
            )

    def load(
        self,
        suspended_operation_id: str,
    ) -> SuspendedExecutionOperationDescriptor | None:
        with self._lock:
            return self._backing.load(suspended_operation_id)

    def load_active_for_continuation(
        self,
        continuation_id: str,
    ) -> SuspendedExecutionOperationDescriptor | None:
        with self._lock:
            return self._backing.load_active_for_continuation(continuation_id)

    def claim(
        self,
        *,
        suspended_operation_id: str,
        expected_materialization_revision: int,
        owner_id: str,
        lease_expires_at,
    ) -> SuspendedOperationClaimResult:
        with self._lock:
            return self._backing.claim(
                suspended_operation_id=suspended_operation_id,
                expected_materialization_revision=expected_materialization_revision,
                owner_id=owner_id,
                lease_expires_at=lease_expires_at,
            )

    def reclaim(
        self,
        *,
        suspended_operation_id: str,
        expected_materialization_revision: int,
        owner_id: str,
        lease_expires_at,
        expected_fence: int | None = None,
    ) -> SuspendedOperationMutationResult:
        with self._lock:
            return self._backing.reclaim(
                suspended_operation_id=suspended_operation_id,
                expected_materialization_revision=expected_materialization_revision,
                owner_id=owner_id,
                lease_expires_at=lease_expires_at,
                expected_fence=expected_fence,
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
        with self._lock:
            return self._backing.mark_consumed(
                suspended_operation_id=suspended_operation_id,
                expected_materialization_revision=expected_materialization_revision,
                owner_id=owner_id,
                fence=fence,
                expected_pause_generation=expected_pause_generation,
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
        with self._lock:
            return self._backing.authority_reblock_from_claimed(
                suspended_operation_id=suspended_operation_id,
                expected_materialization_revision=expected_materialization_revision,
                expected_pause_generation=expected_pause_generation,
                expected_owner_id=expected_owner_id,
                expected_fence=expected_fence,
                next_pause_generation=next_pause_generation,
                next_continuation=next_continuation,
                next_governed_correlation=next_governed_correlation,
                next_invocation_scope_id=next_invocation_scope_id,
                next_authority_scope=next_authority_scope,
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
        with self._lock:
            return self._backing.abandon(
                suspended_operation_id=suspended_operation_id,
                expected_materialization_revision=expected_materialization_revision,
                reason=reason,
                owner_id=owner_id,
                fence=fence,
                expected_pause_generation=expected_pause_generation,
            )


__all__ = [
    "InMemorySuspendedExecutionOperationStore",
    "SuspendedOperationStoreInvariantError",
    "StaleClaimError",
]
