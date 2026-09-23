# © Artur Czarnecki. All rights reserved.

"""Suspended execution operation store contract (UCA-6C-R6)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from intergrax.contracts.execution_continuation import PendingExecutionContinuation
from intergrax.contracts.execution.suspended_operation.claim import (
    SuspendedOperationAbandonReason,
    SuspendedOperationClaimResult,
    SuspendedOperationMutationResult,
)
from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedExecutionOperationDescriptor,
)
from intergrax.contracts.execution.suspended_operation.authority_scope import (
    SuspendedOperationAuthorityScope,
)
from intergrax.contracts.agent_governance_hitl import LogicalInvocationFingerprint
from intergrax.contracts.governed_continuation_correlation import (
    GovernedContinuationCorrelation,
)


class SuspendedExecutionOperationStore(ABC):
    """Pluginable durable work store — no SQL/ORM/vendor runtime state in contract."""

    @property
    @abstractmethod
    def is_durable(self) -> bool:
        """Whether descriptors survive process restart."""

    @abstractmethod
    def prepare(
        self,
        descriptor: SuspendedExecutionOperationDescriptor,
    ) -> SuspendedOperationMutationResult:
        """Insert descriptor in PREPARED state."""

    @abstractmethod
    def block(
        self,
        *,
        suspended_operation_id: str,
        expected_materialization_revision: int,
        continuation: PendingExecutionContinuation,
        governed_correlation: GovernedContinuationCorrelation,
    ) -> SuspendedOperationMutationResult:
        """CAS PREPARED → BLOCKED with continuation + HITL correlation validation."""

    @abstractmethod
    def load(
        self,
        suspended_operation_id: str,
    ) -> SuspendedExecutionOperationDescriptor | None:
        """Load descriptor by durable entity key."""

    @abstractmethod
    def load_active_for_continuation(
        self,
        continuation_id: str,
    ) -> SuspendedExecutionOperationDescriptor | None:
        """Return exactly 0 or 1 active BLOCKED/CLAIMED descriptor; fail if >1."""

    @abstractmethod
    def load_active_for_logical_invocation(
        self,
        logical_invocation_fingerprint: LogicalInvocationFingerprint,
    ) -> SuspendedExecutionOperationDescriptor | None:
        """Return exactly 0 or 1 active descriptor for the logical invocation fingerprint."""

    @abstractmethod
    def claim(
        self,
        *,
        suspended_operation_id: str,
        expected_materialization_revision: int,
        owner_id: str,
        lease_expires_at: datetime,
    ) -> SuspendedOperationClaimResult:
        """CAS BLOCKED → CLAIMED with new fence and lease."""

    @abstractmethod
    def reclaim(
        self,
        *,
        suspended_operation_id: str,
        expected_materialization_revision: int,
        owner_id: str,
        lease_expires_at: datetime,
        expected_fence: int | None = None,
    ) -> SuspendedOperationMutationResult:
        """Reclaim expired CLAIMED lease; store mints monotonic fence for new owner."""

    @abstractmethod
    def mark_consumed(
        self,
        *,
        suspended_operation_id: str,
        expected_materialization_revision: int,
        owner_id: str,
        fence: int,
        expected_pause_generation: int | None = None,
    ) -> SuspendedOperationMutationResult:
        """CAS CLAIMED → CONSUMED for terminal successful re-entry."""

    @abstractmethod
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
        """Terminal ABANDONED with typed reason."""

    @abstractmethod
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
        """CAS CLAIMED → BLOCKED for the next authority pause (same logical invocation)."""


__all__ = ["SuspendedExecutionOperationStore"]
