# © Artur Czarnecki. All rights reserved.

"""Process-local typed transport slot for suspended-work resume authority."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from intergrax.contracts.execution.suspended_operation.claim_authority import (
    SuspendedOperationClaimAuthority,
)
from intergrax.contracts.execution.suspended_operation.resume_authority_context import (
    ExecutionSuspendedWorkResumeAuthorityContext,
)


class ExecutionSuspendedWorkResumeAuthorityTransportConflictError(RuntimeError):
    """Deliver rejected: slot bound to a different continuation."""


class ExecutionSuspendedWorkResumeAuthorityTransportAuthorityConflictError(
    RuntimeError,
):
    """Deliver rejected: slot already holds different authority for continuation."""


class ExecutionSuspendedWorkResumeAuthorityTransportReplacementOutcome(StrEnum):
    APPLIED = "applied"
    IDEMPOTENT = "idempotent"
    NOT_FOUND = "not_found"
    CONTINUATION_MISMATCH = "continuation_mismatch"
    STALE_AUTHORITY = "stale_authority"


@dataclass
class ExecutionSuspendedWorkResumeAuthorityTransport:
    """Explicit DI slot — not a store, not durable."""

    _pending: ExecutionSuspendedWorkResumeAuthorityContext | None = field(
        default=None,
        repr=False,
    )

    def deliver(self, context: ExecutionSuspendedWorkResumeAuthorityContext) -> None:
        pending = self._pending
        if pending is None:
            self._pending = context
            return
        if pending.continuation_id != context.continuation_id:
            raise ExecutionSuspendedWorkResumeAuthorityTransportConflictError(
                "transport slot already bound to a different continuation",
            )
        if pending.claim_authority == context.claim_authority:
            return
        raise ExecutionSuspendedWorkResumeAuthorityTransportAuthorityConflictError(
            "transport slot already holds different authority; use explicit replacement",
        )

    def replace_for_continuation(
        self,
        *,
        continuation_id: str,
        expected_authority: SuspendedOperationClaimAuthority,
        replacement: ExecutionSuspendedWorkResumeAuthorityContext,
    ) -> ExecutionSuspendedWorkResumeAuthorityTransportReplacementOutcome:
        if replacement.continuation_id != continuation_id:
            return ExecutionSuspendedWorkResumeAuthorityTransportReplacementOutcome.CONTINUATION_MISMATCH
        pending = self._pending
        if pending is None:
            return ExecutionSuspendedWorkResumeAuthorityTransportReplacementOutcome.NOT_FOUND
        if pending.continuation_id != continuation_id:
            return ExecutionSuspendedWorkResumeAuthorityTransportReplacementOutcome.CONTINUATION_MISMATCH
        if pending.claim_authority == replacement.claim_authority:
            return ExecutionSuspendedWorkResumeAuthorityTransportReplacementOutcome.IDEMPOTENT
        if pending.claim_authority != expected_authority:
            return ExecutionSuspendedWorkResumeAuthorityTransportReplacementOutcome.STALE_AUTHORITY
        self._pending = replacement
        return ExecutionSuspendedWorkResumeAuthorityTransportReplacementOutcome.APPLIED

    def peek(self) -> ExecutionSuspendedWorkResumeAuthorityContext | None:
        return self._pending

    def discard_for_continuation(self, continuation_id: str) -> bool:
        pending = self._pending
        if pending is None:
            return False
        if pending.continuation_id != continuation_id:
            return False
        self._pending = None
        return True

    def take_for_continuation(
        self,
        continuation_id: str,
    ) -> SuspendedOperationClaimAuthority | None:
        pending = self._pending
        if pending is None:
            return None
        if pending.continuation_id != continuation_id:
            return None
        self._pending = None
        return pending.claim_authority


@dataclass
class ProductionSuspendedWorkAuthorityTelemetry:
    claim_successes: int = 0
    reclaim_successes: int = 0
    authority_snapshots_created: int = 0
    authority_snapshots_transported: int = 0
    authority_refreshes_from_store: int = 0
    production_resume_attempts: int = 0
    toolruntime_calls: int = 0
    terminal_writes: int = 0


__all__ = [
    "ExecutionSuspendedWorkResumeAuthorityTransport",
    "ExecutionSuspendedWorkResumeAuthorityTransportAuthorityConflictError",
    "ExecutionSuspendedWorkResumeAuthorityTransportConflictError",
    "ExecutionSuspendedWorkResumeAuthorityTransportReplacementOutcome",
    "ProductionSuspendedWorkAuthorityTelemetry",
]
