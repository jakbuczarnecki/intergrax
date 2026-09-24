# © Artur Czarnecki. All rights reserved.

"""Process-local typed transport slot for suspended-work resume authority."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.contracts.execution.suspended_operation.claim_authority import (
    SuspendedOperationClaimAuthority,
)
from intergrax.contracts.execution.suspended_operation.resume_authority_context import (
    ExecutionSuspendedWorkResumeAuthorityContext,
)


@dataclass
class ExecutionSuspendedWorkResumeAuthorityTransport:
    """Explicit DI slot — not a store, not durable."""

    _pending: ExecutionSuspendedWorkResumeAuthorityContext | None = field(
        default=None,
        repr=False,
    )

    def deliver(self, context: ExecutionSuspendedWorkResumeAuthorityContext) -> None:
        self._pending = context

    def peek(self) -> ExecutionSuspendedWorkResumeAuthorityContext | None:
        return self._pending

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
    "ProductionSuspendedWorkAuthorityTelemetry",
]
