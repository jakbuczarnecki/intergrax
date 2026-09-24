# © Artur Czarnecki. All rights reserved.

"""Typed transport for caller-held claim authority into production HITL resume."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution.suspended_operation.claim_authority import (
    SuspendedOperationClaimAuthority,
)


@dataclass(frozen=True, slots=True)
class ExecutionSuspendedWorkResumeAuthorityContext:
    """Ephemeral carrier from EE claim/reclaim owner to production resume caller."""

    continuation_id: str
    claim_authority: SuspendedOperationClaimAuthority


__all__ = ["ExecutionSuspendedWorkResumeAuthorityContext"]
