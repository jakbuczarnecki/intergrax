# © Artur Czarnecki. All rights reserved.

"""Typed claim / mutation outcomes for suspended operation store (UCA-6C-R6)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedExecutionOperationDescriptor,
)


class SuspendedOperationClaimOutcome(StrEnum):
    CLAIMED = "claimed"
    STALE_REVISION = "stale_revision"
    ALREADY_CLAIMED = "already_claimed"
    STALE_CLAIM = "stale_claim"
    TERMINAL = "terminal"
    NOT_FOUND = "not_found"
    INVALID_STATE = "invalid_state"
    CONTINUATION_MISMATCH = "continuation_mismatch"
    IDENTITY_MISMATCH = "identity_mismatch"


class SuspendedOperationMutationOutcome(StrEnum):
    APPLIED = "applied"
    ALREADY_ACTIVE = "already_active"
    STALE_REVISION = "stale_revision"
    INVALID_STATE = "invalid_state"
    TERMINAL = "terminal"
    NOT_FOUND = "not_found"
    CONTINUATION_MISMATCH = "continuation_mismatch"
    IDENTITY_MISMATCH = "identity_mismatch"
    STALE_CLAIM = "stale_claim"


class SuspendedOperationAbandonReason(StrEnum):
    ORPHAN_PREPARED = "orphan_prepared"
    CONTINUATION_TERMINAL = "continuation_terminal"
    CORRELATION_MISMATCH = "correlation_mismatch"
    GOVERNANCE_DENIED = "governance_denied"
    CORRUPT_PAYLOAD = "corrupt_payload"
    HUMAN_DENIED = "human_denied"
    HUMAN_ESCALATED = "human_escalated"
    CANCELLED = "cancelled"


class SuspendedOperationClaimResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: SuspendedOperationClaimOutcome
    descriptor: SuspendedExecutionOperationDescriptor | None = None


class SuspendedOperationMutationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: SuspendedOperationMutationOutcome
    descriptor: SuspendedExecutionOperationDescriptor | None = None


__all__ = [
    "SuspendedOperationAbandonReason",
    "SuspendedOperationClaimOutcome",
    "SuspendedOperationClaimResult",
    "SuspendedOperationMutationOutcome",
    "SuspendedOperationMutationResult",
]
