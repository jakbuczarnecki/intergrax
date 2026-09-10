# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical partial fan-out / child slot recovery contracts (NPSC-5E/R3)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.execution_identity import AttemptId, ExecutionId
from intergrax.contracts.execution_retry import ExecutionFailureKind
from intergrax.contracts.orchestration_topology import (
    OrchestrationSlotId,
    OrchestrationTopologyExecutionId,
)


class SlotRecoveryDisposition(StrEnum):
    """Durable per-slot recovery disposition."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    WAITING_FOR_HUMAN = "waiting_for_human"
    CANCELLED = "cancelled"
    RECOVERING = "recovering"
    UNKNOWN_UNSAFE = "unknown_unsafe"


class SlotRecoveryPolicyAction(StrEnum):
    """Narrow recovery policy outcome for one failed slot."""

    RECOVER = "recover"
    PRESERVE_FAILURE = "preserve_failure"
    WAIT = "wait"
    CANCEL = "cancel"


class PartialRecoveryReason(StrEnum):
    """Typed reason for initiating partial topology recovery."""

    TRANSIENT_SLOT_FAILURE = "transient_slot_failure"
    INTERRUPTED_SLOT = "interrupted_slot"
    ALREADY_COMPLETE = "already_complete"
    DUPLICATE_REQUEST = "duplicate_request"


class PartialRecoveryErrorCode(StrEnum):
    """Fail-closed partial recovery validation codes."""

    WRONG_TOPOLOGY = "wrong_topology"
    WRONG_SLOT = "wrong_slot"
    WRONG_ROOT = "wrong_root"
    WRONG_REVISION = "wrong_revision"
    STALE_CHECKPOINT = "stale_checkpoint"
    TERMINAL_SLOT = "terminal_slot"
    SLOT_NOT_RECOVERABLE = "slot_not_recoverable"
    AUTHORITY_REVOKED = "authority_revoked"
    GOVERNANCE_DENIED = "governance_denied"
    TRUST_DENIED = "trust_denied"
    PARENT_CANCELLED = "parent_cancelled"
    SLOT_CANCELLED = "slot_cancelled"
    UNKNOWN_UNSAFE = "unknown_unsafe"
    PERMANENT_FAILURE = "permanent_failure"
    REQUIRE_HUMAN = "require_human"
    SEALED_PARENT = "sealed_parent"
    TENANT_MISMATCH = "tenant_mismatch"


class PartialRecoveryError(ValueError):
    """Fail-closed partial recovery boundary error."""

    __slots__ = ("code",)

    def __init__(self, message: str, *, code: PartialRecoveryErrorCode) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class PartialRecoveryRequest:
    """Identity-bound intent to recover one exact failed topology slot."""

    root_execution_id: ExecutionId
    topology_execution_id: OrchestrationTopologyExecutionId
    slot_id: OrchestrationSlotId
    source_checkpoint_revision: int
    source_attempt_id: AttemptId
    recovery_reason: PartialRecoveryReason

    def __post_init__(self) -> None:
        if self.source_checkpoint_revision < 1:
            raise ValueError("source_checkpoint_revision must be >= 1")


class SlotRecoveryPolicyRequest(BaseModel):
    """Inputs for narrow slot recovery policy evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    disposition: SlotRecoveryDisposition
    failure_kind: ExecutionFailureKind | None = None
    has_unknown_side_effect: bool = False
    cancelled: bool = False
    waiting_for_human: bool = False
    parent_cancelled: bool = False


class SlotRecoveryPolicyResult(BaseModel):
    """Typed slot recovery policy outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: SlotRecoveryPolicyAction
    reason: str = Field(default="", max_length=512)


def evaluate_slot_recovery_policy(
    request: SlotRecoveryPolicyRequest,
) -> SlotRecoveryPolicyResult:
    """Decide whether one slot may be recovered without scheduling or attempt mint."""
    if request.parent_cancelled:
        return SlotRecoveryPolicyResult(
            action=SlotRecoveryPolicyAction.CANCEL,
            reason="parent execution cancelled",
        )
    if request.cancelled:
        return SlotRecoveryPolicyResult(
            action=SlotRecoveryPolicyAction.CANCEL,
            reason="slot cancelled",
        )
    if request.waiting_for_human:
        return SlotRecoveryPolicyResult(
            action=SlotRecoveryPolicyAction.WAIT,
            reason="governed human continuation required",
        )
    if request.disposition is SlotRecoveryDisposition.SUCCEEDED:
        return SlotRecoveryPolicyResult(
            action=SlotRecoveryPolicyAction.PRESERVE_FAILURE,
            reason="successful slot is terminal",
        )
    if request.disposition is SlotRecoveryDisposition.UNKNOWN_UNSAFE:
        return SlotRecoveryPolicyResult(
            action=SlotRecoveryPolicyAction.PRESERVE_FAILURE,
            reason="unknown side-effect slot cannot be blindly replayed",
        )
    if request.has_unknown_side_effect:
        return SlotRecoveryPolicyResult(
            action=SlotRecoveryPolicyAction.PRESERVE_FAILURE,
            reason="unknown side-effect",
        )
    if request.failure_kind in {
        ExecutionFailureKind.NON_RETRYABLE_PERMANENT,
        ExecutionFailureKind.GOVERNANCE_DENIED,
        ExecutionFailureKind.AUTHORITY_DENIED,
        ExecutionFailureKind.TRUST_DENIED,
        ExecutionFailureKind.TERMINAL_DENY,
        ExecutionFailureKind.CONTRACT_ERROR,
        ExecutionFailureKind.BUDGET_EXHAUSTED,
        ExecutionFailureKind.DEADLINE_EXCEEDED,
    }:
        return SlotRecoveryPolicyResult(
            action=SlotRecoveryPolicyAction.PRESERVE_FAILURE,
            reason=f"non-recoverable failure kind: {request.failure_kind}",
        )
    if request.disposition in {
        SlotRecoveryDisposition.FAILED,
        SlotRecoveryDisposition.INTERRUPTED,
        SlotRecoveryDisposition.RECOVERING,
    }:
        return SlotRecoveryPolicyResult(
            action=SlotRecoveryPolicyAction.RECOVER,
            reason="failed slot eligible for exact recovery",
        )
    return SlotRecoveryPolicyResult(
        action=SlotRecoveryPolicyAction.PRESERVE_FAILURE,
        reason=f"unsupported disposition: {request.disposition}",
    )


__all__ = [
    "PartialRecoveryError",
    "PartialRecoveryErrorCode",
    "PartialRecoveryReason",
    "PartialRecoveryRequest",
    "SlotRecoveryDisposition",
    "SlotRecoveryPolicyAction",
    "SlotRecoveryPolicyRequest",
    "SlotRecoveryPolicyResult",
    "evaluate_slot_recovery_policy",
]
