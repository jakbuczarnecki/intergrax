# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bridge contract between ERL recovery/governance outcomes and execution lifecycle port."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.enterprise_reliability.recovery_decision import RecoveryLifecycleAction


class RecoveryLifecycleDecisionContextRefs(BaseModel):
    """Opaque references to upstream ERL decisions — not duplicated decision payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_ref: str = Field(min_length=1, max_length=512)
    resolution_context_ref: str = Field(min_length=1, max_length=512)
    recovery_context_ref: str = Field(min_length=1, max_length=512)


class RecoveryLifecycleHandoffRequest(BaseModel):
    """
    ERL recommends this lifecycle action for the execution subsystem.

    Bridge only — recovery and governance decisions remain authoritative elsewhere.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str = Field(min_length=1, max_length=256)
    correlation_id: str = Field(min_length=1, max_length=256)
    contract_id: str = Field(min_length=1, max_length=256)
    execution_ref: str = Field(min_length=1, max_length=512)
    lifecycle_action: RecoveryLifecycleAction
    decision_context: RecoveryLifecycleDecisionContextRefs
    governance_result_ref: str | None = None


class LifecycleHandoffDisposition(StrEnum):
    """Outcome of attempting to hand lifecycle intent to execution — not execution state."""

    HANDED_OFF = "handed_off"
    BLOCKED = "blocked"
    APPROVAL_REQUIRED = "approval_required"
    ESCALATED = "escalated"
    PORT_UNAVAILABLE = "port_unavailable"


class RecoveryLifecycleHandoffResult(BaseModel):
    """Immutable handoff outcome for observability and downstream correlation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    disposition: LifecycleHandoffDisposition
    request: RecoveryLifecycleHandoffRequest | None = None
    rationale: str = ""


def recovery_lifecycle_decision_context_refs(
    *,
    evidence_ref: str,
    correlation_id: str,
    contract_id: str,
) -> RecoveryLifecycleDecisionContextRefs:
    """Stable platform refs for resolution and recovery context (no decision duplication)."""
    return RecoveryLifecycleDecisionContextRefs(
        evidence_ref=evidence_ref,
        resolution_context_ref=f"erl:resolution:{correlation_id}:{contract_id}",
        recovery_context_ref=f"erl:recovery:{correlation_id}:{contract_id}",
    )


def governance_result_ref(
    *,
    correlation_id: str,
    contract_id: str,
    disposition_token: str,
) -> str:
    """Reference to the governance evaluation outcome applied at handoff time."""
    return f"erl:governance:{disposition_token}:{correlation_id}:{contract_id}"


__all__ = [
    "LifecycleHandoffDisposition",
    "RecoveryLifecycleDecisionContextRefs",
    "RecoveryLifecycleHandoffRequest",
    "RecoveryLifecycleHandoffResult",
    "governance_result_ref",
    "recovery_lifecycle_decision_context_refs",
]
