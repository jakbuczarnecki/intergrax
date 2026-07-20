# © Artur Czarnecki. All rights reserved.

"""Governed Continuation — composition over existing Nexus interrupt + HITL evidence.

Reusable capability: execution may block for governance, receive continuation
evidence, then resume via the existing Nexus / HITL paths.

This module is **not** a runtime. It does not pause, decide, authorize, poll,
or resume. Callers compose:

```text
surface blocker → ExecutionInterrupt → governance decision → evidence → Nexus resume
```

External Work is the first consumer (``ContinuationReason.QUOTE``). Future
reasons (security, legal, …) reuse the same shape without quote-specific types.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Final, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.execution_interrupt import ExecutionInterrupt, InterruptType
from intergrax.contracts.external_work import QuoteAcceptanceEvidence

SCHEMA_GOVERNED_CONTINUATION_REQUEST_V1: Final = "governed_continuation_request.v1"
SCHEMA_CONTINUATION_EVIDENCE_REFS_V1: Final = "continuation_evidence_refs.v1"

META_CONTINUATION_REASON: Final = "continuation.reason"
META_CONTINUATION_CORRELATION: Final = "continuation.correlation"

_NON_EMPTY = Field(min_length=1)


class ContinuationReason(StrEnum):
    """Why continuation is blocked — independent of commercial/domain logic."""

    QUOTE = "quote"
    SECURITY = "security"
    LEGAL = "legal"
    PROCUREMENT = "procurement"
    COMPLIANCE = "compliance"
    PUBLICATION = "publication"


class ContinuationEvidenceRefs(BaseModel):
    """Reason-agnostic pointers into existing governance artifacts.

    Does not authorize continuation. Domain evidence (e.g. quote acceptance)
    remains in its specialized contract and carries the same refs.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["continuation_evidence_refs.v1"] = (
        SCHEMA_CONTINUATION_EVIDENCE_REFS_V1
    )
    reason: ContinuationReason
    hitl_decision_id: str | None = None
    interrupt_id: str | None = None
    policy_decision_ref: str | None = None

    @field_validator("hitl_decision_id", "interrupt_id", "policy_decision_ref")
    @classmethod
    def _strip_optional_refs(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class GovernedContinuationRequest(BaseModel):
    """Surfaced continuation blocker for composition with Nexus interrupts.

    Tier-2 / callers may construct this. Governance and Nexus own decisions
    and resume; this type never evaluates approval.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["governed_continuation_request.v1"] = (
        SCHEMA_GOVERNED_CONTINUATION_REQUEST_V1
    )
    reason: ContinuationReason
    task_id: str = _NON_EMPTY
    run_id: str = _NON_EMPTY
    source_agent_id: str = _NON_EMPTY
    source_step_id: str | None = None
    prompt: str = Field(min_length=1)
    correlation: Mapping[str, Any] = Field(default_factory=dict)
    context: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("task_id", "run_id", "source_agent_id", "prompt")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("field must be non-empty")
        return normalized

    @field_validator("source_step_id")
    @classmethod
    def _strip_optional_step(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


def compose_continuation_interrupt(
    request: GovernedContinuationRequest,
    *,
    interrupt_id: str | None = None,
) -> ExecutionInterrupt:
    """Map a continuation request onto the existing ``ExecutionInterrupt`` model."""
    metadata: dict[str, Any] = {
        META_CONTINUATION_REASON: request.reason.value,
        META_CONTINUATION_CORRELATION: dict(request.correlation),
        **dict(request.context),
    }
    kwargs: dict[str, Any] = {
        "interrupt_type": InterruptType.HUMAN_JUDGMENT_REQUIRED,
        "source_agent_id": request.source_agent_id,
        "source_step_id": request.source_step_id,
        "task_id": request.task_id,
        "run_id": request.run_id,
        "blocking": True,
        "recommended_action": AgentDecisionType.REQUEST_HUMAN,
        "metadata": metadata,
    }
    if interrupt_id is not None and interrupt_id.strip():
        kwargs["interrupt_id"] = interrupt_id.strip()
    return ExecutionInterrupt(**kwargs)


def compose_continuation_agent_decision(
    request: GovernedContinuationRequest,
    interrupt: ExecutionInterrupt,
) -> AgentDecision:
    """AgentDecision that Nexus interrupt handling already understands."""
    return AgentDecision(
        type=AgentDecisionType.INTERRUPT,
        reason=f"governed_continuation:{request.reason.value}",
        interrupt_id=interrupt.interrupt_id,
        payload={
            "interrupt_type": InterruptType.HUMAN_JUDGMENT_REQUIRED.value,
            "blocking": True,
            "recommended_action": AgentDecisionType.REQUEST_HUMAN,
            META_CONTINUATION_REASON: request.reason.value,
            META_CONTINUATION_CORRELATION: dict(request.correlation),
            "prompt": request.prompt,
            **dict(request.context),
        },
    )


def continuation_reason_from_interrupt(
    interrupt: ExecutionInterrupt,
) -> ContinuationReason | None:
    """Read generic continuation reason from interrupt metadata, if present."""
    raw = interrupt.metadata.get(META_CONTINUATION_REASON)
    if raw is None:
        return None
    try:
        return ContinuationReason(str(raw).strip().lower())
    except ValueError:
        return None


def continuation_evidence_refs_from_quote_acceptance(
    acceptance: QuoteAcceptanceEvidence,
) -> ContinuationEvidenceRefs:
    """QUOTE specialization: reuse existing acceptance refs without reinterpretation."""
    return ContinuationEvidenceRefs(
        reason=ContinuationReason.QUOTE,
        hitl_decision_id=acceptance.hitl_decision_id,
        interrupt_id=acceptance.interrupt_id,
        policy_decision_ref=acceptance.policy_decision_ref,
    )


def attach_continuation_refs_to_quote_acceptance(
    acceptance: QuoteAcceptanceEvidence,
    *,
    hitl_decision_id: str | None = None,
    interrupt_id: str | None = None,
    policy_decision_ref: str | None = None,
) -> QuoteAcceptanceEvidence:
    """Propagate governance identity into quote acceptance without deciding."""
    updates: dict[str, Any] = {}
    if hitl_decision_id is not None:
        updates["hitl_decision_id"] = hitl_decision_id
    if interrupt_id is not None:
        updates["interrupt_id"] = interrupt_id
    if policy_decision_ref is not None:
        updates["policy_decision_ref"] = policy_decision_ref
    if not updates:
        return acceptance
    return acceptance.model_copy(update=updates)
