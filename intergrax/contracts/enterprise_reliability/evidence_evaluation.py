# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Evidence evaluation — platform quality gate before resolution planning (ERL)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.enterprise_reliability.evidence import ExternalEffectEvidenceVerdict
from intergrax.contracts.enterprise_reliability.lifecycle import UncertaintyStateRecord
from intergrax.contracts.enterprise_reliability.outcome import ExternalEffectOutcome
from intergrax.contracts.enterprise_reliability.reconciliation_evidence import (
    ExternalEffectEvidence,
    ExternalEffectEvidenceConfidence,
)

SCHEMA_EVIDENCE_EVALUATION_CONTEXT_V1: Final = "evidence_evaluation_context.v1"
SCHEMA_EVIDENCE_EVALUATION_RESULT_V1: Final = "evidence_evaluation_result.v1"


class EvidenceEvaluationOutcome(StrEnum):
    """Whether collected evidence may proceed to resolution planning — not a business verdict."""

    READY_FOR_DECISION = "ready_for_decision"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    EVALUATION_FAILED = "evaluation_failed"


class EvidenceEvaluationContext(BaseModel):
    """
    Domain-neutral evidence bundle for evaluation.

    Integrations attach provider payloads behind ``evidence_ref``; core evaluates
    completeness, provenance, consistency, availability, and confidence only.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_EVIDENCE_EVALUATION_CONTEXT_V1
    tenant_id: str = Field(min_length=1, max_length=256)
    correlation_id: str = Field(min_length=1, max_length=256)
    contract_id: str = Field(min_length=1, max_length=256)
    evidence_items: tuple[ExternalEffectEvidence, ...] = ()
    collected_at: datetime | None = None


class EvidenceEvaluationResult(BaseModel):
    """Outcome of platform evidence evaluation — does not authorize business acceptance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_EVIDENCE_EVALUATION_RESULT_V1
    outcome: EvidenceEvaluationOutcome
    rationale: str = Field(default="", max_length=512)
    primary_evidence_ref: str | None = Field(default=None, max_length=512)


class EvidenceEvaluationError(ValueError):
    """Evidence evaluation inputs violate platform invariants."""


def build_evidence_evaluation_context(
    *,
    tenant_id: str,
    correlation_id: str,
    contract_id: str,
    evidence_items: tuple[ExternalEffectEvidence, ...],
    collected_at: datetime | None = None,
) -> EvidenceEvaluationContext:
    """Construct evaluation context from reconciliation or probe materialization output."""
    return EvidenceEvaluationContext(
        tenant_id=tenant_id,
        correlation_id=correlation_id,
        contract_id=contract_id,
        evidence_items=evidence_items,
        collected_at=collected_at,
    )


def _definitive_verdict(
    evidence: ExternalEffectEvidence,
) -> ExternalEffectEvidenceVerdict | None:
    if evidence.confidence is not ExternalEffectEvidenceConfidence.DEFINITIVE:
        return None
    if evidence.verdict is ExternalEffectEvidenceVerdict.INSUFFICIENT:
        return None
    return evidence.verdict


def evaluate_evidence_collection(
    *,
    state: UncertaintyStateRecord,
    context: EvidenceEvaluationContext,
) -> EvidenceEvaluationResult:
    """
    Evaluate evidence quality for resolution readiness.

    Does not decide payment or business outcomes — only whether the platform may
    invoke resolution planning with the supplied evidence bundle.
    """
    if state.effect_outcome is not ExternalEffectOutcome.UNKNOWN:
        return EvidenceEvaluationResult(
            outcome=EvidenceEvaluationOutcome.EVALUATION_FAILED,
            rationale="evaluation_requires_unknown_effect_outcome",
        )
    if context.correlation_id != state.correlation_id:
        return EvidenceEvaluationResult(
            outcome=EvidenceEvaluationOutcome.EVALUATION_FAILED,
            rationale="correlation_reference_mismatch",
        )
    items = context.evidence_items
    if not items:
        return EvidenceEvaluationResult(
            outcome=EvidenceEvaluationOutcome.INSUFFICIENT_EVIDENCE,
            rationale="missing_evidence",
        )

    for item in items:
        link = item.operation_link
        if link.tenant_id != context.tenant_id:
            return EvidenceEvaluationResult(
                outcome=EvidenceEvaluationOutcome.EVALUATION_FAILED,
                rationale="invalid_provenance_tenant",
            )
        if link.correlation_id != context.correlation_id:
            return EvidenceEvaluationResult(
                outcome=EvidenceEvaluationOutcome.EVALUATION_FAILED,
                rationale="invalid_provenance_correlation",
            )
        if link.contract_id != context.contract_id:
            return EvidenceEvaluationResult(
                outcome=EvidenceEvaluationOutcome.EVALUATION_FAILED,
                rationale="invalid_provenance_contract",
            )

    definitive_verdicts: set[ExternalEffectEvidenceVerdict] = set()
    for item in items:
        verdict = _definitive_verdict(item)
        if verdict is not None:
            definitive_verdicts.add(verdict)

    if len(definitive_verdicts) > 1:
        return EvidenceEvaluationResult(
            outcome=EvidenceEvaluationOutcome.CONFLICTING_EVIDENCE,
            rationale="conflicting_definitive_verdicts",
            primary_evidence_ref=items[0].evidence_ref,
        )

    if not definitive_verdicts:
        return EvidenceEvaluationResult(
            outcome=EvidenceEvaluationOutcome.INSUFFICIENT_EVIDENCE,
            rationale="insufficient_evidence_for_decision",
            primary_evidence_ref=items[0].evidence_ref,
        )

    return EvidenceEvaluationResult(
        outcome=EvidenceEvaluationOutcome.READY_FOR_DECISION,
        rationale="evidence_sufficient_for_resolution_planning",
        primary_evidence_ref=items[0].evidence_ref,
    )


__all__ = [
    "EvidenceEvaluationContext",
    "EvidenceEvaluationError",
    "EvidenceEvaluationOutcome",
    "EvidenceEvaluationResult",
    "SCHEMA_EVIDENCE_EVALUATION_CONTEXT_V1",
    "SCHEMA_EVIDENCE_EVALUATION_RESULT_V1",
    "build_evidence_evaluation_context",
    "evaluate_evidence_collection",
]
