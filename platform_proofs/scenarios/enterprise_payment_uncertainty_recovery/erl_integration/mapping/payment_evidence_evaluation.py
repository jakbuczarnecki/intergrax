"""Payment-domain evidence rules — outputs only platform ``EvidenceEvaluationOutcome``."""

from __future__ import annotations

from intergrax.contracts.enterprise_reliability.evidence import ExternalEffectEvidenceVerdict
from intergrax.contracts.enterprise_reliability.evidence_evaluation import (
    EvidenceEvaluationContext,
    EvidenceEvaluationOutcome,
)
from intergrax.contracts.enterprise_reliability.reconciliation_evidence import (
    ExternalEffectEvidence,
    ExternalEffectEvidenceConfidence,
)

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.payment_reconciliation_evidence import (
    PaymentReconciliationEvidence,
)

_RECON_UNAVAILABLE = "unavailable"
_TIER_UNAVAILABLE = "UNAVAILABLE"
_SETTLEMENT_SETTLED = "SETTLED"
_SETTLEMENT_NOT_SETTLED = "NOT_SETTLED"


def _definitive_platform_verdict(
    item: ExternalEffectEvidence,
) -> ExternalEffectEvidenceVerdict | None:
    if item.confidence is not ExternalEffectEvidenceConfidence.DEFINITIVE:
        return None
    if item.verdict is ExternalEffectEvidenceVerdict.INSUFFICIENT:
        return None
    return item.verdict


def _payment_bundle_incomplete(payment: PaymentReconciliationEvidence) -> bool:
    if not payment.external_effect_reference.strip():
        return True
    if payment.reconciliation_availability == _RECON_UNAVAILABLE:
        return True
    if payment.source_reliability_tier == _TIER_UNAVAILABLE:
        return True
    if payment.reconciliation_availability != _RECON_UNAVAILABLE:
        if not payment.psp_confirmation_id or not payment.psp_confirmation_id.strip():
            return True
    return False


def _probe_payment_conflict(
    verdict: ExternalEffectEvidenceVerdict,
    payment: PaymentReconciliationEvidence,
) -> bool:
    if verdict is ExternalEffectEvidenceVerdict.DEFINITIVE_SUCCESS:
        if not payment.funds_captured:
            return True
        if payment.settlement_status != _SETTLEMENT_SETTLED:
            return True
    if verdict is ExternalEffectEvidenceVerdict.DEFINITIVE_FAILURE:
        if payment.funds_captured:
            return True
        if payment.settlement_status == _SETTLEMENT_SETTLED:
            return True
    return False


def advise_payment_evidence_outcome(
    *,
    context: EvidenceEvaluationContext,
    payment: PaymentReconciliationEvidence,
) -> tuple[EvidenceEvaluationOutcome, str]:
    """
    Evaluate payment reconciliation attributes against materialized probe evidence.

    Decisions are driven by evidence content (PSP id, settlement, SoR consistency),
    not by scenario variant identifiers.
    """
    if payment.correlation_id != context.correlation_id:
        return (
            EvidenceEvaluationOutcome.EVALUATION_FAILED,
            "payment_correlation_mismatch",
        )

    if _payment_bundle_incomplete(payment):
        return (
            EvidenceEvaluationOutcome.INSUFFICIENT_EVIDENCE,
            "payment_reconciliation_bundle_incomplete",
        )

    definitive_verdicts: set[ExternalEffectEvidenceVerdict] = set()
    for item in context.evidence_items:
        verdict = _definitive_platform_verdict(item)
        if verdict is None:
            continue
        if _probe_payment_conflict(verdict, payment):
            return (
                EvidenceEvaluationOutcome.CONFLICTING_EVIDENCE,
                "payment_probe_settlement_mismatch",
            )
        definitive_verdicts.add(verdict)

    if len(definitive_verdicts) > 1:
        return (
            EvidenceEvaluationOutcome.CONFLICTING_EVIDENCE,
            "conflicting_definitive_probe_verdicts",
        )

    if not definitive_verdicts:
        return (
            EvidenceEvaluationOutcome.INSUFFICIENT_EVIDENCE,
            "no_definitive_probe_with_payment_bundle",
        )

    return (
        EvidenceEvaluationOutcome.READY_FOR_DECISION,
        "payment_evidence_consistent_with_probe",
    )
