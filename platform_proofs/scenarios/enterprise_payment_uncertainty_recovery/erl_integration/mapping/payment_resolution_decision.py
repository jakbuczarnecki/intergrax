"""Payment-domain resolution rules — outputs only platform ``ResolutionDecision``."""

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
from intergrax.contracts.enterprise_reliability.resolution_decision import (
    ResolutionDecision,
    ResolutionPlatformAction,
)

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.payment_reconciliation_evidence import (
    PaymentReconciliationEvidence,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.mapping.payment_evidence_evaluation import (
    advise_payment_evidence_outcome,
)


def decide_payment_resolution(
    *,
    evidence: ExternalEffectEvidence,
    payment: PaymentReconciliationEvidence,
    tenant_id: str,
    contract_id: str,
    correlation_id: str,
) -> ResolutionDecision:
    """
    Map evaluated payment evidence and reconciliation probe verdict to platform resolution.

    Driven by evidence content — never by scenario variant identifiers.
    """
    if evidence.confidence is not ExternalEffectEvidenceConfidence.DEFINITIVE:
        return ResolutionDecision(
            action=ResolutionPlatformAction.ESCALATE,
            rationale="payment_truth_unavailable_inconclusive_probe",
        )
    if evidence.verdict is ExternalEffectEvidenceVerdict.INSUFFICIENT:
        return ResolutionDecision(
            action=ResolutionPlatformAction.ESCALATE,
            rationale="payment_truth_unavailable_insufficient_probe",
        )

    context = EvidenceEvaluationContext(
        tenant_id=tenant_id,
        correlation_id=correlation_id,
        contract_id=contract_id,
        evidence_items=(evidence,),
    )
    outcome, eval_rationale = advise_payment_evidence_outcome(
        context=context,
        payment=payment,
    )
    if outcome is EvidenceEvaluationOutcome.INSUFFICIENT_EVIDENCE:
        return ResolutionDecision(
            action=ResolutionPlatformAction.ESCALATE,
            rationale=f"payment_truth_unavailable:{eval_rationale}",
        )
    if outcome is EvidenceEvaluationOutcome.CONFLICTING_EVIDENCE:
        return ResolutionDecision(
            action=ResolutionPlatformAction.ESCALATE,
            rationale=f"payment_evidence_conflict:{eval_rationale}",
        )
    if outcome is EvidenceEvaluationOutcome.EVALUATION_FAILED:
        return ResolutionDecision(
            action=ResolutionPlatformAction.ESCALATE,
            rationale=eval_rationale,
        )

    if evidence.verdict is ExternalEffectEvidenceVerdict.DEFINITIVE_SUCCESS:
        return ResolutionDecision(
            action=ResolutionPlatformAction.CONTINUE,
            rationale="payment_confirmed_continue_business_process",
        )
    if evidence.verdict is ExternalEffectEvidenceVerdict.DEFINITIVE_FAILURE:
        return ResolutionDecision(
            action=ResolutionPlatformAction.STOP,
            rationale="payment_failed_stop_affected_process",
        )

    return ResolutionDecision(
        action=ResolutionPlatformAction.ESCALATE,
        rationale="payment_resolution_verdict_indeterminate",
    )
