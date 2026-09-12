# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Evidence evaluation orchestration — reconciliation output toward resolution readiness."""

from __future__ import annotations

from intergrax.contracts.enterprise_reliability.evidence_evaluation import (
    EvidenceEvaluationOutcome,
    EvidenceEvaluationResult,
    build_evidence_evaluation_context,
    evaluate_evidence_collection,
)
from intergrax.contracts.enterprise_reliability.lifecycle import UncertaintyStateRecord
from intergrax.contracts.enterprise_reliability.plugin_spi import (
    EvidenceEvaluationRequest,
    EvidenceEvaluatorStrategy,
)
from intergrax.contracts.enterprise_reliability.reconciliation_evidence import (
    ExternalEffectEvidence,
)


class EvidenceEvaluationOrchestrationError(ValueError):
    """Evidence evaluation could not complete for the current episode."""


def evaluate_external_effect_evidence(
    *,
    state: UncertaintyStateRecord,
    evidence: ExternalEffectEvidence,
    tenant_id: str,
    contract_id: str,
    additional_evidence: tuple[ExternalEffectEvidence, ...] = (),
    evaluator_strategy: EvidenceEvaluatorStrategy | None = None,
) -> EvidenceEvaluationResult:
    """
    Evaluate reconciliation evidence before resolution planning.

    Optional ``evaluator_strategy`` extends platform rules; abstention preserves
    platform evaluation. Evaluator failures surface as ``EVALUATION_FAILED``.
    """
    context = build_evidence_evaluation_context(
        tenant_id=tenant_id,
        correlation_id=state.correlation_id,
        contract_id=contract_id,
        evidence_items=(evidence, *additional_evidence),
        collected_at=evidence.obtained_at,
    )
    platform_result = evaluate_evidence_collection(state=state, context=context)
    if evaluator_strategy is None:
        return platform_result

    request = EvidenceEvaluationRequest(
        state=state,
        context=context,
        platform_result=platform_result,
    )
    try:
        strategy_advice = evaluator_strategy.evaluate(request)
    except Exception:
        return EvidenceEvaluationResult(
            outcome=EvidenceEvaluationOutcome.EVALUATION_FAILED,
            rationale="evaluator_unavailable",
            primary_evidence_ref=evidence.evidence_ref,
        )
    if strategy_advice is None:
        return platform_result
    return EvidenceEvaluationResult(
        outcome=strategy_advice.outcome,
        rationale=strategy_advice.rationale or platform_result.rationale,
        primary_evidence_ref=platform_result.primary_evidence_ref,
    )


__all__ = [
    "EvidenceEvaluationOrchestrationError",
    "evaluate_external_effect_evidence",
]
