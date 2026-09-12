"""Scenario payment evidence evaluator — ``EvidenceEvaluatorStrategy`` implementation."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.enterprise_reliability.evidence_evaluation import (
    EvidenceEvaluationOutcome,
)
from intergrax.contracts.enterprise_reliability.plugin_spi import (
    EvidenceEvaluationRequest,
    EvidenceEvaluatorAdvice,
)

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.constants import (
    SCENARIO_EVIDENCE_REF_PREFIX,
    SCENARIO_PAYMENT_EVIDENCE_EVALUATOR_ID,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.failures import (
    PaymentReconciliationEvidenceLookupError,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.payment_reconciliation_evidence import (
    PaymentReconciliationEvidenceLookupPort,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.mapping.payment_evidence_evaluation import (
    advise_payment_evidence_outcome,
)


def _applies_to_context(request: EvidenceEvaluationRequest) -> bool:
    items = request.context.evidence_items
    if not items:
        return False
    return any(
        item.evidence_ref.startswith(SCENARIO_EVIDENCE_REF_PREFIX) for item in items
    )


@dataclass(frozen=True, slots=True)
class PaymentEvidenceEvaluatorPlugin:
    """
    Replaceable payment evidence quality extension behind platform SPI.

    No registry hook — composition passes this instance to
    ``evaluate_external_effect_evidence(..., evaluator_strategy=...)``.
    """

    _lookup: PaymentReconciliationEvidenceLookupPort

    @property
    def plugin_id(self) -> str:
        return SCENARIO_PAYMENT_EVIDENCE_EVALUATOR_ID

    def evaluate(
        self,
        request: EvidenceEvaluationRequest,
    ) -> EvidenceEvaluatorAdvice | None:
        if not _applies_to_context(request):
            return None

        try:
            payment = self._lookup.lookup_by_correlation_id(request.context.correlation_id)
        except PaymentReconciliationEvidenceLookupError:
            return EvidenceEvaluatorAdvice(
                outcome=EvidenceEvaluationOutcome.INSUFFICIENT_EVIDENCE,
                rationale="payment_evidence_lookup_failed",
            )

        outcome, rationale = advise_payment_evidence_outcome(
            context=request.context,
            payment=payment,
        )
        return EvidenceEvaluatorAdvice(outcome=outcome, rationale=rationale)
