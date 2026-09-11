"""Two-stage verification and terminal decision services."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.verification.contracts import (
    HypothesisVerificationCollection,
    IdentityHypothesisVerification,
    ProductIdentificationDecision,
    ProductIdentificationVerificationOutcome,
    ProductIdentificationVerificationRequest,
)
from platform_proofs.scenarios.verified_product_identification.application.verification.decision_policy import (
    ProductIdentificationDecisionPolicy,
)
from platform_proofs.scenarios.verified_product_identification.application.verification.verification_policy import (
    IdentityVerificationPolicy,
)


@dataclass(frozen=True, slots=True)
class IdentityVerificationService:
    verification_policy: IdentityVerificationPolicy

    def verify(
        self,
        request: ProductIdentificationVerificationRequest,
    ) -> HypothesisVerificationCollection:
        rows: list[IdentityHypothesisVerification] = []
        for evaluated in request.ranked_hypotheses.hypotheses:
            rows.append(
                self.verification_policy.verify_hypothesis(
                    evaluated,
                    request.query_context,
                )
            )
        return HypothesisVerificationCollection(rows=tuple(rows))


@dataclass(frozen=True, slots=True)
class ProductIdentificationDecisionService:
    decision_policy: ProductIdentificationDecisionPolicy

    def decide(
        self,
        request: ProductIdentificationVerificationRequest,
        verification: HypothesisVerificationCollection,
    ) -> ProductIdentificationDecision:
        evaluated = request.ranked_hypotheses.hypotheses
        return self.decision_policy.decide(
            evaluated_hypotheses=evaluated,
            verification_rows=verification.rows,
            query_context=request.query_context,
            empty_input_rejection_evidence=request.empty_input_rejection_evidence,
        )


@dataclass(frozen=True, slots=True)
class ProductIdentificationVerificationService:
    identity_verification_service: IdentityVerificationService
    decision_service: ProductIdentificationDecisionService

    def run(
        self,
        request: ProductIdentificationVerificationRequest,
    ) -> ProductIdentificationVerificationOutcome:
        verification = self.identity_verification_service.verify(request)
        decision = self.decision_service.decide(request, verification)
        return ProductIdentificationVerificationOutcome(decision=decision)
