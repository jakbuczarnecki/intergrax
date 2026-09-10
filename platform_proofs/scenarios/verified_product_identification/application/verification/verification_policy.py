"""Per-hypothesis verification policy seam."""

from __future__ import annotations

from typing import Protocol

from platform_proofs.scenarios.verified_product_identification.application.contracts.identification_context import (
    ProductIdentificationQueryContext,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.contracts import (
    EvaluatedIdentityHypothesis,
)
from platform_proofs.scenarios.verified_product_identification.application.verification.constraint_evaluation import (
    evaluate_negative_constraint,
    evaluate_required_constraint,
)
from platform_proofs.scenarios.verified_product_identification.application.verification.contracts import (
    ConstraintRequirementStatus,
    ContradictedRequirementEvidence,
    HypothesisVerificationState,
    IdentityHypothesisVerification,
    MissingRequirement,
    VerifiedRequirementEvidence,
)
from platform_proofs.scenarios.verified_product_identification.application.verification.identity_sufficiency import (
    identity_evidence_materially_sufficient,
)


class IdentityVerificationPolicy(Protocol):
    def verify_hypothesis(
        self,
        evaluated: EvaluatedIdentityHypothesis,
        query_context: ProductIdentificationQueryContext,
    ) -> IdentityHypothesisVerification:
        """Evaluate one hypothesis against authoritative request context."""


class DeterministicIdentityVerificationPolicy:
    """Rule-based per-hypothesis verification — no models or scores."""

    def verify_hypothesis(
        self,
        evaluated: EvaluatedIdentityHypothesis,
        query_context: ProductIdentificationQueryContext,
    ) -> IdentityHypothesisVerification:
        hypothesis = evaluated.hypothesis
        blocking = evaluated.contradiction_evaluation.internal_blocking

        supported: list[VerifiedRequirementEvidence] = []
        contradicted: list[ContradictedRequirementEvidence] = []
        missing: list[MissingRequirement] = []

        for constraint in query_context.required_constraints:
            status, ok_row, bad_row, miss_row = evaluate_required_constraint(
                hypothesis,
                constraint,
            )
            if status is ConstraintRequirementStatus.SUPPORTED and ok_row is not None:
                supported.append(ok_row)
            elif status is ConstraintRequirementStatus.CONTRADICTED and bad_row is not None:
                contradicted.append(bad_row)
            elif status is ConstraintRequirementStatus.MISSING and miss_row is not None:
                missing.append(miss_row)

        for negative in query_context.negative_constraints:
            status, bad_row = evaluate_negative_constraint(hypothesis, negative)
            if status is ConstraintRequirementStatus.CONTRADICTED and bad_row is not None:
                contradicted.append(bad_row)

        for user_missing in query_context.missing_user_distinguishing_requirements:
            missing.append(
                MissingRequirement(
                    attribute_name=user_missing.attribute_name,
                    origin=user_missing.origin,
                    requirement_id=user_missing.requirement_id,
                )
            )

        identity_ok = identity_evidence_materially_sufficient(
            hypothesis,
            evaluated.evidence_profile,
        )

        if blocking or contradicted:
            state = HypothesisVerificationState.CONTRADICTED
            eligible = False
        elif missing or not identity_ok:
            state = HypothesisVerificationState.INCOMPLETE
            eligible = False
        else:
            state = HypothesisVerificationState.SUPPORTED
            eligible = True

        return IdentityHypothesisVerification(
            hypothesis_id=hypothesis.hypothesis_id,
            eligible_for_verification=eligible,
            verification_state=state,
            supported_requirements=tuple(supported),
            contradicted_requirements=tuple(contradicted),
            missing_requirements=tuple(missing),
            blocking_contradictions=blocking,
            identity_evidence_sufficient=identity_ok,
        )
