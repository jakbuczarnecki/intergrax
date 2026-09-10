"""Terminal outcome selection policy."""

from __future__ import annotations

from typing import Protocol

from platform_proofs.scenarios.verified_product_identification.application.contracts.identification_context import (
    HypothesisRejectionEvidence,
    MissingRequirementOrigin,
    ProductIdentificationQueryContext,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.contracts import (
    IdentityContradiction,
    IdentityEvidenceStrengthClass,
    IdentityEvidenceType,
    ProductIdentityHypothesis,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.contracts import (
    EvaluatedIdentityHypothesis,
)
from platform_proofs.scenarios.verified_product_identification.application.verification.contracts import (
    ContradictedRequirementEvidence,
    HypothesisVerificationState,
    IdentityHypothesisVerification,
    MissingRequirement,
    ProductIdentificationDecision,
    ProductIdentificationDecisionReasonCode,
    ProductIdentificationOutcome,
)


class ProductIdentificationDecisionPolicy(Protocol):
    def decide(
        self,
        *,
        evaluated_hypotheses: tuple[EvaluatedIdentityHypothesis, ...],
        verification_rows: tuple[IdentityHypothesisVerification, ...],
        query_context: ProductIdentificationQueryContext,
        empty_input_rejection_evidence: tuple[HypothesisRejectionEvidence, ...],
    ) -> ProductIdentificationDecision:
        """Select one terminal business outcome."""


class DeterministicProductIdentificationDecisionPolicy:
    """Deterministic precedence tree — no scores or margins."""

    def decide(
        self,
        *,
        evaluated_hypotheses: tuple[EvaluatedIdentityHypothesis, ...],
        verification_rows: tuple[IdentityHypothesisVerification, ...],
        query_context: ProductIdentificationQueryContext,
        empty_input_rejection_evidence: tuple[HypothesisRejectionEvidence, ...],
    ) -> ProductIdentificationDecision:
        if not evaluated_hypotheses:
            return _decide_empty_input(
                query_context=query_context,
                empty_input_rejection_evidence=empty_input_rejection_evidence,
            )

        viable = tuple(
            row.hypothesis_id
            for row in verification_rows
            if row.eligible_for_verification
        )
        contradicted_all = bool(verification_rows) and all(
            row.verification_state is HypothesisVerificationState.CONTRADICTED
            for row in verification_rows
        )

        if not viable and contradicted_all:
            return _no_match_decision(
                evaluated_hypotheses=evaluated_hypotheses,
                verification_rows=verification_rows,
                reason=ProductIdentificationDecisionReasonCode.ALL_HYPOTHESES_CONTRADICTED,
            )

        if len(viable) >= 2:
            unresolved = _unresolved_distinguishing_requirements(
                viable_ids=viable,
                evaluated_hypotheses=evaluated_hypotheses,
                query_context=query_context,
            )
            return ProductIdentificationDecision(
                outcome=ProductIdentificationOutcome.AMBIGUOUS,
                verified_hypothesis_id=None,
                verified_member_refs=(),
                evaluated_hypotheses=evaluated_hypotheses,
                decision_evidence=(),
                decision_contradicted_requirements=(),
                decision_contradictions=(),
                missing_requirements=unresolved,
                ambiguity_candidates=viable,
                decision_reason_code=ProductIdentificationDecisionReasonCode.MULTIPLE_VIABLE_IDENTITIES,
            )

        if len(viable) == 1:
            verified_id = viable[0]
            verified_evaluated = _find_evaluated(evaluated_hypotheses, verified_id)
            row = _find_verification_row(verification_rows, verified_id)
            return ProductIdentificationDecision(
                outcome=ProductIdentificationOutcome.VERIFIED,
                verified_hypothesis_id=verified_id,
                verified_member_refs=verified_evaluated.hypothesis.members,
                evaluated_hypotheses=evaluated_hypotheses,
                decision_evidence=row.supported_requirements,
                decision_contradicted_requirements=(),
                decision_contradictions=(),
                missing_requirements=(),
                ambiguity_candidates=(),
                decision_reason_code=ProductIdentificationDecisionReasonCode.UNIQUE_IDENTITY_SUPPORTED,
            )

        return _insufficient_decision(
            evaluated_hypotheses=evaluated_hypotheses,
            verification_rows=verification_rows,
            query_context=query_context,
        )


def _decide_empty_input(
    *,
    query_context: ProductIdentificationQueryContext,
    empty_input_rejection_evidence: tuple[HypothesisRejectionEvidence, ...],
) -> ProductIdentificationDecision:
    if empty_input_rejection_evidence:
        return ProductIdentificationDecision(
            outcome=ProductIdentificationOutcome.NO_MATCH,
            verified_hypothesis_id=None,
            verified_member_refs=(),
            evaluated_hypotheses=(),
            decision_evidence=(),
            decision_contradicted_requirements=(),
            decision_contradictions=(),
            missing_requirements=(),
            ambiguity_candidates=(),
            decision_reason_code=ProductIdentificationDecisionReasonCode.ALL_HYPOTHESES_CONTRADICTED,
            catalog_rejection_evidence=empty_input_rejection_evidence,
        )

    missing = _missing_from_query_context(query_context)
    if not missing:
        missing = (
            MissingRequirement(
                attribute_name="evaluated_hypotheses",
                origin=MissingRequirementOrigin.CATALOG,
                requirement_id="empty_input_without_rejection",
            ),
        )

    return ProductIdentificationDecision(
        outcome=ProductIdentificationOutcome.INSUFFICIENT_INFORMATION,
        verified_hypothesis_id=None,
        verified_member_refs=(),
        evaluated_hypotheses=(),
        decision_evidence=(),
        decision_contradicted_requirements=(),
        decision_contradictions=(),
        missing_requirements=missing,
        ambiguity_candidates=(),
        decision_reason_code=ProductIdentificationDecisionReasonCode.EMPTY_INPUT_WITHOUT_REJECTION_EVIDENCE,
    )


def _no_match_decision(
    *,
    evaluated_hypotheses: tuple[EvaluatedIdentityHypothesis, ...],
    verification_rows: tuple[IdentityHypothesisVerification, ...],
    reason: ProductIdentificationDecisionReasonCode,
) -> ProductIdentificationDecision:
    contradictions: list[IdentityContradiction] = []
    contradicted_requirements: list[ContradictedRequirementEvidence] = []
    for row in verification_rows:
        contradictions.extend(row.blocking_contradictions)
        for item in row.contradicted_requirements:
            contradicted_requirements.append(item)
            contradictions.extend(item.contradicting_contradictions)
    return ProductIdentificationDecision(
        outcome=ProductIdentificationOutcome.NO_MATCH,
        verified_hypothesis_id=None,
        verified_member_refs=(),
        evaluated_hypotheses=evaluated_hypotheses,
        decision_evidence=(),
        decision_contradicted_requirements=tuple(contradicted_requirements),
        decision_contradictions=tuple(contradictions),
        missing_requirements=(),
        ambiguity_candidates=(),
        decision_reason_code=reason,
    )


def _insufficient_decision(
    *,
    evaluated_hypotheses: tuple[EvaluatedIdentityHypothesis, ...],
    verification_rows: tuple[IdentityHypothesisVerification, ...],
    query_context: ProductIdentificationQueryContext,
) -> ProductIdentificationDecision:
    missing: list[MissingRequirement] = []
    for row in verification_rows:
        missing.extend(row.missing_requirements)
    missing.extend(_missing_from_query_context(query_context))

    reason = ProductIdentificationDecisionReasonCode.NO_VIABLE_HYPOTHESIS
    if any(
        row.verification_state is HypothesisVerificationState.INCOMPLETE
        and row.missing_requirements
        for row in verification_rows
    ):
        reason = ProductIdentificationDecisionReasonCode.MISSING_REQUIRED_CATALOG_EVIDENCE
    if query_context.missing_user_distinguishing_requirements:
        reason = ProductIdentificationDecisionReasonCode.MISSING_DISTINGUISHING_FACT
    if not missing:
        missing.append(
            MissingRequirement(
                attribute_name="identity_evidence",
                origin=MissingRequirementOrigin.CATALOG,
                requirement_id="identity_not_sufficient",
            )
        )

    return ProductIdentificationDecision(
        outcome=ProductIdentificationOutcome.INSUFFICIENT_INFORMATION,
        verified_hypothesis_id=None,
        verified_member_refs=(),
        evaluated_hypotheses=evaluated_hypotheses,
        decision_evidence=(),
        decision_contradicted_requirements=(),
        decision_contradictions=(),
        missing_requirements=tuple(_dedupe_missing(missing)),
        ambiguity_candidates=(),
        decision_reason_code=reason,
    )


def _missing_from_query_context(
    query_context: ProductIdentificationQueryContext,
) -> tuple[MissingRequirement, ...]:
    return tuple(
        MissingRequirement(
            attribute_name=item.attribute_name,
            origin=item.origin,
            requirement_id=item.requirement_id,
        )
        for item in query_context.missing_user_distinguishing_requirements
    )


def _unresolved_distinguishing_requirements(
    *,
    viable_ids: tuple[str, ...],
    evaluated_hypotheses: tuple[EvaluatedIdentityHypothesis, ...],
    query_context: ProductIdentificationQueryContext,
) -> tuple[MissingRequirement, ...]:
    user_missing = _missing_from_query_context(query_context)
    if user_missing:
        return user_missing
    return (
        MissingRequirement(
            attribute_name="variant",
            origin=MissingRequirementOrigin.CATALOG,
            requirement_id="unresolved_competing_identity",
        ),
    )


def _find_evaluated(
    evaluated_hypotheses: tuple[EvaluatedIdentityHypothesis, ...],
    hypothesis_id: str,
) -> EvaluatedIdentityHypothesis:
    for item in evaluated_hypotheses:
        if item.hypothesis.hypothesis_id == hypothesis_id:
            return item
    raise ValueError("hypothesis_id not found in evaluated_hypotheses")


def _find_verification_row(
    verification_rows: tuple[IdentityHypothesisVerification, ...],
    hypothesis_id: str,
) -> IdentityHypothesisVerification:
    for row in verification_rows:
        if row.hypothesis_id == hypothesis_id:
            return row
    raise ValueError("hypothesis_id not found in verification_rows")


def _dedupe_missing(
    items: list[MissingRequirement],
) -> tuple[MissingRequirement, ...]:
    seen: set[str] = set()
    ordered: list[MissingRequirement] = []
    for item in items:
        if item.requirement_id in seen:
            continue
        seen.add(item.requirement_id)
        ordered.append(item)
    return tuple(ordered)
