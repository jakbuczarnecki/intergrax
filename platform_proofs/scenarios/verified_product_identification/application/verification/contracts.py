"""Immutable terminal verification and abstention contracts (5C10)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from platform_proofs.scenarios.verified_product_identification.application.contracts.failures import (
    CatalogSearchFailure,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.identification_context import (
    HypothesisRejectionEvidence,
    MissingRequirementOrigin,
    ProductIdentificationQueryContext,
)
from platform_proofs.scenarios.verified_product_identification.application.identity.contracts import (
    IdentityContradiction,
    IdentityEvidence,
    IdentityHypothesisMember,
)
from platform_proofs.scenarios.verified_product_identification.application.identity_evaluation.contracts import (
    EvaluatedIdentityHypothesis,
    RankedIdentityHypothesisCollection,
)


class ProductIdentificationOutcome(StrEnum):
    VERIFIED = "verified"
    AMBIGUOUS = "ambiguous"
    INSUFFICIENT_INFORMATION = "insufficient_information"
    NO_MATCH = "no_match"


class ProductIdentificationDecisionReasonCode(StrEnum):
    UNIQUE_IDENTITY_SUPPORTED = "unique_identity_supported"
    MULTIPLE_VIABLE_IDENTITIES = "multiple_viable_identities"
    MISSING_DISTINGUISHING_FACT = "missing_distinguishing_fact"
    MISSING_REQUIRED_CATALOG_EVIDENCE = "missing_required_catalog_evidence"
    ALL_HYPOTHESES_CONTRADICTED = "all_hypotheses_contradicted"
    REQUIRED_CONSTRAINT_CONFLICT = "required_constraint_conflict"
    NO_VIABLE_HYPOTHESIS = "no_viable_hypothesis"
    EMPTY_INPUT_WITHOUT_REJECTION_EVIDENCE = "empty_input_without_rejection_evidence"


class ConstraintRequirementStatus(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    MISSING = "missing"


class HypothesisVerificationState(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True)
class VerifiedRequirementEvidence:
    attribute_name: str
    expected_value: str
    catalog_value: str
    supporting_evidence: tuple[IdentityEvidence, ...]

    def __post_init__(self) -> None:
        if not self.attribute_name.strip():
            raise ValueError("attribute_name must be non-empty")
        if not isinstance(self.supporting_evidence, tuple):
            raise TypeError("supporting_evidence must be a tuple")


@dataclass(frozen=True, slots=True)
class ContradictedRequirementEvidence:
    attribute_name: str
    expected_value: str
    catalog_value: str
    contradicting_evidence: tuple[IdentityEvidence, ...]
    contradicting_contradictions: tuple[IdentityContradiction, ...]

    def __post_init__(self) -> None:
        if not self.attribute_name.strip():
            raise ValueError("attribute_name must be non-empty")
        if not isinstance(self.contradicting_evidence, tuple):
            raise TypeError("contradicting_evidence must be a tuple")
        if not isinstance(self.contradicting_contradictions, tuple):
            raise TypeError("contradicting_contradictions must be a tuple")


@dataclass(frozen=True, slots=True)
class MissingRequirement:
    attribute_name: str
    origin: MissingRequirementOrigin
    requirement_id: str

    def __post_init__(self) -> None:
        if not self.attribute_name.strip():
            raise ValueError("attribute_name must be non-empty")
        if not self.requirement_id.strip():
            raise ValueError("requirement_id must be non-empty")


@dataclass(frozen=True, slots=True)
class IdentityHypothesisVerification:
    hypothesis_id: str
    eligible_for_verification: bool
    verification_state: HypothesisVerificationState
    supported_requirements: tuple[VerifiedRequirementEvidence, ...]
    contradicted_requirements: tuple[ContradictedRequirementEvidence, ...]
    missing_requirements: tuple[MissingRequirement, ...]
    blocking_contradictions: tuple[IdentityContradiction, ...]
    identity_evidence_sufficient: bool

    def __post_init__(self) -> None:
        if not self.hypothesis_id.strip():
            raise ValueError("hypothesis_id must be non-empty")


@dataclass(frozen=True, slots=True)
class HypothesisVerificationCollection:
    rows: tuple[IdentityHypothesisVerification, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.rows, tuple):
            raise TypeError("rows must be a tuple")


@dataclass(frozen=True, slots=True)
class ProductIdentificationDecision:
    outcome: ProductIdentificationOutcome
    verified_hypothesis_id: str | None
    verified_member_refs: tuple[IdentityHypothesisMember, ...]
    evaluated_hypotheses: tuple[EvaluatedIdentityHypothesis, ...]
    decision_evidence: tuple[VerifiedRequirementEvidence, ...]
    decision_contradicted_requirements: tuple[ContradictedRequirementEvidence, ...]
    decision_contradictions: tuple[IdentityContradiction, ...]
    missing_requirements: tuple[MissingRequirement, ...]
    ambiguity_candidates: tuple[str, ...]
    decision_reason_code: ProductIdentificationDecisionReasonCode
    catalog_rejection_evidence: tuple[HypothesisRejectionEvidence, ...] = ()
    detail_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.evaluated_hypotheses, tuple):
            raise TypeError("evaluated_hypotheses must be a tuple")
        if not isinstance(self.verified_member_refs, tuple):
            raise TypeError("verified_member_refs must be a tuple")
        if not isinstance(self.decision_evidence, tuple):
            raise TypeError("decision_evidence must be a tuple")
        if not isinstance(self.decision_contradicted_requirements, tuple):
            raise TypeError("decision_contradicted_requirements must be a tuple")
        if not isinstance(self.decision_contradictions, tuple):
            raise TypeError("decision_contradictions must be a tuple")
        if not isinstance(self.missing_requirements, tuple):
            raise TypeError("missing_requirements must be a tuple")
        if not isinstance(self.ambiguity_candidates, tuple):
            raise TypeError("ambiguity_candidates must be a tuple")
        if not isinstance(self.catalog_rejection_evidence, tuple):
            raise TypeError("catalog_rejection_evidence must be a tuple")
        _validate_decision_invariants(self)


@dataclass(frozen=True, slots=True)
class ProductIdentificationVerificationRequest:
    ranked_hypotheses: RankedIdentityHypothesisCollection
    query_context: ProductIdentificationQueryContext
    empty_input_rejection_evidence: tuple[HypothesisRejectionEvidence, ...] = ()

    def __post_init__(self) -> None:
        if type(self.empty_input_rejection_evidence) is not tuple:
            raise TypeError("empty_input_rejection_evidence must be a tuple")


@dataclass(frozen=True, slots=True)
class ProductIdentificationVerificationOutcome:
    """Business decision or infrastructure failure — never both."""

    decision: ProductIdentificationDecision | None = None
    failure: CatalogSearchFailure | None = None

    def __post_init__(self) -> None:
        if self.decision is not None and self.failure is not None:
            raise ValueError("decision and failure are mutually exclusive")
        if self.decision is None and self.failure is None:
            raise ValueError("decision or failure must be present")


def _validate_decision_invariants(decision: ProductIdentificationDecision) -> None:
    outcome = decision.outcome
    verified_id = decision.verified_hypothesis_id

    if outcome is ProductIdentificationOutcome.VERIFIED:
        if verified_id is None:
            raise ValueError("VERIFIED requires verified_hypothesis_id")
        if not decision.verified_member_refs:
            raise ValueError("VERIFIED requires verified_member_refs")
        if decision.ambiguity_candidates:
            raise ValueError("VERIFIED must not include ambiguity_candidates")
        return

    if verified_id is not None:
        raise ValueError(f"{outcome.value} must not set verified_hypothesis_id")
    if decision.verified_member_refs:
        raise ValueError(f"{outcome.value} must not set verified_member_refs")

    if outcome is ProductIdentificationOutcome.AMBIGUOUS:
        if len(decision.ambiguity_candidates) < 2:
            raise ValueError("AMBIGUOUS requires at least two ambiguity_candidates")
        return

    if outcome is ProductIdentificationOutcome.NO_MATCH:
        has_rejection = bool(
            decision.decision_contradictions
            or decision.decision_contradicted_requirements
            or decision.catalog_rejection_evidence
        )
        if not has_rejection:
            raise ValueError("NO_MATCH requires rejection evidence in decision payload")
        return

    if outcome is ProductIdentificationOutcome.INSUFFICIENT_INFORMATION:
        if not decision.missing_requirements:
            raise ValueError("INSUFFICIENT_INFORMATION requires missing_requirements")
