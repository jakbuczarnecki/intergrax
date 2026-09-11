"""Immutable targeted clarification selection contracts (5C11)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from platform_proofs.scenarios.verified_product_identification.application.contracts.identification_context import (
    MissingRequirementOrigin,
    ProductIdentificationQueryContext,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.source_identity_fact import (
    SourceIdentityFact,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductIdentifierType,
)
from platform_proofs.scenarios.verified_product_identification.application.verification.contracts import (
    ProductIdentificationDecision,
)


class ClarificationRequirementKind(StrEnum):
    ATTRIBUTE_VALUE = "attribute_value"
    IDENTIFIER_VALUE = "identifier_value"
    IDENTITY_DISCRIMINATOR = "identity_discriminator"
    USER_MISSING_FACT = "user_missing_fact"


class NoClarificationReason(StrEnum):
    DECISION_ALREADY_TERMINAL = "decision_already_terminal"
    NO_USER_ANSWERABLE_REQUIREMENT = "no_user_answerable_requirement"
    CATALOG_EVIDENCE_ONLY_GAP = "catalog_evidence_only_gap"
    NO_DISCRIMINATING_FACT = "no_discriminating_fact"
    NO_UNRESOLVED_COMPETITOR = "no_unresolved_competitor"
    NO_DISCRIMINATOR_AVAILABLE = "no_discriminator_available"


@dataclass(frozen=True, slots=True)
class ClarificationDiscriminationMetrics:
    """Discrete discrimination view — no scores or probabilities."""

    known_hypothesis_count: int
    total_competing_hypothesis_count: int
    distinct_known_value_count: int
    eliminable_hypothesis_count: int
    has_complete_coverage: bool

    def __post_init__(self) -> None:
        if type(self.known_hypothesis_count) is not int or self.known_hypothesis_count < 0:
            raise ValueError("known_hypothesis_count must be a non-negative int")
        if (
            type(self.total_competing_hypothesis_count) is not int
            or self.total_competing_hypothesis_count < 0
        ):
            raise ValueError("total_competing_hypothesis_count must be a non-negative int")
        if (
            type(self.distinct_known_value_count) is not int
            or self.distinct_known_value_count < 0
        ):
            raise ValueError("distinct_known_value_count must be a non-negative int")
        if (
            type(self.eliminable_hypothesis_count) is not int
            or self.eliminable_hypothesis_count < 0
        ):
            raise ValueError("eliminable_hypothesis_count must be a non-negative int")
        if self.known_hypothesis_count > self.total_competing_hypothesis_count:
            raise ValueError("known_hypothesis_count cannot exceed total_competing_hypothesis_count")


@dataclass(frozen=True, slots=True)
class ClarificationRequirementProvenance:
    """Typed traceability — reconstructable without natural language."""

    affected_hypothesis_ids: tuple[str, ...]
    supporting_source_facts: tuple[SourceIdentityFact, ...]
    origin: MissingRequirementOrigin | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.affected_hypothesis_ids, tuple):
            raise TypeError("affected_hypothesis_ids must be a tuple")
        if not isinstance(self.supporting_source_facts, tuple):
            raise TypeError("supporting_source_facts must be a tuple")
        for hypothesis_id in self.affected_hypothesis_ids:
            if not hypothesis_id.strip():
                raise ValueError("affected_hypothesis_ids must be non-empty strings")


@dataclass(frozen=True, slots=True)
class ClarificationRequirement:
    requirement_id: str
    kind: ClarificationRequirementKind
    attribute_name: str
    origin: MissingRequirementOrigin | None
    reason: str
    discrimination: ClarificationDiscriminationMetrics
    provenance: ClarificationRequirementProvenance
    identifier_type: ProductIdentifierType | None = None
    candidate_values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.requirement_id.strip():
            raise ValueError("requirement_id must be non-empty")
        if not self.attribute_name.strip():
            raise ValueError("attribute_name must be non-empty")
        if not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if not isinstance(self.candidate_values, tuple):
            raise TypeError("candidate_values must be a tuple")
        if self.kind is ClarificationRequirementKind.IDENTIFIER_VALUE:
            if self.identifier_type is None:
                raise ValueError("identifier clarification requires identifier_type")
        elif self.identifier_type is not None:
            raise ValueError("non-identifier clarification must not set identifier_type")


@dataclass(frozen=True, slots=True)
class ClarificationSelectionRequest:
    decision: ProductIdentificationDecision
    query_context: ProductIdentificationQueryContext

    def __post_init__(self) -> None:
        if self.decision is None:
            raise ValueError("decision must be present")


@dataclass(frozen=True, slots=True)
class ClarificationSelectionResult:
    clarification_required: bool
    primary_requirement: ClarificationRequirement | None
    alternate_requirements: tuple[ClarificationRequirement, ...]
    no_clarification_reason: NoClarificationReason | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.alternate_requirements, tuple):
            raise TypeError("alternate_requirements must be a tuple")
        if self.clarification_required:
            if self.primary_requirement is None:
                raise ValueError("clarification_required requires primary_requirement")
            if self.no_clarification_reason is not None:
                raise ValueError("clarification_required must not set no_clarification_reason")
        else:
            if self.primary_requirement is not None:
                raise ValueError("no clarification must not set primary_requirement")
            if self.no_clarification_reason is None:
                raise ValueError("no clarification requires no_clarification_reason")
        primary_id = self.primary_requirement.requirement_id if self.primary_requirement else None
        for alternate in self.alternate_requirements:
            if primary_id is not None and alternate.requirement_id == primary_id:
                raise ValueError("alternate must not duplicate primary requirement_id")
