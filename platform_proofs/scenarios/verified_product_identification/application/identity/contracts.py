"""Immutable product identity hypothesis contracts — propositions, not verified truth."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductIdentifierType,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
    source_ref_sort_key,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source_identity import (
    source_ref_set_sha256,
)
from platform_proofs.scenarios.verified_product_identification.application.fusion.contracts import (
    FusedOfferCandidateCollection,
    OfferChannelEvidence,
)


class IdentityEvidenceType(StrEnum):
    """Closed evidence categories for cross-offer identity support."""

    EXACT_IDENTIFIER_MATCH = "exact_identifier_match"
    MODEL_NUMBER_MATCH = "model_number_match"
    BRAND_MATCH = "brand_match"
    STRUCTURED_ATTRIBUTE_MATCH = "structured_attribute_match"
    TITLE_TOKEN_SUPPORT = "title_token_support"
    SEMANTIC_SUPPORT = "semantic_support"


class IdentityContradictionType(StrEnum):
    """Discrete evidence against a same-product hypothesis."""

    IDENTIFIER_CONFLICT = "identifier_conflict"
    MODEL_NUMBER_CONFLICT = "model_number_conflict"
    BRAND_CONFLICT = "brand_conflict"
    STRUCTURED_ATTRIBUTE_CONFLICT = "structured_attribute_conflict"


class IdentityEvidenceStrengthClass(StrEnum):
    """Discrete strength — not a confidence score."""

    STRONG = "strong"
    WEAK = "weak"


@dataclass(frozen=True, slots=True)
class IdentityEvidenceProvenance:
    """Traceability for one evidence or contradiction item."""

    left_source_ref: SourceRecordRef
    right_source_ref: SourceRecordRef
    source_field: str
    normalization_rule: str

    def __post_init__(self) -> None:
        if not self.source_field.strip():
            raise ValueError("source_field must be non-empty")
        if not self.normalization_rule.strip():
            raise ValueError("normalization_rule must be non-empty")


@dataclass(frozen=True, slots=True)
class IdentityEvidence:
    """Typed cross-offer support — explains what matched, between whom, and how."""

    evidence_type: IdentityEvidenceType
    source_refs: tuple[SourceRecordRef, SourceRecordRef]
    attribute_key: str
    normalized_value: str
    strength_class: IdentityEvidenceStrengthClass
    provenance: IdentityEvidenceProvenance
    identifier_type: ProductIdentifierType | None = None

    def __post_init__(self) -> None:
        if len(self.source_refs) != 2:
            raise ValueError("source_refs must contain exactly two references")
        if not self.attribute_key.strip():
            raise ValueError("attribute_key must be non-empty")
        if not self.normalized_value.strip():
            raise ValueError("normalized_value must be non-empty")
        ordered = tuple(sorted(self.source_refs, key=source_ref_sort_key))
        if self.source_refs != ordered:
            raise ValueError("source_refs must be deterministically ordered")


@dataclass(frozen=True, slots=True)
class IdentityContradiction:
    """Typed cross-offer conflict — discrete, not negative confidence."""

    contradiction_type: IdentityContradictionType
    source_refs: tuple[SourceRecordRef, SourceRecordRef]
    attribute_key: str
    left_normalized_value: str
    right_normalized_value: str
    provenance: IdentityEvidenceProvenance
    identifier_type: ProductIdentifierType | None = None

    def __post_init__(self) -> None:
        if len(self.source_refs) != 2:
            raise ValueError("source_refs must contain exactly two references")
        if not self.attribute_key.strip():
            raise ValueError("attribute_key must be non-empty")
        if not self.left_normalized_value.strip():
            raise ValueError("left_normalized_value must be non-empty")
        if not self.right_normalized_value.strip():
            raise ValueError("right_normalized_value must be non-empty")
        ordered = tuple(sorted(self.source_refs, key=source_ref_sort_key))
        if self.source_refs != ordered:
            raise ValueError("source_refs must be deterministically ordered")
        if self.left_normalized_value == self.right_normalized_value:
            raise ValueError("contradiction values must differ")


@dataclass(frozen=True, slots=True)
class IdentityHypothesisMember:
    """Linkage from one hypothesis member back to fused offer context."""

    source_ref: SourceRecordRef
    fused_rank: int
    fusion_evidence: tuple[OfferChannelEvidence, ...]

    def __post_init__(self) -> None:
        if type(self.fused_rank) is not int or self.fused_rank < 0:
            raise ValueError("fused_rank must be a non-negative int")
        if not isinstance(self.fusion_evidence, tuple):
            raise TypeError("fusion_evidence must be a tuple")


@dataclass(frozen=True, slots=True)
class ProductIdentityHypothesis:
    """Explicit reversible proposition — these offers may describe the same product."""

    hypothesis_id: str
    members: tuple[IdentityHypothesisMember, ...]
    evidence: tuple[IdentityEvidence, ...]
    contradictions: tuple[IdentityContradiction, ...]

    def __post_init__(self) -> None:
        if not self.hypothesis_id.strip():
            raise ValueError("hypothesis_id must be non-empty")
        if not isinstance(self.members, tuple) or len(self.members) < 1:
            raise ValueError("members must be a non-empty tuple")
        if not isinstance(self.evidence, tuple):
            raise TypeError("evidence must be a tuple")
        if not isinstance(self.contradictions, tuple):
            raise TypeError("contradictions must be a tuple")
        member_refs = [member.source_ref for member in self.members]
        if len(set(member_refs)) != len(member_refs):
            raise ValueError("members must be unique by source_ref")
        expected_id = source_ref_set_sha256(tuple(member_refs))
        if self.hypothesis_id != expected_id:
            raise ValueError("hypothesis_id must match deterministic member digest")


@dataclass(frozen=True, slots=True)
class ProductIdentityHypothesisCollection:
    """Deterministic hypothesis list — ordering is not identity confidence."""

    hypotheses: tuple[ProductIdentityHypothesis, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.hypotheses, tuple):
            raise TypeError("hypotheses must be a tuple")


@dataclass(frozen=True, slots=True)
class IdentityHypothesisConfiguration:
    """Bounded identity hypothesis input policy."""

    max_candidates: int = 20

    def __post_init__(self) -> None:
        if type(self.max_candidates) is not int or self.max_candidates <= 0:
            raise ValueError("max_candidates must be a positive int")


@dataclass(frozen=True, slots=True)
class ProductIdentityHypothesisRequest:
    """Validated identity hypothesis request over fused offer candidates."""

    fused_candidates: FusedOfferCandidateCollection
    configuration: IdentityHypothesisConfiguration = IdentityHypothesisConfiguration()

    def __post_init__(self) -> None:
        candidate_count = len(self.fused_candidates.candidates)
        if candidate_count > self.configuration.max_candidates:
            raise ValueError("fused candidate count exceeds max_candidates")
