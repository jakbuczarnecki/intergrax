"""Immutable contracts for VPI query understanding (raw input → typed query)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from platform_proofs.scenarios.verified_product_identification.application.contracts.identification_context import (
    MissingDistinguishingRequirement,
    NegativeAttributeConstraint,
    ProductIdentificationQueryContext,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.product_identification_query import (
    ProductIdentificationQuery,
)
from platform_proofs.scenarios.verified_product_identification.application.contracts.queries import (
    StructuredAttributeConstraint,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductIdentifier,
    ProductIdentifierType,
)
from platform_proofs.scenarios.verified_product_identification.application.observability.contracts import (
    ProductIdentificationInputOrigin,
)

MAX_RAW_QUERY_CHARS = 4096


@dataclass(frozen=True, slots=True)
class RawProductIdentificationRequest:
    """Scenario-owned immutable raw user request — no benchmark or provider fields."""

    raw_text: str
    correlation_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.raw_text) is not str:
            raise TypeError("raw_text must be str")
        stripped = self.raw_text.strip()
        if not stripped:
            raise ValueError("raw_text must be non-empty after strip")
        if len(self.raw_text) > MAX_RAW_QUERY_CHARS:
            raise ValueError(f"raw_text exceeds MAX_RAW_QUERY_CHARS ({MAX_RAW_QUERY_CHARS})")
        if self.correlation_id is not None and not self.correlation_id.strip():
            raise ValueError("correlation_id must be non-empty when present")


@dataclass(frozen=True, slots=True)
class QuerySourceSpan:
    """Provenance anchor in the original raw_text (offsets are UTF-8 code units / str indices)."""

    start_offset: int
    end_offset: int
    fragment: str | None = None

    def __post_init__(self) -> None:
        if type(self.start_offset) is not int or type(self.end_offset) is not int:
            raise TypeError("offsets must be int")
        if self.start_offset < 0 or self.end_offset < self.start_offset:
            raise ValueError("invalid span offsets")
        if self.fragment is not None and type(self.fragment) is not str:
            raise TypeError("fragment must be str or None")


class ExtractionCertainty(StrEnum):
    DIRECT = "direct"
    DETERMINISTIC = "deterministic"
    INFERRED = "inferred"
    UNCERTAIN = "uncertain"


class QueryUnderstandingStatus(StrEnum):
    SUCCESS = "success"
    REJECTED = "rejected"


class QueryUnderstandingIssueCode(StrEnum):
    CONFLICTING_USER_CONSTRAINT = "conflicting_user_constraint"
    UNSUPPORTED_ATTRIBUTE_EXPRESSION = "unsupported_attribute_expression"
    AMBIGUOUS_IDENTIFIER_TYPE = "ambiguous_identifier_type"
    INVALID_IDENTIFIER = "invalid_identifier"
    NO_ACTIONABLE_SEMANTICS = "no_actionable_semantics"


@dataclass(frozen=True, slots=True)
class QueryUnderstandingIssue:
    code: QueryUnderstandingIssueCode
    detail: str | None = None

    def __post_init__(self) -> None:
        if type(self.detail) is not str and self.detail is not None:
            raise TypeError("detail must be str or None")


@dataclass(frozen=True, slots=True)
class ExtractedIdentifierRecord:
    identifier: ProductIdentifier
    raw_value: str
    normalized_value: str
    normalization_rule: str
    source_span: QuerySourceSpan
    certainty: ExtractionCertainty

    def __post_init__(self) -> None:
        if type(self.raw_value) is not str or not self.raw_value:
            raise ValueError("raw_value must be non-empty str")
        if type(self.normalized_value) is not str or not self.normalized_value:
            raise ValueError("normalized_value must be non-empty str")
        if not self.normalization_rule.strip():
            raise ValueError("normalization_rule must be non-empty")


@dataclass(frozen=True, slots=True)
class ExtractedConstraintRecord:
    constraint: StructuredAttributeConstraint
    source_span: QuerySourceSpan
    certainty: ExtractionCertainty
    raw_value: str
    normalized_value: str


@dataclass(frozen=True, slots=True)
class ExtractedNegativeConstraintRecord:
    constraint: NegativeAttributeConstraint
    source_span: QuerySourceSpan
    certainty: ExtractionCertainty
    raw_value: str
    normalized_value: str


@dataclass(frozen=True, slots=True)
class ExtractedSoftPreferenceRecord:
    preference: StructuredAttributeConstraint
    source_span: QuerySourceSpan
    certainty: ExtractionCertainty
    raw_value: str
    normalized_value: str


@dataclass(frozen=True, slots=True)
class QueryInterpretationCandidate:
    """Optional interpreter output — merged under deterministic authority."""

    identifiers: tuple[ExtractedIdentifierRecord, ...] = ()
    required_constraints: tuple[ExtractedConstraintRecord, ...] = ()
    negative_constraints: tuple[ExtractedNegativeConstraintRecord, ...] = ()
    soft_preferences: tuple[ExtractedSoftPreferenceRecord, ...] = ()
    missing_requirements: tuple[MissingDistinguishingRequirement, ...] = ()


@dataclass(frozen=True, slots=True)
class QueryUnderstandingExtractionBundle:
    identifiers: tuple[ExtractedIdentifierRecord, ...] = ()
    required_constraints: tuple[ExtractedConstraintRecord, ...] = ()
    negative_constraints: tuple[ExtractedNegativeConstraintRecord, ...] = ()
    soft_preferences: tuple[ExtractedSoftPreferenceRecord, ...] = ()
    missing_requirements: tuple[MissingDistinguishingRequirement, ...] = ()


@dataclass(frozen=True, slots=True)
class QueryUnderstandingObservedPayload:
    raw_input_sha256_prefix: str
    extracted_identifiers: tuple[ExtractedIdentifierRecord, ...]
    required_constraints: tuple[ExtractedConstraintRecord, ...]
    negative_constraints: tuple[ExtractedNegativeConstraintRecord, ...]
    soft_preferences: tuple[ExtractedSoftPreferenceRecord, ...]
    issues: tuple[QueryUnderstandingIssue, ...]
    search_text: str | None


def raw_input_fingerprint(raw_text: str) -> str:
    digest = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    return digest[:16]


@dataclass(frozen=True, slots=True)
class ProductIdentificationQueryUnderstandingResult:
    """Authoritative query-understanding outcome — not a verification decision."""

    query: ProductIdentificationQuery | None
    issues: tuple[QueryUnderstandingIssue, ...]
    extraction: QueryUnderstandingExtractionBundle
    observation: QueryUnderstandingObservedPayload
    pipeline_input_origin: ProductIdentificationInputOrigin = (
        ProductIdentificationInputOrigin.RAW_QUERY
    )

    @property
    def status(self) -> QueryUnderstandingStatus:
        if self.query is not None:
            return QueryUnderstandingStatus.SUCCESS
        return QueryUnderstandingStatus.REJECTED

    def __post_init__(self) -> None:
        if self.query is None and not self.issues:
            raise ValueError("failed understanding requires at least one issue")
        if self.query is not None:
            if type(self.query.verification_context) is not ProductIdentificationQueryContext:
                raise TypeError("query.verification_context must be ProductIdentificationQueryContext")
            blocking = (
                QueryUnderstandingIssueCode.CONFLICTING_USER_CONSTRAINT,
                QueryUnderstandingIssueCode.AMBIGUOUS_IDENTIFIER_TYPE,
                QueryUnderstandingIssueCode.NO_ACTIONABLE_SEMANTICS,
            )
            if any(issue.code in blocking for issue in self.issues):
                raise ValueError("SUCCESS result cannot carry blocking issues")


class ProductIdentificationQueryInterpreter(Protocol):
    def interpret(
        self,
        request: RawProductIdentificationRequest,
    ) -> QueryInterpretationCandidate:
        ...
