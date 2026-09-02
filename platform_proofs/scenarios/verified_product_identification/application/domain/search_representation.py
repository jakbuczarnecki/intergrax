"""Derived search representations — retrieval-only, never source truth."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductIdentifierType,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)

SEARCH_REPRESENTATION_DERIVATION_VERSION = "v1"


@dataclass(frozen=True, slots=True)
class ExactIdentifierTerm:
    """One conservative exact lookup term derived from a typed source identifier."""

    identifier_type: ProductIdentifierType
    source_value: str
    lookup_value: str
    source_field: str


@dataclass(frozen=True, slots=True)
class ExactSearchRepresentation:
    """Exact identifier lookup representation for one source offer."""

    source_ref: SourceRecordRef
    terms: tuple[ExactIdentifierTerm, ...]


@dataclass(frozen=True, slots=True)
class StructuredTextFragment:
    """Bounded textual fragment from structured source content."""

    source_field: str
    text: str


@dataclass(frozen=True, slots=True)
class LexicalSearchRepresentation:
    """Lexical retrieval representation preserving logical field boundaries."""

    source_ref: SourceRecordRef
    title: str | None
    brand: str | None
    description: str | None
    structured_text_fragments: tuple[StructuredTextFragment, ...]


@dataclass(frozen=True, slots=True)
class StructuredAttribute:
    """One bounded structured attribute derived from source fields."""

    canonical_key: str | None
    source_key: str
    source_value: str
    normalized_text_value: str
    typed_value: str | int | float | bool | None
    source_field: str


@dataclass(frozen=True, slots=True)
class StructuredSearchRepresentation:
    """Structured retrieval representation over heterogeneous source attributes."""

    source_ref: SourceRecordRef
    attributes: tuple[StructuredAttribute, ...]


@dataclass(frozen=True, slots=True)
class SemanticContributingField:
    """One source-attributable field contributing to semantic text."""

    source_field: str
    text: str


@dataclass(frozen=True, slots=True)
class SemanticSearchRepresentation:
    """Deterministic semantic text input for later embedding — no vectors here."""

    source_ref: SourceRecordRef
    semantic_text: str
    contributing_fields: tuple[SemanticContributingField, ...]


@dataclass(frozen=True, slots=True)
class DerivedOfferSearchRepresentation:
    """Immutable derived search envelope for one source offer."""

    source_ref: SourceRecordRef
    exact: ExactSearchRepresentation
    lexical: LexicalSearchRepresentation
    structured: StructuredSearchRepresentation
    semantic: SemanticSearchRepresentation
    derivation_version: str
