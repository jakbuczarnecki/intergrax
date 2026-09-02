"""Derived search representations — retrieval-only, never source truth."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductIdentifierType,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)

SEARCH_REPRESENTATION_DERIVATION_VERSION = "v2"


@dataclass(frozen=True, slots=True)
class ExactIdentifierTerm:
    """One conservative exact lookup term derived from a typed source identifier."""

    identifier_type: ProductIdentifierType
    source_value: str
    lookup_value: str
    source_field: str

    def __post_init__(self) -> None:
        if not self.source_value.strip():
            msg = "ExactIdentifierTerm.source_value must be non-empty"
            raise ValueError(msg)
        if not self.lookup_value.strip():
            msg = "ExactIdentifierTerm.lookup_value must be non-empty"
            raise ValueError(msg)
        if not self.source_field.strip():
            msg = "ExactIdentifierTerm.source_field must be non-empty"
            raise ValueError(msg)


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

    def __post_init__(self) -> None:
        if not self.source_key.strip():
            msg = "StructuredAttribute.source_key must be non-empty"
            raise ValueError(msg)
        if not self.source_value.strip():
            msg = "StructuredAttribute.source_value must be non-empty"
            raise ValueError(msg)
        if not self.source_field.strip():
            msg = "StructuredAttribute.source_field must be non-empty"
            raise ValueError(msg)


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

    def __post_init__(self) -> None:
        if not self.text.strip():
            msg = "SemanticContributingField.text must be non-empty"
            raise ValueError(msg)
        if not self.source_field.strip():
            msg = "SemanticContributingField.source_field must be non-empty"
            raise ValueError(msg)


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

    def __post_init__(self) -> None:
        if not self.derivation_version.strip():
            msg = "DerivedOfferSearchRepresentation.derivation_version must be non-empty"
            raise ValueError(msg)
        channel_refs = (
            self.exact.source_ref,
            self.lexical.source_ref,
            self.structured.source_ref,
            self.semantic.source_ref,
        )
        for channel_name, channel_ref in (
            ("exact", channel_refs[0]),
            ("lexical", channel_refs[1]),
            ("structured", channel_refs[2]),
            ("semantic", channel_refs[3]),
        ):
            if channel_ref != self.source_ref:
                msg = (
                    "DerivedOfferSearchRepresentation channel source_ref mismatch: "
                    f"envelope={self.source_ref!r} {channel_name}={channel_ref!r}"
                )
                raise ValueError(msg)
