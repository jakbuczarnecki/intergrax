"""Direct per-offer identity facts for query-to-source verification (5C10 handoff)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductIdentifierType,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)


class SourceIdentityFactKind(StrEnum):
    IDENTIFIER = "identifier"
    STRUCTURED_ATTRIBUTE = "structured_attribute"


@dataclass(frozen=True, slots=True)
class SourceIdentityFactProvenance:
    """Traceability from one catalog field to a normalized fact value."""

    source_field: str
    normalization_rule: str
    source_value: str

    def __post_init__(self) -> None:
        if not self.source_field.strip():
            raise ValueError("source_field must be non-empty")
        if not self.normalization_rule.strip():
            raise ValueError("normalization_rule must be non-empty")
        if not self.source_value.strip():
            raise ValueError("source_value must be non-empty")


@dataclass(frozen=True, slots=True)
class SourceIdentityFact:
    """One authoritative catalog value carried by a single source offer."""

    source_ref: SourceRecordRef
    fact_kind: SourceIdentityFactKind
    attribute_key: str
    normalized_value: str
    provenance: SourceIdentityFactProvenance
    identifier_type: ProductIdentifierType | None = None

    def __post_init__(self) -> None:
        if not self.attribute_key.strip():
            raise ValueError("attribute_key must be non-empty")
        if not self.normalized_value.strip():
            raise ValueError("normalized_value must be non-empty")
        if self.fact_kind is SourceIdentityFactKind.IDENTIFIER:
            if self.identifier_type is None:
                raise ValueError("identifier facts require identifier_type")
        elif self.fact_kind is SourceIdentityFactKind.STRUCTURED_ATTRIBUTE:
            if self.identifier_type is not None:
                raise ValueError("structured attribute facts must not set identifier_type")
