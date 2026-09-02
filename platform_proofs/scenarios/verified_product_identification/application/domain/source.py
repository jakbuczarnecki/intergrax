"""Immutable source-truth models — verification authority, not search representation."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductOfferId,
)


def _require_non_empty_str(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{field_name} must not have leading or trailing whitespace")


@dataclass(frozen=True, slots=True)
class ProductSourceProvenance:
    """Provenance for immutable catalog source records."""

    catalog_id: str
    source_revision: str | None = None
    ingestion_batch: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty_str(self.catalog_id, field_name="ProductSourceProvenance.catalog_id")


@dataclass(frozen=True, slots=True)
class SourceRecordRef:
    """Lightweight reference from a derived candidate back to source truth."""

    offer_id: ProductOfferId
    catalog_id: str
    source_revision: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty_str(self.catalog_id, field_name="SourceRecordRef.catalog_id")


@dataclass(frozen=True, slots=True)
class ProductSourceRecord:
    """Immutable source-truth offer record — never a search index projection."""

    offer_id: ProductOfferId
    record_payload_ref: str
    provenance: ProductSourceProvenance

    def __post_init__(self) -> None:
        _require_non_empty_str(
            self.record_payload_ref,
            field_name="ProductSourceRecord.record_payload_ref",
        )
