"""Typed product identifier and offer identity models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ProductIdentifierType(StrEnum):
    """Canonical identifier families for exact lookup."""

    GTIN = "gtin"
    MPN = "mpn"
    SKU = "sku"
    PRODUCT_ID = "product_id"


def _require_non_empty_str(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class ProductOfferId:
    """Canonical immutable identity for one catalog source offer."""

    value: str

    def __post_init__(self) -> None:
        normalized = _require_non_empty_str(self.value, field_name="ProductOfferId.value")
        if normalized != self.value:
            object.__setattr__(self, "value", normalized)


@dataclass(frozen=True, slots=True)
class ProductIdentifier:
    """Typed identifier used for exact lookup — not a derived search token."""

    identifier_type: ProductIdentifierType
    value: str
    source_field: str | None = None

    def __post_init__(self) -> None:
        normalized = _require_non_empty_str(self.value, field_name="ProductIdentifier.value")
        if normalized != self.value:
            object.__setattr__(self, "value", normalized)
        source_field = self.source_field
        if source_field is not None:
            normalized_source = _require_non_empty_str(
                source_field,
                field_name="ProductIdentifier.source_field",
            )
            if normalized_source != source_field:
                object.__setattr__(self, "source_field", normalized_source)
