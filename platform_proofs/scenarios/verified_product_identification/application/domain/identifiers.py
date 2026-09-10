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


class ProductIdentifierIdentityScope(StrEnum):
    """Cross-offer product identity comparability — distinct from exact retrieval."""

    GLOBAL = "global"
    MANUFACTURER_SCOPED = "manufacturer_scoped"
    SOURCE_LOCAL = "source_local"


def identity_scope_for_identifier_type(
    identifier_type: ProductIdentifierType,
) -> ProductIdentifierIdentityScope:
    """Classify whether an identifier family is globally comparable for identity."""
    if identifier_type is ProductIdentifierType.GTIN:
        return ProductIdentifierIdentityScope.GLOBAL
    if identifier_type is ProductIdentifierType.MPN:
        return ProductIdentifierIdentityScope.MANUFACTURER_SCOPED
    if identifier_type in (ProductIdentifierType.SKU, ProductIdentifierType.PRODUCT_ID):
        return ProductIdentifierIdentityScope.SOURCE_LOCAL
    raise ValueError(f"unsupported identifier type: {identifier_type}")


def _require_non_empty_str(value: str, *, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{field_name} must not have leading or trailing whitespace")


@dataclass(frozen=True, slots=True)
class ProductOfferId:
    """Canonical immutable identity for one catalog source offer."""

    value: str

    def __post_init__(self) -> None:
        _require_non_empty_str(self.value, field_name="ProductOfferId.value")


@dataclass(frozen=True, slots=True)
class ProductIdentifier:
    """Typed identifier used for exact lookup — not a derived search token."""

    identifier_type: ProductIdentifierType
    value: str
    source_field: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty_str(self.value, field_name="ProductIdentifier.value")
        if self.source_field is not None:
            _require_non_empty_str(
                self.source_field,
                field_name="ProductIdentifier.source_field",
            )
