"""Conservative exact identifier normalization for derived lookup terms."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductIdentifierType,
)

_GTIN_KEY_MARKERS = ("/gtin8", "/gtin12", "/gtin13", "/gtin14", "gtin")


def classify_wdc_identifier_type(source_key: str) -> ProductIdentifierType | None:
    """Map one WDC identifier key to a supported exact lookup family."""
    normalized_key = source_key.casefold()
    if any(marker in normalized_key for marker in _GTIN_KEY_MARKERS):
        return ProductIdentifierType.GTIN
    if "mpn" in normalized_key:
        return ProductIdentifierType.MPN
    if "sku" in normalized_key:
        return ProductIdentifierType.SKU
    if "productid" in normalized_key:
        return ProductIdentifierType.PRODUCT_ID
    return None


def normalize_exact_lookup_value(
    identifier_type: ProductIdentifierType,
    source_value: str,
) -> str:
    """Derive a conservative lookup value without mutating the source value."""
    lookup = _unwrap_wdc_bracket_notation(source_value.strip())
    if identifier_type is ProductIdentifierType.GTIN:
        return "".join(character for character in lookup if character.isdigit())
    return lookup


def _unwrap_wdc_bracket_notation(value: str) -> str:
    if len(value) >= 2 and value.startswith("[") and value.endswith("]"):
        return value[1:-1].strip()
    return value
