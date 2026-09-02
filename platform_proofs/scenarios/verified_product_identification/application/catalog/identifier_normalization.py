"""Conservative exact identifier normalization for derived lookup terms."""

from __future__ import annotations

import re

from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductIdentifierType,
)

_GTIN_SOURCE_KEYS = frozenset(
    {
        "/gtin8",
        "/gtin12",
        "/gtin13",
        "/gtin14",
    }
)
_VALID_GTIN_LENGTHS = frozenset({8, 12, 13, 14})
_GTIN_SEPARATOR_CHARS = frozenset({" ", "-", ".", "\t"})


def classify_wdc_identifier_type(source_key: str) -> ProductIdentifierType | None:
    """Map one WDC identifier key to a supported exact lookup family."""
    normalized_key = source_key.casefold()
    if normalized_key in _GTIN_SOURCE_KEYS:
        return ProductIdentifierType.GTIN
    if normalized_key.endswith("/mpn") or normalized_key == "mpn":
        return ProductIdentifierType.MPN
    if normalized_key.endswith("/sku") or normalized_key == "sku":
        return ProductIdentifierType.SKU
    if normalized_key.endswith("/productid") or normalized_key == "productid":
        return ProductIdentifierType.PRODUCT_ID
    return None


def normalize_exact_lookup_value(
    identifier_type: ProductIdentifierType,
    source_value: str,
) -> str:
    """Derive a conservative lookup value without mutating the source value."""
    lookup = _unwrap_wdc_bracket_notation(source_value.strip())
    if identifier_type is ProductIdentifierType.GTIN:
        return _normalize_gtin_lookup_value(lookup)
    return lookup


def _normalize_gtin_lookup_value(lookup: str) -> str:
    if not lookup:
        return ""
    if "," in lookup:
        return ""
    if lookup.isdigit():
        return lookup if len(lookup) in _VALID_GTIN_LENGTHS else ""

    if not all(
        character.isdigit() or character in _GTIN_SEPARATOR_CHARS for character in lookup
    ):
        return ""

    digit_runs = re.findall(r"\d+", lookup)
    if not digit_runs:
        return ""

    independently_valid_runs = [
        digit_run for digit_run in digit_runs if len(digit_run) in _VALID_GTIN_LENGTHS
    ]
    if len(independently_valid_runs) >= 2:
        return ""

    candidate = "".join(digit_runs)
    if len(candidate) not in _VALID_GTIN_LENGTHS:
        return ""
    return candidate


def _unwrap_wdc_bracket_notation(value: str) -> str:
    if len(value) >= 2 and value.startswith("[") and value.endswith("]"):
        return value[1:-1].strip()
    return value
