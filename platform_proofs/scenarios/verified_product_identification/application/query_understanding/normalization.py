"""Deterministic normalization helpers for query understanding."""

from __future__ import annotations

import re

from platform_proofs.scenarios.verified_product_identification.application.catalog.identifier_normalization import (
    normalize_exact_lookup_value,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductIdentifierType,
)

_WHITESPACE_RUN = re.compile(r"\s+")

# Canonical attribute keys aligned with 5C12 pipeline tests and structured retrieval.
CANONICAL_CAPACITY = "capacity"
CANONICAL_INTERFACE = "interface"
CANONICAL_MEMORY_TYPE = "memory_type"
CANONICAL_ECC = "ecc"

_CAPACITY_PATTERN = re.compile(
    r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>TB|GB|MB)\b",
    re.IGNORECASE,
)
_INTERFACE_TOKENS: tuple[tuple[str, str], ...] = (
    ("nvme", "NVMe"),
    ("sata", "SATA"),
    ("pcie", "PCIe"),
    ("usb-c", "USB-C"),
)
_MEMORY_TYPE_PATTERN = re.compile(r"\b(DDR[345])\b", re.IGNORECASE)


def normalize_search_text(raw_text: str) -> str:
    """Trim and collapse internal whitespace only — no semantic rewriting."""
    collapsed = _WHITESPACE_RUN.sub(" ", raw_text.strip())
    return collapsed


def normalize_capacity_token(raw: str) -> str | None:
    match = _CAPACITY_PATTERN.search(raw)
    if match is None:
        return None
    unit = match.group("unit").upper()
    num = match.group("num")
    if num.endswith(".0"):
        num = num[:-2]
    return f"{num}{unit}"


def normalize_interface_token(raw: str) -> str | None:
    lowered = raw.casefold()
    for needle, canonical in _INTERFACE_TOKENS:
        if needle in lowered:
            return canonical
    return None


def normalize_memory_type_token(raw: str) -> str | None:
    match = _MEMORY_TYPE_PATTERN.search(raw)
    if match is None:
        return None
    return match.group(1).upper()


def normalize_ecc_required(raw: str) -> bool | None:
    lowered = raw.casefold()
    if re.search(r"\bwithout\s+ecc\b", lowered) or re.search(r"\bno\s+ecc\b", lowered):
        return None
    if re.search(r"\becc\b", lowered):
        return True
    return None


def normalize_identifier_value(
    identifier_type: ProductIdentifierType,
    raw_token: str,
) -> tuple[str, str]:
    """Return (normalized_value, normalization_rule). Empty normalized means invalid."""
    raw_stripped = raw_token.strip()
    normalized = normalize_exact_lookup_value(identifier_type, raw_stripped)
    if not normalized:
        return "", "invalid_for_type"
    if identifier_type is ProductIdentifierType.GTIN:
        rule = "gtin_exact_lookup"
    else:
        rule = "exact_lookup_trim"
    return normalized, rule
