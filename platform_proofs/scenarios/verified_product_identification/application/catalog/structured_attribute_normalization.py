"""Scenario-owned structured attribute normalization policy."""

from __future__ import annotations

import re
from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.domain.json_value import (
    JsonValue,
)

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class DefaultStructuredAttributeNormalizationPolicy:
    """Minimal normalization — no ontology mapping in v1."""

    def canonical_key(self, *, source_key: str, source_field: str) -> str | None:
        del source_field
        stripped = source_key.strip()
        return stripped if stripped else None

    def normalized_text_value(self, *, source_value: str) -> str:
        return _WHITESPACE_RE.sub(" ", source_value.strip())

    def typed_value(self, *, raw_value: JsonValue) -> str | int | float | bool | None:
        if raw_value is None:
            return None
        if isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, int):
            return raw_value
        if isinstance(raw_value, float):
            return raw_value
        if isinstance(raw_value, str):
            stripped = raw_value.strip()
            return stripped if stripped else None
        return None


_DEFAULT_POLICY = DefaultStructuredAttributeNormalizationPolicy()


def normalize_structured_query_attribute_name(attribute_name: str) -> str:
    """Deterministic query-side attribute key normalization (index/query invariant)."""
    stripped = attribute_name.strip()
    if not stripped:
        raise ValueError("Structured query attribute_name must be non-empty after normalization")
    return stripped


def normalize_structured_query_value(value: str) -> str:
    """Deterministic query-side value normalization (index/query invariant)."""
    normalized = _DEFAULT_POLICY.normalized_text_value(source_value=value)
    if not normalized:
        raise ValueError("Structured query value must be non-empty after normalization")
    return normalized
