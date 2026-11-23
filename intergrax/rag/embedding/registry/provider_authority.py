# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Embedding provider slug authority delegated to Integrations catalog."""

from __future__ import annotations

from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    UnknownIntegrationError,
)
from intergrax.integrations.registry.catalog import get_entry

_LAB_EMBEDDING_PROVIDER = "ollama"
_EMBEDDING_CATEGORY = IntegrationCategory.EMBEDDING_PROVIDER


def lab_embedding_provider_slug() -> str:
    """Named laboratory default slug; reachable only through an explicit LAB opt-in."""
    return _LAB_EMBEDDING_PROVIDER


def validate_embedding_provider_slug(slug: str) -> str:
    """Validate provider slug against Integrations catalog authority."""
    normalized = slug.strip().lower()
    if not normalized:
        raise ValueError("provider must be a non-empty string slug")
    try:
        entry = get_entry(normalized)
    except UnknownIntegrationError as exc:
        raise ValueError(
            f"unknown embedding provider slug {normalized!r}; slug is not registered in the Integrations catalog"
        ) from exc
    if _EMBEDDING_CATEGORY not in entry.categories:
        raise ValueError(
            f"integration slug {normalized!r} is not registered for category {_EMBEDDING_CATEGORY.value!r}"
        )
    return normalized


__all__ = [
    "lab_embedding_provider_slug",
    "validate_embedding_provider_slug",
]
