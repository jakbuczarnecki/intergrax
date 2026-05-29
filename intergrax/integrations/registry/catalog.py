# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Integration slug catalog (Phase M.3)."""

from __future__ import annotations

from typing import Iterator

from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationEntry,
    IntegrationMetadata,
    UnknownIntegrationError,
    normalize_category,
)

_CATALOG: dict[str, IntegrationEntry] = {}


def register_integration(entry: IntegrationEntry, *, override: bool = False) -> None:
    """Register or replace a provider factory (used by providers and tests)."""
    if entry.slug in _CATALOG and not override:
        raise ValueError(f"Integration slug '{entry.slug}' is already registered.")
    _CATALOG[entry.slug] = entry


def unregister_integration(slug: str) -> None:
    _CATALOG.pop(slug, None)


def clear_catalog() -> None:
    """Test helper — reset catalog to empty."""
    _CATALOG.clear()


def get_entry(slug: str) -> IntegrationEntry:
    normalized = slug.strip().lower()
    try:
        return _CATALOG[normalized]
    except KeyError as exc:
        raise UnknownIntegrationError(normalized) from exc


def iter_entries() -> Iterator[IntegrationEntry]:
    yield from _CATALOG.values()


def list_slugs(*, category: str | IntegrationCategory | None = None) -> list[str]:
    if category is None:
        return sorted(_CATALOG)
    normalized = normalize_category(category)
    return sorted(
        entry.slug
        for entry in _CATALOG.values()
        if normalized in entry.categories
    )


def metadata_for_slug(slug: str) -> IntegrationMetadata:
    return get_entry(slug).metadata


def catalog_snapshot() -> dict[str, IntegrationEntry]:
    return dict(_CATALOG)
