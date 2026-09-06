# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Integration slug catalog (Phase M.3)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Iterator

from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationEntry,
    IntegrationMetadata,
    UnknownIntegrationError,
    normalize_category,
)

if TYPE_CHECKING:
    from intergrax.integrations.registry.contract_spec import IntegrationContractSpec

_CATALOG: dict[str, IntegrationEntry] = {}


def register_integration(entry: IntegrationEntry, *, override: bool = False) -> None:
    """Register or replace a provider factory (used by providers and tests)."""
    normalized_slug = entry.slug.strip().lower()
    if normalized_slug in _CATALOG and not override:
        raise ValueError(f"Integration slug '{normalized_slug}' is already registered.")
    _CATALOG[normalized_slug] = IntegrationEntry(
        slug=normalized_slug,
        categories=entry.categories,
        factory=entry.factory,
        status=entry.status,
        env_prefix=entry.env_prefix,
        description=entry.description,
        requires_local_container=entry.requires_local_container,
        contract_specs=entry.contract_specs,
    )


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


def augment_integration_contract_specs(
    slug: str,
    *,
    categories: tuple[IntegrationCategory, ...] = (),
    contract_specs: Iterable[IntegrationContractSpec],
) -> IntegrationEntry:
    """Append provider-owned contract specs and categories to an existing catalog row."""
    from intergrax.integrations.core.manifest import IntegrationManifest
    from intergrax.integrations.registry.contract_spec import (
        validate_contract_spec_identity,
        validate_contract_specs_against_manifest,
    )

    normalized_slug = slug.strip().lower()
    entry = get_entry(normalized_slug)
    merged_categories: list[IntegrationCategory] = list(entry.categories)
    for category in categories:
        if category not in merged_categories:
            merged_categories.append(category)

    existing_categories = {spec.category for spec in entry.contract_specs}
    merged_specs = list(entry.contract_specs)
    for spec in contract_specs:
        validate_contract_spec_identity(
            slug=normalized_slug,
            spec=spec,
            observed_provider_id=spec.provider_id,
        )
        if spec.category in existing_categories:
            msg = (
                f"Integration {normalized_slug!r}: contract spec category "
                f"{spec.category!r} is already registered"
            )
            raise ValueError(msg)
        existing_categories.add(spec.category)
        merged_specs.append(spec)

    merged_manifest = IntegrationManifest(
        slug=normalized_slug,
        categories=tuple(merged_categories),
        status=entry.status,
        env_prefix=entry.env_prefix,
        description=entry.description,
        requires_local_container=entry.requires_local_container,
    )
    validate_contract_specs_against_manifest(merged_manifest, merged_specs)

    updated = IntegrationEntry(
        slug=entry.slug,
        categories=tuple(merged_categories),
        factory=entry.factory,
        status=entry.status,
        env_prefix=entry.env_prefix,
        description=entry.description,
        requires_local_container=entry.requires_local_container,
        contract_specs=tuple(merged_specs),
    )
    register_integration(updated, override=True)
    return updated
