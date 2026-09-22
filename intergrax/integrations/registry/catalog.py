# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Integration slug catalog (Phase M.3)."""

from __future__ import annotations

import contextlib
import contextvars
import threading
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationEntry,
    IntegrationMetadata,
    UnknownIntegrationError,
    normalize_category,
)

if TYPE_CHECKING:
    from intergrax.integrations.registry.contract_spec import IntegrationContractSpec

from intergrax.contracts.integration_catalog_revision import CatalogRevision
from intergrax.integrations.registry.catalog_revision import compute_catalog_state_digest

_CATALOG: dict[str, IntegrationEntry] = {}
_CATALOG_GENERATION = 0
_CATALOG_STATE_LOCK = threading.RLock()
_REGISTRATION_SINK: contextvars.ContextVar[dict[str, IntegrationEntry] | None] = (
    contextvars.ContextVar("integration_catalog_registration_sink", default=None)
)


@contextlib.contextmanager
def catalog_state_lock():
    with _CATALOG_STATE_LOCK:
        yield


@contextlib.contextmanager
def isolated_catalog_registration(sink: dict[str, IntegrationEntry]):
    token = _REGISTRATION_SINK.set(sink)
    try:
        yield
    finally:
        _REGISTRATION_SINK.reset(token)


def _catalog_entries_for_revision() -> dict[str, IntegrationEntry]:
    return dict(_CATALOG)


def _atomic_replace_catalog_entries(entries: dict[str, IntegrationEntry]) -> None:
    _CATALOG.clear()
    _CATALOG.update(entries)


def read_catalog_revision_under_lock() -> CatalogRevision:
    """Read authoritative revision; caller must hold ``catalog_state_lock``."""
    digest = compute_catalog_state_digest(_CATALOG)
    return CatalogRevision(generation=_CATALOG_GENERATION, state_digest=digest)


def set_catalog_generation_for_tests(generation: int) -> None:
    """Test helper — set generation under catalog lock (used with ``clear_catalog``)."""
    _assign_catalog_generation(generation)


def _assign_catalog_generation(generation: int) -> None:
    global _CATALOG_GENERATION
    _CATALOG_GENERATION = generation


def _apply_material_change_if_digest_differs(before_digest: str) -> None:
    global _CATALOG_GENERATION
    after_digest = compute_catalog_state_digest(_CATALOG)
    if after_digest != before_digest:
        _CATALOG_GENERATION += 1


def register_integration(entry: IntegrationEntry, *, override: bool = False) -> None:
    """Register or replace a provider factory (used by providers and tests)."""
    normalized_slug = entry.slug.strip().lower()
    sink = _REGISTRATION_SINK.get()
    target = sink if sink is not None else _CATALOG
    if normalized_slug in target and not override:
        raise ValueError(f"Integration slug '{normalized_slug}' is already registered.")
    normalized_entry = IntegrationEntry(
        slug=normalized_slug,
        categories=entry.categories,
        factory=entry.factory,
        status=entry.status,
        env_prefix=entry.env_prefix,
        description=entry.description,
        requires_local_container=entry.requires_local_container,
        contract_specs=entry.contract_specs,
    )
    if sink is not None:
        sink[normalized_slug] = normalized_entry
        return
    with _CATALOG_STATE_LOCK:
        before_digest = compute_catalog_state_digest(_CATALOG)
        _CATALOG[normalized_slug] = normalized_entry
        _apply_material_change_if_digest_differs(before_digest)


def unregister_integration(slug: str) -> None:
    sink = _REGISTRATION_SINK.get()
    if sink is not None:
        sink.pop(slug, None)
        return
    normalized = slug.strip().lower()
    with _CATALOG_STATE_LOCK:
        if normalized not in _CATALOG:
            return
        before_digest = compute_catalog_state_digest(_CATALOG)
        _CATALOG.pop(normalized, None)
        _apply_material_change_if_digest_differs(before_digest)


def clear_catalog() -> None:
    """Test helper — reset catalog and revision generation to initial state."""
    with _CATALOG_STATE_LOCK:
        _CATALOG.clear()
        _assign_catalog_generation(0)


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
