# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register integrations from manifests or plugin classes."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from intergrax.integrations.contracts.base import IntegrationEntry, IntegrationFactory
from intergrax.integrations.core.manifest import IntegrationManifest
from intergrax.integrations.core.plugin import IntegrationPlugin, integration_manifest_for_plugin
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.providers.layout import SLUG_CATEGORY
from intergrax.integrations.registry.contract_spec import (
    B1_TYPED_CONTRACT_CATEGORIES,
    EXPLICIT_CONTRACT_SPEC_PROVIDER_KEYS,
    manifest_category_values,
    validate_contract_specs_against_manifest,
)

if TYPE_CHECKING:
    from intergrax.integrations.registry.contract_spec import IntegrationContractSpec


def _required_explicit_categories(manifest: IntegrationManifest) -> frozenset[str]:
    """Categories that must supply provider-owned contract specs (fail-closed)."""
    slug = manifest.slug.strip().lower()
    manifest_categories = manifest_category_values(manifest)
    required: set[str] = set()
    if slug in SLUG_CATEGORY:
        required.update(manifest_categories & B1_TYPED_CONTRACT_CATEGORIES)
    for provider_id, category in EXPLICIT_CONTRACT_SPEC_PROVIDER_KEYS:
        if provider_id == slug and category in manifest_categories:
            required.add(category)
    return frozenset(required)


def _resolve_contract_specs(
    manifest: IntegrationManifest,
    explicit: Iterable[IntegrationContractSpec] | None,
) -> tuple[IntegrationContractSpec, ...]:
    """Resolve canonical contract specs from explicit rows or transitional built-in capture."""
    if explicit is not None:
        specs = tuple(explicit)
        validate_contract_specs_against_manifest(manifest, specs)
        return specs

    slug = manifest.slug.strip().lower()
    required_categories = _required_explicit_categories(manifest)
    if required_categories:
        categories = ", ".join(sorted(required_categories))
        msg = (
            f"Integration {slug!r} requires explicit contract_specs for typed categories: "
            f"{categories}"
        )
        raise ValueError(msg)

    from intergrax.integrations.registry.contract_capture import capture_builtin_contract_specs

    return capture_builtin_contract_specs(manifest)


def register_from_manifest(
    manifest: IntegrationManifest,
    factory: IntegrationFactory,
    *,
    override: bool = False,
    contract_specs: Iterable[IntegrationContractSpec] | None = None,
) -> IntegrationManifest:
    """Register catalog row from manifest + factory; returns manifest for Tier-3 imports."""
    resolved_specs = _resolve_contract_specs(manifest, contract_specs)
    register_integration(
        IntegrationEntry(
            slug=manifest.slug,
            categories=manifest.categories,
            factory=factory,
            status=manifest.status,
            env_prefix=manifest.env_prefix,
            description=manifest.description,
            requires_local_container=manifest.requires_local_container,
            contract_specs=resolved_specs,
        ),
        override=override,
    )
    return manifest


def register_integration_plugin(
    plugin: type[IntegrationPlugin],
    *,
    override: bool = False,
    contract_specs: Iterable[IntegrationContractSpec] | None = None,
) -> IntegrationManifest:
    """Register catalog row from an :class:`IntegrationPlugin` implementation."""

    def _factory(**kwargs: Any) -> Any:
        return plugin.create_integration(**kwargs)

    manifest = integration_manifest_for_plugin(plugin)
    return register_from_manifest(manifest, _factory, override=override, contract_specs=contract_specs)
