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
from intergrax.integrations.registry.contract_spec import (
    manifest_category_values,
    typed_contract_categories,
    validate_contract_specs_against_manifest,
    validate_required_explicit_categories,
)

if TYPE_CHECKING:
    from intergrax.integrations.registry.contract_spec import IntegrationContractSpec


def _required_explicit_categories(manifest: IntegrationManifest) -> frozenset[str]:
    """Categories that must supply provider-owned contract specs (fail-closed)."""
    manifest_categories = manifest_category_values(manifest)
    return frozenset(manifest_categories & typed_contract_categories())


def _resolve_contract_specs(
    manifest: IntegrationManifest,
    explicit: Iterable[IntegrationContractSpec] | None,
) -> tuple[IntegrationContractSpec, ...]:
    """Resolve canonical contract specs from explicit provider-owned declarations only."""
    slug = manifest.slug.strip().lower()
    required_categories = _required_explicit_categories(manifest)
    if explicit is not None:
        specs = tuple(explicit)
        validate_contract_specs_against_manifest(manifest, specs)
        validate_required_explicit_categories(manifest, specs, required_categories)
        return specs

    if required_categories:
        categories = ", ".join(sorted(required_categories))
        msg = (
            f"Integration {slug!r} requires explicit contract_specs for typed categories: "
            f"{categories}"
        )
        raise ValueError(msg)

    return ()


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
