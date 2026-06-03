# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register integrations from manifests or plugin classes."""

from __future__ import annotations

from typing import Any

from intergrax.integrations.contracts.base import IntegrationEntry, IntegrationFactory
from intergrax.integrations.core.manifest import IntegrationManifest
from intergrax.integrations.core.plugin import IntegrationPlugin, integration_manifest_for_plugin
from intergrax.integrations.registry.catalog import register_integration


def register_from_manifest(
    manifest: IntegrationManifest,
    factory: IntegrationFactory,
    *,
    override: bool = False,
) -> IntegrationManifest:
    """Register catalog row from manifest + factory; returns manifest for Tier-3 imports."""
    register_integration(
        IntegrationEntry(
            slug=manifest.slug,
            categories=manifest.categories,
            factory=factory,
            status=manifest.status,
            env_prefix=manifest.env_prefix,
            description=manifest.description,
        ),
        override=override,
    )
    return manifest


def register_integration_plugin(
    plugin: type[IntegrationPlugin],
    *,
    override: bool = False,
) -> IntegrationManifest:
    """Register catalog row from an :class:`IntegrationPlugin` implementation."""

    def _factory(**kwargs: Any) -> Any:
        return plugin.create_integration(**kwargs)

    manifest = integration_manifest_for_plugin(plugin)
    return register_from_manifest(manifest, _factory, override=override)
