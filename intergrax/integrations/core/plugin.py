# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Integration plugin protocol — explicit type + factory (§7.1.4)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from intergrax.integrations.core.manifest import IntegrationManifest


@runtime_checkable
class IntegrationPlugin(Protocol):
    """
    Optional class-based registration for custom integrations.

    Implement on a class and pass the class to :class:`IntegrationProfile` or
    :func:`register_integration_plugin`.
    """

    @classmethod
    def integration_manifest(cls) -> IntegrationManifest:
        """Catalog identity for this provider."""

    @classmethod
    def create_integration(cls, **kwargs: Any) -> Any:
        """Factory invoked by :func:`intergrax.integrations.registry.factory.resolve`."""


def integration_manifest_for_plugin(plugin_type: type[IntegrationPlugin]) -> IntegrationManifest:
    manifest = plugin_type.integration_manifest()
    if not isinstance(manifest, IntegrationManifest):
        raise TypeError(
            f"{plugin_type.__qualname__}.integration_manifest() must return IntegrationManifest"
        )
    return manifest
