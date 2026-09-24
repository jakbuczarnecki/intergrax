# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Integration plugin protocol — explicit type + factory (§7.1.4)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from intergrax.integrations.contracts.catalog_factory import IntegrationFactoryConfigValue
from intergrax.integrations.contracts.manifest import IntegrationManifest

if TYPE_CHECKING:
    from intergrax.integrations.registry.contract_spec import IntegrationContractSpec
    from intergrax.runtime.integrations.contracts import PlatformIntegrationContract


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
    def integration_contract_specs(cls) -> tuple[IntegrationContractSpec, ...]:
        """
        Provider-owned contract declarations used when ``contract_specs`` is omitted
        at registration time. Return ``()`` only for categories that do not require
        typed contract specs.
        """

    @classmethod
    def create_integration(
        cls,
        **kwargs: IntegrationFactoryConfigValue,
    ) -> PlatformIntegrationContract:
        """Factory invoked by :func:`intergrax.integrations.registry.factory.resolve`."""


def integration_manifest_for_plugin(plugin_type: type[IntegrationPlugin]) -> IntegrationManifest:
    manifest = plugin_type.integration_manifest()
    if not isinstance(manifest, IntegrationManifest):
        raise TypeError(
            f"{plugin_type.__qualname__}.integration_manifest() must return IntegrationManifest"
        )
    return manifest
