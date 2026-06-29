# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p2.factories import create_brave_search_provider as _legacy_create_brave_search_provider

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.search_provider.brave.integration import (
    BRAVE_SEARCH_PROVIDER_PROVIDER_ID,
    BraveSearchProviderIntegration,
    BraveSearchProviderIntegrationConfig,
    BraveSearchProviderClient,
)

__all__ = [
    "create_brave_search_provider",
    "create_brave_search_provider_integration",
]


def create_brave_search_provider_integration(
    *,
    client: BraveSearchProviderClient | None = None,
    enabled: bool = False,
) -> BraveSearchProviderIntegration:
    """
    Build a contract-based Brave search provider integration.

    The legacy facade (create_brave_search_provider) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Brave search provider integration requires an injected client when enabled=True",
        )
    if client is not None:
        return BraveSearchProviderIntegration.from_client(client, enabled=enabled)
    return BraveSearchProviderIntegration.for_provider(
        provider_id=BRAVE_SEARCH_PROVIDER_PROVIDER_ID,
        display_name="Brave",
        config=BraveSearchProviderIntegrationConfig(enabled=enabled),
    )


def create_brave_search_provider(**kwargs: object) -> BraveSearchProviderIntegration:
    """Compatibility shim — constructs BraveSearchProviderIntegration from legacy runtime."""
    runtime = _legacy_create_brave_search_provider(**kwargs)
    if isinstance(runtime, BraveSearchProviderIntegration):
        return runtime
    return BraveSearchProviderIntegration.from_runtime(runtime)
