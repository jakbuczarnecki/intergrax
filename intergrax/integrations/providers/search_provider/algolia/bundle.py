# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p7.factories import create_algolia_search_provider as _legacy_create_algolia_search_provider

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.search_provider.algolia.integration import (
    ALGOLIA_SEARCH_PROVIDER_PROVIDER_ID,
    AlgoliaSearchProviderIntegration,
    AlgoliaSearchProviderIntegrationConfig,
    AlgoliaSearchProviderClient,
)

__all__ = [
    "create_algolia_search_provider",
    "create_algolia_search_provider_integration",
]


def create_algolia_search_provider_integration(
    *,
    client: AlgoliaSearchProviderClient | None = None,
    enabled: bool = False,
) -> AlgoliaSearchProviderIntegration:
    """
    Build a contract-based Algolia search provider integration.

    The legacy facade (create_algolia_search_provider) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Algolia search provider integration requires an injected client when enabled=True",
        )
    if client is not None:
        return AlgoliaSearchProviderIntegration.from_client(client, enabled=enabled)
    return AlgoliaSearchProviderIntegration.for_provider(
        provider_id=ALGOLIA_SEARCH_PROVIDER_PROVIDER_ID,
        display_name="Algolia",
        config=AlgoliaSearchProviderIntegrationConfig(enabled=enabled),
    )


def create_algolia_search_provider(**kwargs: object) -> AlgoliaSearchProviderIntegration:
    """Compatibility shim — constructs AlgoliaSearchProviderIntegration from legacy runtime."""
    runtime = _legacy_create_algolia_search_provider(**kwargs)
    if isinstance(runtime, AlgoliaSearchProviderIntegration):
        return runtime
    return AlgoliaSearchProviderIntegration.from_runtime(runtime)
