# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p8.factories import create_perplexity_search_provider

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.search_provider.perplexity.integration import (
    PERPLEXITY_SEARCH_PROVIDER_PROVIDER_ID,
    PerplexitySearchProviderIntegration,
    PerplexitySearchProviderIntegrationConfig,
    PerplexitySearchProviderClient,
)

__all__ = [
    "create_perplexity_search_provider",
    "create_perplexity_search_provider_integration",
]


def create_perplexity_search_provider_integration(
    *,
    client: PerplexitySearchProviderClient | None = None,
    enabled: bool = False,
) -> PerplexitySearchProviderIntegration:
    """
    Build a contract-based Perplexity search provider integration.

    The legacy facade (create_perplexity_search_provider) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Perplexity search provider integration requires an injected client when enabled=True",
        )
    if client is not None:
        return PerplexitySearchProviderIntegration.from_client(client, enabled=enabled)
    return PerplexitySearchProviderIntegration.for_provider(
        provider_id=PERPLEXITY_SEARCH_PROVIDER_PROVIDER_ID,
        display_name="Perplexity",
        config=PerplexitySearchProviderIntegrationConfig(enabled=enabled),
    )
