# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p3.factories import create_tavily_search_provider as _legacy_create_tavily_search_provider

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.search_provider.tavily.integration import (
    TAVILY_SEARCH_PROVIDER_PROVIDER_ID,
    TavilySearchProviderIntegration,
    TavilySearchProviderIntegrationConfig,
    TavilySearchProviderClient,
)

__all__ = [
    "create_tavily_search_provider",
    "create_tavily_search_provider_integration",
]


def create_tavily_search_provider_integration(
    *,
    client: TavilySearchProviderClient | None = None,
    enabled: bool = False,
) -> TavilySearchProviderIntegration:
    """
    Build a contract-based Tavily search provider integration.

    The legacy facade (create_tavily_search_provider) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Tavily search provider integration requires an injected client when enabled=True",
        )
    if client is not None:
        return TavilySearchProviderIntegration.from_client(client, enabled=enabled)
    return TavilySearchProviderIntegration.for_provider(
        provider_id=TAVILY_SEARCH_PROVIDER_PROVIDER_ID,
        display_name="Tavily",
        config=TavilySearchProviderIntegrationConfig(enabled=enabled),
    )


def create_tavily_search_provider(**kwargs: object) -> TavilySearchProviderIntegration:
    """Compatibility shim — constructs TavilySearchProviderIntegration from legacy runtime."""
    runtime = _legacy_create_tavily_search_provider(**kwargs)
    if isinstance(runtime, TavilySearchProviderIntegration):
        return runtime
    return TavilySearchProviderIntegration.from_runtime(runtime)
