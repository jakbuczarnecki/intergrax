# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p2.factories import create_serpapi_search_provider as _legacy_create_serpapi_search_provider

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.search_provider.serpapi.integration import (
    SERPAPI_SEARCH_PROVIDER_PROVIDER_ID,
    SerpapiSearchProviderIntegration,
    SerpapiSearchProviderIntegrationConfig,
    SerpapiSearchProviderClient,
)

__all__ = [
    "create_serpapi_search_provider",
    "create_serpapi_search_provider_integration",
]


def create_serpapi_search_provider_integration(
    *,
    client: SerpapiSearchProviderClient | None = None,
    enabled: bool = False,
) -> SerpapiSearchProviderIntegration:
    """
    Build a contract-based Serpapi search provider integration.

    The legacy facade (create_serpapi_search_provider) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Serpapi search provider integration requires an injected client when enabled=True",
        )
    if client is not None:
        return SerpapiSearchProviderIntegration.from_client(client, enabled=enabled)
    return SerpapiSearchProviderIntegration.for_provider(
        provider_id=SERPAPI_SEARCH_PROVIDER_PROVIDER_ID,
        display_name="Serpapi",
        config=SerpapiSearchProviderIntegrationConfig(enabled=enabled),
    )


def create_serpapi_search_provider(**kwargs: object) -> SerpapiSearchProviderIntegration:
    """Compatibility shim — constructs SerpapiSearchProviderIntegration from legacy runtime."""
    runtime = _legacy_create_serpapi_search_provider(**kwargs)
    if isinstance(runtime, SerpapiSearchProviderIntegration):
        return runtime
    return SerpapiSearchProviderIntegration.from_runtime(runtime)
