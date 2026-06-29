# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.search_provider import SearchProvider
from intergrax.integrations.providers.search_provider.google_places.config import GooglePlacesIntegrationConfig
from intergrax.integrations.providers.search_provider.google_places.opens import open_google_places_search_provider


def create_google_places_search_provider(**config_overrides: object) -> GooglePlacesSearchProviderIntegration:
    config = GooglePlacesIntegrationConfig.from_env(**config_overrides)
    return GooglePlacesSearchProviderIntegration.from_runtime(open_google_places_search_provider(config))

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.search_provider.google_places.integration import (
    GOOGLE_PLACES_SEARCH_PROVIDER_PROVIDER_ID,
    GooglePlacesSearchProviderIntegration,
    GooglePlacesSearchProviderIntegrationConfig,
    GooglePlacesSearchProviderClient,
)


def create_google_places_search_provider_integration(
    *,
    client: GooglePlacesSearchProviderClient | None = None,
    enabled: bool = False,
) -> GooglePlacesSearchProviderIntegration:
    """
    Build a contract-based Google Places search provider integration.

    The legacy facade (create_google_places_search_provider) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Google Places search provider integration requires an injected client when enabled=True",
        )
    if client is not None:
        return GooglePlacesSearchProviderIntegration.from_client(client, enabled=enabled)
    return GooglePlacesSearchProviderIntegration.for_provider(
        provider_id=GOOGLE_PLACES_SEARCH_PROVIDER_PROVIDER_ID,
        display_name="Google Places",
        config=GooglePlacesSearchProviderIntegrationConfig(enabled=enabled),
    )
