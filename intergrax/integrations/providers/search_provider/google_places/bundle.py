# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.search_provider import SearchProvider
from intergrax.integrations.providers.search_provider.google_places.config import GooglePlacesIntegrationConfig
from intergrax.integrations.providers.search_provider.google_places.opens import open_google_places_search_provider


def create_google_places_search_provider(**config_overrides: object) -> SearchProvider:
    config = GooglePlacesIntegrationConfig.from_env(**config_overrides)
    return open_google_places_search_provider(config)
