# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Optional, Sequence

from intergrax.integrations.providers.search_provider.google_places.config import GooglePlacesIntegrationConfig
from intergrax.integrations.providers.search_provider.google_places.web_client import GooglePlacesProvider
from intergrax.websearch.schemas.query_spec import QuerySpec
from intergrax.websearch.schemas.search_hit import SearchHit


class GooglePlacesSearchProvider:
    def __init__(self, provider: GooglePlacesProvider) -> None:
        self._provider = provider

    @property
    def web_search_provider(self) -> GooglePlacesProvider:
        return self._provider

    def search(self, query: str, *, limit: int = 10) -> Sequence[SearchHit]:
        spec = QuerySpec(query=query, top_k=max(1, int(limit)))
        return self._provider.search(spec)


def open_google_places_web_search_provider(
    config: GooglePlacesIntegrationConfig,
    *,
    provider: Optional[GooglePlacesProvider] = None,
    session: Optional[object] = None,
) -> GooglePlacesProvider:
    if provider is not None:
        return provider
    timeout = int(config.timeout_seconds or GooglePlacesProvider._DEFAULT_TIMEOUT)
    return GooglePlacesProvider(api_key=config.api_key, session=session, timeout=timeout)


def open_google_places_search_provider(
    config: GooglePlacesIntegrationConfig,
    *,
    provider: Optional[GooglePlacesProvider] = None,
    session: Optional[object] = None,
) -> GooglePlacesSearchProvider:
    return GooglePlacesSearchProvider(
        open_google_places_web_search_provider(config, provider=provider, session=session)
    )
