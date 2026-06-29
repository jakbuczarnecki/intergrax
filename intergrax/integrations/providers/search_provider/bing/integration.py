# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bing search provider integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.search_provider import SearchProvider
from intergrax.runtime.integrations.categories.search import SearchProviderIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

BING_SEARCH_PROVIDER_PROVIDER_ID = "bing"


class BingSearchProviderIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Bing search provider integration."""

    pass


BingSearchProviderClient = SearchProvider

class BingSearchProviderIntegration(SearchProviderIntegrationContract):
    """
    Single public Bing search provider entrypoint.

    Legacy catalog factory (create_bing_integration) owns catalog behavior; legacy factories use from_client().
    """

    config: BingSearchProviderIntegrationConfig = BingSearchProviderIntegrationConfig()
    _client: BingSearchProviderClient | None = PrivateAttr(default=None)
    


    def search(self, query: str, *, limit: int = 10) -> list[dict[str, object]]:
        return self._require_client().search(query, limit=limit)


    def _require_client(self) -> SearchProvider:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


    @classmethod
    def from_client(
        cls,
        client: BingSearchProviderClient,
        *,
        enabled: bool = False,
    ) -> BingSearchProviderIntegration:
        integration = cls.for_provider(
            provider_id=BING_SEARCH_PROVIDER_PROVIDER_ID,
            display_name="Bing",
            config=BingSearchProviderIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> BingSearchProviderClient | None:
        return self._client

SearchProvider.register(BingSearchProviderIntegration)
