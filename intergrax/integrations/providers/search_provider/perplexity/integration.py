# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Perplexity search provider integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.search_provider import SearchProvider
from intergrax.runtime.integrations.categories.search import SearchProviderIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

PERPLEXITY_SEARCH_PROVIDER_PROVIDER_ID = "perplexity"


class PerplexitySearchProviderIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Perplexity search provider integration."""

    pass


PerplexitySearchProviderClient = SearchProvider

class PerplexitySearchProviderIntegration(SearchProviderIntegrationContract):
    """
    Single public Perplexity search provider entrypoint.

    Legacy catalog factory (create_perplexity_search_provider) owns catalog behavior; legacy factories use from_client().
    """

    config: PerplexitySearchProviderIntegrationConfig = PerplexitySearchProviderIntegrationConfig()
    _client: PerplexitySearchProviderClient | None = PrivateAttr(default=None)
    


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
        client: PerplexitySearchProviderClient,
        *,
        enabled: bool = False,
    ) -> PerplexitySearchProviderIntegration:
        integration = cls.for_provider(
            provider_id=PERPLEXITY_SEARCH_PROVIDER_PROVIDER_ID,
            display_name="Perplexity",
            config=PerplexitySearchProviderIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> PerplexitySearchProviderClient | None:
        return self._client

SearchProvider.register(PerplexitySearchProviderIntegration)
