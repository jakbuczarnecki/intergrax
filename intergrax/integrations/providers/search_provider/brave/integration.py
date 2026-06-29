# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Brave search provider integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.search_provider import SearchProvider
from intergrax.runtime.integrations.categories.search import SearchProviderIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

BRAVE_SEARCH_PROVIDER_PROVIDER_ID = "brave"


class BraveSearchProviderIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Brave search provider integration."""

    pass


BraveSearchProviderClient = SearchProvider

class BraveSearchProviderIntegration(SearchProviderIntegrationContract):
    """
    Single public Brave search provider entrypoint.

    Legacy catalog factory (create_brave_search_provider) owns catalog behavior; legacy factories use from_client().
    """

    config: BraveSearchProviderIntegrationConfig = BraveSearchProviderIntegrationConfig()
    _client: BraveSearchProviderClient | None = PrivateAttr(default=None)
    


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
        client: BraveSearchProviderClient,
        *,
        enabled: bool = False,
    ) -> BraveSearchProviderIntegration:
        integration = cls.for_provider(
            provider_id=BRAVE_SEARCH_PROVIDER_PROVIDER_ID,
            display_name="Brave",
            config=BraveSearchProviderIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> BraveSearchProviderClient | None:
        return self._client

SearchProvider.register(BraveSearchProviderIntegration)
