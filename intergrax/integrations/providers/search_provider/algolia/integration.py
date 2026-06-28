# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Algolia search provider integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.search import SearchProviderIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

ALGOLIA_SEARCH_PROVIDER_PROVIDER_ID = "algolia"


class AlgoliaSearchProviderIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Algolia search provider integration."""

    pass


@runtime_checkable
class AlgoliaSearchProviderClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class AlgoliaSearchProviderIntegration(SearchProviderIntegrationContract):
    """
    Algolia search provider integration.

    The legacy facade (create_algolia_search_provider) remains separate and backward-compatible.
    """

    config: AlgoliaSearchProviderIntegrationConfig = AlgoliaSearchProviderIntegrationConfig()
    _client: AlgoliaSearchProviderClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: AlgoliaSearchProviderClient,
        *,
        enabled: bool = False,
    ) -> AlgoliaSearchProviderIntegration:
        integration = cls.for_provider(
            provider_id=ALGOLIA_SEARCH_PROVIDER_PROVIDER_ID,
            display_name="Algolia",
            config=AlgoliaSearchProviderIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> AlgoliaSearchProviderClient | None:
        return self._client
