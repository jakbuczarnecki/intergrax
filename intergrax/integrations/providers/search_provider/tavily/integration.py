# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tavily search provider integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.search import SearchProviderIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

TAVILY_SEARCH_PROVIDER_PROVIDER_ID = "tavily"


class TavilySearchProviderIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Tavily search provider integration."""

    pass


@runtime_checkable
class TavilySearchProviderClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class TavilySearchProviderIntegration(SearchProviderIntegrationContract):
    """
    Tavily search provider integration.

    The legacy facade (create_tavily_search_provider) remains separate and backward-compatible.
    """

    config: TavilySearchProviderIntegrationConfig = TavilySearchProviderIntegrationConfig()
    _client: TavilySearchProviderClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: TavilySearchProviderClient,
        *,
        enabled: bool = False,
    ) -> TavilySearchProviderIntegration:
        integration = cls.for_provider(
            provider_id=TAVILY_SEARCH_PROVIDER_PROVIDER_ID,
            display_name="Tavily",
            config=TavilySearchProviderIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> TavilySearchProviderClient | None:
        return self._client
