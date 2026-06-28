# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bing search provider integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.search import SearchProviderIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

BING_SEARCH_PROVIDER_PROVIDER_ID = "bing"


class BingSearchProviderIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Bing search provider integration."""

    pass


@runtime_checkable
class BingSearchProviderClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class BingSearchProviderIntegration(SearchProviderIntegrationContract):
    """
    Bing search provider integration.

    The legacy facade (create_bing_integration) remains separate and backward-compatible.
    """

    config: BingSearchProviderIntegrationConfig = BingSearchProviderIntegrationConfig()
    _client: BingSearchProviderClient | None = PrivateAttr(default=None)

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
