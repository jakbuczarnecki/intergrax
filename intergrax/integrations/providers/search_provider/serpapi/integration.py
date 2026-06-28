# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Serpapi search provider integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.search import SearchProviderIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

SERPAPI_SEARCH_PROVIDER_PROVIDER_ID = "serpapi"


class SerpapiSearchProviderIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Serpapi search provider integration."""

    pass


@runtime_checkable
class SerpapiSearchProviderClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class SerpapiSearchProviderIntegration(SearchProviderIntegrationContract):
    """
    Serpapi search provider integration.

    The legacy facade (create_serpapi_search_provider) remains separate and backward-compatible.
    """

    config: SerpapiSearchProviderIntegrationConfig = SerpapiSearchProviderIntegrationConfig()
    _client: SerpapiSearchProviderClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: SerpapiSearchProviderClient,
        *,
        enabled: bool = False,
    ) -> SerpapiSearchProviderIntegration:
        integration = cls.for_provider(
            provider_id=SERPAPI_SEARCH_PROVIDER_PROVIDER_ID,
            display_name="Serpapi",
            config=SerpapiSearchProviderIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> SerpapiSearchProviderClient | None:
        return self._client
