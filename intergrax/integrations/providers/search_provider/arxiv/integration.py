# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Arxiv search provider integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.search import SearchProviderIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

ARXIV_SEARCH_PROVIDER_PROVIDER_ID = "arxiv"


class ArxivSearchProviderIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Arxiv search provider integration."""

    pass


@runtime_checkable
class ArxivSearchProviderClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class ArxivSearchProviderIntegration(SearchProviderIntegrationContract):
    """
    Arxiv search provider integration.

    The legacy facade (create_arxiv_search_provider) remains separate and backward-compatible.
    """

    config: ArxivSearchProviderIntegrationConfig = ArxivSearchProviderIntegrationConfig()
    _client: ArxivSearchProviderClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: ArxivSearchProviderClient,
        *,
        enabled: bool = False,
    ) -> ArxivSearchProviderIntegration:
        integration = cls.for_provider(
            provider_id=ARXIV_SEARCH_PROVIDER_PROVIDER_ID,
            display_name="Arxiv",
            config=ArxivSearchProviderIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> ArxivSearchProviderClient | None:
        return self._client
