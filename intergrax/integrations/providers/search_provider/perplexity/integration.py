# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Perplexity search provider integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.search import SearchProviderIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

PERPLEXITY_SEARCH_PROVIDER_PROVIDER_ID = "perplexity"


class PerplexitySearchProviderIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Perplexity search provider integration."""

    pass


@runtime_checkable
class PerplexitySearchProviderClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class PerplexitySearchProviderIntegration(SearchProviderIntegrationContract):
    """
    Perplexity search provider integration.

    The legacy facade (create_perplexity_search_provider) remains separate and backward-compatible.
    """

    config: PerplexitySearchProviderIntegrationConfig = PerplexitySearchProviderIntegrationConfig()
    _client: PerplexitySearchProviderClient | None = PrivateAttr(default=None)

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
