# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Reddit search provider integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.search import SearchProviderIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

REDDIT_SEARCH_PROVIDER_PROVIDER_ID = "reddit"


class RedditSearchProviderIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Reddit search provider integration."""

    pass


@runtime_checkable
class RedditSearchProviderClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class RedditSearchProviderIntegration(SearchProviderIntegrationContract):
    """
    Reddit search provider integration.

    The legacy facade (create_reddit_search_provider) remains separate and backward-compatible.
    """

    config: RedditSearchProviderIntegrationConfig = RedditSearchProviderIntegrationConfig()
    _client: RedditSearchProviderClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: RedditSearchProviderClient,
        *,
        enabled: bool = False,
    ) -> RedditSearchProviderIntegration:
        integration = cls.for_provider(
            provider_id=REDDIT_SEARCH_PROVIDER_PROVIDER_ID,
            display_name="Reddit",
            config=RedditSearchProviderIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> RedditSearchProviderClient | None:
        return self._client
