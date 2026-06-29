# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Reddit search provider integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.search_provider import SearchProvider
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
    Single public Reddit search provider entrypoint.

    Legacy catalog factory (create_reddit_search_provider) delegates to this class.
    """

    config: RedditSearchProviderIntegrationConfig = RedditSearchProviderIntegrationConfig()
    _client: RedditSearchProviderClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(
        cls,
        runtime: Any,
        *,
        enabled: bool = True,
    ) -> RedditSearchProviderIntegration:
        integration = cls.for_provider(
            provider_id=REDDIT_SEARCH_PROVIDER_PROVIDER_ID,
            display_name="Reddit",
            config=RedditSearchProviderIntegrationConfig(enabled=enabled),
        )
        integration._runtime = runtime
        return integration


    def search(self, query: str, *, limit: int = 10) -> list[dict[str, object]]:
        return self._require_runtime().search(query, limit=limit)


    def _require_runtime(self) -> Any:
        private = object.__getattribute__(self, "__pydantic_private__")
        runtime = private.get("_runtime")
        if runtime is None:
            runtime = private.get("_backend")
        if runtime is None:
            runtime = private.get("_inner")
        if runtime is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a runtime delegate for catalog operations",
            )
        return runtime


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

SearchProvider.register(RedditSearchProviderIntegration)
