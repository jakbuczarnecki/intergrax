# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bing search provider integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.search_provider import SearchProvider
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
    Single public Bing search provider entrypoint.

    Legacy catalog factory (create_bing_integration) delegates to this class.
    """

    config: BingSearchProviderIntegrationConfig = BingSearchProviderIntegrationConfig()
    _client: _BingSearchProviderClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(
        cls,
        runtime: Any,
        *,
        enabled: bool = True,
    ) -> BingSearchProviderIntegration:
        integration = cls.for_provider(
            provider_id=BING_SEARCH_PROVIDER_PROVIDER_ID,
            display_name="Bing",
            config=BingSearchProviderIntegrationConfig(enabled=enabled),
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
        client: _BingSearchProviderClient,
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

SearchProvider.register(BingSearchProviderIntegration)
