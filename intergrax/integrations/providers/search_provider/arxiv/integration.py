# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Arxiv search provider integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.search_provider import SearchProvider
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
    Single public Arxiv search provider entrypoint.

    Legacy catalog factory (create_arxiv_search_provider) delegates to this class.
    """

    config: ArxivSearchProviderIntegrationConfig = ArxivSearchProviderIntegrationConfig()
    _client: ArxivSearchProviderClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(
        cls,
        runtime: Any,
        *,
        enabled: bool = True,
    ) -> ArxivSearchProviderIntegration:
        integration = cls.for_provider(
            provider_id=ARXIV_SEARCH_PROVIDER_PROVIDER_ID,
            display_name="Arxiv",
            config=ArxivSearchProviderIntegrationConfig(enabled=enabled),
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

SearchProvider.register(ArxivSearchProviderIntegration)
