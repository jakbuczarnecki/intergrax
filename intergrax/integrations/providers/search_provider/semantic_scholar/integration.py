# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Semantic Scholar search provider integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.search_provider import SearchProvider
from intergrax.runtime.integrations.categories.search import SearchProviderIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

SEMANTIC_SCHOLAR_SEARCH_PROVIDER_PROVIDER_ID = "semantic_scholar"


class SemanticScholarSearchProviderIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Semantic Scholar search provider integration."""

    pass


@runtime_checkable
class SemanticScholarSearchProviderClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class SemanticScholarSearchProviderIntegration(SearchProviderIntegrationContract):
    """
    Single public Semantic Scholar search provider entrypoint.

    Legacy catalog factory (create_semantic_scholar_search_provider) delegates to this class.
    """

    config: SemanticScholarSearchProviderIntegrationConfig = SemanticScholarSearchProviderIntegrationConfig()
    _client: SemanticScholarSearchProviderClient | None = PrivateAttr(default=None)
    _runtime: Any | None = PrivateAttr(default=None)

    @classmethod
    def from_runtime(
        cls,
        runtime: Any,
        *,
        enabled: bool = True,
    ) -> SemanticScholarSearchProviderIntegration:
        integration = cls.for_provider(
            provider_id=SEMANTIC_SCHOLAR_SEARCH_PROVIDER_PROVIDER_ID,
            display_name="Semantic Scholar",
            config=SemanticScholarSearchProviderIntegrationConfig(enabled=enabled),
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
        client: SemanticScholarSearchProviderClient,
        *,
        enabled: bool = False,
    ) -> SemanticScholarSearchProviderIntegration:
        integration = cls.for_provider(
            provider_id=SEMANTIC_SCHOLAR_SEARCH_PROVIDER_PROVIDER_ID,
            display_name="Semantic Scholar",
            config=SemanticScholarSearchProviderIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> SemanticScholarSearchProviderClient | None:
        return self._client

SearchProvider.register(SemanticScholarSearchProviderIntegration)
