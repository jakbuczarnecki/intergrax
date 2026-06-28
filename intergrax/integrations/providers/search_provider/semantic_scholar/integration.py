# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Semantic Scholar search provider integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

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
    Semantic Scholar search provider integration.

    The legacy facade (create_semantic_scholar_search_provider) remains separate and backward-compatible.
    """

    config: SemanticScholarSearchProviderIntegrationConfig = SemanticScholarSearchProviderIntegrationConfig()
    _client: SemanticScholarSearchProviderClient | None = PrivateAttr(default=None)

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
