# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Cohere Rerank rerank provider integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.search import RerankProviderIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

COHERE_RERANK_RERANK_PROVIDER_PROVIDER_ID = "cohere_rerank"


class CohereRerankRerankProviderIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Cohere Rerank rerank provider integration."""

    pass


@runtime_checkable
class CohereRerankRerankProviderClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class CohereRerankRerankProviderIntegration(RerankProviderIntegrationContract):
    """
    Cohere Rerank rerank provider integration.

    The legacy facade (create_cohere_rerank_provider) remains separate and backward-compatible.
    """

    config: CohereRerankRerankProviderIntegrationConfig = CohereRerankRerankProviderIntegrationConfig()
    _client: CohereRerankRerankProviderClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: CohereRerankRerankProviderClient,
        *,
        enabled: bool = False,
    ) -> CohereRerankRerankProviderIntegration:
        integration = cls.for_provider(
            provider_id=COHERE_RERANK_RERANK_PROVIDER_PROVIDER_ID,
            display_name="Cohere Rerank",
            config=CohereRerankRerankProviderIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> CohereRerankRerankProviderClient | None:
        return self._client
