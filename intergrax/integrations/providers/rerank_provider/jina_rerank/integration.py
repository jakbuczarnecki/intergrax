# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Jina Rerank rerank provider integration (INTEGRATIONS-2D)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import PrivateAttr

from intergrax.runtime.integrations.categories.search import RerankProviderIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

JINA_RERANK_RERANK_PROVIDER_PROVIDER_ID = "jina_rerank"


class JinaRerankRerankProviderIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Jina Rerank rerank provider integration."""

    pass


@runtime_checkable
class JinaRerankRerankProviderClient(Protocol):
    """Injectable client facade — no vendor SDK or network I/O in the integration class."""

    async def ping(self) -> None:
        """Lightweight connectivity check."""


class JinaRerankRerankProviderIntegration(RerankProviderIntegrationContract):
    """
    Jina Rerank rerank provider integration.

    The legacy facade (create_jina_rerank_provider) remains separate and backward-compatible.
    """

    config: JinaRerankRerankProviderIntegrationConfig = JinaRerankRerankProviderIntegrationConfig()
    _client: JinaRerankRerankProviderClient | None = PrivateAttr(default=None)

    @classmethod
    def from_client(
        cls,
        client: JinaRerankRerankProviderClient,
        *,
        enabled: bool = False,
    ) -> JinaRerankRerankProviderIntegration:
        integration = cls.for_provider(
            provider_id=JINA_RERANK_RERANK_PROVIDER_PROVIDER_ID,
            display_name="Jina Rerank",
            config=JinaRerankRerankProviderIntegrationConfig(enabled=enabled),
        )
        integration._client = client
        return integration

    @property
    def client(self) -> JinaRerankRerankProviderClient | None:
        return self._client
