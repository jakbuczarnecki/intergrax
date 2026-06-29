# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Jina Rerank rerank provider integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.rerank_provider import RerankProvider
from intergrax.runtime.integrations.categories.search import RerankProviderIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

JINA_RERANK_RERANK_PROVIDER_PROVIDER_ID = "jina_rerank"


class JinaRerankRerankProviderIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Jina Rerank rerank provider integration."""

    pass


JinaRerankRerankProviderClient = RerankProvider

class JinaRerankRerankProviderIntegration(RerankProviderIntegrationContract):
    """
    Single public Jina Rerank rerank provider entrypoint.

    Legacy catalog factory (create_jina_rerank_provider) owns catalog behavior; legacy factories use from_client().
    """

    config: JinaRerankRerankProviderIntegrationConfig = JinaRerankRerankProviderIntegrationConfig()
    _client: JinaRerankRerankProviderClient | None = PrivateAttr(default=None)
    

    def name(self):
        return self._require_client().name()

    def rerank(self, query, documents, top_n: int | None = None):
        return self._require_client().rerank(query, documents, top_n=top_n)

    def _require_client(self) -> RerankProvider:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a catalog client for operations",
            )
        return self._client


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

