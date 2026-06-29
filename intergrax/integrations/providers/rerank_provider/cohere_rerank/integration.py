# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Cohere Rerank rerank provider integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""

from __future__ import annotations

from typing import Sequence

from pydantic import PrivateAttr

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.rerank_provider import RerankProvider
from intergrax.runtime.integrations.categories.search import RerankProviderIntegrationContract
from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig

COHERE_RERANK_RERANK_PROVIDER_PROVIDER_ID = "cohere_rerank"


class CohereRerankRerankProviderIntegrationConfig(CategoryIntegrationConfig):
    """Typed config for Cohere Rerank rerank provider integration."""

    pass


CohereRerankRerankProviderClient = RerankProvider

class CohereRerankRerankProviderIntegration(RerankProviderIntegrationContract):
    """
    Single public Cohere Rerank rerank provider entrypoint.

    Legacy catalog factory (create_cohere_rerank_provider) owns catalog behavior; legacy factories use from_client().
    """

    config: CohereRerankRerankProviderIntegrationConfig = CohereRerankRerankProviderIntegrationConfig()
    _client: CohereRerankRerankProviderClient | None = PrivateAttr(default=None)
    

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

