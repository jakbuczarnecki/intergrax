# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.rerank_provider import RerankProvider
from intergrax.integrations.providers.rerank_provider.cohere_rerank.adapter import _CohereRerankProvider
from intergrax.integrations.providers.rerank_provider.cohere_rerank.config import CohereRerankIntegrationConfig


def create_cohere_rerank_provider(**config_overrides: object) -> CohereRerankRerankProviderIntegration:
    return CohereRerankRerankProviderIntegration.from_client(_CohereRerankProvider(CohereRerankIntegrationConfig.from_env(**config_overrides)))

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.rerank_provider.cohere_rerank.integration import (
    COHERE_RERANK_RERANK_PROVIDER_PROVIDER_ID,
    CohereRerankRerankProviderIntegration,
    CohereRerankRerankProviderIntegrationConfig,
    CohereRerankRerankProviderClient,
)


def create_cohere_rerank_rerank_provider_integration(
    *,
    client: CohereRerankRerankProviderClient | None = None,
    enabled: bool = False,
) -> CohereRerankRerankProviderIntegration:
    """
    Build a contract-based Cohere Rerank rerank provider integration.

    Compatibility shim — constructs Integration via from_store (create_cohere_rerank_provider) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Cohere Rerank rerank provider integration requires an injected client when enabled=True",
        )
    if client is not None:
        return CohereRerankRerankProviderIntegration.from_client(client, enabled=enabled)
    return CohereRerankRerankProviderIntegration.for_provider(
        provider_id=COHERE_RERANK_RERANK_PROVIDER_PROVIDER_ID,
        display_name="Cohere Rerank",
        config=CohereRerankRerankProviderIntegrationConfig(enabled=enabled),
    )
