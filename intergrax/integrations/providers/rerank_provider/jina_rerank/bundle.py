# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.rerank_provider import RerankProvider
from intergrax.integrations.providers.rerank_provider.jina_rerank.adapter import JinaRerankProvider
from intergrax.integrations.providers.rerank_provider.jina_rerank.config import JinaRerankIntegrationConfig


def create_jina_rerank_provider(**config_overrides: object) -> RerankProvider:
    return JinaRerankProvider(JinaRerankIntegrationConfig.from_env(**config_overrides))

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.rerank_provider.jina_rerank.integration import (
    JINA_RERANK_RERANK_PROVIDER_PROVIDER_ID,
    JinaRerankRerankProviderIntegration,
    JinaRerankRerankProviderIntegrationConfig,
    JinaRerankRerankProviderClient,
)


def create_jina_rerank_rerank_provider_integration(
    *,
    client: JinaRerankRerankProviderClient | None = None,
    enabled: bool = False,
) -> JinaRerankRerankProviderIntegration:
    """
    Build a contract-based Jina Rerank rerank provider integration.

    The legacy facade (create_jina_rerank_provider) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Jina Rerank rerank provider integration requires an injected client when enabled=True",
        )
    if client is not None:
        return JinaRerankRerankProviderIntegration.from_client(client, enabled=enabled)
    return JinaRerankRerankProviderIntegration.for_provider(
        provider_id=JINA_RERANK_RERANK_PROVIDER_PROVIDER_ID,
        display_name="Jina Rerank",
        config=JinaRerankRerankProviderIntegrationConfig(enabled=enabled),
    )
