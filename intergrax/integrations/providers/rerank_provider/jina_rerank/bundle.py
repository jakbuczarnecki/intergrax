# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.rerank_provider import RerankProvider
from intergrax.integrations.providers.rerank_provider.jina_rerank.adapter import JinaRerankProvider
from intergrax.integrations.providers.rerank_provider.jina_rerank.config import JinaRerankIntegrationConfig


def create_jina_rerank_provider(**config_overrides: object) -> RerankProvider:
    return JinaRerankProvider(JinaRerankIntegrationConfig.from_env(**config_overrides))
