# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.rerank_provider import RerankProvider
from intergrax.integrations.providers.rerank_provider.cohere_rerank.adapter import CohereRerankProvider
from intergrax.integrations.providers.rerank_provider.cohere_rerank.config import CohereRerankIntegrationConfig


def create_cohere_rerank_provider(**config_overrides: object) -> RerankProvider:
    return CohereRerankProvider(CohereRerankIntegrationConfig.from_env(**config_overrides))
