# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""vLLM embedding provider integration (P2-002-B2 catalog registration)."""

from __future__ import annotations

from intergrax.integrations.providers.embedding_provider.vllm.config import (
    VllmEmbeddingProviderIntegrationConfig,
)
from intergrax.runtime.integrations.categories.embedding import EmbeddingProviderIntegrationContract

VLLM_EMBEDDING_PROVIDER_ID = "vllm"


class VllmEmbeddingProviderIntegration(EmbeddingProviderIntegrationContract):
    """Catalog-only embedding provider boundary; runtime binding pending B3."""

    config: VllmEmbeddingProviderIntegrationConfig = VllmEmbeddingProviderIntegrationConfig()
