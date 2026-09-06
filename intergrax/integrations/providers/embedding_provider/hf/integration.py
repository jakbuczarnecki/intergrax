# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""HuggingFace embedding provider integration (P2-002-B2 catalog registration)."""

from __future__ import annotations

from intergrax.integrations.providers.embedding_provider.hf.config import (
    HfEmbeddingProviderIntegrationConfig,
)
from intergrax.runtime.integrations.categories.embedding import EmbeddingProviderIntegrationContract

HF_EMBEDDING_PROVIDER_ID = "hf"


class HfEmbeddingProviderIntegration(EmbeddingProviderIntegrationContract):
    """Catalog-only embedding provider boundary; runtime binding pending B3."""

    config: HfEmbeddingProviderIntegrationConfig = HfEmbeddingProviderIntegrationConfig()
