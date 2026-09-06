# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""llama.cpp embedding provider integration (P2-002-B2 catalog registration)."""

from __future__ import annotations

from intergrax.integrations.providers.embedding_provider.llama_cpp.config import (
    LlamaCppEmbeddingProviderIntegrationConfig,
)
from intergrax.runtime.integrations.categories.embedding import EmbeddingProviderIntegrationContract

LLAMA_CPP_EMBEDDING_PROVIDER_ID = "llama_cpp"


class LlamaCppEmbeddingProviderIntegration(EmbeddingProviderIntegrationContract):
    """Catalog-only embedding provider boundary; runtime binding pending B3."""

    config: LlamaCppEmbeddingProviderIntegrationConfig = LlamaCppEmbeddingProviderIntegrationConfig()
