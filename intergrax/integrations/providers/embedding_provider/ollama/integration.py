# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Ollama embedding provider integration (P2-002-B2 catalog registration)."""

from __future__ import annotations

from intergrax.integrations.providers.embedding_provider.ollama.config import (
    OllamaEmbeddingProviderIntegrationConfig,
)
from intergrax.runtime.integrations.categories.embedding import EmbeddingProviderIntegrationContract

OLLAMA_EMBEDDING_PROVIDER_ID = "ollama"


class OllamaEmbeddingProviderIntegration(EmbeddingProviderIntegrationContract):
    """Catalog-only embedding provider boundary; runtime binding pending B3."""

    config: OllamaEmbeddingProviderIntegrationConfig = OllamaEmbeddingProviderIntegrationConfig()
