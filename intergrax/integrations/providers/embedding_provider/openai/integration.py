# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""OpenAI embedding provider integration (P2-002-B2 catalog registration)."""

from __future__ import annotations

from intergrax.integrations.providers.embedding_provider.openai.config import (
    OpenaiEmbeddingProviderIntegrationConfig,
)
from intergrax.runtime.integrations.categories.embedding import EmbeddingProviderIntegrationContract

OPENAI_EMBEDDING_PROVIDER_ID = "openai"


class OpenaiEmbeddingProviderIntegration(EmbeddingProviderIntegrationContract):
    """Catalog-only embedding provider boundary; runtime binding pending B3."""

    config: OpenaiEmbeddingProviderIntegrationConfig = OpenaiEmbeddingProviderIntegrationConfig()
