# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog factories for Ollama embedding provider."""

from __future__ import annotations

from intergrax.integrations.providers.embedding_provider.ollama.config import (
    OllamaEmbeddingProviderIntegrationConfig,
)
from intergrax.integrations.providers.embedding_provider.ollama.integration import (
    OLLAMA_EMBEDDING_PROVIDER_ID,
    OllamaEmbeddingProviderIntegration,
)

__all__ = [
    "create_ollama_embedding_provider",
    "create_ollama_embedding_provider_integration",
]


def create_ollama_embedding_provider_integration(
    *,
    enabled: bool = False,
) -> OllamaEmbeddingProviderIntegration:
    return OllamaEmbeddingProviderIntegration.for_provider(
        provider_id=OLLAMA_EMBEDDING_PROVIDER_ID,
        display_name="Ollama",
        config=OllamaEmbeddingProviderIntegrationConfig(enabled=enabled),
    )


def create_ollama_embedding_provider(**config_overrides: object) -> OllamaEmbeddingProviderIntegration:
    enabled = bool(config_overrides.get("enabled", False))
    return create_ollama_embedding_provider_integration(enabled=enabled)
