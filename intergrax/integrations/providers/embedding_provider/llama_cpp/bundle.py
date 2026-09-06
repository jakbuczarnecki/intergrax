# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog factories for llama.cpp embedding provider."""

from __future__ import annotations

from intergrax.integrations.providers.embedding_provider.llama_cpp.config import (
    LlamaCppEmbeddingProviderIntegrationConfig,
)
from intergrax.integrations.providers.embedding_provider.llama_cpp.integration import (
    LLAMA_CPP_EMBEDDING_PROVIDER_ID,
    LlamaCppEmbeddingProviderIntegration,
)

__all__ = [
    "create_llama_cpp_embedding_provider",
    "create_llama_cpp_embedding_provider_integration",
]


def create_llama_cpp_embedding_provider_integration(
    *,
    enabled: bool = False,
) -> LlamaCppEmbeddingProviderIntegration:
    return LlamaCppEmbeddingProviderIntegration.for_provider(
        provider_id=LLAMA_CPP_EMBEDDING_PROVIDER_ID,
        display_name="llama.cpp",
        config=LlamaCppEmbeddingProviderIntegrationConfig(enabled=enabled),
    )


def create_llama_cpp_embedding_provider(**config_overrides: object) -> LlamaCppEmbeddingProviderIntegration:
    enabled = bool(config_overrides.get("enabled", False))
    return create_llama_cpp_embedding_provider_integration(enabled=enabled)
