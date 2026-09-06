# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-owned Ollama embedding runtime binding (P2-002-B3)."""

from __future__ import annotations

from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider
from intergrax.rag.embedding.contracts.runtime_binding import (
    EmbeddingProviderRuntimeBindingContext,
    EmbeddingProviderRuntimeBinder,
)


class OllamaEmbeddingProviderRuntimeBinder:
    """Construct OllamaEmbeddingProvider when Ollama is selected."""

    def bind(self, context: EmbeddingProviderRuntimeBindingContext) -> EmbeddingProvider:
        from intergrax.rag.embedding.providers.ollama_embedding_provider import (
            OllamaEmbeddingProvider,
        )

        if context.model is not None:
            return OllamaEmbeddingProvider(context.model)
        return OllamaEmbeddingProvider()


OLLAMA_EMBEDDING_PROVIDER_RUNTIME_BINDER: EmbeddingProviderRuntimeBinder = (
    OllamaEmbeddingProviderRuntimeBinder()
)

__all__ = ["OLLAMA_EMBEDDING_PROVIDER_RUNTIME_BINDER", "OllamaEmbeddingProviderRuntimeBinder"]
