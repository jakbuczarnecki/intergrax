# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-owned llama.cpp embedding runtime binding (P2-002-B3)."""

from __future__ import annotations

from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider
from intergrax.rag.embedding.contracts.runtime_binding import (
    EmbeddingProviderRuntimeBindingContext,
    EmbeddingProviderRuntimeBinder,
)


class LlamaCppEmbeddingProviderRuntimeBinder:
    """Construct LlamaCppEmbeddingProvider when llama.cpp is selected."""

    def bind(self, context: EmbeddingProviderRuntimeBindingContext) -> EmbeddingProvider:
        from intergrax.rag.embedding.providers.llama_cpp_embedding_provider import (
            LlamaCppEmbeddingProvider,
        )

        if context.model is not None:
            return LlamaCppEmbeddingProvider(context.model)
        return LlamaCppEmbeddingProvider()


LLAMA_CPP_EMBEDDING_PROVIDER_RUNTIME_BINDER: EmbeddingProviderRuntimeBinder = (
    LlamaCppEmbeddingProviderRuntimeBinder()
)

__all__ = [
    "LLAMA_CPP_EMBEDDING_PROVIDER_RUNTIME_BINDER",
    "LlamaCppEmbeddingProviderRuntimeBinder",
]
