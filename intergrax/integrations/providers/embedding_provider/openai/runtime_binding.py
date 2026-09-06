# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-owned OpenAI embedding runtime binding (P2-002-B3)."""

from __future__ import annotations

from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider
from intergrax.rag.embedding.contracts.runtime_binding import (
    EmbeddingProviderRuntimeBindingContext,
    EmbeddingProviderRuntimeBinder,
)


class OpenaiEmbeddingProviderRuntimeBinder:
    """Construct OpenAIEmbeddingProvider when OpenAI is selected."""

    def bind(self, context: EmbeddingProviderRuntimeBindingContext) -> EmbeddingProvider:
        from intergrax.rag.embedding.providers.openai_embedding_provider import (
            OpenAIEmbeddingProvider,
        )

        if context.model is not None:
            return OpenAIEmbeddingProvider(context.model)
        return OpenAIEmbeddingProvider()


OPENAI_EMBEDDING_PROVIDER_RUNTIME_BINDER: EmbeddingProviderRuntimeBinder = (
    OpenaiEmbeddingProviderRuntimeBinder()
)

__all__ = ["OPENAI_EMBEDDING_PROVIDER_RUNTIME_BINDER", "OpenaiEmbeddingProviderRuntimeBinder"]
