# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-owned vLLM embedding runtime binding (P2-002-B3)."""

from __future__ import annotations

from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider
from intergrax.rag.embedding.contracts.runtime_binding import (
    EmbeddingProviderRuntimeBindingContext,
    EmbeddingProviderRuntimeBinder,
)


class VllmEmbeddingProviderRuntimeBinder:
    """Construct VllmEmbeddingProvider when vLLM is selected."""

    def bind(self, context: EmbeddingProviderRuntimeBindingContext) -> EmbeddingProvider:
        from intergrax.rag.embedding.providers.vllm_embedding_provider import VllmEmbeddingProvider

        if context.model is not None:
            return VllmEmbeddingProvider(context.model)
        return VllmEmbeddingProvider()


VLLM_EMBEDDING_PROVIDER_RUNTIME_BINDER: EmbeddingProviderRuntimeBinder = (
    VllmEmbeddingProviderRuntimeBinder()
)

__all__ = ["VLLM_EMBEDDING_PROVIDER_RUNTIME_BINDER", "VllmEmbeddingProviderRuntimeBinder"]
