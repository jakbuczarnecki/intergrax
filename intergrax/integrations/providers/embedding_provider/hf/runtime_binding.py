# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider-owned HuggingFace embedding runtime binding (P2-002-B3)."""

from __future__ import annotations

from intergrax.integrations.providers.embedding_provider.hf.config import (
    HfEmbeddingProviderIntegrationConfig,
)
from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider
from intergrax.rag.embedding.contracts.runtime_binding import (
    EmbeddingProviderRuntimeBindingContext,
    EmbeddingProviderRuntimeBinder,
)


class HfEmbeddingProviderRuntimeBinder:
    """Construct HFEmbeddingProvider when HuggingFace is selected."""

    def bind(self, context: EmbeddingProviderRuntimeBindingContext) -> EmbeddingProvider:
        from intergrax.rag.embedding.providers.hf_embedding_provider import HFEmbeddingProvider

        device = None
        batch_size = None
        if context.execution_config is not None:
            device = context.execution_config.device
            batch_size = context.execution_config.batch_size

        typed_options = HfEmbeddingProviderIntegrationConfig.model_validate(
            dict(context.integration_options),
        )
        if typed_options.device is not None:
            device = typed_options.device
        if typed_options.batch_size is not None:
            batch_size = typed_options.batch_size

        if batch_size is not None:
            return HFEmbeddingProvider(context.model, device=device, batch_size=batch_size)
        return HFEmbeddingProvider(context.model, device=device)


HF_EMBEDDING_PROVIDER_RUNTIME_BINDER: EmbeddingProviderRuntimeBinder = (
    HfEmbeddingProviderRuntimeBinder()
)

__all__ = ["HF_EMBEDDING_PROVIDER_RUNTIME_BINDER", "HfEmbeddingProviderRuntimeBinder"]
