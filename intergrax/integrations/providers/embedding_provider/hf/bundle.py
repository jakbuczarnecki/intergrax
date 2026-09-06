# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog factories for HuggingFace embedding provider."""

from __future__ import annotations

from intergrax.integrations.providers.embedding_provider.hf.config import (
    HfEmbeddingProviderIntegrationConfig,
)
from intergrax.integrations.providers.embedding_provider.hf.integration import (
    HF_EMBEDDING_PROVIDER_ID,
    HfEmbeddingProviderIntegration,
)

__all__ = [
    "create_hf_embedding_provider",
    "create_hf_embedding_provider_integration",
]


def create_hf_embedding_provider_integration(
    *,
    enabled: bool = False,
) -> HfEmbeddingProviderIntegration:
    return HfEmbeddingProviderIntegration.for_provider(
        provider_id=HF_EMBEDDING_PROVIDER_ID,
        display_name="HuggingFace",
        config=HfEmbeddingProviderIntegrationConfig(enabled=enabled),
    )


def create_hf_embedding_provider(**config_overrides: object) -> HfEmbeddingProviderIntegration:
    enabled = bool(config_overrides.get("enabled", False))
    return create_hf_embedding_provider_integration(enabled=enabled)
