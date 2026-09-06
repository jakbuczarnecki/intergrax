# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog factories for vLLM embedding provider."""

from __future__ import annotations

from intergrax.integrations.providers.embedding_provider.vllm.config import (
    VllmEmbeddingProviderIntegrationConfig,
)
from intergrax.integrations.providers.embedding_provider.vllm.integration import (
    VLLM_EMBEDDING_PROVIDER_ID,
    VllmEmbeddingProviderIntegration,
)

__all__ = [
    "create_vllm_embedding_provider",
    "create_vllm_embedding_provider_integration",
]


def create_vllm_embedding_provider_integration(
    *,
    enabled: bool = False,
) -> VllmEmbeddingProviderIntegration:
    return VllmEmbeddingProviderIntegration.for_provider(
        provider_id=VLLM_EMBEDDING_PROVIDER_ID,
        display_name="vLLM",
        config=VllmEmbeddingProviderIntegrationConfig(enabled=enabled),
    )


def create_vllm_embedding_provider(**config_overrides: object) -> VllmEmbeddingProviderIntegration:
    enabled = bool(config_overrides.get("enabled", False))
    return create_vllm_embedding_provider_integration(enabled=enabled)
