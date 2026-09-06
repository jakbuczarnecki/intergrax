# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog factories for OpenAI embedding provider."""

from __future__ import annotations

from intergrax.integrations.providers.embedding_provider.openai.config import (
    OpenaiEmbeddingProviderIntegrationConfig,
)
from intergrax.integrations.providers.embedding_provider.openai.integration import (
    OPENAI_EMBEDDING_PROVIDER_ID,
    OpenaiEmbeddingProviderIntegration,
)

__all__ = [
    "create_openai_embedding_provider",
    "create_openai_embedding_provider_integration",
]


def create_openai_embedding_provider_integration(
    *,
    enabled: bool = False,
) -> OpenaiEmbeddingProviderIntegration:
    return OpenaiEmbeddingProviderIntegration.for_provider(
        provider_id=OPENAI_EMBEDDING_PROVIDER_ID,
        display_name="OpenAI",
        config=OpenaiEmbeddingProviderIntegrationConfig(enabled=enabled),
    )


def create_openai_embedding_provider(**config_overrides: object) -> OpenaiEmbeddingProviderIntegration:
    enabled = bool(config_overrides.get("enabled", False))
    return create_openai_embedding_provider_integration(enabled=enabled)
