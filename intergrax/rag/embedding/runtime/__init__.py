# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Canonical embedding provider runtime resolver (IntegrationProfile → EmbeddingProvider)."""

from intergrax.rag.embedding.runtime.resolver import (
    bind_embedding_provider,
    resolve_embedding_provider_slug,
)

from intergrax.rag.embedding.registry.provider_authority import validate_embedding_provider_slug

__all__ = [
    "bind_embedding_provider",
    "resolve_embedding_provider_slug",
    "validate_embedding_provider_slug",
]
