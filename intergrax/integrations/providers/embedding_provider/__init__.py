# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""First-party embedding provider Integrations catalog packages (P2-002-B2)."""

from __future__ import annotations

from intergrax.integrations.providers.embedding_provider.register_all import (
    EMBEDDING_PROVIDER_SLUGS,
    register_embedding_provider_integrations,
)

__all__ = [
    "EMBEDDING_PROVIDER_SLUGS",
    "register_embedding_provider_integrations",
]
