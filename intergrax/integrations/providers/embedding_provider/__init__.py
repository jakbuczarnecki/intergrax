# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""First-party embedding provider Integrations catalog packages (P2-002-B2)."""

from __future__ import annotations

__all__ = [
    "EMBEDDING_PROVIDER_SLUGS",
    "register_embedding_provider_integrations",
]


def __getattr__(name: str):
    if name == "register_embedding_provider_integrations":
        from intergrax.integrations.providers.embedding_provider.register_all import (
            register_embedding_provider_integrations,
        )

        return register_embedding_provider_integrations
    if name == "EMBEDDING_PROVIDER_SLUGS":
        from intergrax.integrations.providers.embedding_provider.register_all import (
            EMBEDDING_PROVIDER_SLUGS,
        )

        return EMBEDDING_PROVIDER_SLUGS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
