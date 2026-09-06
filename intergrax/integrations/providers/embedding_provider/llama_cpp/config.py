# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed config for llama.cpp embedding provider integration."""

from __future__ import annotations

from intergrax.runtime.integrations.contracts import PlatformIntegrationConfig


class LlamaCppEmbeddingProviderIntegrationConfig(PlatformIntegrationConfig):
    """Catalog-boundary config for llama.cpp embedding provider."""

    base_url: str | None = None
    credential_ref: str | None = None
