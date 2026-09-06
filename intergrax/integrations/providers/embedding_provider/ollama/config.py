# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed config for Ollama embedding provider integration."""

from __future__ import annotations

from intergrax.runtime.integrations.contracts import PlatformIntegrationConfig


class OllamaEmbeddingProviderIntegrationConfig(PlatformIntegrationConfig):
    """Catalog-boundary config for Ollama embedding provider."""

    base_url: str | None = None
