# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed config for HuggingFace embedding provider integration."""

from __future__ import annotations

from intergrax.runtime.integrations.contracts import PlatformIntegrationConfig


class HfEmbeddingProviderIntegrationConfig(PlatformIntegrationConfig):
    """Catalog-boundary config for HuggingFace embedding provider."""

    device: str | None = None
    batch_size: int | None = None
