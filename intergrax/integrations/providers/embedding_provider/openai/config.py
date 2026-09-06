# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed config for OpenAI embedding provider integration."""

from __future__ import annotations

from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig


class OpenaiEmbeddingProviderIntegrationConfig(CategoryIntegrationConfig):
    """Catalog-boundary config for OpenAI embedding provider."""

    base_url: str | None = None
    credential_ref: str | None = None
