# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``ollama`` embedding provider integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="ollama",
    categories=(IntegrationCategory.EMBEDDING_PROVIDER,),
    status=IntegrationStatus.STABLE,
    env_prefix="INTERGRAX_OLLAMA_EMBEDDING",
    description="Ollama local embedding provider.",
)
