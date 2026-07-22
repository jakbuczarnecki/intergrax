# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``ollama`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="ollama",
    categories=(IntegrationCategory.MODEL_SERVING_RUNTIME,),
    status=IntegrationStatus.STABLE,
    env_prefix="INTERGRAX_OLLAMA",
    description="Ollama self-hosted model serving runtime (health + list_models)",
)
