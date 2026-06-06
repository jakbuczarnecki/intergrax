# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``huggingface_hub`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="huggingface_hub",
    categories=(IntegrationCategory.OBJECT_STORAGE,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_HUGGINGFACE_HUB',
    description='huggingface_hub integration (Phase M.6 P4)',
)
