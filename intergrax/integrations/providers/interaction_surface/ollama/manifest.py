# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``ollama`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="ollama",
    categories=(IntegrationCategory.INTERACTION_SURFACE,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_OLLAMA',
    description='ollama integration (Phase M.6 P4)',
)
