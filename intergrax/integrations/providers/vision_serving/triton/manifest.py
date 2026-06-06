# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``triton`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="triton",
    categories=(IntegrationCategory.VISION_SERVING,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_TRITON',
    description='triton integration (Phase M.6 P6)',
)
