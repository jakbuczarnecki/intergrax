# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``backblaze_b2`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="backblaze_b2",
    categories=(IntegrationCategory.OBJECT_STORAGE,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_BACKBLAZE_B2',
    description='backblaze_b2 integration (Phase M.6 P6)',
)
