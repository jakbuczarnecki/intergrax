# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``google_drive`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="google_drive",
    categories=(IntegrationCategory.OBJECT_STORAGE,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_GOOGLE_DRIVE',
    description='google_drive integration (Phase M.7 P7)',
)
