# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``gcs`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="gcs",
    categories=(IntegrationCategory.OBJECT_STORAGE,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_GCS',
    description='gcs integration (Phase M.6 P2/P3)',
)
