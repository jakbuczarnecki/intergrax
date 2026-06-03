# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``minio`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="minio",
    categories=(IntegrationCategory.OBJECT_STORAGE,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_MINIO',
    description='minio integration (Phase M.7)',
)
