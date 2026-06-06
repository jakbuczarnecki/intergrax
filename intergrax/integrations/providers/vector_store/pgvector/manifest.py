# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``pgvector`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="pgvector",
    categories=(IntegrationCategory.VECTOR_STORE,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_PGVECTOR',
    description='pgvector integration (Phase M.6 P4)',
)
