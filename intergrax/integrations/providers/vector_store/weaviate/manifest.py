# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``weaviate`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="weaviate",
    categories=(IntegrationCategory.VECTOR_STORE,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_WEAVIATE',
    description='weaviate integration (Phase M.7)',
)
