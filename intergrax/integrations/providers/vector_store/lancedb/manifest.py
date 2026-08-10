# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``lancedb`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="lancedb",
    categories=(IntegrationCategory.VECTOR_STORE,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_LANCEDB',
    description='lancedb integration (BETA: native ownership ABI is incomplete)',
)
