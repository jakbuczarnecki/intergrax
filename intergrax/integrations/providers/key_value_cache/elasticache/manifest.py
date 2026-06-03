# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``elasticache`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="elasticache",
    categories=(IntegrationCategory.KEY_VALUE_CACHE,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_ELASTICACHE',
    description='elasticache integration (Phase M.6 P2/P3)',
)
