# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``upstash_redis`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="upstash_redis",
    categories=(IntegrationCategory.KEY_VALUE_CACHE,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_UPSTASH_REDIS',
    description='upstash_redis integration (Phase M.7 P7)',
)
