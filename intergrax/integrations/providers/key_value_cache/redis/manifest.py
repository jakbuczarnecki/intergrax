# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``redis`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="redis",
    categories=(IntegrationCategory.KEY_VALUE_CACHE,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_REDIS',
    description='Redis — KV cache, idempotency, rate limits, semaphores (via create_redis_integration)',
)
