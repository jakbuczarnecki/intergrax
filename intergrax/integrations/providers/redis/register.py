# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register Redis in the integration catalog (Phase M.4)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.redis.bundle import create_redis_key_value_cache
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_redis_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.REDIS.value,
            categories=(IntegrationCategory.KEY_VALUE_CACHE,),
            factory=create_redis_key_value_cache,
            status=IntegrationStatus.STABLE,
            env_prefix="INTERGRAX_REDIS",
            description="Redis — KV cache, idempotency, rate limits, semaphores (via create_redis_integration)",
        ),
        override=override,
    )
