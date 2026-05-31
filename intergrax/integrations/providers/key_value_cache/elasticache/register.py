# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register elasticache."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.key_value_cache.elasticache.bundle import create_elasticache_key_value_cache
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_elasticache_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.ELASTICACHE.value,
            categories=(IntegrationCategory.KEY_VALUE_CACHE,),
            factory=create_elasticache_key_value_cache,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_ELASTICACHE",
            description="elasticache integration (Phase M.6 P2/P3)",
        ),
        override=override,
    )
