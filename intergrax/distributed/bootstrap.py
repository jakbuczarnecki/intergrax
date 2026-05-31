# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Distributed layer composition root.

Redis MUST be composed via ``IntegrationProfile`` / ``integrations.providers.key_value_cache.redis``.
Legacy ``DistributedProviderRegistry`` delegates to the catalog factory.
"""

from __future__ import annotations

from typing import Any, Optional

from intergrax.distributed.registry import DistributedProviderRegistry
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.key_value_cache.redis import RedisKVStore, register_redis_integration
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug


def resolve_redis_kv_store(
    profile: Optional[IntegrationProfile] = None,
    **config_overrides: object,
) -> Any:
    """Catalog-first Redis KV resolution (Phase M.9)."""
    register_redis_integration(override=True)
    active = profile or IntegrationProfile(key_value_cache=IntegrationSlug.REDIS)
    if active.key_value_cache is None:
        active = active.model_copy(update={"key_value_cache": IntegrationSlug.REDIS})
    return active.resolve(
        IntegrationCategory.KEY_VALUE_CACHE,
        config=active.options_for_slug(IntegrationSlug.REDIS) | dict(config_overrides),
    )


def bootstrap_default_providers(
    registry: DistributedProviderRegistry,
    *,
    profile: Optional[IntegrationProfile] = None,
) -> None:
    """
    Register default distributed providers via Integration Library.

    ``registry.create("redis")`` still works; instances are built from ``IntegrationProfile``.
    """
    register_redis_integration(override=True)
    registry.register("redis", RedisKVStore)
    _ = profile
