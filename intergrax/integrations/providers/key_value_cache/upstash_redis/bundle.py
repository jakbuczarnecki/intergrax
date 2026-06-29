# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p8.factories import create_upstash_redis_key_value_cache as _legacy_create_upstash_redis_key_value_cache

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.key_value_cache.upstash_redis.integration import (
    UPSTASH_REDIS_KEY_VALUE_CACHE_PROVIDER_ID,
    UpstashRedisKeyValueCacheIntegration,
    UpstashRedisKeyValueCacheIntegrationConfig,
    UpstashRedisKeyValueCacheClient,
)

__all__ = [
    "create_upstash_redis_key_value_cache",
    "create_upstash_redis_key_value_cache_integration",
]


def create_upstash_redis_key_value_cache_integration(
    *,
    client: UpstashRedisKeyValueCacheClient | None = None,
    enabled: bool = False,
) -> UpstashRedisKeyValueCacheIntegration:
    """
    Build a contract-based Upstash Redis key value cache integration.

    The legacy facade (create_upstash_redis_key_value_cache) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Upstash Redis key value cache integration requires an injected client when enabled=True",
        )
    if client is not None:
        return UpstashRedisKeyValueCacheIntegration.from_client(client, enabled=enabled)
    return UpstashRedisKeyValueCacheIntegration.for_provider(
        provider_id=UPSTASH_REDIS_KEY_VALUE_CACHE_PROVIDER_ID,
        display_name="Upstash Redis",
        config=UpstashRedisKeyValueCacheIntegrationConfig(enabled=enabled),
    )


def create_upstash_redis_key_value_cache(**kwargs: object) -> UpstashRedisKeyValueCacheIntegration:
    """Compatibility shim — constructs UpstashRedisKeyValueCacheIntegration from legacy runtime."""
    runtime = _legacy_create_upstash_redis_key_value_cache(**kwargs)
    if isinstance(runtime, UpstashRedisKeyValueCacheIntegration):
        return runtime
    return UpstashRedisKeyValueCacheIntegration.from_client(runtime)
