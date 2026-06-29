# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p2.factories import create_memcached_key_value_cache as _legacy_create_memcached_key_value_cache

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.key_value_cache.memcached.integration import (
    MEMCACHED_KEY_VALUE_CACHE_PROVIDER_ID,
    MemcachedKeyValueCacheIntegration,
    MemcachedKeyValueCacheIntegrationConfig,
    MemcachedKeyValueCacheClient,
)

__all__ = [
    "create_memcached_key_value_cache",
    "create_memcached_key_value_cache_integration",
]


def create_memcached_key_value_cache_integration(
    *,
    client: MemcachedKeyValueCacheClient | None = None,
    enabled: bool = False,
) -> MemcachedKeyValueCacheIntegration:
    """
    Build a contract-based Memcached key value cache integration.

    The legacy facade (create_memcached_key_value_cache) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Memcached key value cache integration requires an injected client when enabled=True",
        )
    if client is not None:
        return MemcachedKeyValueCacheIntegration.from_client(client, enabled=enabled)
    return MemcachedKeyValueCacheIntegration.for_provider(
        provider_id=MEMCACHED_KEY_VALUE_CACHE_PROVIDER_ID,
        display_name="Memcached",
        config=MemcachedKeyValueCacheIntegrationConfig(enabled=enabled),
    )


def create_memcached_key_value_cache(**kwargs: object) -> MemcachedKeyValueCacheIntegration:
    """Compatibility shim — constructs MemcachedKeyValueCacheIntegration from legacy runtime."""
    runtime = _legacy_create_memcached_key_value_cache(**kwargs)
    if isinstance(runtime, MemcachedKeyValueCacheIntegration):
        return runtime
    return MemcachedKeyValueCacheIntegration.from_runtime(runtime)
