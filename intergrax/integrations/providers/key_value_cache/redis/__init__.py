# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Redis integration public exports.

Redis is an optional integration dependency; keep this package importable in
minimal runtime images that only need to register the integration catalog.
"""

from __future__ import annotations

from typing import Any

from intergrax.utils.lazy_export import export_from_bundle, export_from_import_path

_EXPORTS = {
    "ENV_REDIS_DB": "intergrax.integrations.providers.key_value_cache.redis.config",
    "ENV_REDIS_KEY_PREFIX": "intergrax.integrations.providers.key_value_cache.redis.config",
    "ENV_REDIS_URL": "intergrax.integrations.providers.key_value_cache.redis.config",
    "RedisIntegrationConfig": "intergrax.integrations.providers.key_value_cache.redis.config",
    "RedisIntegrationBundle": "intergrax.integrations.providers.key_value_cache.redis.bundle",
    "RedisKeyValueCache": "intergrax.integrations.providers.key_value_cache.redis.adapter",
    "RedisKVStore": "intergrax.distributed.providers.redis_kv_store",
    "create_redis_client": "intergrax.integrations.providers.key_value_cache.redis.client",
    "create_redis_execution_semaphore": "intergrax.integrations.providers.key_value_cache.redis.bundle",
    "create_redis_idempotency_store": "intergrax.integrations.providers.key_value_cache.redis.bundle",
    "create_redis_integration": "intergrax.integrations.providers.key_value_cache.redis.bundle",
    "create_redis_key_value_cache": "intergrax.integrations.providers.key_value_cache.redis.bundle",
    "create_redis_key_value_cache_integration": "intergrax.integrations.providers.key_value_cache.redis.bundle",
    "create_redis_kv_store": "intergrax.integrations.providers.key_value_cache.redis.bundle",
    "create_redis_rate_limiter": "intergrax.integrations.providers.key_value_cache.redis.bundle",
    "create_redis_rerank_cache": "intergrax.integrations.providers.key_value_cache.redis.bundle",
    "register_redis_integration": "intergrax.integrations.providers.key_value_cache.redis.register",
    "resolve_redis_config": "intergrax.integrations.providers.key_value_cache.redis.client",
}

__all__ = sorted(
    set(_EXPORTS)
    | {
        "REDIS_KEY_VALUE_CACHE_PROVIDER_ID",
        "RedisKeyValueCacheIntegration",
        "RedisKeyValueCacheIntegrationConfig",
        "RedisKeyValueCacheClient",
    }
)

_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "REDIS_KEY_VALUE_CACHE_PROVIDER_ID",
        "RedisKeyValueCacheIntegration",
        "RedisKeyValueCacheIntegrationConfig",
        "RedisKeyValueCacheClient",
    }
)


def __getattr__(name: str) -> Any:
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.key_value_cache.redis import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = export_from_import_path(module_name, name)
    globals()[name] = value
    return value
