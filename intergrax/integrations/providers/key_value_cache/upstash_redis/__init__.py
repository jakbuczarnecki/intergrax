# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "UPSTASH_REDIS_KEY_VALUE_CACHE_PROVIDER_ID",
    "UpstashRedisKeyValueCacheIntegration",
    "UpstashRedisKeyValueCacheIntegrationConfig",
    "UpstashRedisKeyValueCacheClient",
    "create_upstash_redis_key_value_cache",
    "create_upstash_redis_key_value_cache_integration",
    "register_upstash_redis_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_upstash_redis_key_value_cache",
        "create_upstash_redis_key_value_cache_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "UPSTASH_REDIS_KEY_VALUE_CACHE_PROVIDER_ID",
        "UpstashRedisKeyValueCacheIntegration",
        "UpstashRedisKeyValueCacheIntegrationConfig",
        "UpstashRedisKeyValueCacheClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "UPSTASH_REDIS_KEY_VALUE_CACHE_PROVIDER_ID",
        "UpstashRedisKeyValueCacheIntegration",
        "UpstashRedisKeyValueCacheIntegrationConfig",
        "UpstashRedisKeyValueCacheClient",
    }
)

def __getattr__(name: str):
    if name == "register_upstash_redis_integration":
        from intergrax.integrations.providers.key_value_cache.upstash_redis.register import register_upstash_redis_integration

        return register_upstash_redis_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.key_value_cache.upstash_redis import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.key_value_cache.upstash_redis import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.key_value_cache.upstash_redis import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
