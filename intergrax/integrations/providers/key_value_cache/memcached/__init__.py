# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "MEMCACHED_KEY_VALUE_CACHE_PROVIDER_ID",
    "MemcachedKeyValueCacheIntegration",
    "MemcachedKeyValueCacheIntegrationConfig",
    "MemcachedKeyValueCacheClient",
    "create_memcached_key_value_cache",
    "create_memcached_key_value_cache_integration",
    "register_memcached_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_memcached_key_value_cache",
        "create_memcached_key_value_cache_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "MEMCACHED_KEY_VALUE_CACHE_PROVIDER_ID",
        "MemcachedKeyValueCacheIntegration",
        "MemcachedKeyValueCacheIntegrationConfig",
        "MemcachedKeyValueCacheClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "MEMCACHED_KEY_VALUE_CACHE_PROVIDER_ID",
        "MemcachedKeyValueCacheIntegration",
        "MemcachedKeyValueCacheIntegrationConfig",
        "MemcachedKeyValueCacheClient",
    }
)

def __getattr__(name: str):
    if name == "register_memcached_integration":
        from intergrax.integrations.providers.key_value_cache.memcached.register import register_memcached_integration

        return register_memcached_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.key_value_cache.memcached import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.key_value_cache.memcached import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.key_value_cache.memcached import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
