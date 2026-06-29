# Memcached (memcached)

Category: `key_value_cache`

## Single public entrypoint

- **`MemcachedKeyValueCacheIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `MemcachedKeyValueCacheIntegration`.
- Contract factory: `create_memcached_key_value_cache_integration()`.
