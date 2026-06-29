# Redis (redis)

Category: `key_value_cache`

## Single public entrypoint

- **`RedisKeyValueCacheIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `RedisKeyValueCacheIntegration`.
- Contract factory: `create_redis_key_value_cache_integration()`.
