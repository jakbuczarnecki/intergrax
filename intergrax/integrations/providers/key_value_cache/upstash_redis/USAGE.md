# Upstash Redis (upstash_redis)

Category: `key_value_cache`

## Single public entrypoint

- **`UpstashRedisKeyValueCacheIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `UpstashRedisKeyValueCacheIntegration`.
- Contract factory: `create_upstash_redis_key_value_cache_integration()`.
