# Elasticache (elasticache)

Category: `key_value_cache`

## Single public entrypoint

- **`ElasticacheKeyValueCacheIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `ElasticacheKeyValueCacheIntegration`.
- Contract factory: `create_elasticache_key_value_cache_integration()`.
