# Elasticache (elasticache)

Category: `key_value_cache`

## Legacy facade

- `create_elasticache_key_value_cache()` remains backward-compatible.

## Contract-based integration

- `ElasticacheKeyValueCacheIntegration` derives from the category-specific contract.
- Factory: `create_elasticache_key_value_cache_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
