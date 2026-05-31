# `elasticache` integration — usage

**Category:** ``key_value_cache``  
**Catalog factory:** ``create_elasticache_key_value_cache()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(key_value_cache=IntegrationSlug.ELASTICACHE)
backend = profile.resolve(IntegrationCategory.KEY_VALUE_CACHE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.key_value_cache.elasticache.bundle import create_elasticache_key_value_cache

backend = create_elasticache_key_value_cache(**config_overrides)
```


## Environment variables

Same as memcached — point ``INTERGRAX_ELASTICACHE_HOST`` / ``PORT`` at the ElastiCache Redis endpoint

## Example

```python
from intergrax.integrations.providers.key_value_cache.elasticache.bundle import create_elasticache_key_value_cache

cache = create_elasticache_key_value_cache(host="my-cluster.xxxxx.cache.amazonaws.com", port=6379)
cache.set("t1", "lock:graph", b"1", ttl_seconds=60)
```

## Notes

Uses the memcached-style duck client adapter. For full Redis semantics prefer ``IntegrationSlug.REDIS`` with the cluster URL.
