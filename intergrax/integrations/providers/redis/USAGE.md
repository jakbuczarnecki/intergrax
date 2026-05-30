# `redis` integration — usage

**Category:** ``key_value_cache``  
**Catalog factory:** ``create_redis_key_value_cache()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(key_value_cache=IntegrationSlug.REDIS)
backend = profile.resolve(IntegrationCategory.KEY_VALUE_CACHE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.redis.bundle import create_redis_key_value_cache

backend = create_redis_key_value_cache(**config_overrides)
```


## Environment variables

`INTERGRAX_REDIS_URL`; optional `INTERGRAX_REDIS_DB`, `INTERGRAX_REDIS_KEY_PREFIX`

## Example

```python
from intergrax.integrations.providers.redis.bundle import create_redis_key_value_cache

cache = create_redis_key_value_cache(url="redis://localhost:6379/0")
cache.set("session:42", b"payload", ttl_seconds=3600)
value = cache.get("session:42")
cache.delete("session:42")
```

## Notes

Bundle also provides idempotency, rate limit, semaphore via ``create_redis_integration()``.
