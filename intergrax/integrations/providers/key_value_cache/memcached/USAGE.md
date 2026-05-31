# `memcached` integration — usage

**Category:** ``key_value_cache``  
**Catalog factory:** ``create_memcached_key_value_cache()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(key_value_cache=IntegrationSlug.MEMCACHED)
backend = profile.resolve(IntegrationCategory.KEY_VALUE_CACHE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.key_value_cache.memcached.bundle import create_memcached_key_value_cache

backend = create_memcached_key_value_cache(**config_overrides)
```


## Environment variables

`INTERGRAX_MEMCACHED_HOST` (default `localhost`), `INTERGRAX_MEMCACHED_PORT` (default `11211`)

## Example

```python
from intergrax.integrations.providers.key_value_cache.memcached.bundle import create_memcached_key_value_cache

cache = create_memcached_key_value_cache(host="127.0.0.1", port=11211)
cache.set("t1", "session:42", b"payload", ttl_seconds=3600)
value = cache.get("t1", "session:42")
cache.delete("t1", "session:42")
cache.close()
```

## Notes

``pymemcache`` opened lazily. Keys are tenant-scoped as ``{tenant_id}:{key}``.
