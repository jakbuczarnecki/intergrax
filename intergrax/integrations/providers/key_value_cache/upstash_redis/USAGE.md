# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

# `upstash_redis` integration — usage

**Category:** `key_value_cache`  
**Catalog factory:** ``create_upstash_redis_key_value_cache()``  
**Env prefix:** ``INTERGRAX_UPSTASH_REDIS_*``

```python
from intergrax.integrations.providers.key_value_cache.upstash_redis.bundle import create_upstash_redis_key_value_cache

backend = create_upstash_redis_key_value_cache()
```
