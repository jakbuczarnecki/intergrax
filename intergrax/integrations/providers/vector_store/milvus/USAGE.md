# `milvus` integration — usage

**Category:** ``vector_store``  
**Catalog factory:** ``create_milvus_vector_store()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(vector_store="milvus")
backend = profile.resolve(IntegrationCategory.VECTOR_STORE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.vector_store.milvus.bundle import create_milvus_vector_store

backend = create_milvus_vector_store(**config_overrides)
```


## Environment variables

`INTERGRAX_MILVUS_URL`, optional `INTERGRAX_MILVUS_API_KEY`

## Example

```python
from intergrax.integrations.providers.vector_store.milvus.bundle import create_milvus_vector_store

store = create_milvus_vector_store(url="http://localhost:19530")
```

## Notes

Requires ``pymilvus`` at runtime.
