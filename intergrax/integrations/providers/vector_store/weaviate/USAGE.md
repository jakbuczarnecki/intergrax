# `weaviate` integration — usage

**Category:** ``vector_store``  
**Catalog factory:** ``create_weaviate_vector_store()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(vector_store=IntegrationSlug.WEAVIATE)
backend = profile.resolve(IntegrationCategory.VECTOR_STORE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.vector_store.weaviate.bundle import create_weaviate_vector_store

backend = create_weaviate_vector_store(**config_overrides)
```


## Environment variables

`INTERGRAX_WEAVIATE_URL`, `INTERGRAX_WEAVIATE_API_KEY`, `INTERGRAX_WEAVIATE_COLLECTION`

## Example

```python
from intergrax.integrations.providers.vector_store.weaviate.bundle import create_weaviate_vector_store

store = create_weaviate_vector_store(url="https://...", collection="docs")
```

## Notes

Requires ``weaviate-client`` at runtime. Catalog bridge via ``VectorStoreBridge``.
