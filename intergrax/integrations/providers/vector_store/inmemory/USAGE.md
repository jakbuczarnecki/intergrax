# `inmemory` integration — usage

**Category:** ``vector_store``  
**Catalog factory:** ``create_inmemory_vector_store()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(vector_store=IntegrationSlug.INMEMORY)
backend = profile.resolve(IntegrationCategory.VECTOR_STORE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.vector_store.inmemory.bundle import create_inmemory_vector_store

backend = create_inmemory_vector_store(**config_overrides)
```


## Environment variables

Optional `INTERGRAX_INMEMORY_TENANT_ID` (default ``default``)

## Example

```python
from intergrax.integrations.providers.vector_store.inmemory.bundle import create_inmemory_vector_store

store = create_inmemory_vector_store(tenant_id="lab")
```

## Notes

Delegates to ``intergrax.rag.vectorstore.providers.inmemory_vectorstore`` — lab / unit tests.
