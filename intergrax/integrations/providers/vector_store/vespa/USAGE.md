# `vespa` integration — usage

**Category:** ``vector_store``  
**Catalog factory:** ``create_vespa_vector_store()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(vector_store="vespa")
backend = profile.resolve(IntegrationCategory.VECTOR_STORE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.vector_store.vespa.bundle import create_vespa_vector_store

backend = create_vespa_vector_store(**config_overrides)
```


## Environment variables

`INTERGRAX_VESPA_URL`, `INTERGRAX_VESPA_COLLECTION`

## Example

```python
from intergrax.integrations.providers.vector_store.vespa.bundle import create_vespa_vector_store

store = create_vespa_vector_store(url="http://localhost:8080", collection="docs")
```

## Notes

Vespa vector search catalog bridge.
