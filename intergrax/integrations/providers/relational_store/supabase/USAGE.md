# `supabase` integration — usage

**Category:** ``relational_store``  
**Catalog factory:** ``create_supabase_relational_store()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(relational_store="supabase")
backend = profile.resolve(IntegrationCategory.RELATIONAL_STORE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.relational_store.supabase.bundle import create_supabase_relational_store

backend = create_supabase_relational_store(**config_overrides)
```


## Environment variables

`INTERGRAX_SUPABASE_DSN` (Postgres connection string)

## Example

```python
from intergrax.integrations.providers.relational_store.supabase.bundle import create_supabase_relational_store

store = create_supabase_relational_store(dsn="postgresql://...")
```

## Notes

Postgres-backed product prototypes.
