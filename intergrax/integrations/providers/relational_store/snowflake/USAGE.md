# `snowflake` integration — usage

**Category:** ``relational_store``  
**Catalog factory:** ``create_snowflake_relational_store()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(relational_store=IntegrationSlug.SNOWFLAKE)
backend = profile.resolve(IntegrationCategory.RELATIONAL_STORE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.relational_store.snowflake.bundle import create_snowflake_relational_store

backend = create_snowflake_relational_store(**config_overrides)
```


## Environment variables

`INTERGRAX_SNOWFLAKE_DSN` or connection components

## Example

```python
from intergrax.integrations.providers.relational_store.snowflake.bundle import create_snowflake_relational_store

store = create_snowflake_relational_store(dsn="snowflake://...")
```

## Notes

SQL facade via ``psycopg``-compatible DSN.
