# `postgresql` integration — usage

**Category:** ``relational_store``  
**Catalog factory:** ``create_postgresql_relational_store()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(relational_store="postgresql")
backend = profile.resolve(IntegrationCategory.RELATIONAL_STORE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.relational_store.postgresql.bundle import create_postgresql_relational_store

backend = create_postgresql_relational_store(**config_overrides)
```


## Environment variables

`INTERGRAX_POSTGRESQL_DSN` or HOST/PORT/USER/PASSWORD/DATABASE; optional `INTERGRAX_POSTGRESQL_SCHEMA`

## Example

```python
from intergrax.integrations.providers.relational_store.postgresql.bundle import create_postgresql_relational_store

store = create_postgresql_relational_store(dsn="postgresql://user:pass@localhost:5432/app")
store.execute("INSERT INTO items (name) VALUES (%s)", ("alpha",))
rows = store.fetch_all("SELECT name FROM items")
store.close()
```

## Notes

``psycopg.connect`` only in ``opens.py``.
