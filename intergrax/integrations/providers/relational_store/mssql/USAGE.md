# `mssql` integration — usage

**Category:** ``relational_store``  
**Catalog factory:** ``create_mssql_relational_store()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(relational_store=IntegrationSlug.MSSQL)
backend = profile.resolve(IntegrationCategory.RELATIONAL_STORE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.relational_store.mssql.bundle import create_mssql_relational_store

backend = create_mssql_relational_store(**config_overrides)
```


## Environment variables

`INTERGRAX_MSSQL_DSN` or `INTERGRAX_MSSQL_CONNECTION_STRING`

## Example

```python
from intergrax.integrations.providers.relational_store.mssql.bundle import create_mssql_relational_store

store = create_mssql_relational_store(connection_string="Driver={ODBC Driver 18 for SQL Server};Server=...")
store.execute("INSERT INTO items (name) VALUES (?)", ("alpha",))
rows = store.fetch_all("SELECT name FROM items")
store.close()
```

## Notes

``pyodbc.connect`` opened lazily.
