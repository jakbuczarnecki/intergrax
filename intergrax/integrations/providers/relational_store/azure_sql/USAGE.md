# `azure_sql` integration — usage

**Category:** ``relational_store``  
**Catalog factory:** ``create_azure_sql_relational_store()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(relational_store="azure_sql")
backend = profile.resolve(IntegrationCategory.RELATIONAL_STORE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.relational_store.azure_sql.bundle import create_azure_sql_relational_store

backend = create_azure_sql_relational_store(**config_overrides)
```


## Environment variables

`INTERGRAX_AZURE_SQL_CONNECTION_STRING` or DSN; optional `INTERGRAX_AZURE_SQL_SCHEMA`

## Example

```python
from intergrax.integrations.providers.relational_store.azure_sql.bundle import create_azure_sql_relational_store

store = create_azure_sql_relational_store(
    connection_string="Driver={ODBC Driver 18 for SQL Server};Server=tcp:....database.windows.net;..."
)
rows = store.fetch_all("SELECT TOP 10 id, name FROM items")
store.close()
```

## Notes

Default ``relational_store`` when ``cloud_platform=azure``. ``pyodbc`` opened lazily.
