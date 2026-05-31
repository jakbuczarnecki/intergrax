# `cloud_sql` integration — usage

**Category:** ``relational_store``  
**Catalog factory:** ``create_cloud_sql_relational_store()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(relational_store=IntegrationSlug.CLOUD_SQL)
backend = profile.resolve(IntegrationCategory.RELATIONAL_STORE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.relational_store.cloud_sql.bundle import create_cloud_sql_relational_store

backend = create_cloud_sql_relational_store(**config_overrides)
```


## Environment variables

`INTERGRAX_CLOUD_SQL_DSN` or connection string components (`HOST`, `USER`, `PASSWORD`, `DATABASE`)

## Example

```python
from intergrax.integrations.providers.relational_store.cloud_sql.bundle import create_cloud_sql_relational_store

store = create_cloud_sql_relational_store(dsn="host=127.0.0.1 user=app password=secret dbname=intergrax")
store.execute("INSERT INTO items (name) VALUES (%s)", ("alpha",))
rows = store.fetch_all("SELECT name FROM items")
store.close()
```

## Notes

Default ``relational_store`` when ``cloud_platform=gcp``. ``pg8000`` opened lazily.
