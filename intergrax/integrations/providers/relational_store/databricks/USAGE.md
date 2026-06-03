# `databricks` integration — usage

**Category:** ``relational_store``  
**Catalog factory:** ``create_databricks_relational_store()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(relational_store="databricks")
backend = profile.resolve(IntegrationCategory.RELATIONAL_STORE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.relational_store.databricks.bundle import create_databricks_relational_store

backend = create_databricks_relational_store(**config_overrides)
```


## Environment variables

`INTERGRAX_DATABRICKS_HOST`, `INTERGRAX_DATABRICKS_HTTP_PATH`, `INTERGRAX_DATABRICKS_TOKEN`; optional CATALOG/SCHEMA

## Example

```python
from intergrax.integrations.providers.relational_store.databricks.bundle import create_databricks_relational_store

store = create_databricks_relational_store(
    host="adb-123.4.azuredatabricks.net",
    http_path="/sql/1.0/warehouses/abc",
    access_token="dapi-...",
)
rows = store.fetch_all("SELECT id, name FROM analytics.events LIMIT 10")
store.close()
```

## Notes

``databricks.sql.connect`` only in ``opens.py``.
