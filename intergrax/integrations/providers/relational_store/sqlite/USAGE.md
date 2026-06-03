# `sqlite` integration — usage

**Category:** ``relational_store``  
**Catalog factory:** ``create_sqlite_relational_store()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(relational_store="sqlite")
backend = profile.resolve(IntegrationCategory.RELATIONAL_STORE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.relational_store.sqlite.bundle import create_sqlite_relational_store

backend = create_sqlite_relational_store(**config_overrides)
```


## Environment variables

`INTERGRAX_SQLITE_DATA_DIR` (directory for `.db` files)

## Example

```python
from intergrax.integrations.providers.relational_store.sqlite.bundle import create_sqlite_relational_store

store = create_sqlite_relational_store(data_dir="build/lab")
store.connect()
store.execute("CREATE TABLE IF NOT EXISTS items (id INTEGER PRIMARY KEY, name TEXT)")
store.fetch_all("SELECT * FROM items")
store.close()
```

## Notes

Bundle also exposes ``create_sqlite_trace_store()``, ``create_sqlite_runtime_event_store()``, etc.
