# `oracle` integration — usage

**Category:** ``relational_store``  
**Catalog factory:** ``create_oracle_relational_store()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(relational_store="oracle")
backend = profile.resolve(IntegrationCategory.RELATIONAL_STORE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.relational_store.oracle.bundle import create_oracle_relational_store

backend = create_oracle_relational_store(**config_overrides)
```


## Environment variables

`INTERGRAX_ORACLE_DSN` or `INTERGRAX_ORACLE_CONNECTION_STRING`

## Example

```python
from intergrax.integrations.providers.relational_store.oracle.bundle import create_oracle_relational_store

store = create_oracle_relational_store(dsn="user/pass@localhost:1521/ORCL")
store.execute("INSERT INTO items (name) VALUES (:1)", ("alpha",))
rows = store.fetch_all("SELECT name FROM items")
store.close()
```

## Notes

``oracledb.connect`` opened lazily in ``_shared/p2/factories.py``.
