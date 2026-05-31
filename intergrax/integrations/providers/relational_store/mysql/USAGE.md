# `mysql` integration — usage

**Category:** ``relational_store``  
**Catalog factory:** ``create_mysql_relational_store()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(relational_store=IntegrationSlug.MYSQL)
backend = profile.resolve(IntegrationCategory.RELATIONAL_STORE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.relational_store.mysql.bundle import create_mysql_relational_store

backend = create_mysql_relational_store(**config_overrides)
```


## Environment variables

`INTERGRAX_MYSQL_DSN` or component vars; optional `INTERGRAX_MYSQL_TENANT_DATABASE`

## Example

```python
from intergrax.integrations.providers.relational_store.mysql.bundle import create_mysql_relational_store

store = create_mysql_relational_store(host="127.0.0.1", user="app", password="secret", database="intergrax")
store.execute("INSERT INTO items (name) VALUES (%s)", ("alpha",))
rows = store.fetch_all("SELECT name FROM items")
store.close()
```

## Notes

``pymysql.connect`` only in ``opens.py``.
