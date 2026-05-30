# `cassandra` integration — usage

**Category:** ``document_store``  
**Catalog factory:** ``create_cassandra_document_store()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(document_store=IntegrationSlug.CASSANDRA)
backend = profile.resolve(IntegrationCategory.DOCUMENT_STORE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.cassandra.bundle import create_cassandra_document_store

backend = create_cassandra_document_store(**config_overrides)
```


## Environment variables

`INTERGRAX_CASSANDRA_CONTACT_POINTS`, `INTERGRAX_CASSANDRA_KEYSPACE`; optional USER/PASSWORD/TABLE

## Example

```python
from intergrax.integrations.providers.cassandra.bundle import create_cassandra_document_store

from intergrax.integrations.contracts.document_store import DocumentRecord

store = create_cassandra_document_store(contact_points="127.0.0.1", keyspace="intergrax")
store.put(DocumentRecord(partition_key="tenant-1", row_key="evt-1", data={"status": "ok"}))
doc = store.get("tenant-1", "evt-1")
result = store.query("tenant-1", limit=50, row_key_prefix="2026-")
store.close()
```

## Notes

Cassandra driver only in ``opens.py``.
