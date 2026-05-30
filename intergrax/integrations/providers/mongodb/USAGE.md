# `mongodb` integration — usage

**Category:** ``document_store``  
**Catalog factory:** ``create_mongodb_document_store()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(document_store=IntegrationSlug.MONGODB)
backend = profile.resolve(IntegrationCategory.DOCUMENT_STORE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.mongodb.bundle import create_mongodb_document_store

backend = create_mongodb_document_store(**config_overrides)
```


## Environment variables

`INTERGRAX_MONGODB_URI`, `INTERGRAX_MONGODB_DATABASE`, `INTERGRAX_MONGODB_COLLECTION`

## Example

```python
from intergrax.integrations.providers.mongodb.bundle import create_mongodb_document_store

from intergrax.integrations.contracts.document_store import DocumentRecord

store = create_mongodb_document_store(uri="mongodb://localhost:27017")
store.put(DocumentRecord(partition_key="tenant-1", row_key="mem-1", data={"topic": "onboarding"}))
doc = store.get("tenant-1", "mem-1")
store.close()
```

## Notes

``pymongo.MongoClient`` only in ``opens.py``.
