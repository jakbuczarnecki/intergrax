# `azure_blob` integration — usage

**Category:** ``object_storage``  
**Catalog factory:** ``create_azure_blob_object_storage()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(object_storage="azure_blob")
backend = profile.resolve(IntegrationCategory.OBJECT_STORAGE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.object_storage.azure_blob.bundle import create_azure_blob_object_storage

backend = create_azure_blob_object_storage(**config_overrides)
```


## Environment variables

`INTERGRAX_AZURE_BLOB_CONTAINER` (required); optional `INTERGRAX_AZURE_BLOB_PREFIX`, `INTERGRAX_AZURE_BLOB_CONNECTION_STRING`, `INTERGRAX_AZURE_BLOB_ACCOUNT_URL`

## Example

```python
from intergrax.integrations.providers.object_storage.azure_blob.bundle import create_azure_blob_object_storage

store = create_azure_blob_object_storage(container="artifacts", prefix="tenant-a")
store.put("exports/run-1.zip", file_bytes, content_type="application/zip")
obj = store.get("exports/run-1.zip")
store.delete("exports/run-1.zip")
```

## Notes

``azure-storage-blob`` only in ``opens.py``. Default ``object_storage`` slug when ``cloud_platform=azure``.
