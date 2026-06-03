# `s3` integration — usage

**Category:** ``object_storage``  
**Catalog factory:** ``create_s3_object_storage()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(object_storage="s3")
backend = profile.resolve(IntegrationCategory.OBJECT_STORAGE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.object_storage.s3.bundle import create_s3_object_storage

backend = create_s3_object_storage(**config_overrides)
```


## Environment variables

`INTERGRAX_S3_BUCKET` (required); optional `INTERGRAX_S3_REGION`, `INTERGRAX_S3_PREFIX`, `INTERGRAX_S3_ENDPOINT_URL`, AWS credential vars

## Example

```python
from intergrax.integrations.providers.object_storage.s3.bundle import create_s3_object_storage

store = create_s3_object_storage(bucket="intergrax-artifacts", region="eu-central-1", prefix="tenant-a")
store.put("exports/run-1.zip", file_bytes, content_type="application/zip")
obj = store.get("exports/run-1.zip")
url = store.presigned_url("exports/run-1.zip", expires_in_seconds=900)
store.delete("exports/run-1.zip")
```

## Notes

boto3 S3 client only in ``opens.py``. With ``IntegrationProfile(cloud_platform='aws')``, ``object_storage`` resolves to ``s3`` by default.
