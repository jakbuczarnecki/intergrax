# `minio` integration — usage

**Category:** ``object_storage``  
**Catalog factory:** ``create_minio_object_storage()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(object_storage="minio")
backend = profile.resolve(IntegrationCategory.OBJECT_STORAGE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.object_storage.minio.bundle import create_minio_object_storage

backend = create_minio_object_storage(**config_overrides)
```


## Environment variables

`INTERGRAX_MINIO_ENDPOINT`, `INTERGRAX_MINIO_ACCESS_KEY`, `INTERGRAX_MINIO_SECRET_KEY`, `INTERGRAX_MINIO_BUCKET`

## Example

```python
from intergrax.integrations.providers.object_storage.minio.bundle import create_minio_object_storage

blobs = create_minio_object_storage(endpoint="http://localhost:9000", bucket="artifacts")
```

## Notes

S3-compatible self-hosted storage (boto3).
