# `gcs` integration — usage

**Category:** ``object_storage``  
**Catalog factory:** ``create_gcs_object_storage()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(object_storage=IntegrationSlug.GCS)
backend = profile.resolve(IntegrationCategory.OBJECT_STORAGE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.object_storage.gcs.bundle import create_gcs_object_storage

backend = create_gcs_object_storage(**config_overrides)
```


## Environment variables

`INTERGRAX_GCS_BUCKET` (required); optional `INTERGRAX_GCS_PREFIX`, `INTERGRAX_GCS_PROJECT_ID`; GCP ADC or service account

## Example

```python
from intergrax.integrations.providers.object_storage.gcs.bundle import create_gcs_object_storage

store = create_gcs_object_storage(bucket="intergrax-artifacts", prefix="tenant-a")
store.put("reports/summary.pdf", pdf_bytes, content_type="application/pdf")
obj = store.get("reports/summary.pdf")
url = store.presigned_url("reports/summary.pdf", expires_in_seconds=900)
store.close()
```

## Notes

``google-cloud-storage`` opened lazily in ``_shared/p2/``. Default ``object_storage`` when ``cloud_platform=gcp``.
