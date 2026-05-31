# `gcp` integration — usage

**Category:** ``cloud_platform``  
**Catalog factory:** ``create_gcp_cloud_platform()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(cloud_platform=IntegrationSlug.GCP)
backend = profile.resolve(IntegrationCategory.CLOUD_PLATFORM)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.cloud_platform.gcp.bundle import create_gcp_cloud_platform

backend = create_gcp_cloud_platform(**config_overrides)
```


## Environment variables

`INTERGRAX_GCP_PROJECT_ID`, `INTERGRAX_GCP_REGION`, `INTERGRAX_GCP_CREDENTIALS_FILE` (or ADC)

## Example

```python
from intergrax.integrations.providers.cloud_platform.gcp.bundle import create_gcp_cloud_platform

platform = create_gcp_cloud_platform(project_id="my-project")
gcs_slug = platform.resolve("object_storage")  # -> "gcs"
```

## Notes

``google-auth`` only in ``opens.py``.
