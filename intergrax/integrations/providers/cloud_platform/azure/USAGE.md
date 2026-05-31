# `azure` integration — usage

**Category:** ``cloud_platform``  
**Catalog factory:** ``create_azure_cloud_platform()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(cloud_platform=IntegrationSlug.AZURE)
backend = profile.resolve(IntegrationCategory.CLOUD_PLATFORM)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.cloud_platform.azure.bundle import create_azure_cloud_platform

backend = create_azure_cloud_platform(**config_overrides)
```


## Environment variables

`INTERGRAX_AZURE_TENANT_ID`, `INTERGRAX_AZURE_CLIENT_ID`, `INTERGRAX_AZURE_CLIENT_SECRET` (or managed identity)

## Example

```python
from intergrax.integrations.providers.cloud_platform.azure.bundle import create_azure_cloud_platform

platform = create_azure_cloud_platform()
blob_slug = platform.resolve("object_storage")  # -> "azure_blob"
```

## Notes

``azure-identity`` only in ``opens.py``.
