# `filesystem` integration — usage

**Category:** ``object_storage``  
**Catalog factory:** ``create_filesystem_object_storage()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(object_storage=IntegrationSlug.FILESYSTEM)
backend = profile.resolve(IntegrationCategory.OBJECT_STORAGE)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.object_storage.filesystem.bundle import create_filesystem_object_storage

backend = create_filesystem_object_storage(**config_overrides)
```


## Environment variables

`INTERGRAX_FILESYSTEM_ROOT_DIR` (default ``build/artifacts``)

## Example

```python
from intergrax.integrations.providers.object_storage.filesystem.bundle import create_filesystem_object_storage

blobs = create_filesystem_object_storage(root_dir="build/lab-artifacts")
```

## Notes

Local artifact store for CI/lab — no cloud SDK.
