# `posthog` integration — usage

**Category:** ``observability_backend``  
**Catalog factory:** ``create_posthog_observability_backend()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(observability_backend=IntegrationSlug.POSTHOG)
backend = profile.resolve(IntegrationCategory.OBSERVABILITY_BACKEND)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.observability_backend.posthog.bundle import create_posthog_observability_backend

backend = create_posthog_observability_backend(**config_overrides)
```


## Environment variables

`INTERGRAX_POSTHOG_API_KEY`, `INTERGRAX_POSTHOG_URL`

## Example

```python
from intergrax.integrations.providers.observability_backend.posthog.bundle import create_posthog_observability_backend

obs = create_posthog_observability_backend(api_key="phc_...")
```

## Notes

Product analytics + event metrics facade.
