# `sentry` integration — usage

**Category:** ``observability_backend``  
**Catalog factory:** ``create_sentry_observability_backend()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(observability_backend="sentry")
backend = profile.resolve(IntegrationCategory.OBSERVABILITY_BACKEND)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.observability_backend.sentry.bundle import create_sentry_observability_backend

backend = create_sentry_observability_backend(**config_overrides)
```


## Environment variables

`INTERGRAX_SENTRY_DSN`, `INTERGRAX_SENTRY_ORG`, `INTERGRAX_SENTRY_AUTH_TOKEN`, optional `INTERGRAX_SENTRY_PROJECT`, `INTERGRAX_SENTRY_ENVIRONMENT`

## Example

```python
from intergrax.integrations.providers.observability_backend.sentry.bundle import create_sentry_observability_backend

obs = create_sentry_observability_backend(
    dsn="https://...@sentry.io/...",
    org="my-org",
    auth_token="sntrys_...",
)
obs.capture_message("agent run failed", level="error")
count = obs.query_instant("is:unresolved").series[0].points[0].value
```

## Notes

Error tracking + issue stats. ``sentry-sdk`` for capture; REST API for issue counts. Complements ``otel``/``langfuse``.
