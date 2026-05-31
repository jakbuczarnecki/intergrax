# `otel` integration — usage

**Category:** ``observability_backend``  
**Catalog factory:** ``create_otel_observability_backend()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(observability_backend=IntegrationSlug.OTEL)
backend = profile.resolve(IntegrationCategory.OBSERVABILITY_BACKEND)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.observability_backend.otel.bundle import create_otel_observability_backend

backend = create_otel_observability_backend(**config_overrides)
```


## Environment variables

`INTERGRAX_OTEL_ENDPOINT` (default `http://localhost:4318`), `INTERGRAX_OTEL_SERVICE_NAME`

## Example

```python
from intergrax.integrations.providers.observability_backend.otel.bundle import create_otel_observability_backend

obs = create_otel_observability_backend(endpoint="http://otel-collector:4318", service_name="intergrax-nexus")
instant = obs.query_instant("intergrax_tasks_total")
range_result = obs.query_range("intergrax_tasks_total", start=1710000000, end=1710003600, step="15s")
```

## Notes

Beta facade over an OTLP-oriented exporter. Inject ``exporter=`` in tests; production wiring may evolve.
