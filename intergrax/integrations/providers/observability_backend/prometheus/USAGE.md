# `prometheus` integration — usage

**Category:** ``observability_backend``  
**Catalog factory:** ``create_prometheus_observability_backend()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

register_default_integrations()
profile = IntegrationProfile(observability_backend=IntegrationSlug.PROMETHEUS)
backend = profile.resolve(IntegrationCategory.OBSERVABILITY_BACKEND)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.observability_backend.prometheus.bundle import create_prometheus_observability_backend

backend = create_prometheus_observability_backend(**config_overrides)
```


## Environment variables

`INTERGRAX_PROMETHEUS_BASE_URL`; optional `INTERGRAX_PROMETHEUS_BEARER_TOKEN`

## Example

```python
from intergrax.integrations.providers.observability_backend.prometheus.bundle import create_prometheus_observability_backend

obs = create_prometheus_observability_backend(base_url="http://prometheus:9090")
instant = obs.query_instant("up")
range_result = obs.query_range("rate(http_requests_total[5m])", start=1710000000, end=1710003600, step="1m")
```

## Notes

PromQL via HTTP API v1.
