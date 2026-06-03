# `clickhouse` integration — usage

**Category:** ``observability_backend``  
**Catalog factory:** ``create_clickhouse_observability_backend()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(observability_backend="clickhouse")
backend = profile.resolve(IntegrationCategory.OBSERVABILITY_BACKEND)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.observability_backend.clickhouse.bundle import create_clickhouse_observability_backend

backend = create_clickhouse_observability_backend(**config_overrides)
```


## Environment variables

`INTERGRAX_CLICKHOUSE_URL`

## Example

```python
from intergrax.integrations.providers.observability_backend.clickhouse.bundle import create_clickhouse_observability_backend

obs = create_clickhouse_observability_backend(base_url="http://localhost:8123")
```

## Notes

High-volume agent event analytics (HTTP SQL facade).
