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
## Contract-based observability integration (INTEGRATIONS-2C)



Clickhouse hosts a contract-based observability vendor integration alongside the legacy query facade.



```python

from intergrax.integrations.providers.observability_backend.clickhouse import (

    create_clickhouse_observability_integration,

)



integration = create_clickhouse_observability_integration(transport=my_transport, enabled=True)

await integration.export(sanitized_envelope)

```



- **``ClickhouseObservabilityIntegration``** derives from **`ObservabilityVendorIntegrationContract`**

- **Legacy query facade** — ``create_clickhouse_observability_backend()`` / ``ObservabilityBackend`` — remains backward-compatible

- **Sanitized envelope only** — accepts policy-sanitized ``ObservabilityExportEnvelope``; rejects raw ``application_attributes``

- **Raw content is not exported** — prompts, documents, RAG chunks, tool args, secrets, PII, and full local paths are excluded

- **Disabled by default** — ``enabled=False`` unless operator opts in

- **Transport required when enabled** — ``enabled=True`` without ``ClickhouseObservabilityTransport`` raises ``IntegrationConfigurationError`` immediately (no silent broken export)



Supported signals: ``events``, ``logs``, ``traces``, ``metrics``.



``register_clickhouse_integration()`` still registers the legacy query facade only; registry v2 / contract registry wiring is deferred.

