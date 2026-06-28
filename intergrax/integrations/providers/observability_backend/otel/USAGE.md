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

register_default_integrations()
profile = IntegrationProfile(observability_backend="otel")
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
## Contract-based observability integration (INTEGRATIONS-2C)



OTel hosts a contract-based observability vendor integration alongside the legacy query facade.



```python

from intergrax.integrations.providers.observability_backend.otel import (

    create_otel_observability_integration,

)



integration = create_otel_observability_integration(transport=my_transport, enabled=True)

await integration.export(sanitized_envelope)

```



- **``OtelObservabilityIntegration``** derives from **`ObservabilityVendorIntegrationContract`**

- **Legacy query facade** — ``create_otel_observability_backend()`` / ``ObservabilityBackend`` — remains backward-compatible

- **Sanitized envelope only** — accepts policy-sanitized ``ObservabilityExportEnvelope``; rejects raw ``application_attributes``

- **Raw content is not exported** — prompts, documents, RAG chunks, tool args, secrets, PII, and full local paths are excluded

- **Disabled by default** — ``enabled=False`` unless operator opts in

- **Transport required when enabled** — ``enabled=True`` without ``OtelObservabilityTransport`` raises ``IntegrationConfigurationError`` immediately (no silent broken export)



Supported signals: ``events``, ``logs``, ``traces``, ``metrics``.



``register_otel_integration()`` still registers the legacy query facade only; registry v2 / contract registry wiring is deferred.

