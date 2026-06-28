# `datadog` integration — usage

**Category:** ``observability_backend``  
**Catalog factory:** ``create_datadog_observability_backend()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(observability_backend="datadog")
backend = profile.resolve(IntegrationCategory.OBSERVABILITY_BACKEND)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.observability_backend.datadog.bundle import create_datadog_observability_backend

backend = create_datadog_observability_backend(**config_overrides)
```


## Environment variables

`INTERGRAX_DATADOG_API_KEY`, optional `INTERGRAX_DATADOG_URL`

## Example

```python
from intergrax.integrations.providers.observability_backend.datadog.bundle import create_datadog_observability_backend

obs = create_datadog_observability_backend(api_key="...")
```

## Notes

Datadog metrics API (instant/range queries).
## Contract-based observability integration (INTEGRATIONS-2C)



Datadog hosts a contract-based observability vendor integration alongside the legacy query facade.



```python

from intergrax.integrations.providers.observability_backend.datadog import (

    create_datadog_observability_integration,

)



integration = create_datadog_observability_integration(transport=my_transport, enabled=True)

await integration.export(sanitized_envelope)

```



- **``DatadogObservabilityIntegration``** derives from **`ObservabilityVendorIntegrationContract`**

- **Legacy query facade** — ``create_datadog_observability_backend()`` / ``ObservabilityBackend`` — remains backward-compatible

- **Sanitized envelope only** — accepts policy-sanitized ``ObservabilityExportEnvelope``; rejects raw ``application_attributes``

- **Raw content is not exported** — prompts, documents, RAG chunks, tool args, secrets, PII, and full local paths are excluded

- **Disabled by default** — ``enabled=False`` unless operator opts in

- **Transport required when enabled** — ``enabled=True`` without ``DatadogObservabilityTransport`` raises ``IntegrationConfigurationError`` immediately (no silent broken export)



Supported signals: ``events``, ``logs``, ``traces``, ``metrics``.



``register_datadog_integration()`` still registers the legacy query facade only; registry v2 / contract registry wiring is deferred.

