# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

# `tempo` integration — usage

**Category:** `observability_backend`  
**Catalog factory:** ``create_tempo_observability_backend()``  
**Env prefix:** ``INTERGRAX_TEMPO_*``

```python
from intergrax.integrations.providers.observability_backend.tempo.bundle import create_tempo_observability_backend

backend = create_tempo_observability_backend()
```
## Contract-based observability integration (INTEGRATIONS-2C)



Tempo hosts a contract-based observability vendor integration alongside the legacy query facade.



```python

from intergrax.integrations.providers.observability_backend.tempo import (

    create_tempo_observability_integration,

)



integration = create_tempo_observability_integration(transport=my_transport, enabled=True)

await integration.export(sanitized_envelope)

```



- **``TempoObservabilityIntegration``** derives from **`ObservabilityVendorIntegrationContract`**

- **Legacy query facade** — ``create_tempo_observability_backend()`` / ``ObservabilityBackend`` — remains backward-compatible

- **Sanitized envelope only** — accepts policy-sanitized ``ObservabilityExportEnvelope``; rejects raw ``application_attributes``

- **Raw content is not exported** — prompts, documents, RAG chunks, tool args, secrets, PII, and full local paths are excluded

- **Disabled by default** — ``enabled=False`` unless operator opts in

- **Transport required when enabled** — ``enabled=True`` without ``TempoObservabilityTransport`` raises ``IntegrationConfigurationError`` immediately (no silent broken export)



Supported signals: ``events``, ``logs``, ``traces``, ``metrics``.



``register_tempo_integration()`` still registers the legacy query facade only; registry v2 / contract registry wiring is deferred.

