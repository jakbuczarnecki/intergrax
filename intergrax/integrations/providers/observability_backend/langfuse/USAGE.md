# `langfuse` integration — usage

**Category:** ``observability_backend``  
**Catalog factory:** ``create_langfuse_observability_backend()``  
**Contract integration (pilot):** ``create_langfuse_observability_integration()``

> Tier-3 (application) wires integrations via catalog factories or ``IntegrationProfile``.
> Tier-2 (agents) must **not** import provider slugs or vendor SDKs.

## Common pattern

```python
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

register_default_integrations()
profile = IntegrationProfile(observability_backend="langfuse")
backend = profile.resolve(IntegrationCategory.OBSERVABILITY_BACKEND)
```

Direct factory (preferred in application ``factory.py``):

```python
from intergrax.integrations.providers.observability_backend.langfuse.bundle import create_langfuse_observability_backend

backend = create_langfuse_observability_backend(**config_overrides)
```

## Contract-based observability integration (INTEGRATIONS-2B pilot)

Langfuse now hosts a contract-based observability vendor integration alongside the legacy query facade.

```python
from intergrax.integrations.providers.observability_backend.langfuse.bundle import (
    create_langfuse_observability_integration,
)

integration = create_langfuse_observability_integration(transport=my_transport, enabled=True)
await integration.export(sanitized_envelope)
```

- **``LangfuseObservabilityIntegration``** derives from **`ObservabilityVendorIntegrationContract`**
- **Legacy query facade** — ``create_langfuse_observability_backend()`` / ``ObservabilityBackend`` — remains backward-compatible
- **Sanitized envelope only** — accepts policy-sanitized ``ObservabilityExportEnvelope``; rejects raw ``application_attributes``
- **Raw content is not exported** — prompts, documents, RAG chunks, tool args, secrets, PII, and full local paths are excluded
- **Transport/client injection** — wire an explicit ``LangfuseObservabilityTransport`` for tests and production; disabled by default without transport

Supported signals: ``events``, ``traces``, ``llm_events``.

## Environment variables

`INTERGRAX_LANGFUSE_URL`, `INTERGRAX_LANGFUSE_API_KEY`

## Example

```python
from intergrax.integrations.providers.observability_backend.langfuse.bundle import create_langfuse_observability_backend

obs = create_langfuse_observability_backend(base_url="https://cloud.langfuse.com", api_key="...")
```

## Notes

LLM/agent trace metrics via HTTP (PromQL-shaped facade). Export delivery uses injectable transport — no vendor SDK in the integration class.
