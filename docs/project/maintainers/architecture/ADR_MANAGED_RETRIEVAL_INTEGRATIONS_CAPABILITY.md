# ADR: Managed Retrieval as Canonical Integrations Capability

**Status:** Accepted / Implemented pending independent audit  
**Date:** 2026-09-03  
**Task:** ADR-MANAGED-RETRIEVAL-CAPABILITY-1 · P2-007-R1

## Problem

`ManagedRetrievalBackend` existed with an OpenAI adapter, but provider selection used a special
composition helper (`try_create_managed_retrieval_from_env`) that hardcoded OpenAI when
`OPENAI_API_KEY` was present. The capability was not represented in `IntegrationCategory`,
`IntegrationProfile`, or the canonical integration catalog.

## Decision

`ManagedRetrievalBackend` is a first-class Integrations capability:

```text
Tool
  → ManagedRetrievalBackend
  → ToolWiringContext.managed_retrieval
  → IntegrationProfile.managed_retrieval
  → IntegrationCategory.MANAGED_RETRIEVAL
  → Canonical Integration Catalog
  → Provider factory (e.g. openai)
  → OpenAIManagedRetrievalAdapter
  → OpenAI SDK
```

## Semantic distinction

| Capability | Role |
|---|---|
| `VectorStore` | Raw vector/embedding persistence and similarity search |
| `DocumentStore` | Application/domain document persistence |
| `SearchProvider` | Generic web/enterprise search |
| `ManagedRetrievalBackend` | Provider-hosted ingestion + managed index lifecycle + hosted retrieval/query API |

Do **not** collapse managed retrieval into `VectorStore`, `DocumentStore`, or `SearchProvider`.

## Canonical ownership

- Provider selection: `IntegrationProfile` + catalog only
- Credentials: provider config (`OpenAIManagedRetrievalConfig`, `OPENAI_API_KEY`)
- Tool semantics: `vector_store_id`, default model, instructions (application/tool config)
- OpenAI SDK: isolated in `intergrax/integrations/providers/managed_retrieval/openai/`

## Provider plugin model

External providers register via existing manifest/plugin path:

1. Provider package implementing `ManagedRetrievalBackend`
2. `IntegrationManifest` with `IntegrationCategory.MANAGED_RETRIEVAL`
3. Provider-owned explicit `IntegrationContractSpec` via `declare_integration_contract`
4. `register_from_manifest(..., contract_specs=...)` or `register_integration_plugin(..., contract_specs=...)`
5. Profile binding: `managed_retrieval=<slug>`

No changes to tool service, `RuntimeToolInvoker`, or tool registry are required for Vendor B.

## Configuration ownership

| Layer | Owns |
|---|---|
| Provider | API key, poll interval, poll attempts |
| Tool/application | `vector_store_id`, default model, query/instruction defaults |

## No default vendor

Generic Integrations core does not assume OpenAI. Without profile binding,
`managed_retrieval` is `None` and tools return `managed_retrieval_not_configured`.

## Rejected alternatives

| Alt | Reason |
|---|---|
| A. Use `VectorStore` | Wrong semantics — raw vectors vs provider-managed hosted retrieval |
| B. Use `DocumentStore` | Wrong semantics — application persistence vs hosted provider lifecycle |
| C. Use `SearchProvider` | Wrong semantics — web/enterprise search vs managed document retrieval |
| D. Special OpenAI materialization helper | Bypassed catalog; removed |
| E. Second managed-retrieval registry | Violates single catalog authority |
| F. Tool directly creates provider | Violates DI and provider neutrality |

## registry_v2

Managed retrieval appears in `registry_v2` only via derived projection from canonical catalog
registration: provider-owned explicit `IntegrationContractSpec` →
`register_from_manifest(..., contract_specs=...)` → Integration Catalog → `registry_v2`.
No manual V2 state.

## P2-003 note

Built-in OpenAI registration uses provider-owned explicit `IntegrationContractSpec`
(`intergrax/integrations/providers/managed_retrieval/openai/contract_spec.py`).
Typed `managed_retrieval` category registration fails closed without explicit `contract_specs`.
No reflection-based contract discovery remains on the registration path.
