# ADR: LLM and Embedding Integration Boundary

**Status:** Accepted — implementation in progress (Phase 1 foundation)
**Date:** 2026-09-05  
**Task:** P2-002-A · P2-002

**Implementation progress:** Phase 1 catalog/category foundation (P2-002-B1) — **IMPLEMENTED_PENDING_INDEPENDENT_AUDIT**. Phase 2 first-party embedding provider catalog registration (P2-002-B2) — **IMPLEMENTED_PENDING_INDEPENDENT_AUDIT**. Phase 2 runtime binding cutover (P2-002-B3) — **IMPLEMENTED_PENDING_INDEPENDENT_AUDIT**. Phase 2 legacy registry removal (P2-002-B4) — **IMPLEMENTED_PENDING_INDEPENDENT_AUDIT** (LLM registry decentralization pending P2-002-C).

## Context

Platform audit finding **P2-002** asked whether `llm_adapters/` and `rag/embedding/` should
remain outside the Integrations catalog, partially migrate, or fully migrate. Integrations
already owns typed platform backend categories (vector store, rerank, managed retrieval,
model serving runtime, secrets store, …) with a single canonical catalog and derived
`registry_v2` projection. LLM and embedding maintain separate provider registries.

## Current state

| Domain | Authority today | Runtime contract |
|---|---|---|
| **Integrations** | Canonical catalog for platform backend categories; `IntegrationProfile` host selection; `registry_v2` derived projection only | Category-specific `*IntegrationContract` (e.g. `VectorStore`, `RerankProvider`) |
| **LLM Adapters** | `LLMProvider` + `LLMAdapterRegistry` + `ModelCatalog` (model metadata) | `LLMAdapter` — generate, stream, tools, structured output, usage |
| **Embedding** | Integrations catalog (`embedding_provider`) + `EmbeddingProfile` + `bind_embedding_provider()` | `EmbeddingProvider` — batch `embed()` → vectors |

Documented boundaries in `INTEGRATIONS.md` and `LLM_ADAPTERS.md` already state that
Integrations does **not** own LLM model invocation and that LLM Adapters are a separate
catalog from Integrations.

`IntegrationCategory` has no `llm_provider` or `embedding_provider` row today.
`model_serving_runtime` (e.g. Ollama host health/list_models) is intentionally distinct from
chat/completion adapters.

## Problem

1. **Ambiguous audit expectation** — “outside Integrations catalog” was read as architectural
   debt, but LLM and embedding are specialized inference domains, not generic integration
   backends.
2. **Partial duplication** — embedding provider discovery/config mirrors Integrations patterns
   (registry, profile, lazy import, central factory map) without reusing catalog authority.
3. **LLM central vendor map** — `LLMAdapterRegistry._BUILTIN_ADAPTERS` is a central slug→module
   table; Integrations architecture forbids this pattern for discovery (P2-003 canon).
4. **Provider vs model identity** — LLM requires separate provider identity (`openai`) and
   model identity (`gpt-4o`); collapsing both into Integrations catalog would blur semantics.

## Decision

Adopt **Option C (hybrid)**:

- **LLM Adapters remain a dedicated domain** — authoritative for provider identity,
  registration, runtime contract, model catalog, routing, failover, token accounting, and
  governance. **Do not** migrate LLM chat/completion into Integrations.
- **Embedding migrates catalog/config/security into Integrations** — new typed category
  `embedding_provider` (parallel to `rerank_provider`). **Retain** `EmbeddingProvider` as the
  RAG-domain runtime contract.
- **Integrations** remains authoritative for platform backend categories and becomes
  authoritative for **embedding provider catalog rows** after migration. It does **not**
  become authoritative for LLM inference runtime.

### Authoritative ownership

| Concern | Authority | Notes |
|---|---|---|
| LLM provider identity (`openai`, `claude`, …) | `llm_adapters` (`LLMProvider`, `LLMAdapterRegistry`) | Not Integrations catalog |
| LLM model identity (`gpt-*`, `claude-*`, …) | `ModelCatalog` (`model_catalog.yaml`) | Separate from provider slug |
| LLM runtime contract | `LLMAdapter` ABC | Streaming, tools, structured output, usage |
| LLM routing / failover | `ModelRouter`, `LLMRoutingEvaluator`, `FailoverLLMAdapter` | Domain runtime router |
| LLM provider registration | `LLMAdapterRegistry` | Evolve to provider-owned explicit registration; remove central `_BUILTIN_ADAPTERS` map |
| Embedding provider identity | Integrations catalog (`embedding_provider`) | `IntegrationProfile.embedding_provider` |
| Embedding model selection | `EmbeddingProfile.model` + host/env | Model is domain config, not catalog slug |
| Embedding runtime contract | `EmbeddingProvider` ABC in `rag/embedding/` | Batch vectors, dimensions, normalization |
| Embedding provider registration | Integrations `register_from_manifest` + provider-owned `runtime_binding.py` | No central registry or factory map |
| Platform backends (vector store, parser, rerank, …) | Integrations catalog | Unchanged |
| Local inference host probe | `model_serving_runtime` Integration | Health/list_models only — not LLM/embedding runtime |
| Connection credentials | Tier-3 host + `SecretsStore` Integration | Domain profiles consume resolved secrets |
| Integrations discovery projection | `registry_v2` | Derived from catalog only; includes `embedding_provider` after migration |
| LLM model metadata projection | `ModelCatalog` | Not `registry_v2` |

### LLM boundary

Integrations **must not** own LLM chat/completion protocol. Forcing LLM through
`PlatformIntegrationContract` would create a leaky abstraction (streaming events, tool-call
envelopes, structured Pydantic validation, per-call token budgets, profile failover chains).

`model_serving_runtime` may supply host reachability for self-hosted stacks shared with LLM
and embedding adapters; it does **not** replace `LLMAdapter`.

### Embedding boundary

Embedding is a **provider category** (like `rerank_provider`) for catalog/config/security,
but RAG owns **embedding semantics** (pipeline, manager, chunk→vector orchestration).

Target flow:

```text
RAG ingest / retriever
  → EmbeddingManager / EmbeddingPipeline
  → EmbeddingProvider (runtime contract)
  → IntegrationProfile.embedding_provider (host selection)
  → Integrations catalog factory
  → provider adapter (OpenAI, Ollama, HF, …)
  → vendor SDK / local runtime
```

### registry_v2 participation

- **Yes** for `embedding_provider` rows after catalog migration (derived projection only).
- **No** for LLM providers or `ModelCatalog` entries.

## Rejected alternatives

| Option | Reason |
|---|---|
| **A. Keep both fully separate** | Leaves embedding catalog duplication; does not align embedding with `rerank_provider` precedent |
| **B. LLM separate; embedding fully in Integrations runtime** | Collapses RAG embedding semantics into generic Integration contract; loses batch/dimension runtime specialization |
| **D. Migrate both LLM and embedding fully into Integrations** | Unacceptable abstraction loss for LLM; high churn across Nexus/runtime/applications with no proportional benefit |

## Migration implications

Implementation is **out of scope** for P2-002-A. Logical blocks:

1. **Contract/catalog alignment** — add `IntegrationCategory.EMBEDDING_PROVIDER`, category
   contract, `IntegrationProfile` slot, `PROVIDER_CATEGORY_CONTRACT_REGISTRY` entry.
2. **Provider migration** — move embedding provider packages to
   `integrations/providers/embedding_provider/<slug>/` with explicit `IntegrationContractSpec`;
   register via `register_from_manifest`.
3. **Runtime binding** — `EmbeddingPipeline` resolves provider via `IntegrationProfile` → catalog
   runtime binder → bound `EmbeddingProvider` adapter (**B3/B4 complete**).
4. **LLM registry hardening** — replace `_BUILTIN_ADAPTERS` central map with provider-owned
   registration modules (same pluginability invariant as P2-003; **no** Integrations migration).
5. **Compatibility removal** — legacy `EmbeddingProviderRegistry`, `create_default_registry()`,
   and central `provider_factory_registration` removed in B4; hosts bind one typed provider.
6. **Tests/docs** — registry, plugin, and profile resolution tests; update RAG operator guide.
7. **Final audit** — independent pass confirming single catalog authority for embedding providers.

## Invariants

1. One authoritative catalog per concern — no parallel embedding provider discovery authority
   after migration.
2. LLM provider/model identity separation preserved (`LLMProvider` ≠ `ModelCatalog.model_id`).
3. No new central vendor `if` maps or reflection-based contract discovery on registration paths.
4. `registry_v2` remains a derived read model only.
5. Tier-0 import boundaries unchanged (`intergrax/` must not import `agents/` or `applications/`).
6. Credentials never appear in contract metadata `public_view()`.

## Rollout phases

| Phase | Scope |
|---|---|
| **0 (now)** | Document decision; no production code changes |
| **1** | Embedding catalog category + first-party provider packages in Integrations | **B1 foundation complete** (category, profile slot, typed contract, registry participation). **B2 catalog registration complete** — five first-party `embedding_provider` packages registered via explicit `IntegrationContractSpec`. **B3 runtime binding complete** — canonical provider selection flows through `IntegrationProfile` + Integrations catalog runtime binders; `EmbeddingProvider` remains RAG runtime contract; env `EmbeddingProfile` retained as compatibility input |
| **2** | `IntegrationProfile` + RAG bootstrap cutover; env-based `EmbeddingProfile` compatibility | **B3 complete** — pending independent audit |
| **3** | Remove legacy `EmbeddingProviderRegistry` bootstrap map | **B4 complete** — legacy registry removed; runtime consumers use bound `EmbeddingProvider`; Integrations is single provider authority |
| **4** | LLM `LLMAdapterRegistry` provider-owned registration (decentralize `_BUILTIN_ADAPTERS`) | **P2-002-C pending** |

## Compatibility policy

- **LLM:** Stable public API (`LLMAdapter`, `LLMProfile`, `LLMAdapterRegistry`) — no breaking
  changes during Option C; registry internal refactor only.
- **Embedding:** Maintain `EmbeddingProvider` ABC and `EmbeddingProfile`; legacy
  `EmbeddingProviderRegistry` and `create_default_registry()` removed in B4.
- **Legacy paths** with no production consumers: remove rather than indefinite shim (per audit
  rule).
