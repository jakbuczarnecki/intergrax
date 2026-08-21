<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# LangChain Independence — Multi-layer Feature Architecture

**Status:** LCI-0A–0C **APPROVED**; LCI-1A–1D **APPROVED**; LCI-2A–2F **APPROVED**; LCI-3A–3D **APPROVED**; LCI-4A–4D **APPROVED**; LCI-5A–5C **APPROVED**; LCI-6A–6E **APPROVED**; Native Ollama regression gate **APPROVED**; LCI-7A–7D **APPROVED**; FINAL SYSTEM GATE **APPROVED**; LCI-8A **APPROVED**; LangChain Independence **COMPLETE / APPROVED**.
**Roadmap status:** COMPLETE / APPROVED — historical LCI program closed; Protocol-v2 residual remediation **ACCEPTED / PLANNED** (2026-08-21).
**Feature plan (1:1):** [`../plan/LANGCHAIN_INDEPENDENCE.md`](../plan/LANGCHAIN_INDEPENDENCE.md)
**Primary anchor domain:** `RAG`
**Related domains:** `LLM_ADAPTERS`, `INTEGRATIONS`, `MEMORY`, `MODALITY`, `ORCHESTRATION`, `PLATFORM_FOUNDATION`, `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE`
**Current active task:** NONE — PROGRAM CLOSED; Protocol-v2 remediation blocks **PLANNED** only
**LangGraph decision:** KEEP_OPTIONAL — retain the optional legacy boundary; any future deprecation/removal requires a separately approved product/architecture decision
**Next task after acceptance:** NONE — PROGRAM CLOSED

**LCI-2F decision:** Loader output, normalization, metadata enrichment, chunking, and contextual enrichment remain `KnowledgeDocument` stages. The native RAG path continues through embedding, indexing, vector storage and retrieval; LangChain loaders, embeddings and splitters remain optional provider-local compatibility paths.

**LCI-3A decision:** The embedding API accepts `KnowledgeDocument`; `EmbeddingResult` stores native documents and readonly float32 vectors separately. Vectors are never stored in native metadata.

**LCI-3B decision:** `IndexingManager`, `IndexingPipeline`, and index strategies accept native `KnowledgeDocument` sequences. Single and Dual Index call native `embed_documents`; Dual TOC records are deterministic native derivatives with scope-safe grouping and preserved lineage. Legacy conversion is now private to the native vector-store manager/provider boundary.

**LCI-3C decision:** Public vector-store manager contracts use immutable native records, explicit `VectorStoreScope`, and native hits containing `KnowledgeDocument`. Tenant and namespace are authoritative routing fields and cannot be supplied through user metadata. Delete and count fail closed unless tenant isolation is proven.

**LCI-3D decision:** The VectorStore provider port is native. Providers map `VectorStoreRecord` directly to provider SDK payloads and return native `VectorStoreHit`. Tenant, namespace and workspace routing are system-owned at every provider boundary; LangChain document compatibility is removed from vector-store paths.

**LCI-4A decision:** Retrieval uses a native immutable hit/result contract containing `KnowledgeDocument`, score and rank. Active retrievers and RAG tools do not expose LangChain `Document`. `VectorStoreHit` remains the provider/vector-store result and is mapped at the retriever boundary. Reranking remains a separate LCI-4B boundary. Graph retrieval remains LCI-4C.

**LCI-4B decision:** Reranking uses one immutable native `RerankerCandidate` containing `KnowledgeDocument` and original retrieval score/rank. `RerankerResult` preserves the original candidate and adds only rerank/fusion score and final rank. Active core rerankers and integration providers do not accept or return LangChain `Document`. Reranking cannot alter document identity, scope, provenance or user metadata. Graph RAG remains LCI-4C.

**LCI-4C decision:** Graph RAG indexers and graph isolation contracts accept native `KnowledgeDocument` values. Document identity, tenant scope, namespace, provenance and user metadata remain owned by the native document boundary. Graph indexers read `document.content` and do not accept or construct LangChain `Document`. Graph retrieval uses the existing native `RetrievalHit` contract. GraphStore backend internals remain unchanged. Auxiliary memory, multimedia, legacy RAG, evaluation and soak paths remain LCI-4D.

**LCI-4C-A1 decision:** `workspace_id` is a canonical system-owned `KnowledgeDocumentScope` field. The canonical identity boundary is `tenant_id + namespace + workspace_id + document_id`; missing `workspace_id` remains backward-compatible and means no explicit workspace partition. User metadata cannot provide or override `workspace_id`. Indexing, vector storage, retrieval, reranking and Graph RAG preserve workspace scope without using metadata as a transport tunnel.

**LCI-4D decision:** Memory indexing, multimedia document loaders, legacy RAG answer contracts, evaluation harnesses and soak tooling use the canonical `KnowledgeDocument` boundary. Auxiliary runtime paths preserve canonical identity, tenant, namespace, workspace and provenance without using user metadata as system transport. No active LCI-4D production module imports or exposes LangChain `Document`. Provider-local loaders and embedding dependencies remain optional compatibility surfaces.

**LCI-5A decision:** Plain-text parsing uses a native Intergrax reader and emits
`ParsedDocumentFragment` directly. The default text parser no longer imports or
constructs LangChain `TextLoader` or LangChain `Document`. Text decoding
preserves supported encoding behavior without introducing a new required
dependency. Provider-local LangChain document loaders remain assigned to
LCI-5C.

**LCI-5B decision:** OpenAI, vLLM and llama.cpp embedding providers use a shared native
OpenAI-compatible embeddings transport built on the OpenAI SDK. No LCI-5B provider
imports or constructs `langchain-openai` `OpenAIEmbeddings`. Provider-specific model,
base URL and credential semantics remain owned by the provider wrappers.
The `EmbeddingProvider` ABI, float32 output, shape and lazy dimension resolution
remain unchanged.

**LCI-5C decision:** Provider-local LangChain document loaders and the Ollama
embedding shim remain compatibility implementations, but their imports are lazy
and controlled by `rag-langchain-loaders` / `rag-langchain-embeddings`. Missing
extras raise a stable provider configuration error; core import, registry
discovery and native RAG contracts remain LangChain-safe. `langchain-ollama`
remains available through explicit compatibility extras while the native Ollama
adapter owns the default path.

**LCI-6A decision:** The target `NativeOllamaAdapter` implements the existing
Intergrax `LLMAdapter` ABI and introduces no second public LLM contract.
`LangChainOllamaAdapter` remains explicitly constructible as the compatibility
and parity baseline. LCI-6B is implementation behind a non-default path,
LCI-6C supplies mandatory live Ollama evidence, LCI-6D makes
`NativeOllamaAdapter` the default resolver target, and LCI-6E moves the
LangChain adapter to compatibility-only packaging behind
`llm-langchain-ollama`. The full target and matrix are in the LCI-6A satellite.

**LCI-2E decision:** Native `RecursiveChunkingStrategy` is the default and core-safe baseline. `LangChainRecursiveChunkingStrategy` is an optional provider behind the `rag-langchain-splitters` extra, loaded only on explicit construction or registry registration; missing the extra produces a stable configuration error without silent fallback.

`ParsedDocumentFragment` is extraction-stage only. `KnowledgeDocument` remains the canonical RAG knowledge ABI.

**Dependency inventory satellite:** [`satellites/LANGCHAIN_INDEPENDENCE_dependency_inventory.md`](satellites/LANGCHAIN_INDEPENDENCE_dependency_inventory.md)

**LCI-6A parity satellite:** [`satellites/OLLAMA_NATIVE_ADAPTER_PARITY_MATRIX.md`](satellites/OLLAMA_NATIVE_ADAPTER_PARITY_MATRIX.md)

---

## Cursor read scope (token budget)

**Do not read this entire file in one session.**

- **Default:** §Purpose, §Strategic decision, §As-built baseline, §Target architecture, §Import zones.
- **Inventory work:** load **only** [`satellites/LANGCHAIN_INDEPENDENCE_dependency_inventory.md`](satellites/LANGCHAIN_INDEPENDENCE_dependency_inventory.md) — not both hub sections and full inventory.
- **Plan / task selection:** [`../plan/LANGCHAIN_INDEPENDENCE.md`](../plan/LANGCHAIN_INDEPENDENCE.md) read-scope block + active `LCI-*` section only.
- **Satellites:** at most **one** `architecture/satellites` or `plan/satellites` file per session unless `RESUME:` cites more.

---

## Purpose

LangChain Independence is a cross-layer platform initiative to **own Intergrax public contracts and core runtime types**, not to ideologically remove every LangChain package from the ecosystem.

Intergrax already defines native LLM message envelopes, adapter contracts, and provider boundaries. The remaining work is to **stop LangChain types from defining the platform ABI** — especially `langchain_core.documents.Document` across the RAG, memory, modality, and integration surfaces — while preserving optional compatibility for provider bridges and legacy paths.

**LKW** (Local Knowledge Workspace) is a **client and proof workload** for platform capabilities, including future LangChain-free RAG paths. LKW does **not** own the migration mechanics; domain plans and this feature plan do.

---

## Strategic decision

```text
LangChain-free core + optional LangChain compatibility/providers.
```

| Rule | Meaning |
|------|---------|
| **Core rule** | No LangChain/LangGraph type may appear in Intergrax public or shared core contracts after migration. |
| **Compatibility rule** | LangChain may remain behind explicit provider or `compat` boundaries, optional extras, and legacy loaders. |
| **LangGraph position** | Not a core dependency today; optional `langgraph-legacy` extra only; LCI-8A decision is `KEEP_OPTIONAL`. |
| **LKW position** | Proof client only; migration ownership stays in Tier-0/Tier-1 domain and feature plans. |

### Current capability classification

| Surface | Current classification | Contract and packaging rule |
|---|---|---|
| **CORE / DEFAULT** | Native Intergrax contracts and providers | Default installation requires none of the LangChain/LangGraph extras. Native LLM paths, native Ollama, and native RAG contracts are canonical. |
| **OPTIONAL COMPATIBILITY** | Selected LangChain loaders, embeddings, splitters, and `LangChainOllamaAdapter` | Keep behind a provider/compatibility boundary, lazy import, named extra, and controlled missing-extra error. |
| **LEGACY OPTIONAL** | LangGraph supervisor/web-search adapters | Not a core dependency; retain under `langgraph-legacy` as the `LCI-8A` optional compatibility boundary. |

Current packaging extras are `llm-langchain-ollama`, `rag-langchain-loaders`,
`rag-langchain-embeddings`, `rag-langchain-splitters`, and
`langgraph-legacy`. Explicit opt-in enables these compatibility paths; the
default installation enables none of them. Transitive package versions are
resolver output, not a stable public contract.

**LCI-8A decision:** `KEEP_OPTIONAL`. The LangGraph package remains isolated
behind explicit opt-in; Nexus, Harness, agent execution and default
application runtime do not require it. The native LangGraph-compatible
skill-pack importer remains a profile-gated format compatibility path and
does not import the package.

Historical or invalid claims that must not be emitted as current state:

- LangChain has been fully removed.
- Intergrax contains no LangChain code.
- LangChain is required by the default/core installation.
- `LangChainOllamaAdapter` is the default Ollama provider.
- `langchain_core.documents.Document` is a native RAG core contract.

---

## As-built baseline (evidence-backed)

Facts verified against repository state at inventory time.

| # | Fact | Evidence |
|---|------|----------|
| 1 | Intergrax defines native LLM messages independent of LangChain | `intergrax/llm/messages.py` — `ChatMessage`, `MessageRole`, `AttachmentRef` |
| 2 | LLM adapter public contract is native; providers map at boundary | `intergrax/llm_adapters/contracts/llm_adapter.py` — `LLMAdapter` uses `ChatMessage`, not LangChain messages |
| 3 | Canonical/default LLM provider paths use native SDKs; remaining LangChain provider code is explicit compatibility-only | Provider modules under `intergrax/llm_adapters/providers`, the native default resolver, and `pyproject.toml` optional compatibility extras |
| 4 | `NativeOllamaAdapter` is the canonical/default Ollama LLM path; `LangChainOllamaAdapter` remains an optional compatibility provider behind `llm-langchain-ollama` | `intergrax/llm_adapters/llm_provider_registry.py`, `intergrax/llm_adapters/providers/native_ollama_adapter.py`, `intergrax/llm_adapters/providers/ollama_adapter.py`, `pyproject.toml` |
| 5 | Core LangChain document contract leaks have been removed. Remaining LangChain production imports are optional provider / compatibility boundaries | `satellites/LANGCHAIN_INDEPENDENCE_dependency_inventory.md` — **0 core contract leaks** and **0 core implementation dependencies** |
| 6 | LangGraph is not a core dependency; guard exists | `scripts/maintenance/check_langgraph_not_required.py`; `pyproject.toml` — `langgraph` only under `[project.optional-dependencies] langgraph-legacy` |
| 7 | Default packaging no longer declares LangChain/LangGraph packages as direct core dependencies after LCI-7A | `pyproject.toml` `[project].dependencies` is free of `langchain*` and `langgraph`; compatibility/provider packages remain in named extras |

**Import audit scale:** 11 direct production/runtime import statements · 46 direct test import statements · 1 direct tooling import · 2 direct LangGraph lazy imports · 0 core contract leaks · 0 core implementation dependencies · 0 project core LangChain dependencies (see inventory satellite §B/§D and `pyproject.toml`). All remaining production imports are optional providers, compatibility-only adapters, or legacy optional orchestration.

**LCI-7B evidence:** [`satellites/LANGCHAIN_FREE_CORE_INSTALLATION_GATE.md`](satellites/LANGCHAIN_FREE_CORE_INSTALLATION_GATE.md) records the default-install proof: zero `langchain*` and `langgraph*` distributions, native/core/RAG/Nexus/Harness smoke PASS.

**LCI-7C evidence:** [`satellites/LANGCHAIN_COMPATIBILITY_INSTALLATION_GATE.md`](satellites/LANGCHAIN_COMPATIBILITY_INSTALLATION_GATE.md) records PASS for all five compatibility families while native defaults remain native. It also records the later deterministic Torch/Transformers incompatibility, caused by Transformers v5 resolving against supported torch 2.2.2, repaired with `transformers>=4.41,<5`, and successfully requalified.

---

## Target architecture

```text
native Intergrax contracts
    ↓
native runtime/domain implementations
    ↓
optional compatibility/provider boundary
    ↓
LangChain packages (optional extras)
```

**Forbidden direction after migration:**

```text
LangChain type → Intergrax public/core contract
```

Native document, message, and tool-call types are defined and versioned by Intergrax first. Provider adapters translate at the outer boundary only.

### Extension guidance

New native/core providers implement the native Intergrax contract. A LangChain
integration belongs behind an optional provider or compatibility boundary with
a lazy import, a named extra, and a controlled missing-extra error. It must
not make LangChain a core dependency or expose LangChain types through a
canonical contract.

---

## Import zones

### Docelowo zabronione

LangChain/LangGraph imports must not appear in:

```text
intergrax/**/contracts/
intergrax/runtime/
agents/
applications/
```

and any other canonical core path where a foreign type would become part of the Intergrax ABI (public re-exports, shared DTOs consumed across domains, Nexus orchestration contracts).

Enforcement begins at **`LCI-0B`** (architecture boundary guard). See [LCI-0B enforcement boundary](.#lci-0b-enforcement-boundary) below.

### Docelowo dozwolone warunkowo

Only behind explicit, reviewable boundaries:

```text
intergrax/compat/langchain/          (LCI-1C — APPROVED)
intergrax/integrations/providers/.../
intergrax/llm_adapters/providers/.../
intergrax/legacy/
tests/
```

Provider paths are allowed only when:

1. LangChain types do not leak into public contracts.
2. The dependency can be optional (missing package does not break core import).
3. Provider maps outward/inward to native Intergrax types at the boundary.
4. No provider object escapes the provider module.

**Note:** `intergrax/compat/langchain` provides the isolated LangChain `Document` bridge (`LCI-1C`, **APPROVED**); canonical native type remains `KnowledgeDocument`; the LangChain bridge remains compatibility-only; **LCI-1D** enforces native document ABI; dependency optionalization and both installation gates are **APPROVED** under **LCI-7A**–**LCI-7C**.

---

## Domain ownership matrix

| Domain | Ownership in LCI |
|--------|------------------|
| **RAG** | Native knowledge document type; ingest, chunking, embedding, indexing, vector store, retrieval, rerank, graph pipelines |
| **LLM_ADAPTERS** | Provider boundaries; native Ollama sequence `LCI-6A`–`LCI-6E`; LangChain Ollama shim optionalized in `LCI-6E` |
| **INTEGRATIONS** | Optional provider loading; document-parser and vector-store bridges (`LCI-3D`, `LCI-5C`); community loader isolation (`LCI-5A`, `LCI-5C`) |
| **MEMORY** | Remove `Document` from session/profile indexing (`LCI-4D`) |
| **MODALITY** | Multimedia smart loaders — native document output at modality boundary (`LCI-4D`) |
| **ORCHESTRATION** | LangGraph legacy boundary (`LCI-8A`); Nexus ingestion native document path (`LCI-2F`) |
| **PLATFORM_FOUNDATION** | Packaging optional extras (`LCI-7A`); lockfile regeneration; documentation closeout (`LCI-7D`) |
| **EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE** | Boundary guard (`LCI-0B`); conformance gate (`LCI-1D`); install gates (`LCI-7B`, `LCI-7C`) |

---

## Migration invariants

1. **No big-bang rewrite** — incremental tasks with acceptance gates.
2. **One migration task active at a time** per stream unless explicitly parallelized in plan.
3. **Contract first, consumer second** — native type before downstream refactors.
4. **No removal before parity proof** — keep LangChain path until native path is proven equivalent.
5. **LKW is client/proof workload** — not owner of document or adapter mechanics.
6. **No application exceptions in platform core** — `applications` cannot justify contract leaks in `intergrax`.
7. **Compatibility bridge ≠ second canonical model** — `compat/langchain` maps; it does not replace native contracts.
8. **Preserve content, metadata, identity, provenance, tenant scope** across migrations.
9. **Provider-specific objects stay inside provider boundary.**
10. **Each stage has its own acceptance gate** (see feature plan).

---

## Current risk assessment

| Risk | Description |
|------|-------------|
| **Contract lock-in** | Historical migration risk: `Document` once appeared in 16 public contracts; the current inventory reports zero core contract leaks. |
| **Packaging risk** | Optional compatibility/provider packages must stay out of default `[project].dependencies`; LCI-7B clean-install evidence is PASS. |
| **Version range risk** | Broad `>=0.3` ranges on multiple LangChain packages increase resolver drift and breaking-change exposure. |
| **Transitive dependency risk** | LangChain meta-packages pull large transitive trees (community, text-splitters, OpenAI shims). |
| **Migration regression risk** | Dual-model transition (LangChain `Document` vs native document) can silently drop metadata or tenant fields. |
| **Dual-model transition risk** | Parallel types during migration require strict conversion tests (`LCI-1D`, conformance gate). |
| **LKW/Ollama parity risk** | LKW proof paths use the native default after `LCI-6D`; the LangChain adapter remains an explicit compatibility baseline, not the default. |

---

## Final acceptance definition (full LCI program)

The LangChain Independence program (`LCI-0A` … `LCI-8A`) is **COMPLETE /
APPROVED**. All listed implementation tasks, the Native Ollama regression gate,
and the FINAL SYSTEM GATE are accepted.

Final architecture statement:

- Default/core installation requires no LangChain.
- Default/core installation requires no LangGraph.
- Native Intergrax contracts/providers are canonical.
- `NativeOllamaAdapter` is the default Ollama adapter.
- Native RAG uses `KnowledgeDocument`.
- Selected LangChain providers remain explicit optional compatibility extras.
- `langgraph-legacy` remains optional under the accepted `KEEP_OPTIONAL` decision.

The validated gates are:

1. Public Intergrax contracts expose **no** LangChain types.
2. Default `pip install` / `uv sync` **does not** require any `langchain*` package for core import of `intergrax`.
3. RAG ingest → index → retrieve → rerank runs on **native document types** with parity proof.
4. Memory and modality paths use native documents at indexing boundaries.
5. Ollama LLM and embedding paths have **native adapters** with parity proof; LangChain Ollama shims are optional extras only.
6. `check_langchain_boundary` (or successor) passes in CI for forbidden zones (`LCI-0B`); LangChain-free core install gate passes (`LCI-7B`).
7. LangChain compatibility is isolated under `compat` and/or provider modules with optional extras.
8. LangGraph remains optional under the accepted `KEEP_OPTIONAL` decision.
9. LKW proof workload passes on LangChain-free core install (LKW as client, not owner).
10. Domain plan rows updated; inventory satellite shows `unclassified occurrences = 0` and zero core contract leaks.

---

## LCI-1A — Native knowledge document contract (summary)

Canonical ABI: **`KnowledgeDocument`** in neutral Tier-0 module `intergrax/knowledge/contracts/document.py` (implementation: **LCI-1B**, status **APPROVED**). KnowledgeDocument remains canonical; the LangChain bridge remains compatibility-only; **LCI-1D** enforces native document ABI; dependency optionalization and both installation gates are **APPROVED** under **LCI-7A**–**LCI-7C**.

| Decision | Value |
|----------|-------|
| Public import | `from intergrax.knowledge.contracts import KnowledgeDocument` |
| Functional owner | RAG |
| Shared consumers | Memory, modality, integrations |
| Replaces | `langchain_core.documents.Document` in public contracts (16 inventory leaks) |
| Schema version | `1` — immutable Pydantic v2, `extra="forbid"`, `frozen=True` |

Sub-models: `KnowledgeDocumentIdentity` (persistent IDs + lineage), `KnowledgeDocumentScope` (required `tenant_id`), `KnowledgeDocumentProvenance` (source trace). Content is non-empty `str` only; binary/media normalized before document creation. Vendor Knowledge models inform semantics but are not imported — fetch-stage → RAG-ready normalization is a separate adapter step.

**Full specification:** [`satellites/LANGCHAIN_INDEPENDENCE_native_document_contract.md`](satellites/LANGCHAIN_INDEPENDENCE_native_document_contract.md)

---

## LCI-0B enforcement boundary

`LCI-0B` adds a deterministic AST-based CI guard that freezes existing LangChain/LangGraph production import violations while blocking new leaks into protected zones.

| Aspect | Policy |
|--------|--------|
| **Scan roots** | `intergrax`, `agents`, `applications` production `*.py` files |
| **Excluded paths** | any path containing `tests`, `__pycache__`, non-`*.py` files, `docker/runtime-context` copies |
| **Allowed production zones** | `intergrax/compat/langchain`, `intergrax/integrations/providers`, `intergrax/llm_adapters/providers`, `intergrax/legacy` |
| **Provider import rule** | LangChain imports under `intergrax/integrations/providers` must be lazy function-bound imports; `TYPE_CHECKING` imports are deferred-only and accepted. Eager provider imports fail the boundary audit. |
| **Protected production zones** | all other scanned production paths (contracts, runtime, RAG, memory, modality, integrations shared bridges, agents, applications, etc.) |
| **Detected namespaces** | `langchain`, `langchain_*`, `langgraph` |
| **Import forms** | `import`, `from`, nested function imports, literal `importlib.import_module("…")`, literal `__import__("…")` |
| **Grandfather model** | `scripts/maintenance/langchain_boundary_grandfather.json` — exact fingerprint entries (`path`, `kind`, `module`, sorted `names`) matched to approved `LCI-INV-####` inventory rows; line numbers are not part of identity |
| **New vs stale** | `current guarded − grandfathered` → `NEW_FORBIDDEN_IMPORT`; `grandfathered − current` → `STALE_GRANDFATHER_ENTRY` (debt removal requires register cleanup) |
| **Automatic baseline update** | **none** — no `--update-baseline` / write mode |
| **CI wiring** | PR smoke (`ci-smoke`) and full governance (`gate-governance-tier`) via `scripts/maintenance/check_langchain_boundary.py` |
| **LangGraph guard relationship** | `check_langgraph_not_required.py` remains — core packaging/non-optional LangGraph dependency guard; `LCI-0B` covers all LangGraph production imports in protected zones with inventory-backed grandfathering |
| **Register maintenance** | When a grandfathered import is removed during debt paydown, delete the matching register entry in the same change |

---

## Protocol v2 langchain independence target invariants (2026-08-18)

Accepted [`LANGCHAIN_INDEPENDENCE`](../../../audit_results/2026-08-18/LANGCHAIN_INDEPENDENCE.md) findings **01–06** (2026-08-21). Remediation **ACCEPTED / PLANNED** — **not implemented** by audit persistence.

1. **Trusted scope injection** — compatibility conversion (`from_langchain_document` and callers) receives trusted canonical scope (`tenant_id`, `namespace`, `workspace_id`) separately from foreign metadata. Foreign LangChain or provider metadata may only match or confirm trusted scope; mismatch fails closed. Untrusted metadata cannot establish tenant/workspace/namespace authority. Cross-link [`RAG`](../../architecture/RAG.md) **RAG-SCOPE-CONTRACT-INTEGRITY** and **IDENTITY_TRUST** where applicable.
2. **Provider-hit identity validation** — legacy/provider conversion (`from_legacy_rag_hit` and equivalent paths) receives expected trusted scope and verifies returned routing identity against it. Provider-returned metadata cannot mint system ownership or provenance authority. Do not create a second document contract.
3. **Conditional provider exemption** — boundary guard allows specific reviewed compatibility boundaries/capabilities, not entire provider directories. Equivalent eager/optional-boundary enforcement applies to every allowed provider family (`integrations/providers`, `llm_adapters/providers`, `compat/langchain`, `legacy`). New provider compatibility use requires explicit qualification.
4. **Robust static dynamic-import detection** — LCI-0B tracks common statically resolvable `importlib` aliases and `import_module` aliases (`import importlib as il`, `from importlib import import_module`, etc.) with adversarial regression fixtures. No need for general Python execution or unrestricted dynamic analysis.
5. **Explicit compatibility packaging semantics** — either native parsing extras remain LangChain-free and compatibility loader dependencies live only in named compatibility extras (`llm-langchain-ollama`, `rag-langchain-loaders`, `rag-langchain-embeddings`, `rag-langchain-splitters`, `langgraph-legacy`), or docs/package contract explicitly declares transitive opt-in (e.g. `parsing-office` / `parsing-pdf` → `langchain-community`).
6. **Historical inventory vs current conformance evidence** — preserve historical migration inventory (`LANGCHAIN_INDEPENDENCE_dependency_inventory.md` pinned to migration-era SHA) as historical evidence; maintain a separate mechanically generated/current conformance evidence record pinned to a specific repository SHA. Architecture must not present stale inventory counts as current-state proof.

Preserved: LangChain-free core strategy; **KEEP_OPTIONAL** LangGraph decision; historical LCI-0A..8A delivery/APPROVED facts; native `KnowledgeDocument` ownership; native Ollama default; optional compatibility philosophy. Protocol-v2 FAIL does not undo historical migration delivery.

---

## Related documents

| Document | Role |
|----------|------|
| [`../plan/LANGCHAIN_INDEPENDENCE.md`](../plan/LANGCHAIN_INDEPENDENCE.md) | Task roadmap `LCI-0A`–`LCI-8A` |
| [`satellites/LANGCHAIN_INDEPENDENCE_dependency_inventory.md`](satellites/LANGCHAIN_INDEPENDENCE_dependency_inventory.md) | Evidence-backed import and package inventory |
| [`satellites/LANGCHAIN_FREE_CORE_INSTALLATION_GATE.md`](satellites/LANGCHAIN_FREE_CORE_INSTALLATION_GATE.md) | LCI-7B default-install evidence |
| [`satellites/LANGCHAIN_COMPATIBILITY_INSTALLATION_GATE.md`](satellites/LANGCHAIN_COMPATIBILITY_INSTALLATION_GATE.md) | LCI-7C optional compatibility evidence |
| [`satellites/LANGCHAIN_INDEPENDENCE_CLOSEOUT.md`](satellites/LANGCHAIN_INDEPENDENCE_CLOSEOUT.md) | LCI-7D closeout receipt |
| [`satellites/LANGCHAIN_INDEPENDENCE_native_document_contract.md`](satellites/LANGCHAIN_INDEPENDENCE_native_document_contract.md) | Native `KnowledgeDocument` contract spec (LCI-1A) |
| [`../plan/satellites/LANGCHAIN_INDEPENDENCE_domain_plan_cross_references.md`](../plan/satellites/LANGCHAIN_INDEPENDENCE_domain_plan_cross_references.md) | Domain plan routing for future rows |
| [`../../architecture/RAG.md`](../../architecture/RAG.md) | Primary anchor domain architecture |
