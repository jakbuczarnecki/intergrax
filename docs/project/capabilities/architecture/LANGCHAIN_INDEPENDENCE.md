<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# LangChain Independence — Multi-layer Feature Architecture

**Status:** LCI-3D **APPROVED**; LCI-4A **APPROVED**; LCI-4B **APPROVED**; LCI-4C-A1 **APPROVED**; LCI-4C **APPROVED**; LCI-4D **APPROVED**; LCI-5A **APPROVED**; LCI-5B **READY_FOR_REVIEW**.
**Roadmap status:** LCI-3C **APPROVED**; LCI-3D **APPROVED**; LCI-4A **APPROVED**; LCI-4B **APPROVED**; LCI-4C-A1 **APPROVED**; LCI-4C **APPROVED**; LCI-4D **APPROVED**; LCI-5A **APPROVED**; LCI-5B **READY_FOR_REVIEW**; LCI-5C **PLANNED / NEXT AFTER ACCEPTANCE**; LCI-6 **PLANNED**; LCI-7 **PLANNED**; LCI-8 **PLANNED**.
**Feature plan (1:1):** [`../plan/LANGCHAIN_INDEPENDENCE.md`](../plan/LANGCHAIN_INDEPENDENCE.md)
**Primary anchor domain:** `RAG`
**Related domains:** `LLM_ADAPTERS`, `INTEGRATIONS`, `MEMORY`, `MODALITY`, `ORCHESTRATION`, `PLATFORM_FOUNDATION`, `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE`
**Current active task:** LCI-5B — Native OpenAI-compatible embedding transport

**LCI-2F decision:** Loader output, normalization, metadata enrichment, chunking, and contextual enrichment remain `KnowledgeDocument` stages. `embed_texts` receives native chunk content; conversion through `to_legacy_rag_document()` occurs only immediately before the still-LangChain indexing, vector-store, and Graph consumers. Embedding contracts remain LCI-3A and indexing remains LCI-3B.

**LCI-3A decision:** The embedding API accepts `KnowledgeDocument`; `EmbeddingResult` stores native documents and readonly float32 vectors separately. Vectors are never stored in native metadata.

**LCI-3B decision:** `IndexingManager`, `IndexingPipeline`, and index strategies accept native `KnowledgeDocument` sequences. Single and Dual Index call native `embed_documents`; Dual TOC records are deterministic native derivatives with scope-safe grouping and preserved lineage. Legacy conversion is now private to the native vector-store manager/provider boundary.

**LCI-3C decision:** Public vector-store manager contracts use immutable native records, explicit `VectorStoreScope`, and native hits containing `KnowledgeDocument`. Tenant and namespace are authoritative routing fields and cannot be supplied through user metadata. Delete and count fail closed unless tenant isolation is proven.

**LCI-3D decision:** The VectorStore provider port is native. Providers map `VectorStoreRecord` directly to provider SDK payloads and return native `VectorStoreHit`. Tenant, namespace and workspace routing are system-owned at every provider boundary; LangChain document compatibility is removed from vector-store paths.

**LCI-4A decision:** Retrieval uses a native immutable hit/result contract containing `KnowledgeDocument`, score and rank. Active retrievers and RAG tools do not expose LangChain `Document`. `VectorStoreHit` remains the provider/vector-store result and is mapped at the retriever boundary. Reranking remains a separate LCI-4B boundary. Graph retrieval remains LCI-4C.

**LCI-4B decision:** Reranking uses one immutable native `RerankerCandidate` containing `KnowledgeDocument` and original retrieval score/rank. `RerankerResult` preserves the original candidate and adds only rerank/fusion score and final rank. Active core rerankers and integration providers do not accept or return LangChain `Document`. Reranking cannot alter document identity, scope, provenance or user metadata. Graph RAG remains LCI-4C.

**LCI-4C decision:** Graph RAG indexers and graph isolation contracts accept native `KnowledgeDocument` values. Document identity, tenant scope, namespace, provenance and user metadata remain owned by the native document boundary. Graph indexers read `document.content` and do not accept or construct LangChain `Document`. Graph retrieval uses the existing native `RetrievalHit` contract. GraphStore backend internals remain unchanged. Auxiliary memory, multimedia, legacy RAG, evaluation and soak paths remain LCI-4D.

**LCI-4C-A1 decision:** `workspace_id` is a canonical system-owned `KnowledgeDocumentScope` field. The canonical identity boundary is `tenant_id + namespace + workspace_id + document_id`; missing `workspace_id` remains backward-compatible and means no explicit workspace partition. User metadata cannot provide or override `workspace_id`. Indexing, vector storage, retrieval, reranking and Graph RAG preserve workspace scope without using metadata as a transport tunnel.

**LCI-4D decision:** Memory indexing, multimedia document loaders, legacy RAG answer contracts, evaluation harnesses and soak tooling use the canonical `KnowledgeDocument` boundary. Auxiliary runtime paths preserve canonical identity, tenant, namespace, workspace and provenance without using user metadata as system transport. No active LCI-4D production module imports or exposes LangChain `Document`. Provider-local loaders and embedding dependencies remain assigned to LCI-5 and LCI-6.

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

**LCI-2E decision:** Native `RecursiveChunkingStrategy` is the default and core-safe baseline. `LangChainRecursiveChunkingStrategy` is an optional provider behind the `rag-langchain-splitters` extra, loaded only on explicit construction or registry registration; missing the extra produces a stable configuration error without silent fallback.

`ParsedDocumentFragment` is extraction-stage only. `KnowledgeDocument` remains the canonical RAG knowledge ABI.

**Dependency inventory satellite:** [`satellites/LANGCHAIN_INDEPENDENCE_dependency_inventory.md`](satellites/LANGCHAIN_INDEPENDENCE_dependency_inventory.md)

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
| **LangGraph position** | Not a core dependency today; optional `langgraph-legacy` extra only; retirement reviewed under `LCI-8A`. |
| **LKW position** | Proof client only; migration ownership stays in Tier-0/Tier-1 domain and feature plans. |

Forbidden documentation claims (targets only, not current state):

- LangChain has been fully removed.
- Default Intergrax install is LangChain-free.
- RAG is already LangChain-independent.
- Native Ollama adapter already exists.
- All integrations are already optional.

---

## As-built baseline (evidence-backed)

Facts verified against repository state at inventory time.

| # | Fact | Evidence |
|---|------|----------|
| 1 | Intergrax defines native LLM messages independent of LangChain | `intergrax/llm/messages.py` — `ChatMessage`, `MessageRole`, `AttachmentRef` |
| 2 | LLM adapter public contract is native; providers map at boundary | `intergrax/llm_adapters/contracts/llm_adapter.py` — `LLMAdapter` uses `ChatMessage`, not LangChain messages |
| 3 | Most non-Ollama LLM providers use native SDKs behind `LLMAdapter` | Provider modules under `intergrax/llm_adapters/providers` (OpenAI, Anthropic, Gemini, etc.) — no `langchain_*` imports except Ollama |
| 4 | Ollama LLM path is provider-bound LangChain today | `intergrax/llm_adapters/providers/ollama_adapter.py` — `LangChainOllamaAdapter`, `ChatOllama`, lazy `langchain_core.messages` |
| 5 | `langchain_core.documents.Document` leaks through core RAG contracts | e.g. `intergrax/rag/document_loaders/contracts/base_document_parser.py`, `intergrax/rag/vectorstore/contracts/vector_store.py`, `intergrax/rag/embedding/contracts/base_embedding_manager.py` — **16 contract files** (see inventory satellite §D) |
| 6 | LangGraph is not a core dependency; guard exists | `scripts/maintenance/check_langgraph_not_required.py`; `pyproject.toml` — `langgraph` only under `[project.optional-dependencies] langgraph-legacy` |
| 7 | Default packaging still installs LangChain packages as core dependencies | `pyproject.toml` `[project].dependencies` — `langchain`, `langchain-core`, `langchain-community`, `langchain-openai`, `langchain-ollama`, `langchain-text-splitters` |

**Import audit scale:** 104 direct production/runtime import statements · 69 direct test import statements · 1 direct tooling import · 2 direct LangGraph lazy imports (see inventory satellite §B).

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
intergrax/compat/langchain/          (LCI-1C — READY_FOR_REVIEW)
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

**Note:** `intergrax/compat/langchain` provides the isolated LangChain `Document` bridge (`LCI-1C`, **APPROVED**); canonical native type remains `KnowledgeDocument`; the LangChain bridge remains compatibility-only; **LCI-1D** enforces native document ABI; full dependency optionalization remains later (**LCI-7A** / **LCI-7B**).

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
| **Contract lock-in** | `Document` in 16 public contracts couples most of RAG, memory, modality, and integrations to LangChain Core. |
| **Packaging risk** | Six LangChain packages in default `[project].dependencies` force install surface and security exposure for all users. |
| **Version range risk** | Broad `>=0.3` ranges on multiple LangChain packages increase resolver drift and breaking-change exposure. |
| **Transitive dependency risk** | LangChain meta-packages pull large transitive trees (community, text-splitters, OpenAI shims). |
| **Migration regression risk** | Dual-model transition (LangChain `Document` vs native document) can silently drop metadata or tenant fields. |
| **Dual-model transition risk** | Parallel types during migration require strict conversion tests (`LCI-1D`, conformance gate). |
| **LKW/Ollama parity risk** | LKW proof paths depend on `LangChainOllamaAdapter` today; native Ollama live proof (`LCI-6C`) and cutover (`LCI-6D`) must precede default resolver switch. |

---

## Final acceptance definition (full LCI program)

The LangChain Independence program (`LCI-0A` … `LCI-8A`) is **complete** when all of the following are true (none are true today):

1. Public Intergrax contracts expose **no** LangChain types.
2. Default `pip install` / `uv sync` **does not** require any `langchain*` package for core import of `intergrax`.
3. RAG ingest → index → retrieve → rerank runs on **native document types** with parity proof.
4. Memory and modality paths use native documents at indexing boundaries.
5. Ollama LLM and embedding paths have **native adapters** with parity proof; LangChain Ollama shims are optional extras only.
6. `check_langchain_boundary` (or successor) passes in CI for forbidden zones (`LCI-0B`); LangChain-free core install gate passes (`LCI-7B`).
7. LangChain compatibility is isolated under `compat` and/or provider modules with optional extras.
8. LangGraph remains optional; retirement decision recorded under `LCI-8A`.
9. LKW proof workload passes on LangChain-free core install (LKW as client, not owner).
10. Domain plan rows updated; inventory satellite shows `unclassified occurrences = 0` and zero core contract leaks.

---

## LCI-1A — Native knowledge document contract (summary)

Canonical ABI: **`KnowledgeDocument`** in neutral Tier-0 module `intergrax/knowledge/contracts/document.py` (implementation: **LCI-1B**, status **APPROVED**). KnowledgeDocument remains canonical; the LangChain bridge remains compatibility-only; **LCI-1D** enforces native document ABI; full dependency optionalization remains later (**LCI-7A** / **LCI-7B**).

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

## Related documents

| Document | Role |
|----------|------|
| [`../plan/LANGCHAIN_INDEPENDENCE.md`](../plan/LANGCHAIN_INDEPENDENCE.md) | Task roadmap `LCI-0A`–`LCI-8A` |
| [`satellites/LANGCHAIN_INDEPENDENCE_dependency_inventory.md`](satellites/LANGCHAIN_INDEPENDENCE_dependency_inventory.md) | Evidence-backed import and package inventory |
| [`satellites/LANGCHAIN_INDEPENDENCE_native_document_contract.md`](satellites/LANGCHAIN_INDEPENDENCE_native_document_contract.md) | Native `KnowledgeDocument` contract spec (LCI-1A) |
| [`../plan/satellites/LANGCHAIN_INDEPENDENCE_domain_plan_cross_references.md`](../plan/satellites/LANGCHAIN_INDEPENDENCE_domain_plan_cross_references.md) | Domain plan routing for future rows |
| [`../../architecture/RAG.md`](../../architecture/RAG.md) | Primary anchor domain architecture |
