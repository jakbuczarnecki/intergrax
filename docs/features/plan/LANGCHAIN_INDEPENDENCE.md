<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# LangChain Independence — Multi-layer Feature Plan

**Status:** LCI-0A **APPROVED**; LCI-0B **APPROVED**; LCI-0C **READY_FOR_REVIEW**; LCI-1A **PLANNED**
**Feature architecture (1:1):** [../architecture/LANGCHAIN_INDEPENDENCE.md](../architecture/LANGCHAIN_INDEPENDENCE.md)
**Primary anchor domain:** RAG
**Related domains:** LLM_ADAPTERS, INTEGRATIONS, MEMORY, MODALITY, ORCHESTRATION, PLATFORM_FOUNDATION, EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE
**Current task:** LCI-0C
**Next task after acceptance:** LCI-1A

**Inventory satellite:** [../architecture/satellites/LANGCHAIN_INDEPENDENCE_dependency_inventory.md](../architecture/satellites/LANGCHAIN_INDEPENDENCE_dependency_inventory.md)

---

## Cursor read scope (token budget)

1. This file — read-scope block + **active LCI-* task only**.
2. Feature architecture — strategic decision + invariants only.
3. One owning domain plan pair for the active task.
4. **On demand (one max):** [satellites/LANGCHAIN_INDEPENDENCE_domain_plan_cross_references.md](satellites/LANGCHAIN_INDEPENDENCE_domain_plan_cross_references.md).

## Roadmap sequence

```text
inventory (LCI-0A) → boundary guard (LCI-0B) → dependency hardening (LCI-0C)
→ native document architecture/implementation/compat/conformance (LCI-1A–1D)
→ ingestion/chunking/ingest (LCI-2A–2F) → embedding/indexing/vector store (LCI-3A–3D)
→ retrieval/reranking/graph/memory/modality leaks (LCI-4A–4D)
→ thin replacements + optionalization (LCI-5A–5C)
→ native Ollama (LCI-6A–6E) → packaging/install gates/closeout (LCI-7A–7D) → LangGraph review (LCI-8A)
```

**LKW note:** LKW is a proof client, not owner of LCI migration mechanics.

---

## LCI-0A — Canonical architecture and LangChain dependency inventory

| Field | Value |
|-------|-------|
| **Priority** | P0 |
| **Status** | APPROVED |
| **Purpose** | Establish canonical architecture, evidence-backed inventory, and migration roadmap. |
| **Owning domain plan** | Feature plan + docs/plan/RAG.md anchor |
| **Dependencies** | None |
| **Exact scope** | Feature architecture/plan pair, inventory satellite, domain xref satellite; mechanical counts and task mapping |
| **Explicit out of scope** | Boundary guard, code/dependency changes, CI |
| **Acceptance criteria** | Inventory rows reconcile with grep; summary totals match detailed rows; unclassified = 0; canonical task IDs preserved |
| **User-visible outcome** | Reviewable roadmap and inventory; no runtime change |

---

## LCI-0B — LangChain architecture boundary guard

| Field | Value |
|-------|-------|
| **Priority** | P0 |
| **Status** | APPROVED |
| **Purpose** | Prevent new LangChain leaks into forbidden zones; grandfather existing violations. |
| **Owning domain plan** | docs/plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md |
| **Dependencies** | LCI-0A accepted |
| **Exact scope** | `check_langchain_boundary.py`, CI wiring, grandfather register from inventory |
| **Explicit out of scope** | Fixing existing leaks |
| **Acceptance criteria** | AST detection of `import` / `from` / nested imports / literal `importlib.import_module` / literal `__import__`; exact grandfather fingerprint comparison; `NEW_FORBIDDEN_IMPORT` failure on new violations; `STALE_GRANDFATHER_ENTRY` failure on orphan register rows; inventory consistency (`LCI-INV-####` path/module/symbol match); PR smoke CI + full governance CI wiring; deterministic unit tests in `tests/unit/architecture/test_langchain_boundary.py` |
| **User-visible outcome** | No new contract leaks without waiver |

---

## LCI-0C — LangChain dependency range hardening

| Field | Value |
|-------|-------|
| **Priority** | P0 |
| **Status** | READY_FOR_REVIEW |
| **Purpose** | Confirm whether meta-package langchain is used; remove from core if clean-install proof shows no use; add controlled upper version ranges; clean-install smoke. |
| **Owning domain plan** | docs/plan/PLATFORM_FOUNDATION.md |
| **Dependencies** | LCI-0B |
| **Exact scope** | Usage audit for langchain meta-package; pyproject range hardening; clean-install smoke script |
| **Explicit out of scope** | Full optionalization of all langchain packages; removing packages from lockfile without proof |
| **Acceptance criteria** | zero exact root langchain imports; unused direct meta-package removed; five remaining packages bounded; lockfile consistent; isolated clean-install proof passed |
| **User-visible outcome** | Tighter, evidence-backed core dependency surface |

---

## LCI-1A — Native knowledge document contract architecture

| Field | Value |
|-------|-------|
| **Priority** | P0 |
| **Status** | PLANNED |
| **Purpose** | Decide structure, semantics, identity, serialization, metadata, and provenance for native knowledge documents. |
| **Owning domain plan** | docs/plan/RAG.md |
| **Dependencies** | LCI-0C |
| **Exact scope** | Architecture + contract spec; mapping table from LangChain Document |
| **Explicit out of scope** | Implementation; LangChain bridge |
| **Acceptance criteria** | Approved native document contract published in architecture/RAG |
| **User-visible outcome** | Stable document ABI specification |

---

## LCI-1B — Native knowledge document contract implementation

| Field | Value |
|-------|-------|
| **Priority** | P0 |
| **Status** | PLANNED |
| **Purpose** | Implement native document type and its validation/serialization only. |
| **Owning domain plan** | docs/plan/RAG.md |
| **Dependencies** | LCI-1A |
| **Exact scope** | Tier-0 type module; serializers/validators; unit tests for native type |
| **Explicit out of scope** | Consumer migration; LangChain bridge |
| **Acceptance criteria** | Native module imports without langchain_core; round-trip serialization tests pass |
| **User-visible outcome** | Native document type available for migrators |

---

## LCI-1C — LangChain document compatibility bridge

| Field | Value |
|-------|-------|
| **Priority** | P0 |
| **Status** | PLANNED |
| **Purpose** | Optional from_langchain_document / to_langchain_document behind compat boundary. |
| **Owning domain plan** | docs/plan/RAG.md |
| **Dependencies** | LCI-1B |
| **Exact scope** | intergrax/compat/langchain/ bridge module behind optional extra |
| **Explicit out of scope** | Making bridge canonical; consumer migration |
| **Acceptance criteria** | Bridge isolated from core imports; optional extra documented |
| **User-visible outcome** | Gradual migration path for compatibility callers |

---

## LCI-1D — Knowledge document conformance gate

| Field | Value |
|-------|-------|
| **Priority** | P0 |
| **Status** | PLANNED |
| **Purpose** | LCI-1D proves that the native knowledge-document module, its serializers, compatibility-independent conformance tests, and native document public exports can be imported and exercised without langchain* installed. |
| **Owning domain plan** | docs/plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md |
| **Dependencies** | LCI-1B |
| **Exact scope** | Conformance test suite for contract, serialization, identity, and metadata; native document module import gate without langchain*; AST/signature checks wired in CI for scope delivered to this point |
| **Explicit out of scope** | Full pipeline migration; full Intergrax core installation without langchain* (remains LCI-7B) |
| **Acceptance criteria** | Contract tests pass; serialization round-trip tests pass; identity and metadata conformance tests pass; native document module imports without langchain* installed; AST/signature checks pass for implemented native document surface |
| **User-visible outcome** | Enforced native document contract hygiene with executed gates |

Full Intergrax core installation without langchain* remains out of scope until LCI-7B.

---

## LCI-2A — Document parser contract migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | Parser contracts stop returning LangChain Document. |
| **Owning domain plan** | docs/plan/RAG.md + docs/plan/INTEGRATIONS.md |
| **Dependencies** | LCI-1D |
| **Exact scope** | document_loaders/contracts parsers; integration parser bridges |
| **Explicit out of scope** | Loaders, normalization, chunking, ingest |
| **Acceptance criteria** | Parser contracts and parser implementations emit native documents; parser unit tests green |
| **User-visible outcome** | Parsed output is native at parser boundary |

---

## LCI-2B — Document loader and handler migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | Loaders, handlers, registry, and documents_loader operate on native contract. |
| **Owning domain plan** | docs/plan/RAG.md |
| **Dependencies** | LCI-2A |
| **Exact scope** | documents_loader.py; handler registry; loader pipelines |
| **Explicit out of scope** | Normalization, chunking, ingest |
| **Acceptance criteria** | Loader/handler paths use native documents end-to-end in unit tests |
| **User-visible outcome** | Filesystem and handler ingest accepts native documents |

---

## LCI-2C — Normalization and metadata pipeline migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | Normalizer, metadata provider, and parser/metadata pipelines use native contract. |
| **Owning domain plan** | docs/plan/RAG.md |
| **Dependencies** | LCI-2B |
| **Exact scope** | normalizers/, metadata/, metadata_pipeline, normalizer_pipeline, parser_pipeline metadata stages |
| **Explicit out of scope** | Chunking, embedding |
| **Acceptance criteria** | Normalization/metadata pipelines preserve fields on native documents |
| **User-visible outcome** | Metadata and normalization native throughout |

---

## LCI-2D — Chunking contract migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | Base chunking contracts, engine, and native strategies use native contract. |
| **Owning domain plan** | docs/plan/RAG.md |
| **Dependencies** | LCI-2C |
| **Exact scope** | document_splitters/contracts, engine, native strategies (excluding LangChain splitter provider) |
| **Explicit out of scope** | LangChain splitter optionalization; ingest |
| **Acceptance criteria** | Chunking contracts and native strategies pass parity unit tests |
| **User-visible outcome** | Chunk boundaries preserved on native documents |

---

## LCI-2E — LangChain splitter optionalization

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | LangChain recursive splitter remains optional provider; native recursive strategy is default baseline. |
| **Owning domain plan** | docs/plan/RAG.md |
| **Dependencies** | LCI-2D |
| **Exact scope** | langchain_recursive_chunking_strategy.py behind optional extra; lazy import and config error |
| **Explicit out of scope** | Mandatory removal of LangChain splitter |
| **Acceptance criteria** | Default chunking path does not import langchain_text_splitters; optional provider works when extra installed |
| **User-visible outcome** | Native chunking default with optional LangChain provider |

---

## LCI-2F — Ingest pipeline native document migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | Full parser → normalization → chunking → ingest path and boundary consumers on native documents. |
| **Owning domain plan** | docs/plan/RAG.md + docs/plan/ORCHESTRATION.md |
| **Dependencies** | LCI-2E |
| **Exact scope** | ingest_pipeline.py, ingest_policy.py, chunk_enricher.py, Nexus ingestion_service.py; e2e ingest proof |
| **Explicit out of scope** | Embedding/indexing migration |
| **Acceptance criteria** | Ingest and Nexus ingestion integration tests pass on native documents |
| **User-visible outcome** | End-to-end ingest API LangChain-free at boundary |

---

## LCI-3A — Embedding contract migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | Neutral embedding contract: embed_texts, embed_one, embed_documents(native document). |
| **Owning domain plan** | docs/plan/RAG.md + docs/plan/LLM_ADAPTERS.md |
| **Dependencies** | LCI-1D |
| **Exact scope** | embedding/contracts, embedding_manager interface; contract tests |
| **Explicit out of scope** | Immediate replacement of all embedding providers |
| **Acceptance criteria** | Contract tests pass; existing providers adapted behind boundary without changing public contract |
| **User-visible outcome** | Embedding API neutral and native-document-centric |

---

## LCI-3B — Indexing contract and strategy migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | Indexing manager, pipeline, and strategies use native document. |
| **Owning domain plan** | docs/plan/RAG.md |
| **Dependencies** | LCI-3A |
| **Exact scope** | indexing/** |
| **Explicit out of scope** | Vector store providers |
| **Acceptance criteria** | Indexing unit/integration tests pass with native documents |
| **User-visible outcome** | Indexing API native at contract |

---

## LCI-3C — Vector store contract migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | Public vector store contract and tenant-safe record semantics stop using LangChain Document; tenant isolation proofs included. |
| **Owning domain plan** | docs/plan/RAG.md |
| **Dependencies** | LCI-3B |
| **Exact scope** | vectorstore/contracts, tenant contracts, core vectorstore_manager |
| **Explicit out of scope** | Provider SDK adapter rewrites |
| **Acceptance criteria** | CRUD/search/isolation contract tests pass on native records |
| **User-visible outcome** | Vector store contract native with tenant proofs |

---

## LCI-3D — Vector store provider adapter migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | Providers and integration bridges map native records to SDK/vendor structures; tenant isolation proofs at provider boundary. |
| **Owning domain plan** | docs/plan/RAG.md + docs/plan/INTEGRATIONS.md |
| **Dependencies** | LCI-3C |
| **Exact scope** | vectorstore/providers, integrations/_shared bridges, integration rag_store modules |
| **Explicit out of scope** | Application tenancy |
| **Acceptance criteria** | Provider adapter tests and isolation proofs pass |
| **User-visible outcome** | Integration vector paths native at boundary |

---

## LCI-4A — Retrieval result contract migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | Native retrieval hit/result contract across retrievers and RAG tools. |
| **Owning domain plan** | docs/plan/RAG.md |
| **Dependencies** | LCI-3C |
| **Exact scope** | retrievers/**, tools/providers/rag |
| **Explicit out of scope** | Reranking, graph |
| **Acceptance criteria** | Retrieval integration tests green on native result types |
| **User-visible outcome** | Search results native at public contract |

---

## LCI-4B — Reranking contract migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | Single native RerankerCandidate; no List[Document] in public contract. |
| **Owning domain plan** | docs/plan/RAG.md + docs/plan/INTEGRATIONS.md |
| **Dependencies** | LCI-4A |
| **Exact scope** | rerankers/**, integration rerank adapters |
| **Explicit out of scope** | Algorithm changes |
| **Acceptance criteria** | Rerank ordering parity tests pass on native candidates |
| **User-visible outcome** | Rerank API native at contract |

---

## LCI-4C — Graph RAG document contract migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | Graph indexers and graph isolation use native document. |
| **Owning domain plan** | docs/plan/RAG.md |
| **Dependencies** | LCI-3B |
| **Exact scope** | intergrax/rag/graph/** |
| **Explicit out of scope** | Neo4j internals |
| **Acceptance criteria** | Graph indexer and isolation tests green |
| **User-visible outcome** | Graph channel native at contract |

---

## LCI-4D — Memory and multimedia document leak cleanup

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | Remaining auxiliary leaks: memory indexing, multimedia loaders, legacy rag_answers, evaluation harness, soak tooling. |
| **Owning domain plan** | docs/plan/MEMORY.md + docs/plan/MODALITY.md + docs/plan/RAG.md |
| **Dependencies** | LCI-3A |
| **Exact scope** | session_turn_index_service, user_profile_manager, multimedia loaders, legacy/rag_answers, evaluation/, vectorstore/soak/ |
| **Explicit out of scope** | Doc generators; test fixtures (migrate with owning feature) |
| **Acceptance criteria** | Target modules import no langchain_core; soak/evaluation smoke pass |
| **User-visible outcome** | Auxiliary runtime paths LangChain-free |

---

## LCI-5A — Native text document loader

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | Native plain-text file loader replaces langchain-community TextLoader in default path. |
| **Owning domain plan** | docs/plan/RAG.md |
| **Dependencies** | LCI-2A |
| **Exact scope** | text_smart_parser.py native loader implementation |
| **Explicit out of scope** | Other community loaders |
| **Acceptance criteria** | Default text ingest works without langchain_community |
| **User-visible outcome** | Plain-text ingest without community loader |

---

## LCI-5B — Native OpenAI embedding provider

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | Direct OpenAI SDK usage replaces langchain-openai OpenAIEmbeddings in default path. |
| **Owning domain plan** | docs/plan/RAG.md + docs/plan/LLM_ADAPTERS.md |
| **Dependencies** | LCI-3A |
| **Exact scope** | openai_embedding_provider.py and shared OpenAI-compatible providers |
| **Explicit out of scope** | All other embedding providers |
| **Acceptance criteria** | Embedding parity tests pass against prior LangChain wrapper baseline |
| **User-visible outcome** | OpenAI embeddings without langchain-openai default |

---

## LCI-5C — LangChain loaders and embeddings optionalization

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | Remaining LangChain loaders/embeddings move to optional extras with lazy import and controlled configuration errors. |
| **Owning domain plan** | docs/plan/INTEGRATIONS.md |
| **Dependencies** | LCI-5A, LCI-5B |
| **Exact scope** | integration document parsers, optional embedding shims, extras wiring |
| **Explicit out of scope** | Native replacements already delivered in 5A/5B |
| **Acceptance criteria** | Missing optional package fails with clear error; core import unaffected |
| **User-visible outcome** | LangChain loaders/embeddings explicitly optional |

---

## LCI-6A — Native Ollama adapter architecture and parity matrix

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | PLANNED |
| **Purpose** | Document parity matrix for messages, streaming, tools, structured output, JSON schema, capability resolution, usage, errors, timeouts, Token Optimization interactions. |
| **Owning domain plan** | docs/plan/LLM_ADAPTERS.md |
| **Dependencies** | LCI-1B |
| **Exact scope** | Architecture spec + parity matrix in LLM_ADAPTERS feature docs |
| **Explicit out of scope** | Implementation; LKW cutover |
| **Acceptance criteria** | Approved parity matrix with explicit pass/fail criteria per dimension |
| **User-visible outcome** | Signed-off native Ollama target behavior |

---

## LCI-6B — Native Ollama adapter implementation

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | PLANNED |
| **Purpose** | Implement native Ollama adapter behind LLMAdapter without LKW cutover. |
| **Owning domain plan** | docs/plan/LLM_ADAPTERS.md |
| **Dependencies** | LCI-6A |
| **Exact scope** | Native adapter module; unit tests; side-by-side harness vs LangChainOllamaAdapter |
| **Explicit out of scope** | Default resolver switch; live proof |
| **Acceptance criteria** | Unit/integration harness passes for implemented surfaces; no UNVERIFIED status on implemented criteria |
| **User-visible outcome** | Native Ollama adapter available behind feature flag |

---

## LCI-6C — Native Ollama live parity proof

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | PLANNED |
| **Purpose** | Mandatory live proof against real Ollama for matrix dimensions marked in-scope. |
| **Owning domain plan** | docs/plan/LLM_ADAPTERS.md |
| **Dependencies** | LCI-6B |
| **Exact scope** | Live proof scripts/tests; recorded evidence artifacts |
| **Explicit out of scope** | LKW default cutover |
| **Acceptance criteria** | Live proof executed and recorded; parity matrix shows no UNVERIFIED for in-scope dimensions |
| **User-visible outcome** | Documented live parity evidence |

---

## LCI-6D — LKW and Token Optimization native Ollama cutover

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | PLANNED |
| **Purpose** | Controlled default resolver switch; LKW and Token Optimization regression suite. |
| **Owning domain plan** | docs/plan/LLM_ADAPTERS.md + LKW IMPLEMENTATION_PLAN (client) |
| **Dependencies** | LCI-6C, LCI-4A, LCI-3C |
| **Exact scope** | Resolver default change; LKW proof workflows; Token Optimization regression tests |
| **Explicit out of scope** | Removing LangChain Ollama shim |
| **Acceptance criteria** | LKW proof and Token Optimization regressions green on native adapter default |
| **User-visible outcome** | Product proof paths use native Ollama |

---

## LCI-6E — LangChain Ollama compatibility optionalization

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | PLANNED |
| **Purpose** | LangChainOllamaAdapter remains optional compatibility provider behind extra. |
| **Owning domain plan** | docs/plan/LLM_ADAPTERS.md |
| **Dependencies** | LCI-6D |
| **Exact scope** | ollama_adapter.py optional extra; tool_calls_from_langchain_message behind compat boundary |
| **Explicit out of scope** | Deleting compatibility shim |
| **Acceptance criteria** | Core install imports without langchain-ollama; compat extra restores LangChain adapter |
| **User-visible outcome** | LangChain Ollama path explicitly optional |

---

## LCI-7A — LangChain optional extras packaging

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | PLANNED |
| **Purpose** | Remove langchain* packages from core dependencies; define extras; regenerate lockfile. |
| **Owning domain plan** | docs/plan/PLATFORM_FOUNDATION.md |
| **Dependencies** | LCI-6E |
| **Exact scope** | pyproject.toml dependency moves; extras map; uv.lock regeneration |
| **Explicit out of scope** | Application packaging |
| **Acceptance criteria** | pyproject has no langchain* in [project].dependencies; lockfile regenerated in CI |
| **User-visible outcome** | Smaller default dependency closure on paper |

---

## LCI-7B — LangChain-free core installation gate

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | PLANNED |
| **Purpose** | Clean environment gate: no langchain* installed; import core; native LLM; minimal native RAG; Nexus/Harness smoke. |
| **Owning domain plan** | docs/plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md |
| **Dependencies** | LCI-7A, LCI-0B |
| **Exact scope** | CI job/script for clean env install gate |
| **Explicit out of scope** | Compatibility bridge tests |
| **Acceptance criteria** | Gate script passes in CI on clean environment |
| **User-visible outcome** | Protected LangChain-free core install claim |

---

## LCI-7C — LangChain compatibility installation gate

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | PLANNED |
| **Purpose** | Separate extras install and compatibility bridge/provider tests. |
| **Owning domain plan** | docs/plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md |
| **Dependencies** | LCI-7A |
| **Exact scope** | Compat extra install job; bridge/provider compatibility tests |
| **Explicit out of scope** | Core install gate |
| **Acceptance criteria** | Compat install gate passes; compatibility tests green |
| **User-visible outcome** | Optional LangChain paths verified independently |

---

## LCI-7D — LangChain independence documentation closeout

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | PLANNED |
| **Purpose** | Update documentation, generators, extension guide, final inventory, and claims. |
| **Owning domain plan** | docs/plan/PLATFORM_FOUNDATION.md |
| **Dependencies** | LCI-7B, LCI-7C |
| **Exact scope** | scripts/docs/generate_integration_usage_docs.py; docs/generators; extension guide; inventory refresh |
| **Explicit out of scope** | Migrating production contracts (already owned by earlier tasks) |
| **Acceptance criteria** | Doc generators run without required LangChain import; final inventory shows zero core contract leaks |
| **User-visible outcome** | Accurate public documentation and claims |

---

## LCI-8A — LangGraph legacy retirement review

| Field | Value |
|-------|-------|
| **Priority** | Optional |
| **Status** | PLANNED |
| **Purpose** | Keep/remove decision after critical path complete. |
| **Owning domain plan** | docs/plan/ORCHESTRATION.md |
| **Dependencies** | LCI-7D |
| **Exact scope** | supervisor_to_state_graph.py, langgraph_nodes.py, langgraph-legacy extra review |
| **Explicit out of scope** | Nexus replacement |
| **Acceptance criteria** | Written keep vs remove decision with rationale and follow-up tasks if needed |
| **User-visible outcome** | Clear legacy orchestration story |

---

