<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# LangChain Independence — Multi-layer Feature Plan

**Status:** LCI-0A ready for review; all other tasks **PLANNED** (implementation not started)
**Feature architecture (1:1):** [../architecture/LANGCHAIN_INDEPENDENCE.md](../architecture/LANGCHAIN_INDEPENDENCE.md)
**Primary anchor domain:** RAG
**Related domains:** LLM_ADAPTERS, INTEGRATIONS, MEMORY, MODALITY, ORCHESTRATION, PLATFORM_FOUNDATION, EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE
**Next task after review:** LCI-0B — LANGCHAIN ARCHITECTURE BOUNDARY GUARD

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
→ ingestion/chunking (LCI-2A–2F) → embedding/indexing/vector store (LCI-3A–3D)
→ retrieval/reranking/graph/memory/modality (LCI-4A–4D, LCI-5A–5C)
→ thin replacements + native Ollama (LCI-6A–6E) → packaging/CI (LCI-7A–7D) → LangGraph review (LCI-8A)
```

**LKW note:** LKW is a proof client, not owner of LCI migration mechanics.

---

## LCI-0A — Canonical architecture and LangChain dependency inventory

| Field | Value |
|-------|-------|
| **Priority** | P0 |
| **Status** | READY_FOR_REVIEW |
| **Purpose** | Establish canonical architecture, evidence-backed inventory, and migration roadmap. |
| **Owning domain plan** | Feature plan + docs/plan/RAG.md anchor |
| **Dependencies** | None |
| **Exact scope** | Feature architecture/plan pair, inventory satellite, domain xref satellite, hub updates |
| **Explicit out of scope** | Boundary guard, code/dependency changes |
| **Acceptance criteria** | All LCI-0A prompt section 11 criteria |
| **User-visible outcome** | Reviewable roadmap; no runtime change |

---

## LCI-0B — LangChain architecture boundary guard

| Field | Value |
|-------|-------|
| **Priority** | P0 |
| **Status** | PLANNED |
| **Purpose** | Prevent new LangChain leaks into forbidden zones; grandfather existing violations. |
| **Owning domain plan** | docs/plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md |
| **Dependencies** | LCI-0A accepted |
| **Exact scope** | check_langchain_boundary.py, CI wiring, grandfather register |
| **Explicit out of scope** | Fixing existing leaks |
| **Acceptance criteria** | CI fails on new forbidden imports |
| **User-visible outcome** | No new contract leaks without waiver |

---

## LCI-0C — Dependency hardening design

| Field | Value |
|-------|-------|
| **Priority** | P0 |
| **Status** | PLANNED |
| **Purpose** | Design optional-extra matrix and minimal core install before mass migration. |
| **Owning domain plan** | docs/plan/PLATFORM_FOUNDATION.md |
| **Dependencies** | LCI-0B |
| **Exact scope** | Extra naming, import-time failure modes, minimal install proof design |
| **Explicit out of scope** | Removing packages from lockfile |
| **Acceptance criteria** | Approved extra map documented |
| **User-visible outcome** | Documented install tiers |

---

## LCI-1A — Native knowledge document architecture

| Field | Value |
|-------|-------|
| **Priority** | P0 |
| **Status** | PLANNED |
| **Purpose** | Define native Intergrax knowledge document contract. |
| **Owning domain plan** | docs/plan/RAG.md |
| **Dependencies** | LCI-0C |
| **Exact scope** | Architecture + contract spec; mapping from Document |
| **Explicit out of scope** | Implementation |
| **Acceptance criteria** | Contract approved |
| **User-visible outcome** | Stable document ABI spec |

---

## LCI-1B — Native knowledge document implementation

| Field | Value |
|-------|-------|
| **Priority** | P0 |
| **Status** | PLANNED |
| **Purpose** | Implement native document type and conversion utilities. |
| **Owning domain plan** | docs/plan/RAG.md |
| **Dependencies** | LCI-1A |
| **Exact scope** | Tier-0 type module; round-trip tests |
| **Explicit out of scope** | Removing LangChain from consumers |
| **Acceptance criteria** | Native type without langchain_core |
| **User-visible outcome** | Native document for migrators |

---

## LCI-1C — LangChain compatibility bridge

| Field | Value |
|-------|-------|
| **Priority** | P0 |
| **Status** | PLANNED |
| **Purpose** | Optional intergrax/compat/langchain/ mappers. |
| **Owning domain plan** | docs/plan/RAG.md |
| **Dependencies** | LCI-1B |
| **Exact scope** | Bridge module behind optional extra |
| **Explicit out of scope** | Making bridge canonical |
| **Acceptance criteria** | Bridge isolated from core imports |
| **User-visible outcome** | Gradual migration path |

---

## LCI-1D — Contract conformance gate

| Field | Value |
|-------|-------|
| **Priority** | P0 |
| **Status** | PLANNED |
| **Purpose** | Tests proving public contracts expose only native types. |
| **Owning domain plan** | docs/plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md |
| **Dependencies** | LCI-1B |
| **Exact scope** | Contract signature tests; AST checks |
| **Explicit out of scope** | Full pipeline migration |
| **Acceptance criteria** | Gate defined |
| **User-visible outcome** | Enforced contract hygiene |

---

## LCI-2A — Document loader and parser boundary migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | Migrate loader/parser contracts and smart parsers. |
| **Owning domain plan** | docs/plan/RAG.md + docs/plan/INTEGRATIONS.md |
| **Dependencies** | LCI-1D |
| **Exact scope** | document_loaders/contracts, parser pipelines, integration bridges |
| **Explicit out of scope** | Chunking, embedding |
| **Acceptance criteria** | Loaders emit native documents |
| **User-visible outcome** | Ingest accepts native documents |

---

## LCI-2B — Chunking pipeline migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | Migrate chunking engine and strategies off Document. |
| **Owning domain plan** | docs/plan/RAG.md |
| **Dependencies** | LCI-2A |
| **Exact scope** | document_splitters/** except langchain recursive |
| **Explicit out of scope** | Embedding/indexing |
| **Acceptance criteria** | Chunking contracts native |
| **User-visible outcome** | Chunk boundaries preserved |

---

## LCI-2C — LangChain recursive chunking replacement

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | Replace RecursiveCharacterTextSplitter with native chunker. |
| **Owning domain plan** | docs/plan/RAG.md |
| **Dependencies** | LCI-2B |
| **Exact scope** | langchain_recursive_chunking_strategy.py |
| **Explicit out of scope** | Removing package from extras |
| **Acceptance criteria** | Parity on golden fixtures |
| **User-visible outcome** | No text-splitters in default path |

---

## LCI-2D — Contextual enrichment migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | Migrate chunk_enricher and contextual steps. |
| **Owning domain plan** | docs/plan/RAG.md |
| **Dependencies** | LCI-2B |
| **Exact scope** | intergrax/rag/contextual/chunk_enricher.py |
| **Explicit out of scope** | Graph indexing |
| **Acceptance criteria** | Native document in/out |
| **User-visible outcome** | Enrichment metadata preserved |

---

## LCI-2E — Ingest pipeline migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | Migrate ingest_pipeline and ingest_policy. |
| **Owning domain plan** | docs/plan/RAG.md |
| **Dependencies** | LCI-2B |
| **Exact scope** | intergrax/rag/ingest/** |
| **Explicit out of scope** | Vector store writes |
| **Acceptance criteria** | Ingest unit tests native |
| **User-visible outcome** | Ingest API LangChain-free |

---

## LCI-2F — Nexus ingestion service migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | Remove Document from Nexus ingestion service. |
| **Owning domain plan** | docs/plan/ORCHESTRATION.md |
| **Dependencies** | LCI-2E |
| **Exact scope** | runtime/nexus/ingestion/ingestion_service.py |
| **Explicit out of scope** | Full Nexus refactor |
| **Acceptance criteria** | Nexus ingestion native |
| **User-visible outcome** | Orchestration ingestion boundary native |

---

## LCI-3A — Embedding provider migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | Replace LangChain embedding wrappers with native/SDK paths. |
| **Owning domain plan** | docs/plan/RAG.md + docs/plan/LLM_ADAPTERS.md |
| **Dependencies** | LCI-1D |
| **Exact scope** | embedding/providers, embedding_manager, contracts |
| **Explicit out of scope** | Vector store providers |
| **Acceptance criteria** | No langchain_openai/ollama in default path |
| **User-visible outcome** | Embedding unchanged |

---

## LCI-3B — Indexing pipeline migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | Migrate indexing strategies and manager. |
| **Owning domain plan** | docs/plan/RAG.md |
| **Dependencies** | LCI-3A |
| **Exact scope** | indexing/** |
| **Explicit out of scope** | Graph indexers |
| **Acceptance criteria** | Index writes parity |
| **User-visible outcome** | Indexing API native |

---

## LCI-3C — Vector store contract and provider migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | Migrate VectorStore contract and providers. |
| **Owning domain plan** | docs/plan/RAG.md + docs/plan/INTEGRATIONS.md |
| **Dependencies** | LCI-3B |
| **Exact scope** | vectorstore/contracts, integration rag_store files |
| **Explicit out of scope** | Retrieval ranking |
| **Acceptance criteria** | CRUD/search on native documents |
| **User-visible outcome** | Vector stores native at contract |

---

## LCI-3D — Tenant isolation contract migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | Migrate tenant isolation contracts. |
| **Owning domain plan** | docs/plan/RAG.md |
| **Dependencies** | LCI-3C |
| **Exact scope** | vectorstore/tenant/** |
| **Explicit out of scope** | Application tenancy |
| **Acceptance criteria** | Isolation proofs pass |
| **User-visible outcome** | Tenant safety unchanged |

---

## LCI-4A — Retrieval pipeline migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | Migrate retrievers and RAG tools. |
| **Owning domain plan** | docs/plan/RAG.md |
| **Dependencies** | LCI-3C |
| **Exact scope** | retrievers/**, tools/providers/rag |
| **Explicit out of scope** | Reranking, graph |
| **Acceptance criteria** | Retrieval integration tests green |
| **User-visible outcome** | Search results native |

---

## LCI-4B — Reranking migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | Migrate reranker contracts and providers. |
| **Owning domain plan** | docs/plan/RAG.md + docs/plan/INTEGRATIONS.md |
| **Dependencies** | LCI-4A |
| **Exact scope** | rerankers/**, integration rerank adapters |
| **Explicit out of scope** | Algorithm changes |
| **Acceptance criteria** | Rerank ordering parity |
| **User-visible outcome** | Rerank API native |

---

## LCI-4C — Memory indexing migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | Remove Document from memory indexing services. |
| **Owning domain plan** | docs/plan/MEMORY.md |
| **Dependencies** | LCI-3A |
| **Exact scope** | session_turn_index_service, user_profile_manager |
| **Explicit out of scope** | STM redesign |
| **Acceptance criteria** | Memory indexing native |
| **User-visible outcome** | Memory layer LangChain-free |

---

## LCI-4D — Graph RAG indexer migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | Migrate graph indexers and isolation contracts. |
| **Owning domain plan** | docs/plan/RAG.md |
| **Dependencies** | LCI-3B |
| **Exact scope** | intergrax/rag/graph/** |
| **Explicit out of scope** | Neo4j internals |
| **Acceptance criteria** | Graph indexer tests green |
| **User-visible outcome** | Graph channel native |

---

## LCI-5A — Modality loader migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | Migrate multimedia smart loaders. |
| **Owning domain plan** | docs/plan/MODALITY.md |
| **Dependencies** | LCI-2A |
| **Exact scope** | intergrax/multimedia/*_smart_loader.py |
| **Explicit out of scope** | Ollama vision internals |
| **Acceptance criteria** | Loaders return native documents |
| **User-visible outcome** | Media ingest native |

---

## LCI-5B — Integration document-parser bridge optionalization

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | Isolate langchain_community loaders to optional extra. |
| **Owning domain plan** | docs/plan/INTEGRATIONS.md |
| **Dependencies** | LCI-2A |
| **Exact scope** | document_parser opens.py, text_smart_parser.py |
| **Explicit out of scope** | Native parser impl |
| **Acceptance criteria** | Missing community pkg safe |
| **User-visible outcome** | Optional community loaders |

---

## LCI-5C — Integration vector-store bridge optionalization

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | PLANNED |
| **Purpose** | Ensure integration vector bridges map to native types. |
| **Owning domain plan** | docs/plan/INTEGRATIONS.md |
| **Dependencies** | LCI-3C |
| **Exact scope** | integrations/_shared bridges |
| **Explicit out of scope** | Provider SDK rewrites |
| **Acceptance criteria** | Bridges at native boundary |
| **User-visible outcome** | Integration layer optional LangChain |

---

## LCI-6A — Legacy RAG answers isolation

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | PLANNED |
| **Purpose** | Retire or isolate legacy rag_answers Document usage. |
| **Owning domain plan** | docs/plan/RAG.md |
| **Dependencies** | LCI-4A |
| **Exact scope** | intergrax/legacy/rag_answers/** |
| **Explicit out of scope** | Modern RAG runtime |
| **Acceptance criteria** | Legacy optional |
| **User-visible outcome** | Legacy clearly marked |

---

## LCI-6B — RAG evaluation harness migration

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | PLANNED |
| **Purpose** | Migrate evaluation harness off Document. |
| **Owning domain plan** | docs/plan/RAG.md |
| **Dependencies** | LCI-4A |
| **Exact scope** | rag/evaluation, vectorstore/soak |
| **Explicit out of scope** | Benchmark semantics |
| **Acceptance criteria** | Harness native |
| **User-visible outcome** | Evaluation tooling native |

---

## LCI-6C — Native Ollama LLM adapter

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | PLANNED |
| **Purpose** | Native Ollama adapter; demote LangChainOllamaAdapter. |
| **Owning domain plan** | docs/plan/LLM_ADAPTERS.md |
| **Dependencies** | LCI-1B, LCI-4A |
| **Exact scope** | ollama_adapter.py, LKW runtime proof |
| **Explicit out of scope** | Token Optimization router |
| **Acceptance criteria** | Tool/structured parity UNVERIFIED until live proof |
| **User-visible outcome** | Ollama without LangChain default |

---

## LCI-6D — Tool-call LangChain helper boundary

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | PLANNED |
| **Purpose** | Move tool_calls_from_langchain_message behind compat boundary. |
| **Owning domain plan** | docs/plan/LLM_ADAPTERS.md |
| **Dependencies** | LCI-6C |
| **Exact scope** | tool_call.py helper |
| **Explicit out of scope** | LLMToolCall contract change |
| **Acceptance criteria** | No LangChain in public contract module |
| **User-visible outcome** | Cleaner adapter contracts |

---

## LCI-6E — LKW proof slice on native stack

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | PLANNED |
| **Purpose** | LKW proof slice on LangChain-free core path. |
| **Owning domain plan** | LKW IMPLEMENTATION_PLAN (client scheduling) |
| **Dependencies** | LCI-6C, LCI-4A, LCI-3C |
| **Exact scope** | LKW proof workflows only |
| **Explicit out of scope** | LKW feature expansion |
| **Acceptance criteria** | LKW proof green on minimal install |
| **User-visible outcome** | Product proof on native platform |

---

## LCI-7A — Packaging and optional extras closeout

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | PLANNED |
| **Purpose** | Remove langchain* from default dependencies. |
| **Owning domain plan** | docs/plan/PLATFORM_FOUNDATION.md |
| **Dependencies** | LCI-6E |
| **Exact scope** | pyproject.toml, lockfile regen |
| **Explicit out of scope** | Application packaging |
| **Acceptance criteria** | Minimal install without LangChain |
| **User-visible outcome** | Smaller default install |

---

## LCI-7B — CI dependency conformance

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | PLANNED |
| **Purpose** | CI checks for boundary and packaging rules. |
| **Owning domain plan** | docs/plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md |
| **Dependencies** | LCI-7A, LCI-0B |
| **Exact scope** | CI workflows, doctor checks |
| **Explicit out of scope** | Full pytest redesign |
| **Acceptance criteria** | CI fails on regression |
| **User-visible outcome** | Protected migration gains |

---

## LCI-7C — Documentation and generator migration

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | PLANNED |
| **Purpose** | Update doc generators and examples. |
| **Owning domain plan** | docs/plan/PLATFORM_FOUNDATION.md |
| **Dependencies** | LCI-1B |
| **Exact scope** | scripts/docs/generate_integration_usage_docs.py |
| **Explicit out of scope** | Public roadmap rewrite |
| **Acceptance criteria** | No LangChain in required doc generators |
| **User-visible outcome** | Accurate docs |

---

## LCI-7D — Test fixture migration

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | PLANNED |
| **Purpose** | Migrate tests from Document fixtures to native factories. |
| **Owning domain plan** | docs/plan/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md |
| **Dependencies** | LCI-1B |
| **Exact scope** | tests/** and application tests |
| **Explicit out of scope** | Production code |
| **Acceptance criteria** | Default tests need no LangChain for core |
| **User-visible outcome** | Clearer tests |

---

## LCI-8A — LangGraph retirement review

| Field | Value |
|-------|-------|
| **Priority** | Optional |
| **Status** | PLANNED |
| **Purpose** | Decide fate of LangGraph adapters. |
| **Owning domain plan** | docs/plan/ORCHESTRATION.md |
| **Dependencies** | LCI-7B |
| **Exact scope** | supervisor_to_state_graph, langgraph_nodes, langgraph-legacy extra |
| **Explicit out of scope** | Nexus replacement |
| **Acceptance criteria** | Written keep vs remove decision |
| **User-visible outcome** | Clear legacy orchestration story |

---

