<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# LangChain Independence — Multi-layer Feature Plan

**Status:** LCI-0A **APPROVED**; LCI-0B **APPROVED**; LCI-0C **APPROVED**; LCI-1A **APPROVED**; LCI-1B **APPROVED**; LCI-1C **APPROVED**; LCI-1D **APPROVED**; LCI-2A **APPROVED**; LCI-2B **APPROVED**; LCI-2C **APPROVED**; LCI-2D **APPROVED**; LCI-2E **APPROVED**; LCI-2F **APPROVED**; LCI-3A **APPROVED**; LCI-3B **APPROVED**; LCI-3C **APPROVED**; LCI-3D-1 **APPROVED**; LCI-3D-2 **APPROVED**; LCI-3D-3 **READY_FOR_REVIEW**; LCI-3D **APPROVED**; LCI-4A **APPROVED**; LCI-4B **APPROVED**; LCI-4C-A1 **APPROVED**; LCI-4C **APPROVED**; LCI-4D **APPROVED**; LCI-5A **APPROVED**; LCI-5B **APPROVED**; LCI-5C **APPROVED**; LCI-6A **APPROVED**; LCI-6B **APPROVED**; LCI-6C **APPROVED**; LCI-6D **APPROVED**; Native Ollama regression gate **APPROVED**; LCI-6E **READY_FOR_REVIEW**
**Feature architecture (1:1):** [../architecture/LANGCHAIN_INDEPENDENCE.md](../architecture/LANGCHAIN_INDEPENDENCE.md)
**Primary anchor domain:** RAG
**Related domains:** LLM_ADAPTERS, INTEGRATIONS, MEMORY, MODALITY, ORCHESTRATION, PLATFORM_FOUNDATION, EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE
**Current active task:** LCI-6E — LangChain Ollama compatibility optionalization
**Next task after acceptance:** LCI-7A

**LCI-4C-A1 decision:** `workspace_id` is a canonical system-owned `KnowledgeDocumentScope` field. The canonical identity boundary is `tenant_id + namespace + workspace_id + document_id`; missing `workspace_id` remains backward-compatible and means no explicit workspace partition. User metadata cannot provide or override `workspace_id`. Indexing, vector storage, retrieval, reranking and Graph RAG preserve workspace scope without using metadata as a transport tunnel.

**LCI-4D decision:** Memory indexing, multimedia document loaders, legacy RAG answer contracts, evaluation harnesses and soak tooling use the canonical `KnowledgeDocument` boundary. Auxiliary runtime paths preserve canonical identity, tenant, namespace, workspace and provenance without using user metadata as system transport. No active LCI-4D production module imports or exposes LangChain `Document`. Provider-local loaders and embedding dependencies remain assigned to LCI-5 and LCI-6.

**LCI-5A decision:** Plain-text parsing uses a native Intergrax reader and emits
`ParsedDocumentFragment` directly. The default text parser no longer imports or
constructs LangChain `TextLoader` or LangChain `Document`. Text decoding
preserves supported encoding behavior without introducing a new required
dependency. Provider-local LangChain document loaders remain assigned to
LCI-5C.

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
| **Owning domain plan** | Feature plan + docs/project/maintainers/plans/RAG.md anchor |
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
| **Owning domain plan** | docs/project/maintainers/plans/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md |
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
| **Status** | APPROVED |
| **Purpose** | Confirm whether meta-package langchain is used; remove from core if clean-install proof shows no use; add controlled upper version ranges; clean-install smoke. |
| **Owning domain plan** | docs/project/maintainers/plans/PLATFORM_FOUNDATION.md |
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
| **Status** | APPROVED |
| **Purpose** | Decide structure, semantics, identity, serialization, metadata, and provenance for native knowledge documents. |
| **Owning domain plan** | docs/project/maintainers/plans/RAG.md |
| **Dependencies** | LCI-0C |
| **Exact scope** | Architecture + contract spec; mapping table from LangChain Document |
| **Explicit out of scope** | Implementation; LangChain bridge |
| **Contract satellite** | [../architecture/satellites/LANGCHAIN_INDEPENDENCE_native_document_contract.md](../architecture/satellites/LANGCHAIN_INDEPENDENCE_native_document_contract.md) |
| **Acceptance criteria** | Canonical location (`intergrax/knowledge/contracts/document.py`, public import `KnowledgeDocument`); complete field contract (`KnowledgeDocument`, `KnowledgeDocumentIdentity`, `KnowledgeDocumentScope`, `KnowledgeDocumentProvenance`); canonical scoped identity; source hierarchy separated from document lineage; strict source/derivative lineage invariants; identity and lineage rules; tenant fail-closed policy; metadata and provenance rules; reserved metadata always rejected; finite JSON numbers; serialization/versioning (`schema_version = 1`); LangChain `Document` mapping table; Vendor Knowledge reuse decision; shared validation reuse decision; explicit boundaries for LCI-1B–LCI-1D; RAG architecture hub entry published |
| **User-visible outcome** | Stable document ABI specification |

---

## LCI-1B — Native knowledge document contract implementation

| Field | Value |
|-------|-------|
| **Priority** | P0 |
| **Status** | APPROVED |
| **Purpose** | Implement native document type and its validation/serialization only. |
| **Owning domain plan** | docs/project/maintainers/plans/RAG.md |
| **Dependencies** | LCI-1A |
| **Exact scope** | Tier-0 type module; serializers/validators; unit tests for native type |
| **Explicit out of scope** | Consumer migration; LangChain bridge |
| **Acceptance criteria** | Native models (`KnowledgeDocument`, identity/scope/provenance sub-models); public import `from intergrax.knowledge.contracts import KnowledgeDocument`; shared validation reuse via `intergrax/knowledge/contracts/validation.py`; deterministic `dump_knowledge_document` / `load_knowledge_document`; LangChain-free `intergrax/knowledge` module; targeted unit tests (`tests/unit/knowledge/contracts/test_document.py`); Vendor Knowledge regression tests |
| **User-visible outcome** | Native document type available for migrators |

---

## LCI-1C — LangChain document compatibility bridge

| Field | Value |
|-------|-------|
| **Priority** | P0 |
| **Status** | APPROVED |
| **Purpose** | Optional from_langchain_document / to_langchain_document behind compat boundary. |
| **Owning domain plan** | docs/project/maintainers/plans/RAG.md |
| **Dependencies** | LCI-1B |
| **Exact scope** | isolated lazy bridge under `intergrax/compat/langchain` |
| **Explicit out of scope** | Making bridge canonical; consumer migration; packaging optionalization (LCI-7A) |
| **Acceptance criteria** | bidirectional mapping; no silent data loss; explicit conflict errors; module import without eager `langchain_core` import; missing-dependency error; tests; packaging move deferred to LCI-7A |
| **User-visible outcome** | Gradual migration path for compatibility callers |

---

## LCI-1D — Knowledge document conformance gate

| Field | Value |
|-------|-------|
| **Priority** | P0 |
| **Status** | APPROVED |
| **Purpose** | LCI-1D proves that the native knowledge-document module, its serializers, compatibility-independent conformance tests, and native document public exports can be imported and exercised without langchain* installed. |
| **Owning domain plan** | docs/project/maintainers/plans/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md |
| **Dependencies** | LCI-1B |
| **Exact scope** | public ABI conformance suite; native AST boundary; blocked-import subprocess proof; CI smoke and full-governance wiring |
| **Explicit out of scope** | Full pipeline migration; full Intergrax core installation without langchain* (remains LCI-7B) |
| **Acceptance criteria** | public models/fields/config validated; serializer signatures validated; source/derivative and metadata proof executed; no langchain*/langgraph*/compat import attempted; checker wired in both CI profiles |
| **User-visible outcome** | Enforced native document contract hygiene with executed gates |

LCI-1D is not the full LangChain-free core installation proof. That remains LCI-7B.

---

## LCI-2A — Document parser contract migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | APPROVED |
| **Purpose** | Parser contracts stop returning LangChain Document. |
| **Owning domain plan** | docs/project/maintainers/plans/RAG.md + docs/project/maintainers/plans/INTEGRATIONS.md |
| **Dependencies** | LCI-1D |
| **Exact scope** | document_loaders/contracts parsers; integration parser bridges |
| **Explicit out of scope** | Loaders, normalization, chunking, ingest |
| **Acceptance criteria** | Parser contracts and parser implementations emit `ParsedDocumentFragment`; parser unit tests green |
| **User-visible outcome** | Parsed output is native at parser boundary |
| **Canonical parser-stage output** | `ParsedDocumentFragment` |
| **KnowledgeDocument construction** | deferred to scoped handler/loader boundary in LCI-2B |
| **Temporary compatibility** | removed in LCI-2B — handler now emits `KnowledgeDocument` |

---

## LCI-2B — Document loader and handler migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | APPROVED |
| **Purpose** | Scoped handler and loader boundary emits `KnowledgeDocument`. |
| **Owning domain plan** | docs/project/maintainers/plans/RAG.md |
| **Dependencies** | LCI-2A |
| **Exact scope** | `BaseDocumentHandler`, `BaseDocumentsLoader`, `DocumentsLoader`, downstream ingest call-site bridges |
| **Explicit out of scope** | Normalization, metadata providers, chunking, embedding, indexing |
| **Acceptance criteria** | Handler/loader paths return `KnowledgeDocument`; required `tenant_id` scope; temporary LangChain bridge around normalization/metadata; ingest call sites convert to LangChain before splitter |
| **User-visible outcome** | Filesystem and handler ingest produce scoped native documents |
| **Handler output** | `KnowledgeDocument` |
| **Loader output** | `KnowledgeDocument` |
| **Scope** | required `tenant_id`, optional `namespace` |
| **Identity** | deterministic per parser fragment (`source` + `position`) |
| **Normalization/metadata bridge** | removed in LCI-2C |
| **Downstream LangChain conversion** | limited to existing ingest call sites |

---

## LCI-2C — Normalization and metadata pipeline migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | APPROVED |
| **Purpose** | Normalizer, metadata provider, and parser/metadata pipelines use native contract. |
| **Owning domain plan** | docs/project/maintainers/plans/RAG.md |
| **Dependencies** | LCI-2B |
| **Exact scope** | normalizers/, metadata/, metadata_pipeline, normalizer_pipeline, parser_pipeline metadata stages |
| **Explicit out of scope** | Chunking, embedding |
| **Acceptance criteria** | Normalization/metadata pipelines preserve fields on native documents |
| **User-visible outcome** | Metadata and normalization native throughout |

| **Normalizer contracts** | `KnowledgeDocument` |
| **Metadata provider contracts** | `KnowledgeDocument` |
| **DocumentsLoader compatibility round-trip** | removed |
| **Runtime handle** | preserved privately |
| **Legacy conversion** | only immediately before splitters |

---

## LCI-2D — Chunking contract migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | APPROVED |
| **Purpose** | Base chunking contracts, engine, and native strategies use native contract. |
| **Owning domain plan** | docs/project/maintainers/plans/RAG.md |
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
| **Status** | APPROVED |
| **Purpose** | LangChain recursive splitter remains an optional provider; native recursive strategy is the default and core-safe baseline. |
| **Owning domain plan** | docs/project/maintainers/plans/RAG.md |
| **Dependencies** | LCI-2D |
| **Exact scope** | `langchain_recursive_chunking_strategy.py` behind `rag-langchain-splitters`; lazy import, explicit registry registration, and stable missing-extra configuration error |
| **Explicit out of scope** | Mandatory removal of LangChain splitter |
| **Acceptance criteria** | Default bootstrap registers only `recursive`, `semantic`, `parent_child`, and `docling`; default chunking path does not import `langchain_text_splitters`; optional provider works after installing `rag-langchain-splitters`; no silent fallback |
| **User-visible outcome** | Native chunking default with optional LangChain provider |

---

## LCI-2F — Ingest pipeline native document migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | APPROVED |
| **Purpose** | Full parser → normalization → chunking → contextual enrichment → ingest path on native documents, with legacy conversion only at downstream consumer boundaries. |
| **Owning domain plan** | docs/project/maintainers/plans/RAG.md + docs/project/maintainers/plans/ORCHESTRATION.md |
| **Dependencies** | LCI-2E |
| **Exact scope** | ingest_pipeline.py, ingest_policy.py, chunk_enricher.py, Nexus ingestion_service.py; e2e ingest proof |
| **Explicit out of scope** | Embedding/indexing migration |
| **Acceptance criteria** | Ingest and Nexus ingestion tests prove native loader-through-contextual stages, raw-text embedding input, and compatibility conversion only immediately before indexing/vector-store/Graph consumers. |
| **User-visible outcome** | End-to-end ingest API is LangChain-free through native contextual enrichment; embedding contracts remain LCI-3A and indexing remains LCI-3B. |

---

## LCI-3A — Embedding contract migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | APPROVED |
| **Purpose** | Neutral embedding contract: embed_texts, embed_one, embed_documents(native document). |
| **Owning domain plan** | docs/project/maintainers/plans/RAG.md + docs/project/maintainers/plans/LLM_ADAPTERS.md |
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
| **Status** | READY_FOR_REVIEW |
| **Purpose** | Indexing manager, pipeline, and strategies use native `KnowledgeDocument`. |
| **Owning domain plan** | docs/project/maintainers/plans/RAG.md |
| **Dependencies** | LCI-3A |
| **Exact scope** | indexing/** |
| **Explicit out of scope** | Vector store providers |
| **Acceptance criteria** | Indexing unit/integration tests pass with native documents, native `embed_documents`, deterministic tenant-safe TOC lineage, and legacy conversion only immediately before `BaseVectorstoreManager`. |
| **User-visible outcome** | Indexing API native at contract |

**Decision:** `IndexingManager`, pipeline, and strategies accept `KnowledgeDocument`. Single and Dual Index use native `embed_documents`. Dual TOC records are native derivatives with deterministic lineage and tenant-safe grouping. Legacy conversion is private to the native vector-store manager/provider boundary.

---

## LCI-3C — Vector store contract migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | APPROVED |
| **Purpose** | Public vector store contract and tenant-safe record semantics stop using LangChain Document; tenant isolation proofs included. |
| **Owning domain plan** | docs/project/maintainers/plans/RAG.md |
| **Dependencies** | LCI-3B |
| **Exact scope** | vectorstore/contracts, tenant contracts, core vectorstore_manager |
| **Explicit out of scope** | Provider SDK adapter rewrites |
| **Acceptance criteria** | CRUD/search/isolation contract tests pass on native records |
| **User-visible outcome** | Vector store contract native with tenant proofs |

**Decision:** Public vector-store manager contracts use native records, explicit scope and native hits. Tenant and namespace are authoritative routing fields, never user metadata. Delete and count fail closed unless tenant isolation is proven.

---

## LCI-3D — Vector store provider adapter migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | READY_FOR_REVIEW |
| **Purpose** | Providers and integration bridges map native records to SDK/vendor structures; tenant isolation proofs at provider boundary. |
| **Owning domain plan** | docs/project/maintainers/plans/RAG.md + docs/project/maintainers/plans/INTEGRATIONS.md |
| **Dependencies** | LCI-3C |
| **Exact scope** | vectorstore/providers, integrations/_shared bridges, integration rag_store modules |
| **Explicit out of scope** | Application tenancy |
| **Acceptance criteria** | Provider adapter tests and isolation proofs pass |
| **User-visible outcome** | Integration vector paths native at boundary |

**Decision:** VectorStore provider port is native. Providers map `VectorStoreRecord` directly to SDK payloads and return native `VectorStoreHit`. Tenant, namespace and workspace routing are system-owned at every provider boundary. LangChain document compatibility is removed from vector-store paths.

---

## LCI-4A — Retrieval result contract migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | APPROVED |
| **Purpose** | Migrate active retrievers and RAG tools to one immutable native hit/result contract. |
| **Owning domain plan** | docs/project/maintainers/plans/RAG.md |
| **Dependencies** | LCI-3D |
| **Exact scope** | retrievers/**, tools/providers/rag, corresponding contracts/tests |
| **Explicit out of scope** | Reranking (LCI-4B), graph retrieval (LCI-4C) |
| **Acceptance criteria** | Native retriever/tool tests, scoped LangChain proofs, inventory and boundary audits pass |
| **User-visible outcome** | Search results native at public contract |

---

## LCI-4B — Reranking contract migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | APPROVED |
| **Purpose** | Single immutable native RerankerCandidate and RerankerResult; no LangChain Document in active rerank contracts. |
| **Owning domain plan** | docs/project/maintainers/plans/RAG.md + docs/project/maintainers/plans/INTEGRATIONS.md |
| **Dependencies** | LCI-4A |
| **Exact scope** | rerankers/**, integration rerank adapters |
| **Explicit out of scope** | Algorithm changes |
| **Acceptance criteria** | Native candidate/result contract, RetrievalHit mapping, provider validation, ordering parity, scope/provenance preservation, inventory and boundary audits pass |
| **User-visible outcome** | Rerank API native at contract |

**Decision:** Reranking uses one immutable native `RerankerCandidate` containing `KnowledgeDocument` and original retrieval score/rank. `RerankerResult` preserves the original candidate and adds only rerank/fusion score and final rank. Active core rerankers and integration providers do not accept or return LangChain `Document`. Reranking cannot alter document identity, scope, provenance or user metadata. Graph RAG remains LCI-4C.

---

## LCI-4C — Graph RAG document contract migration

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | READY_FOR_REVIEW |
| **Purpose** | Graph indexers, graph isolation and the active graph retrieval boundary use native documents. |
| **Owning domain plan** | docs/project/maintainers/plans/RAG.md |
| **Dependencies** | LCI-4B |
| **Exact scope** | intergrax/rag/graph/**, active Graph RAG retriever boundary, corresponding tests, inventory and documentation |
| **Explicit out of scope** | Neo4j, Memgraph, FalkorDB and other GraphStore internals; auxiliary memory, multimedia, legacy RAG, evaluation and soak paths |
| **Acceptance criteria** | Native KnowledgeDocument indexer/isolation contracts, native RetrievalHit graph results, no active Graph RAG LangChain document shapes, targeted tests and boundary audits pass |
| **User-visible outcome** | Graph channel native at contract |

**Decision:** Graph RAG indexers and graph isolation contracts accept native `KnowledgeDocument` values. Identity, tenant scope, namespace, provenance and user metadata remain owned by the native document boundary; user metadata cannot become authoritative graph scope. Graph retrieval uses the existing native `RetrievalHit` contract. GraphStore backend internals remain unchanged.

---

## LCI-4D — Memory and multimedia document leak cleanup

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | READY_FOR_REVIEW |
| **Purpose** | Remaining auxiliary leaks: memory indexing, multimedia loaders, legacy rag_answers, evaluation harness, soak tooling. |
| **Owning domain plan** | docs/project/maintainers/plans/MEMORY.md + docs/project/maintainers/plans/MODALITY.md + docs/project/maintainers/plans/RAG.md |
| **Dependencies** | LCI-4C |
| **Exact scope** | session_turn_index_service, user_profile_manager, multimedia loaders, legacy/rag_answers, evaluation/, vectorstore/soak/ |
| **Explicit out of scope** | Doc generators; test fixtures (migrate with owning feature) |
| **Acceptance criteria** | Target modules import no langchain_core; soak/evaluation smoke pass |
| **User-visible outcome** | Auxiliary runtime paths use the canonical native document boundary |

---

## LCI-5A — Native text document loader

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | READY_FOR_REVIEW |
| **Purpose** | Native plain-text file loader replaces langchain-community TextLoader in default path. |
| **Owning domain plan** | docs/project/maintainers/plans/RAG.md |
| **Dependencies** | LCI-2A |
| **Exact scope** | text_smart_parser.py native loader implementation |
| **Explicit out of scope** | Other community loaders |
| **Acceptance criteria** | Default text ingest works without importing or constructing `langchain_community` `TextLoader`; `ParsedDocumentFragment` output and metadata parity are covered by targeted tests |
| **User-visible outcome** | Plain-text ingest without community loader |

---

## LCI-5B — Native OpenAI embedding provider

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | READY_FOR_REVIEW |
| **Purpose** | Direct OpenAI SDK usage replaces langchain-openai OpenAIEmbeddings in the OpenAI-compatible embedding providers. |
| **Owning domain plan** | docs/project/maintainers/plans/RAG.md + docs/project/maintainers/plans/LLM_ADAPTERS.md |
| **Dependencies** | LCI-3A |
| **Exact scope** | openai_embedding_provider.py, vllm_embedding_provider.py, llama_cpp_embedding_provider.py and one shared native transport |
| **Explicit out of scope** | All other embedding providers |
| **Acceptance criteria** | Native transport ordering, validation, batching, lazy dimensions and provider parity tests pass against the prior wrapper baseline |
| **User-visible outcome** | OpenAI-compatible embeddings without langchain-openai provider construction |

---

## LCI-5C — LangChain loaders and embeddings optionalization

| Field | Value |
|-------|-------|
| **Priority** | P1 |
| **Status** | READY_FOR_REVIEW |
| **Purpose** | Remaining LangChain loaders/embeddings move to optional extras with lazy import and controlled configuration errors. |
| **Owning domain plan** | docs/project/maintainers/plans/INTEGRATIONS.md |
| **Dependencies** | LCI-5A, LCI-5B |
| **Exact scope** | integration document parsers, optional embedding shims, extras wiring |
| **Explicit out of scope** | Native replacements already delivered in 5A/5B |
| **Acceptance criteria** | Missing optional package fails with clear error; core import unaffected |
| **User-visible outcome** | LangChain loaders/embeddings explicitly optional |

**Closeout:** LCI-5C is `APPROVED`. The existing provider IDs, registration
behavior and `ParsedDocumentFragment` output ABI are preserved. LCI-6A is
`APPROVED`; LCI-6B is `APPROVED`; LCI-6C is `READY_FOR_REVIEW`; LCI-6D is
`NEXT AFTER ACCEPTANCE`.
This task does not optionalize the chat Ollama adapter.

---

## LCI-6A — Native Ollama adapter architecture and parity matrix

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | READY_FOR_REVIEW |
| **Purpose** | Document parity matrix for messages, streaming, tools, structured output, JSON schema, capability resolution, usage, errors, timeouts, Token Optimization interactions. |
| **Owning domain plan** | docs/project/maintainers/plans/LLM_ADAPTERS.md |
| **Dependencies** | LCI-1B |
| **Exact scope** | Architecture spec + parity matrix in [`../architecture/satellites/OLLAMA_NATIVE_ADAPTER_PARITY_MATRIX.md`](../architecture/satellites/OLLAMA_NATIVE_ADAPTER_PARITY_MATRIX.md) |
| **Explicit out of scope** | Implementation; LKW cutover |
| **Acceptance criteria** | Reviewable parity matrix with explicit pass/fail criteria per dimension; no runtime, resolver, LKW, Token Optimization, dependency, or test changes |
| **User-visible outcome** | Frozen native Ollama target behavior, READY_FOR_REVIEW; implementation and live proof remain pending |

---

## LCI-6B — Native Ollama adapter implementation

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | APPROVED |
| **Purpose** | Implement native Ollama adapter behind LLMAdapter without LKW cutover. |
| **Owning domain plan** | docs/project/maintainers/plans/LLM_ADAPTERS.md |
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
| **Status** | APPROVED |
| **Purpose** | Mandatory live proof against real Ollama for matrix dimensions marked in-scope. |
| **Owning domain plan** | docs/project/maintainers/plans/LLM_ADAPTERS.md |
| **Dependencies** | LCI-6B |
| **Exact scope** | Live proof scripts/tests; recorded evidence artifacts |
| **Explicit out of scope** | LKW default cutover |
| **Acceptance criteria** | Live proof executed and recorded; parity matrix shows no UNVERIFIED for in-scope dimensions |
| **User-visible outcome** | Documented live parity evidence; rows 040–042 remain LIVE_NOT_REPRODUCIBLE and rows 043–044 remain PROVIDER_PREVENTS_REPRODUCTION with deterministic LCI-6B support |

---

## LCI-6D — LKW and Token Optimization native Ollama cutover

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | APPROVED |
| **Purpose** | Controlled default resolver switch; LKW and Token Optimization regression suite. |
| **Owning domain plan** | docs/project/maintainers/plans/LLM_ADAPTERS.md + LKW IMPLEMENTATION_PLAN (client) |
| **Dependencies** | LCI-6C, LCI-4A, LCI-3C |
| **Exact scope** | Resolver default change; LKW proof workflows; Token Optimization regression tests |
| **Explicit out of scope** | Removing LangChain Ollama shim |
| **Acceptance criteria** | LKW proof and Token Optimization regressions green on native adapter default; native Ollama regression gate accepted |
| **User-visible outcome** | Product proof paths use native Ollama |

---

## LCI-6E — LangChain Ollama compatibility optionalization

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | READY_FOR_REVIEW |
| **Purpose** | LangChainOllamaAdapter remains optional compatibility provider behind extra. |
| **Owning domain plan** | docs/project/maintainers/plans/LLM_ADAPTERS.md |
| **Dependencies** | LCI-6D |
| **Exact scope** | ollama_adapter.py optional extra `llm-langchain-ollama`; tool_calls_from_langchain_message behind compat boundary; provider-based multimedia Ollama detection |
| **Explicit out of scope** | Deleting compatibility shim |
| **Acceptance criteria** | Core/native imports do not require langchain-ollama; missing extra has a stable configuration error; compat extra restores the LangChain adapter; native registry default remains unchanged |
| **User-visible outcome** | LangChain Ollama path explicitly optional |

---

## LCI-7A — LangChain optional extras packaging

| Field | Value |
|-------|-------|
| **Priority** | P2 |
| **Status** | PLANNED |
| **Purpose** | Remove langchain* packages from core dependencies; define extras; regenerate lockfile. |
| **Owning domain plan** | docs/project/maintainers/plans/PLATFORM_FOUNDATION.md |
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
| **Owning domain plan** | docs/project/maintainers/plans/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md |
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
| **Owning domain plan** | docs/project/maintainers/plans/EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md |
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
| **Owning domain plan** | docs/project/maintainers/plans/PLATFORM_FOUNDATION.md |
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
| **Owning domain plan** | docs/project/maintainers/plans/ORCHESTRATION.md |
| **Dependencies** | LCI-7D |
| **Exact scope** | supervisor_to_state_graph.py, langgraph_nodes.py, langgraph-legacy extra review |
| **Explicit out of scope** | Nexus replacement |
| **Acceptance criteria** | Written keep vs remove decision with rationale and follow-up tasks if needed |
| **User-visible outcome** | Clear legacy orchestration story |

---

