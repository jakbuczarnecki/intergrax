# RAG — Implementation Plan

**Architecture (1:1):** [`architecture/RAG.md`](../../architecture/RAG.md)  
**Hub:** [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md)  
**Strategy:** [`guides/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../technical/guides/INTERGRAX_DEVELOPMENT_STRATEGY.md)

> When implementing this layer, read **only** the architecture doc and **this plan hub** (`plan/satellites/` satellites on demand).

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (RAG plan).

- **Implement / audit default:** Hub §6 · [`plan/satellites/`](plan/satellites) satellites on demand. **On demand (one max):** [`plan/satellites/RAG_audit_history.md`](plan/satellites/RAG_audit_history.md) · [`plan/satellites/RAG_embedded_detail.md`](plan/satellites/RAG_embedded_detail.md). Phase AUDIT-IDEAL — **Planned** / open rows only. §6.1 maintenance queues — open P0/P1 only
- **Use** `Read` with offset/limit — open `### 6.1*` / Phase rows (**P0/P1**, Status ≠ Done) only.
- **Skip** `(closed)`, `(complete)`, `Archived`, **Done** unless re-validating a cited gap.
- **Architecture hub:** [`architecture/RAG.md`](../../architecture/RAG.md) read-scope block only.
- **Audit slice:** [`guides/audit_slices/RAG.md`](../../technical/guides/audit_slices/RAG.md).
- **Satellites:** at most **one** `plan/satellites/` file per session unless RESUME cites more.

---

## Satellite registers (read on demand)

Large historical registers moved out of the hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited gap ID.

| Satellite | Contents |
|-----------|----------|
| [`plan/satellites/RAG_audit_history.md`](plan/satellites/RAG_audit_history.md) | audit history |
| [`plan/satellites/RAG_embedded_detail.md`](plan/satellites/RAG_embedded_detail.md) | embedded detail |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.


---

## Phase AUDIT-IDEAL — RAG gap register (layer 14)

**Source:** Post-L3 audit vs [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../../technical/guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §3.6, §7.7  
**Master register:** [`plan/AUDIT_IDEAL_2026.md`](AUDIT_IDEAL_2026.md) · Band **2ay** · queue **§6.1au**  
**Engine depth audit:** 2026-06-10 — full register in [`architecture/RAG.md`](../../architecture/RAG.md) §Engine depth audit register

| ID | AUDIT § | Gap | Priority | Status | M-RAG |
|----|---------|-----|----------|--------|-------|
| AUDIT-IDEAL-14.1 | §14 RAG | Graph RAG production profile (shared with MEMORY) | P1 | **Done** | M-RAG.12 (stable) |
| AUDIT-IDEAL-14.3 | §14 RAG | Wire `RagProfile.query_expansion` to retrieval path | P0 | **Done** | M-RAG.23 |
| AUDIT-IDEAL-14.4 | §14 RAG | Dual-index + hierarchical retriever default bootstrap | P1 | **Done** | M-RAG.24 |
| AUDIT-IDEAL-14.5 | §14 RAG | Retrieval poisoning defense on `rag.retrieve` catalog path | P1 | **Done** | M-RAG.25 |
| AUDIT-IDEAL-14.6 | §14 RAG | Large-corpus async ingest (stream / job orchestration) | P1 | **Done** | M-RAG.26 |
| AUDIT-IDEAL-14.7 | §14 RAG | OpenTelemetry spans on RAG retrieve + ingest hot path | P2 | **Done** | M-RAG.27 |
| AUDIT-IDEAL-14.8 | §14 RAG · §3.7.1 | Universal GraphRAG platform — backend registry, lifecycle, retrieval hardening | P1 | **Done** (G1–G3; G4 optional) | M-RAG-GRAPH (M-RAG.38–M-RAG.47, M-RAG.48, M-RAG.52) |

**Note:** AUDIT-IDEAL-14.2 (retrieval poisoning on product hosts) is owned by [`plan/MEMORY.md`](MEMORY.md) + UAEP security wiring — Nexus `rag.retrieve` (catalog) path.

**Delivery rule:** One **AUDIT-IDEAL-\*** ID per PR (when applicable) → update this table + master register → gate green. Additional GAP-RAG rows without AUDIT-IDEAL IDs use M-RAG.\* only.

**Engine audit (2026-06-13):** Maturity **L3 implementation / L3 control plane** — **Frozen**. Closeout: [Phase M-RAG-CONVERGE](#phase-m-rag-converge--doc--diagnostics-closeout-2026-06-13).

---

## Active cross-feature work — LangChain Independence

| Task | Priority | Status | Deliverable | Next |
|------|----------|--------|-------------|------|
| LCI-2B | P1 | APPROVED | scoped native handler and loader boundary | LCI-2C |
| LCI-2C | P1 | APPROVED | native normalization and metadata pipeline | LCI-2D |
| LCI-2D | P1 | APPROVED | native chunking contract and derivative lineage | LCI-2E |
| LCI-2E | P1 | APPROVED | optional LangChain recursive splitter | LCI-2F |
| LCI-2F | P1 | APPROVED | end-to-end native ingest boundary | LCI-3A |
| LCI-3A | P1 | APPROVED | native embedding contract and result | LCI-3B |
| LCI-3B | P1 | APPROVED | native indexing contract and TOC lineage | LCI-3C |
| LCI-3C | P1 | APPROVED | native vector-store records and tenant isolation | LCI-3D |
| LCI-3D | P1 | APPROVED | native provider adapters with scoped native records and hits | LCI-4A |
| LCI-4A | P1 | APPROVED | native immutable retrieval hit/result contract and native RAG tool output | LCI-4B |
| LCI-4B | P1 | APPROVED | native immutable reranker candidate/result contract and RetrievalHit boundary | LCI-4C |
| LCI-4C-A1 | P1 | READY_FOR_REVIEW | canonical workspace-aware KnowledgeDocumentScope ABI propagation | LCI-4D |
| LCI-4C | P1 | READY_FOR_REVIEW | native Graph RAG indexer, isolation and retrieval document contract | LCI-4D |
| LCI-4D | P1 | READY_FOR_REVIEW | native auxiliary memory, multimedia, legacy answer, evaluation and soak document paths | LCI-5A |

LCI-3D closes the vector-store provider boundary. The VectorStore provider port is
native: active providers accept `VectorStoreRecord` and `VectorStoreScope`, map
`KnowledgeDocument` directly to provider SDK payloads, and return native
`VectorStoreHit` values. Tenant, namespace, and workspace routing are
system-owned at each provider boundary. LangChain `Document` compatibility is
absent from vector-store paths. Retrieval result migration is complete for active retrievers and RAG tools under LCI-4A; reranking is complete under LCI-4B; graph retrieval is the active LCI-4C review target.

**Contract spec:** [`../features/architecture/satellites/LANGCHAIN_INDEPENDENCE_native_document_contract.md`](../../capabilities/architecture/satellites/LANGCHAIN_INDEPENDENCE_native_document_contract.md) · **Feature plan:** [`features/plan/LANGCHAIN_INDEPENDENCE.md`](../../capabilities/plan/LANGCHAIN_INDEPENDENCE.md)

LCI-4B reranking preserves the native `KnowledgeDocument`, identity, scope,
provenance and user metadata while adding only rerank/fusion scores and final
rank. LCI-4C Graph RAG indexers and graph isolation contracts accept native
`KnowledgeDocument` values, and graph retrieval uses the existing native
`RetrievalHit` contract. GraphStore backends remain tenant-bound where applicable;
namespace and workspace isolation are enforced by the bound Graph indexer scope
fence before writes. A single indexer instance cannot switch tenant, namespace or
workspace after binding. GraphStore backend internals remain unchanged.

LCI-4D is READY_FOR_REVIEW for auxiliary memory, multimedia, legacy RAG,
evaluation and soak paths. These paths use `KnowledgeDocument` and native
`VectorStoreRecord`/`VectorStoreScope` boundaries while preserving identity,
scope, provenance and source metadata. LCI-5A remains PLANNED / NEXT AFTER
ACCEPTANCE.

LCI-4C-A1 records that `workspace_id` is a canonical system-owned
`KnowledgeDocumentScope` field. The canonical identity boundary is
`tenant_id + namespace + workspace_id + document_id`. Missing `workspace_id`
remains backward-compatible and means no explicit workspace partition. User
metadata cannot provide or override `workspace_id`; indexing, vector storage,
retrieval, reranking and Graph RAG preserve workspace scope without using
metadata as a transport tunnel.

---
