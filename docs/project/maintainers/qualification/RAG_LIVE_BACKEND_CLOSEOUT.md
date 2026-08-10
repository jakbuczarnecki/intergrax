# RAG-LIVE-15E — Multi-Backend Live Qualification Closeout

**Status:** `READY_FOR_REVIEW`  
**Closeout date:** 2026-08-10  
**Repository:** `development` @ `b4e068c84b62be28a310b99eefe82f0664bafe74`  
**Decision owner:** RAG-LIVE-15E audit

Append-only closeout record. This is the final gateway to all live-backend
qualification evidence. It does not rewrite historical PROD-13 or per-provider
live records.

## 1. Decision

| Item | Result |
|---|---|
| **Global** | `PRODUCTION_QUALIFIED_WITH_LIMITATIONS` |
| **Deployment** | `APPROVED WITH EXPLICIT LIMITATIONS` |
| **RAG-LIVE track** | `CLOSED` |
| **Post-15D semantic drift** | None (`b4e068c8` — category A only) |

All targeted stable live backends have consistent, linked evidence. No further
RAG-LIVE implementation task is planned.

## 2. Final provider matrix

| Provider / surface | Offline / harness | Live status | Evidence |
|---|---|---|---|
| Qdrant | `QUALIFIED_OFFLINE_CONTRACT` | `LIVE_QUALIFIED` | [`RAG_PRODUCTION_QUALIFICATION.md`](RAG_PRODUCTION_QUALIFICATION.md) (RAG-PROD-13) |
| PgVector | `QUALIFIED_OFFLINE_CONTRACT` | `LIVE_QUALIFIED` | [`RAG_PGVECTOR_LIVE_QUALIFICATION.md`](RAG_PGVECTOR_LIVE_QUALIFICATION.md) (RAG-LIVE-15A-R2) |
| Chroma | `QUALIFIED_OFFLINE_CONTRACT` | `LIVE_QUALIFIED` | [`RAG_CHROMA_LIVE_QUALIFICATION.md`](RAG_CHROMA_LIVE_QUALIFICATION.md) (RAG-LIVE-15B-R2) |
| Neo4j GraphRAG baseline | `CANONICAL_HARNESS_QUALIFIED` | `LIVE_QUALIFIED_BASELINE` | [`RAG_NEO4J_LIVE_BASELINE_QUALIFICATION.md`](RAG_NEO4J_LIVE_BASELINE_QUALIFICATION.md) (RAG-LIVE-15C-R2) |
| Neo4j publication-generation fencing | `CANONICAL_HARNESS_QUALIFIED` | `LIVE_QUALIFIED` | [`RAG_NEO4J_GENERATION_FENCING_QUALIFICATION.md`](RAG_NEO4J_GENERATION_FENCING_QUALIFICATION.md) (RAG-LIVE-15D-R2) |
| Canonical GraphRAG | `CANONICAL_HARNESS_QUALIFIED` | `LIVE_NEO4J_BASELINE + LIVE_NEO4J_GENERATION_FENCING` | 15C + 15D records above |

**Beta / non-qualified (no live promotion):** Weaviate, LanceDB, Typesense,
Pinecone, Milvus, Vespa — catalog `BETA`; no stable live qualification claim.

## 3. Evidence chain

Required accepted ancestors (all present in closeout HEAD):

| SHA | Role |
|---|---|
| `a27cc92e5720702e8e910408500baf560c22e868` | RAG-LIVE-15B Chroma live qualification |
| `4d4626475abee892a7c295d4fa32d968c2d639c2` | Neo4j graph evidence ownership scope |
| `23239c003551dbd6edc31fb8e2125f86e40cda95` | RAG-LIVE-15C Neo4j baseline live qualification |
| `c3ad18537e2f62670aae21d719f3a58e92670bf3` | Neo4j publication-generation fencing runtime |
| `52e837900f3f0b074e4e9bd0e9e3c92e440a04d2` | RAG-LIVE-15D-R2 live evidence record |

Post-15D commit review:

| SHA | Classification | Note |
|---|---|---|
| `b4e068c84b62be28a310b99eefe82f0664bafe74` | A | `intergrax/integrations/_shared/p3/factories.py` unused-import cleanup only |

No category C commit exists after live evidence. No live gate refresh required.

**Key live run identifiers (verified in source records):**

- PgVector: `3f8fb8cd26ac41a882ec1621b40bbd12`, `44251fc2e09445cf9554821404d47c00`
- Chroma: `e49d90098a7f44ff8fec3c91e3e75010`, `8336fad58be14d828422a6f1e207a748`
- Neo4j generation fencing: `f466a5f53a9b4bbda004851f43c55172`, `2de4cc1062fb4a97aa3011776764a319` (20 contention iterations, 0 failures each)

Production handoff: [`RAG_PRODUCTION_HANDOFF.md`](RAG_PRODUCTION_HANDOFF.md).  
Architecture: [`../../architecture/RAG.md`](../../architecture/RAG.md).

## 4. GraphRAG status

- Baseline: `LIVE_QUALIFIED_BASELINE` only — does not independently claim generation fencing.
- Generation fencing: `LIVE_QUALIFIED` by RAG-LIVE-15D-R2.
- Combined canonical claim: `CANONICAL_HARNESS_QUALIFIED + LIVE_NEO4J_BASELINE + LIVE_NEO4J_GENERATION_FENCING`.

## 5. Global production status

`PRODUCTION_QUALIFIED_WITH_LIMITATIONS` is the final production contract, not
an open task state. The platform is production-qualified for explicitly
qualified surfaces while retaining architectural and deployment limitations
documented in the handoff.

## 6. Deployment limitations (must remain explicit)

1. No distributed transaction across vector, TOC, graph or other stores.
2. No exactly-once publication claim.
3. Default `SourceOperationCoordinator` is process-local; multi-process deployment needs a durable coordinator.
4. Physical stale records may remain; generation fencing guarantees logical visibility, not immediate physical deletion.
5. Live provider qualification is environment-specific; no universal capacity, latency, durability, cloud topology or backend-version claim.
6. Plugins are trusted installed Python code, not sandboxed.
7. Optional/beta providers without live evidence remain outside stable live qualification.
8. Legacy Neo4j graph schema is not silently migrated or reinterpreted.

## 7. Provider / environment boundaries

Live evidence covers repository-owned qualification environments only:

- Qdrant: real backend, scope isolation, adversarial foreign-document isolation, ownership, logical-ID parity, replacement, scoped delete, cleanup, bounded soak (RAG-PROD-13).
- PgVector: PostgreSQL + pgvector server path, no memory fallback, server-side search, two live passes, bounded soak (15A-R2).
- Chroma: HTTP/server 1.4.1 path, no embedded fallback, server-side scope where applicable, two live passes, bounded soak (15B-R2).
- Neo4j: 5.26 Community, driver 5.28.4, full GraphScope baseline (15C-R2); generation authority, G1/G2, stale writer, contention and cleanup gates (15D-R2).

## 8. Audit exceptions

**`AUDIT_EXCEPTION: MIXED_COMMIT_OWNERSHIP`**

RAG-LIVE-15D-R2 evidence paths were committed in
`52e837900f3f0b074e4e9bd0e9e3c92e440a04d2`. That commit also contained unrelated
concurrent Vendor Knowledge / workspace changes from pre-existing staged state.
RAG qualification evidence was independently audited by path and accepted. No
history rewrite was performed because `development` is shared and the commit
had already been pushed. This is a provenance exception, not a runtime or
qualification defect.

## 9. Reopening criteria

Reopen RAG live-backend qualification only when:

- `KnowledgeDocument` ABI changes
- `VectorStoreScope` semantics change
- logical vector ID law changes
- source ownership / replacement law changes
- publication generation visibility semantics change
- GraphScope / `RagEvidence` ownership model changes
- material Qdrant adapter change
- material PgVector adapter change
- material Chroma HTTP adapter change
- material Neo4j GraphRAG adapter change
- `SourceOperationCoordinator` semantics change
- introduction of durable coordinator with new guarantees
- new stable backend promotion
- public RAG plugin ABI change
- backend/server version changes requiring refreshed qualification evidence

**Do not reopen** for ordinary docs changes, application consumers, Vendor
Knowledge work or unrelated integrations.

## 10. Final roadmap state

| Task | State |
|---|---|
| RAG-LIVE-15A | `CLOSED` |
| RAG-LIVE-15B | `CLOSED` |
| RAG-LIVE-15C | `CLOSED` |
| RAG-LIVE-15D | `CLOSED` |
| RAG-LIVE-15E | `CLOSED` |
| **RAG-LIVE track** | **`CLOSED`** |

**Next RAG-LIVE task:** none.
