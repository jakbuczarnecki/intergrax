# RAG FINAL PRODUCTION HANDOFF

**Status:** `PRODUCTION_QUALIFIED_WITH_LIMITATIONS`
**Production deployment:** `APPROVED WITH EXPLICIT LIMITATIONS`
**Canonical runtime defect:** `NONE OPEN FROM PROD-13`
**Qualification date:** 2026-08-10
**Evidence owner:** RAG-PROD-13 and append-only RAG-LIVE-15A-R2,
RAG-LIVE-15B-R2 and RAG-LIVE-15C-R2 records

This is the final RAG-PROD-14 handoff. It closes the qualification decision; it
does not promote the platform to unrestricted `PRODUCTION_QUALIFIED`.

## Post-PROD-13 drift review

The required PROD-13 ancestor
`592ec39b791a5a55474544de9c053b107ad412c5` is present in the current
development history. The commits after it were reviewed as follows:

- `205bd1d2611b9aa86d0deca36e56f61b45291f8c` — category B: workspace
  application consumption of the canonical scoped vector cleanup contract;
- `31d3e0a5330ab0b4814a1168a4a867ab93500cb4` — category A: optional provider
  dependency loading and registry protection, without changing canonical RAG
  identity, scope, ownership, replacement or generation semantics.

No post-PROD-13 commit materially changed canonical RAG behavior.

RAG-LIVE-15A-R2 subsequently qualified the native PgVector provider against
the repository-owned PostgreSQL + pgvector Docker service. RAG-LIVE-15B-R2
subsequently qualified the native Chroma provider against the repository-owned
Chroma 1.4.1 HTTP service. These are append-only live evidence updates; they
do not rewrite the historical PROD-13 record.

RAG-LIVE-15C-R2 subsequently qualified the accepted Neo4j GraphRAG baseline
against the repository-owned Neo4j 5.26 Community Docker service. This is an
append-only live evidence update; it does not qualify publication-generation
fencing or rewrite the historical PROD-13 record.

## Approved production surface

The following native surface is approved under the evidence levels stated in
the qualification record:

- native `KnowledgeDocument` ABI;
- native recursive chunking;
- native ingest → retrieval;
- dense, hybrid and hierarchical retrieval;
- dual-index lifecycle;
- source-scoped reingest and exact source ownership;
- portable logical vector IDs;
- tenant/namespace/workspace isolation contract;
- same-source serialization;
- publication generation visibility fencing;
- canonical GraphRAG harness lifecycle;
- chunker, retriever and reranker plugin architecture;
- native-core LangChain optionality;
- stable vector provider contracts for Qdrant, PgVector and Chroma.

These rows do not all have the same evidence type: they combine offline
contract evidence, canonical harness evidence and provider-specific live gates.

## Live-qualified surface

`LIVE_QUALIFIED` applies to **Qdrant, PgVector and Chroma** under their separate
qualification records. The Qdrant live scope covered:

- tenant, namespace, workspace and combined-scope isolation;
- exact source ownership;
- logical-ID parity;
- source replacement;
- scoped delete;
- foreign-scope preservation;
- bounded soak.

This evidence does not generalize to every Qdrant deployment, network or
storage topology, or to universal production capacity and SLO claims.

The PgVector live scope covered the repository-owned PostgreSQL + pgvector
Docker service, explicit native vector dimension, server-side tenant,
namespace and workspace isolation, exact logical-ID ownership, source
replacement, same-basename preservation, scoped delete, metadata filtering,
fail-closed behavior and a bounded 50-record/5-round soak. See the
[`RAG-LIVE-15A-R2 record`](RAG_PGVECTOR_LIVE_QUALIFICATION.md).

The Chroma live scope covered the repository-owned Chroma 1.4.1 HTTP/server
service, heartbeat readiness, server-side tenant/namespace/workspace
isolation, exact logical-ID ownership, source replacement, same-basename
preservation, scoped delete, metadata filtering, reconstruction, propagated
failure behavior and a bounded 50-record/5-round soak. See the
[`RAG-LIVE-15B-R2 record`](RAG_CHROMA_LIVE_QUALIFICATION.md).

## Offline and harness-only surfaces

- **PgVector:** `QUALIFIED_OFFLINE_CONTRACT + LIVE_QUALIFIED` for the
  repository-owned qualification environment.
- **Chroma:** `QUALIFIED_OFFLINE_CONTRACT + LIVE_QUALIFIED` for the
  repository-owned qualification environment.
- **Canonical GraphRAG:** `CANONICAL_HARNESS_QUALIFIED`.
- **Neo4j GraphRAG baseline:** `LIVE_QUALIFIED_BASELINE`.
- **Neo4j publication-generation fencing:** `NOT LIVE_QUALIFIED`.

`NOT LIVE_QUALIFIED` is an evidence boundary, not a runtime defect.

## Deployment contract

Production operators must:

1. use an authoritative `KnowledgeDocument` scope;
2. propagate tenant, namespace and workspace explicitly;
3. select a provider according to its qualification level;
4. provide exact source ownership when changed-source replacement is enabled;
5. use a durable `SourceOperationCoordinator` for multi-process or
   multi-worker production;
6. use the process-local coordinator only for a single-process ownership
   domain;
7. make retries and recovery tolerate partial vector, TOC or GraphRAG
   publication;
8. not assume a distributed transaction or exactly-once processing;
9. monitor and reclaim stale physical generation records where appropriate;
10. install plugin packages as trusted code;
11. keep plugin/provider secrets outside `KnowledgeDocument` metadata;
12. repeat live provider qualification in the actual production infrastructure
    before making deployment-specific SLO claims.

## Guarantees and non-guarantees

### Guaranteed by canonical contract

| Guarantee | Boundary |
|---|---|
| Document identity and lineage ABI | Native `KnowledgeDocument` contract |
| Exact `VectorStoreScope` | Tenant/namespace/workspace routing |
| Portable logical vector-ID domain | Physical provider IDs remain internal |
| Exact ownership for replacement-capable providers | Scoped source enumeration |
| Fail-closed unsupported replacement | No append-only substitute for changed sources |
| Deterministic generation visibility | Newer publication hides stale evidence |
| Canonical same-source operation key | Serialization contract; coordinator durability is deployment-specific |
| Canonical graph stale-evidence fencing | Qualified harness only |
| Plugin contract boundaries | Native ABI and composition/registration rules |

### Not guaranteed

| Non-guarantee | Consequence |
|---|---|
| Exactly-once ingestion | Retries may repeat work |
| Distributed cross-store transaction | Vector, TOC and graph writes are not one transaction |
| Automatic rollback of every partial publication | Recovery is an operator/application responsibility |
| Zero stale physical records | Reclamation may be required |
| Universal live PgVector or Chroma behavior | Qualification is environment-specific; no universal backend claim |
| Live Neo4j generation fencing | No live evidence |
| Universal Qdrant performance | No universal capacity or SLO claim |
| Multi-process safety with the default coordinator | Durable coordination is required |
| Safety or sandboxing of arbitrary Python plugin code | Plugins are trusted installed code |

## Provider deployment decision

See [`RAG.md`](../../architecture/RAG.md) for the detailed capability matrix.

| Provider | Catalog | Offline evidence | Live deployment decision |
|---|---|---|---|
| Qdrant | `STABLE` | Qualified | Live-qualified; recommended when the qualified live path is required |
| PgVector | `STABLE` | Qualified | Live-qualified by RAG-LIVE-15A-R2 in the repository-owned environment |
| Chroma | `STABLE` | Qualified | Live-qualified by RAG-LIVE-15B-R2 in the repository-owned environment |
| Neo4j GraphRAG baseline | `STABLE` | Canonical harness qualified | `LIVE_QUALIFIED_BASELINE` by RAG-LIVE-15C-R2 in the repository-owned environment |
| Weaviate / LanceDB / Typesense | `BETA` | — | Source replacement unsupported |
| Pinecone / Milvus / Vespa | `BETA` | — | No live qualification claim |
| InMemory | Harness/test taxonomy | Canonical harness use | Use for harness/tests according to current taxonomy |

## GraphRAG handoff

Approved at canonical-harness level:

- canonical GraphRAG architecture;
- graph indexing and retrieval;
- source replacement;
- stale topology fencing;
- shared evidence preservation.

Evidence combines the canonical `InMemoryGraphStore` harness with the
RAG-LIVE-15C-R2 live Neo4j baseline. The live gate covered scoped indexing,
exact source ownership, shared evidence, source replacement, safe unlink,
canonical traversal, failure semantics and cleanup. It did not qualify
publication-generation fencing, concurrent generation takeover or visibility
handoff. Neo4j limitations do not block the native RAG platform as a whole.

## Extension handoff

The [`RAG_EXTENSION_GUIDE.md`](../../technical/guides/RAG_EXTENSION_GUIDE.md)
is authoritative:

- `PUBLIC_EXTERNAL_PLUGIN`: chunker, retriever, reranker;
- parser and vector extensions: Integration Library;
- embedding, loader, metadata and `GraphIndexer`: current
  composition/internal registration model.

The native core cannot be bypassed. LangChain remains permitted only as an
optional implementation behind native contracts.

## Production deployment checklist

- [ ] Choose a qualified vector backend.
- [ ] Configure explicit tenant/namespace/workspace scope.
- [ ] Configure a durable coordinator for multi-worker deployment.
- [ ] Validate source ownership support.
- [ ] Run the provider-specific live isolation/lifecycle gate.
- [ ] Configure bounded retries and timeouts.
- [ ] Configure observability.
- [ ] Verify cleanup and recovery procedures.
- [ ] Validate the optional GraphRAG backend if enabled.
- [ ] Validate external plugins in the target environment.
- [ ] Preserve canonical qualification evidence with the deployment release.

## Evidence index

1. [`RAG.md`](../../architecture/RAG.md) — canonical architecture and provider
   taxonomy.
2. [`RAG_PRODUCTION_QUALIFICATION.md`](RAG_PRODUCTION_QUALIFICATION.md) —
   detailed RAG-PROD-13 evidence ledger.
3. [`RAG_PGVECTOR_LIVE_QUALIFICATION.md`](RAG_PGVECTOR_LIVE_QUALIFICATION.md) —
   RAG-LIVE-15A-R2 PgVector live evidence.
4. [`RAG_CHROMA_LIVE_QUALIFICATION.md`](RAG_CHROMA_LIVE_QUALIFICATION.md) —
   RAG-LIVE-15B-R2 Chroma live evidence.
5. [`RAG_NEO4J_LIVE_BASELINE_QUALIFICATION.md`](RAG_NEO4J_LIVE_BASELINE_QUALIFICATION.md) —
  RAG-LIVE-15C-R2 Neo4j GraphRAG live baseline evidence.
6. [`RAG_EXTENSION_GUIDE.md`](../../technical/guides/RAG_EXTENSION_GUIDE.md) —
   extension and plugin contracts.
7. [`LANGCHAIN_INDEPENDENCE_native_document_contract.md`](../../capabilities/architecture/satellites/LANGCHAIN_INDEPENDENCE_native_document_contract.md)
   — `KnowledgeDocument` ABI detail.
8. [`RAG.md` historical plan](../plans/RAG.md) — implementation history only.

Accepted PROD-13 evidence commits are
`a93c68c138fea4a8758df9e3aca43fd454f521c0` and
`592ec39b791a5a55474544de9c053b107ad412c5`. The current production decision
is defined by this handoff and the linked documents, not by searching history.

## Reopening criteria

Reopen RAG production qualification when any of the following changes:

- `KnowledgeDocument` ABI;
- `VectorStoreScope` semantics;
- logical-ID invariants;
- source-ownership contract;
- replacement lifecycle;
- generation or lease semantics;
- promotion of a new stable vector backend;
- material Qdrant adapter behavior;
- durable coordinator behavior;
- GraphRAG generation model;
- live Neo4j qualification is added;
- a new public RAG plugin ABI is introduced.

Documentation-only edits, application-level consumers using unchanged
contracts and unrelated vendor/session work do not automatically reopen
qualification.

## Closure

**Roadmap:** `RAG-PROD-14 CLOSED`
**Current live qualification:** `RAG-LIVE-15A-R2, RAG-LIVE-15B-R2 and
RAG-LIVE-15C-R2 COMPLETE`
**Global status remains:** `PRODUCTION_QUALIFIED_WITH_LIMITATIONS`
**Next:** `RAG-LIVE-15D NOT STARTED`
