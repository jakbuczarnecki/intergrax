# RAG-LIVE-15C-R2 — Neo4j Live GraphRAG Baseline Qualification

**Status:** `READY_FOR_REVIEW`  
**Neo4j GraphRAG baseline:** `LIVE_QUALIFIED_BASELINE`  
**Canonical GraphRAG:** `CANONICAL_HARNESS_QUALIFIED + LIVE_NEO4J_BASELINE`  
**Generation fencing:** `NOT LIVE_QUALIFIED`  
**Global:** `PRODUCTION_QUALIFIED_WITH_LIMITATIONS`  
**Qualification date:** 2026-08-10

This is an append-only qualification record. It qualifies the accepted
scoped Neo4j GraphRAG runtime against the repository-owned Docker service. It
does not redesign GraphRAG ownership and does not qualify publication
generation fencing.

## Repository and environment

- Start HEAD/origin: `4d4626475abee892a7c295d4fa32d968c2d639c2`
  (`development`).
- Final HEAD/origin: `2d39fe9b998bfaacdd73b9a82e3387250e182816`
  (`development`); the concurrent vendor-knowledge fast-forward was
  preserved.
- Required accepted ancestor: `4d4626475abee892a7c295d4fa32d968c2d639c2`.
- Qualification test:
  `tests/integration/rag/test_neo4j_live_qualification.py`.
- Docker Compose:
  `infra/docker/neo4j/docker-compose.yml`.
- Server image: `neo4j:5.26-community`.
- Driver: `neo4j==5.28.4` (`neo4j>=5.26,<6`).
- Bolt: `bolt://localhost:7687`; HTTP: `http://localhost:7474`.
- Container health: `healthy`; `cypher-shell RETURN 1` succeeded.
- Provider open: production `create_neo4j_graph_store` path,
  `Neo4jRagGraphStore`, `verify_connectivity()` and live `RETURN 1`.
- InMemory GraphStore fallback: not used. The vector side of the canonical
  harness used scoped InMemory vectors only as retrieval seed data.
- Qualification state: unique run-scoped tenant, namespace, workspace and
  source identifiers; compose qualification volume was not purged.
- Unrelated concurrent work, including commit
  `2d39fe9b998bfaacdd73b9a82e3387250e182816` and the working-tree change in
  `intergrax/integrations/_shared/p3/factories.py`, was preserved.

The exact environment-gated command was:

```text
INTERGRAX_RUN_NEO4J_LIVE=1 uv run pytest tests/integration/rag/test_neo4j_live_qualification.py -q -s
```

Only provider setup/open failures can skip. After provider open, all runtime,
Cypher, ownership, traversal and cleanup failures fail the test.

## Live runs

### Live run 1 — PASS

- Run identifier: `e3bceeaa7ede4018b0a6d885dcc665fd`.
- Scope, ownership, shared evidence, replacement, unlink and traversal:
  PASS.
- Failure semantics and bounded transaction-boundary check: PASS.
- Soak: 48 graph evidence items, 5 traversal rounds, observed p95
  `4.14 ms`; no canonical graph SLO was applied.
- Cleanup: PASS.

### Live run 2 — PASS

- Run identifier: `539ca616c59c4114803a15012aeff8cd`.
- Scope, ownership, shared evidence, replacement, unlink and traversal:
  PASS.
- Failure semantics and bounded transaction-boundary check: PASS.
- Soak: 48 graph evidence items, 5 traversal rounds, observed p95
  `4.73 ms`; no canonical graph SLO was applied.
- Cleanup: PASS.

Both runs started from a new clean qualification scope and completed cleanup.

## Gate evidence

### Scope matrix and identity

The live gate created:

1. tenant A / namespace A / workspace A;
2. tenant A / namespace B / workspace A;
3. tenant A / namespace A / workspace B;
4. tenant A / namespace B / workspace B;
5. tenant B / namespace A / workspace A.

The same logical entity IDs and names were used in every matrix scope.
Traversal, `find_nodes`, chunk lookup and canonical retrieval returned only the
requested combined scope. Raw read-only Cypher showed five physical
`RagEntity` records with the same logical ID and five distinct `scope_key`
values. `RagChunk` physical records also carried `scope_key`; no cross-scope
physical collision or foreign metadata overwrite was observed.

### Canonical ingest and ownership

All graph data was produced by `KnowledgeDocument`-derived documents flowing
through the canonical `IngestPipeline`, splitter, vector manager, GraphRAG
indexer and live `Neo4jRagGraphStore`. The test did not create the graph
through hand-written qualification Cypher.

Exact ownership by tenant, namespace, workspace and `source_id` passed.
Two distinct source IDs used the same basename-like `same.txt` metadata and
overlapping entity/relationship topology. Their `RagEvidence` records were
distinguishable exactly by `source_id`; generation metadata was present and
preserved.

Read-only Cypher confirmed:

- scope-aware `RagEntity` and `RagChunk` physical identity;
- `RagEvidence.source_id` plus tenant/namespace/workspace scope;
- evidence links to nodes, edges and chunks;
- one shared relation with two source supporters;
- no cross-scope physical identity collision.

### Shared support and pruning

Source A and source B supported the same `Alpha Node -> Beta Node` topology.
Each source had its own `RagEvidence`. Unlinking A removed only A evidence;
the relation remained traversable through B. Unlinking B removed the last
support and pruned the semantic relation and orphan-only graph objects.

### Replacement

- Version 1: A supported `Alpha -> Beta` and `Beta -> Gamma`.
- Version 2: A changed to `Alpha -> Quasar` and `Quasar -> Zeta`.
- Version 3: A was reduced to `Alpha -> Quasar`.
- Repeated version 3 was idempotent.

Old A-only topology and stale A evidence were removed. Current topology
remained visible, B support was preserved, and orphan-only nodes/relations
were pruned.

### Scope-safe unlink and traversal

Unlinking A in the exact combined scope preserved B, every other namespace
and workspace, tenant B, and the matrix records. Canonical GraphRAG
retrieval/traversal showed current topology, hid stale topology, excluded
foreign scope, and retained B-supported shared topology.

### Failure semantics

After successful provider open, the gate verified that invalid scope,
malformed results, Cypher failures, ownership lookup/unlink failures and
traversal backend failures fail closed by raising. An idempotent zero-row
unlink returned zero and did not claim a removal. No corrupted shared state
was introduced.

### Transaction boundary

Neo4j graph writes are bounded by the driver session transaction path using
`session.execute_write`. This is a Neo4j transaction-bound statement only.
There is no cross-store transaction claim, exactly-once claim, or vector/TOC/
graph atomicity claim.

## Offline regression and validation

Targeted GraphRAG regression:

```text
uv run pytest tests/unit/rag/graph/test_neo4j_rag_graph_store.py tests/unit/rag/graph/test_graph_rag_neo4j_prod_contract.py tests/unit/rag/graph/test_graph_rag_retriever.py tests/unit/rag/graph/test_graph_tenant_isolation.py tests/unit/rag/graph/test_graph_lifecycle_delete_sync.py tests/integration/rag/test_graph_reingest_qualification.py -q
```

Result: `13 passed`.

Additional validation:

- Ruff changed Python: PASS.
- Compose config: PASS.
- Relative documentation links: PASS.
- Mojibake: PASS.
- `git diff --check`: PASS.

## Cleanup and limitations

Both runs removed all run-owned `RagEvidence`, `RagChunk`, `RagEntity` and
`RAG_REL` records through exact source/scoped cleanup and verified no
run-scoped graph objects remained. Unrelated graph data was not touched.

This evidence is limited to the repository-owned Neo4j 5.26 Community Docker
service, local Bolt topology, pinned Python driver and tested native runtime.
The soak records latency observations only; it is not a universal capacity,
durability or production SLO claim.

Generation metadata exists and is preserved, but R2 does not qualify stale
generation takeover, concurrent generation fencing, publication visibility
handoff or stale topology suppression under race. Those remain exclusively
`RAG-LIVE-15D`.

## Decision

All R2 baseline gates passed twice:

```text
Neo4j GraphRAG baseline = LIVE_QUALIFIED_BASELINE
Canonical GraphRAG = CANONICAL_HARNESS_QUALIFIED + LIVE_NEO4J_BASELINE
Generation fencing = NOT LIVE_QUALIFIED
Global = PRODUCTION_QUALIFIED_WITH_LIMITATIONS
```

Next: `RAG-LIVE-15D` — `NOT STARTED`.
