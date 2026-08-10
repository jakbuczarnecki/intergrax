# RAG-LIVE-15D-R2 — Neo4j Publication-Generation Fencing Live Qualification

**Status:** `READY_FOR_REVIEW`  
**Neo4j GraphRAG baseline:** `LIVE_QUALIFIED_BASELINE`  
**Neo4j publication-generation fencing:** `LIVE_QUALIFIED`  
**Canonical GraphRAG:** `CANONICAL_HARNESS_QUALIFIED + LIVE_NEO4J_BASELINE + LIVE_NEO4J_GENERATION_FENCING`  
**Global:** `PRODUCTION_QUALIFIED_WITH_LIMITATIONS`  
**Qualification date:** 2026-08-10

This is an append-only qualification record. It live-qualifies the accepted
Neo4j publication-generation fencing runtime against the repository-owned
Docker service. It does not redesign generation fencing and does not reopen
RAG-LIVE-15C baseline decisions.

## Repository and environment

- Start HEAD/origin: `c3ad18537e2f62670aae21d719f3a58e92670bf3` (`development`).
- Required accepted ancestor: `c3ad18537e2f62670aae21d719f3a58e92670bf3`.
- Qualification test:
  `tests/integration/rag/test_neo4j_generation_fencing_qualification.py`.
- Accepted smoke helper reuse only for coordinator/topology patterns:
  `tests/integration/rag/test_neo4j_generation_fencing_smoke.py`.
- Docker Compose: `infra/docker/neo4j/docker-compose.yml`.
- Server image: `neo4j:5.26-community`.
- Driver: `neo4j==5.28.4` (`neo4j>=5.26,<6`).
- Bolt: `bolt://localhost:7687`; HTTP: `http://localhost:7474`.
- Container health: `healthy`; `verify_connectivity` and `RETURN 1` passed.
- Provider open: production `create_neo4j_graph_store` path and
  `Neo4jRagGraphStore`; no InMemory fallback.
- Qualification state: unique run-scoped tenant, namespace, workspace and
  source identifiers per run.
- Concurrent unrelated working-tree changes were preserved.

The exact environment-gated command was:

```text
INTERGRAX_RUN_NEO4J_LIVE=1 uv run pytest tests/integration/rag/test_neo4j_generation_fencing_qualification.py -q -s
```

Only provider setup/open failures can skip. After provider open, all runtime,
Cypher, coordinator, traversal and cleanup failures fail the test.

## Live runs

### Live run 1 — PASS

- Run identifier: `f466a5f53a9b4bbda004851f43c55172`.
- G1/G2 takeover, late stale writer, reverse completion, reduced topology,
  generation-specific cleanup, shared support, scope isolation,
  server-side `active_pairs` filtering, partial publication, idempotency,
  coordinator failure, GraphRAG retrieval fencing, InMemory parity inside the
  same module, contention loop and cleanup: PASS.
- Contention: `iterations=20`, `failures=0`.

### Live run 2 — PASS

- Run identifier: `2de4cc1062fb4a97aa3011776764a319`.
- Same gate coverage as live run 1: PASS.
- Contention: `iterations=20`, `failures=0`.

Both runs started from a new clean qualification scope and completed cleanup.

## Gate evidence

### Generation authority

- Authoritative `SourceOperationCoordinator` via
  `InProcessSourceOperationCoordinator`.
- Exact `RagSourceOperationKey`: tenant + namespace + workspace + source_id.
- Generation creation used only `acquire()`, `publication_generation()` and
  `promote_publication()`; no fabricated active-generation Cypher.

### G1/G2 takeover — PASS

- G1 topology `Alpha -> Beta -> Gamma` became invisible after G2 promotion.
- G2 topology `Alpha -> Quasar -> Zeta` remained authoritative.

### Late stale writer — PASS

- Physical G1 `RagEvidence` remained in Neo4j after delayed G1 writes.
- G1 topology stayed invisible; G2 remained visible; G1 could not overwrite G2.

### Reverse completion order — PASS

- `start G1 -> start G2 -> promote G2 -> late G1 write`: resolved to G2.
- `partial G1 -> G2 takeover -> G2 finish -> late G1 cleanup`: resolved to G2.
- Deterministic lease barriers only; no sleep-based races.

### Reduced topology — PASS

- G1 `A -> B -> C -> D`, G2 `A -> B`; only G2-visible `A -> B` traversed.

### Generation-specific cleanup — PASS

- `unlink_source_generation(source, G1, scope)` removed only G1 evidence.
- G2 traversal unchanged; repeat cleanup idempotent.

### Shared support — PASS

- Source B current support kept `X -> Y` visible after source A G2 promotion.
- A/G1 cleanup removed only A stale support; removing B removed `X -> Y` when
  no other current support remained.

### Scope isolation — PASS

- Independent generations across other namespace, workspace and tenant scopes.
- Promoting A/G2 in the main scope did not fence evidence in other scopes.

### Server-side fencing — PASS

- Source inspection of `visibility_query_params`, `find_nodes`, `neighbors` and
  `chunk_ids_for_nodes` showed `coordinator_bound` and `active_pairs`.
- Live query capture proved generation predicates and active pairs are passed
  into Cypher before topology/evidence reads.

### Find / traversal / chunk lookups — PASS

- Fencing applied consistently to `find_nodes`, `neighbors`,
  `chunk_ids_for_nodes`, `node_ids_for_chunks` and canonical `GraphRagRetriever`
  expansion.

### Partial publication — PASS

- Unpromoted G1 physical records remained.
- Versioned G1 chunk/topology reads were invisible; G2 visible.
- Physical-stale-vs-logical-visible distinction recorded via physical evidence
  counts plus fenced chunk/traversal assertions.

### Idempotency — PASS

- Repeated current-generation writes did not explode `RagEvidence` counts.
- Visible topology unchanged; stale G1 not resurrected.

### Coordinator failure — PASS

- After versioned evidence existed, injected coordinator lookup failure at the
  GraphStore boundary failed closed (`RuntimeError` on Neo4j reads; empty fenced
  reads on InMemory for versioned evidence).

### Unbound coordinator — PASS

- Versioned evidence with no coordinator was not visible.
- Legacy `generation IS NULL` evidence remained visible per compatibility law.

### InMemory parity — PASS

- Same essential scenarios on `InMemoryGraphStore` matched Neo4j visible-result
  law for takeover, late stale writer, cleanup, shared support and coordinator
  failure.

### Contention loop — PASS

- `iterations=20`, `failures=0` deterministic handoff evidence.

### Cleanup — PASS

- Each run removed qualification-owned evidence, chunks, entities, `RAG_REL` and
  run-owned source state; foreign data untouched.

## Offline regression

Targeted offline regression passed:

```text
uv run pytest \
  tests/unit/rag/graph/test_graph_generation_fencing.py \
  tests/unit/rag/graph/test_neo4j_rag_graph_store.py \
  tests/unit/rag/graph/test_graph_rag_neo4j_prod_contract.py \
  tests/integration/rag/test_same_source_reingest_serialization.py \
  tests/unit/rag/graph/test_graph_rag_retriever.py \
  tests/integration/rag/test_neo4j_generation_fencing_smoke.py -q
```

Result: `27 passed`.

## Transaction / guarantee boundary

### Qualified

- generation visibility fencing;
- stale writer suppression at read/traversal boundaries;
- generation-specific cleanup safety;
- scoped generation authority across tenant/namespace/workspace.

### Not claimed

- distributed cross-store transaction;
- exactly-once publication;
- zero stale physical records;
- synchronous reclamation;
- multi-process coordinator durability unless a durable coordinator is used.

## Remaining limitations

- Global closeout remains `PRODUCTION_QUALIFIED_WITH_LIMITATIONS` until
  RAG-LIVE-15E.
- Physical orphan entities without node evidence may remain visible under the
  accepted compatibility law; logical versioned evidence remains fenced.
- Qualification used the process-local coordinator; durable coordinator
  behavior is deployment-specific.
