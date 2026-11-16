# T-IGX-041-P — PDC-X batch B-P8-R4: repair of finding G-23 (GraphRAG evidence of a re-ingested source resolves its active publication generation under the key scoped by the root document id)

## Scope

Product ticket of the second P8 repair loop, by `DELEGATION-PDC-LOOP2.md` (Commission table row "Second loop" and Amendment 1, which widens the write set by the graph indexer carrier file) and finding G-23 of `PDC_INDEPENDENT_AUDIT.md` section 13.5. At `76516c4dd` ingest promotes the publication of a source under a `RagSourceOperationKey` that carries `publication_scope_id` (the root document id), and the vector side resolves it with the same key, while the graph side (`intergrax/rag/graph/providers/inmemory_graph_store.py`, `intergrax/rag/graph/generation_visibility.py`) resolves it with a key without `publication_scope_id`, so the graph evidence of a re-ingested source is invisible; the graph evidence carries no root document id because `HeuristicGraphIndexer._publication_metadata` does not write it. The ticket makes the indexer stamp the root document id on versioned evidence and makes the in-memory store and the shared visibility helper use it as `publication_scope_id`; the visibility law and the fail-closed behaviour are unchanged, and negative tests prove that a wrong or missing scope id, another source or another tenant, namespace or workspace does not become visible. The Neo4j/Cypher store is not changed and stays a declared residual. After the implementation only the covering tests run (the new test file and the two reproduction tests named in G-23); no EBH-7 replay, no broad suite, no new pin. It is not a certification verdict.

## Plan digest

Plan digest: a6511d13fbed012f233ba9f39f8066bb9b31597a033546332c21696749d56fc1

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-041-P.md`](plans/PLAN_T-IGX-041-P.md).

## Test contract

```text
uv run pytest tests/unit/rag/graph/test_graph_publication_scope_visibility.py tests/integration/rag/test_graph_reingest_qualification.py::test_graphrag_source_reingest_replaces_vectors_and_graph_safely tests/integration/rag/test_same_source_reingest_serialization.py::test_stale_graph_topology_is_inactive_after_generation_takeover -q
uv run ruff check intergrax/rag/graph/generation_visibility.py intergrax/rag/graph/providers/inmemory_graph_store.py intergrax/rag/graph/indexer/heuristic_graph_indexer.py tests/unit/rag/graph/test_graph_publication_scope_visibility.py
uv run pyright intergrax/rag/graph/generation_visibility.py intergrax/rag/graph/providers/inmemory_graph_store.py intergrax/rag/graph/indexer/heuristic_graph_indexer.py tests/unit/rag/graph/test_graph_publication_scope_visibility.py
uv run python scripts/qualification/pdc_x_wbs.py --check
```

## Out of scope

- The Neo4j/Cypher store (declared residual with its reopen condition in the plan), the other graph indexers, the ingest and vector sides, `intergrax/distributed/`, any new coordinator API.
- Everything under `intergrax/applications/_shared/`, `intergrax/contracts/`, `scripts/governance/`, `.github/`, `pyproject.toml`, `uv.lock`, every existing test, the other findings of the global audit, the EBH-7 L3 replay and any new EBH-7 pin, the IIAC activation and rebind, FRZ promotion, OD signatures, certification.

## Lifecycle state

plan_pending
