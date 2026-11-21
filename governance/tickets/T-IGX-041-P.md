# T-IGX-041-P — PDC-X batch B-P8-R4: repair of finding G-23 (GraphRAG evidence of a re-ingested source resolves its active publication generation under the key scoped by the root document id)

## Scope

Product ticket of the second P8 repair loop, by `DELEGATION-PDC-LOOP2.md` (Commission table row "Second loop" and Amendment 1, which widens the write set by the graph indexer carrier file) and finding G-23 of `PDC_INDEPENDENT_AUDIT.md` section 13.5. At `76516c4dd` ingest promotes the publication of a source under a `RagSourceOperationKey` that carries `publication_scope_id` (the root document id), and the vector side resolves it with the same key, while the graph side (`intergrax/rag/graph/providers/inmemory_graph_store.py`, `intergrax/rag/graph/generation_visibility.py`) resolves it with a key without `publication_scope_id`, so the graph evidence of a re-ingested source is invisible; the graph evidence carries no root document id because `HeuristicGraphIndexer._publication_metadata` does not write it. The ticket makes the indexer stamp the root document id on versioned evidence and makes the in-memory store and the shared visibility helper use it as `publication_scope_id`; the visibility law and the fail-closed behaviour are unchanged, and negative tests prove that a wrong or missing scope id, another source or another tenant, namespace or workspace does not become visible. The Neo4j/Cypher store is not changed and stays a declared residual. After the implementation only the covering tests run (the new test file and the two reproduction tests named in G-23); no EBH-7 replay, no broad suite, no new pin. It is not a certification verdict.

## Plan digest

Plan digest: a6511d13fbed012f233ba9f39f8066bb9b31597a033546332c21696749d56fc1

Canonical plan: [`governance/tickets/plans/PLAN_T-IGX-041-P.md`](plans/PLAN_T-IGX-041-P.md).

## Approved

Approved: operator-delegate @ 2026-11-16T06:55:00+01:00
Delegation: docs/project/maintainers/qualification/platform_domain_completeness/operator_decisions/DELEGATION-PDC-LOOP2.md@04a741d47aa10c69cb19cf344338fb2fff09d918

Delegate checks (scope of `DELEGATION-PDC-LOOP2.md` including Amendment 1; the record is on `origin/development`, merge commit `04a741d47`, file last changed by `1e68151af`; plan read at commit `951f03f5e` of the branch `pdc-x-plan-041pq`, whose ancestor is `76516c4dd`):

- The plan digest recomputed with `scripts/governance/plan_digest.py` equals the digest above (`a6511d13fbed012f233ba9f39f8066bb9b31597a033546332c21696749d56fc1`).
- The write set of the plan (4 entries) equals the `writes` tuple of batch `B-P8-R4` in `pdc_x_wbs.py` (compared as lists, equal; `pdc_x_wbs.py --check` OK, so the generated WBS and DAG match). It is exactly the three files of Amendment 1 (`generation_visibility.py`, `inmemory_graph_store.py`, `heuristic_graph_indexer.py`) plus the new test file; nothing under `intergrax/applications/_shared/`, `intergrax/contracts/`, `scripts/governance/` or `.github/`. The branch diff against `development` touches only the two tickets, the two plans, the WBS, the DAG and `pdc_x_wbs.py`.
- Factual sentences and file:line read against the code at the branch (about 60 references): `ingest_pipeline.py:345-352, 422-429, 503-513, 511, 527`; `source_operation.py:47, 56-60, 70-71, 75-76, 97-123, 138-140, 175-182`; `vectorstore_manager.py:345-360` (root id at `:353`, passed at `:359`); `publication_visibility.py:14-31`; `generation_visibility.py:48-68, 65-68, 71-94, 82-87, 97-118, 110`; `inmemory_graph_store.py:143-178, 329-356, 343-348` (extra metadata keys survive `_metadata`, so node evidence will carry the new key); `heuristic_graph_indexer.py:51, 60, 72, 77-89, 85-88` (only tenant, namespace, workspace, source id and generation are written today); `cypher_rag_graph_store.py:145-160, 181, 196, 236, 296`; `chunk_document.py:56-58, 165`; `compat/langchain/documents.py:274`; `llm_graph_indexer.py:70-71, 79-85`; `community_report_graph_indexer.py:54-64`; the test file lines `test_graph_reingest_qualification.py:129, 172, 230`, `test_same_source_reingest_serialization.py:174, 438`, `fixtures/__init__.py:6` (the `metadata` parameter exists), `test_graph_generation_fencing.py:28-30, 33-39, 58-72`; `changed_since_impact_map.v1.json:171-176`, `impact_map.py:15-18`; `RAG_RED_CLASSIFICATION.md:56`; the audit row G-23 (section 13.5, `assert set()`). The new test file does not exist yet, as the plan says. All true.
- Probe by the delegate (temporary edit of the three product files in the delegate worktree, reverted, nothing committed): with the carrier, the scope helper and the scoped key in `_evidence`, `test_graphrag_source_reingest_replaces_vectors_and_graph_safely` and `test_stale_graph_topology_is_inactive_after_generation_takeover` both pass (2 passed). This supports the plan's expectation; the implementer still states the observed result.
- No gate is weakened or deleted: the visibility law, the fail-closed behaviour and the existing key fields stay, one narrowing field is added; unscoped callers keep the unscoped key; every new test is a strengthening and the plan keeps the Neo4j store unchanged as the declared residual R1 with its reopen condition. No registry, scheduler, engine, trust, policy or audit authority is added; the plan states it is not a certification verdict and states the targeted-test condition (new test file and the two reproduction tests only; no EBH-7 replay, no broad suite, no new pin).
- The implementer session and this delegate session are different sessions; the delegate edited nothing except this approval block and the lifecycle state. Not checked by the delegate: the behaviour of the new tests (not written yet), the Neo4j path, and the root-id semantics beyond the code read above.
- Merge gate (as the record states): authorizations 1 and 3 of `DELEGATION-PDC-STANDING.md`, green CI with the known red classes only (IIAC activation staleness, EBH-7 identity), head equal to the verified SHA. Write-set amendments are not delegated.

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

## Implementation checkpoint

- Tests first (`d6b8d0593`): `tests/unit/rag/graph/test_graph_publication_scope_visibility.py`, 23 cases. On the unchanged code 16 failed (the positives, the re-ingest case, the other-source and the other-tenant/namespace/workspace cases whose control assertion needs the main store to see its own evidence, the shared-helper cases that call the new keyword, the carrier cases) and 7 passed (wrong, missing and blank scope id, the unscoped compatibility case, the unversioned-carrier case, the scope-id-from-metadata negatives): log `.tmp/proof/pdc-x-041p/red.log` (not committed).
- Repair: `generation_visibility.py` (constant `GRAPH_PUBLICATION_SCOPE_METADATA_KEY`, helper `publication_scope_from_metadata`, keyword `publication_scope_ids` on `resolve_scope_active_generations` and `visibility_query_params`), `inmemory_graph_store.py` (one keyword in `_evidence`, one import), `heuristic_graph_indexer.py` (one assignment in `_publication_metadata`, one import). The Neo4j/Cypher store and every other file are untouched (residual R1 stays).
- Green after, one pytest command (the Test contract: the new file and the two reproduction node ids): 25 passed (`.tmp/proof/pdc-x-041p/green.log`). Both reproduction tests (`test_graphrag_source_reingest_replaces_vectors_and_graph_safely`, `test_stale_graph_topology_is_inactive_after_generation_takeover`) pass.
- ruff check and ruff format --check on the four files of the Write set: clean. pyright: 7 errors, all in lines of `inmemory_graph_store.py` this change does not touch (`_scope_for_metadata`, `_remove_evidence`), present at `origin/development`; none in the changed lines or in the other three files. `pdc_x_wbs.py --check` OK.
- No other test was run: no existing graph unit test, no integration directory, no EBH-7 replay, no broad suite, no new pin.
- Deviation from the signed plan (no change of scope, design or write set): the three product files did not pass the pre-commit and push ruff gates at `origin/development` (25 findings in `inmemory_graph_store.py`, format differences in all three; the plan assumed clean files). Commit `f044fcdfc` applies `ruff check --fix` and `ruff format` to exactly those three files (annotations, import order and line wrapping only, no behavior change) before the repair commit; hooks were not skipped. The test file was adjusted once after the red run (an exact-id visibility helper in the other-source negative, because unlinked nodes are visible by the store's existing rule).

## Lifecycle state

plan_approved
