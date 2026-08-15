# RAG-LIVE-15B-R2 — Chroma Live Qualification

**Status:** `READY_FOR_REVIEW`
**Provider status:** `QUALIFIED_OFFLINE_CONTRACT + LIVE_QUALIFIED`
**Global status:** `PRODUCTION_QUALIFIED_WITH_LIMITATIONS`
**Qualification date:** 2026-08-10

This is an append-only live qualification record. It does not rewrite the
historical RAG-PROD-13 evidence.

## Repository and environment

- Start HEAD/origin: `2f8ce0c6d9666a3edee6ca8e91ac7bfa4aa81e81`
  (`development`).
- Final shared HEAD/origin at validation: `0cf8560ed1bd15dc6b92c20046ff114aab2615e1`
  (`development`); this concurrent Slack test commit did not touch RAG files.
- Required accepted ancestor: `a09d75b6c5c0bda24527813809a90b9bf3cc62ba`.
- Qualification test: `tests/integration/rag/vectorstore/test_chroma_live_qualification.py`.
- Docker Compose: `infra/docker/chromadb/docker-compose.yml`.
- Client: `chromadb==1.4.1`.
- Server: `chromadb/chroma:1.4.1`, repository-owned Docker service.
- Container health: healthy; `/api/v2/heartbeat` succeeded.
- Runtime path: production Chroma opener, `HttpClient` over localhost HTTP,
  followed by heartbeat readiness and real server operations.
- Embedded `PersistentClient`/`Client` fallback path: not used.
- Qualification state: unique collection names and run-scoped tenant/source IDs.

## Exact commands and live runs

Both runs used the environment gate:

```text
INTERGRAX_RUN_CHROMA_LIVE=1 uv run pytest tests/integration/rag/vectorstore/test_chroma_live_qualification.py -q -s
```

### Live run 1 — PASS

- Run identifier: `e49d90098a7f44ff8fec3c91e3e75010`
- Scope matrix, adversarial isolation and server-side filter evidence: PASS
- Identity, ownership, replacement, same-basename and scoped delete: PASS
- Metadata and reconstruction: PASS
- Failure semantics: PASS
- Soak: 50 records, 5 query rounds, p95 `47.75 ms`, threshold `5000.00 ms`
- Cleanup: PASS

### Live run 2 — PASS

- Run identifier: `8336fad58be14d828422a6f1e207a748`
- Scope matrix, adversarial isolation and server-side filter evidence: PASS
- Identity, ownership, replacement, same-basename and scoped delete: PASS
- Metadata and reconstruction: PASS
- Failure semantics: PASS
- Soak: 50 records, 5 query rounds, p95 `51.28 ms`, threshold `5000.00 ms`
- Cleanup: PASS

Both runs opened the provider before lifecycle assertions. Only setup/open
connectivity failures can skip; provider, server, ownership and contract
failures after open are test failures.

## Gate evidence

- **Isolation:** tenant A/B; tenant A namespace A/B; workspace A/B; combined
  reverse scope; foreign records had stronger similarity and never escaped.
- **Server-side filter:** live spies captured Chroma `query(where=...)` with
  tenant, namespace and workspace predicates; metadata predicates were
  combined with scope predicates. Static source checks confirmed the Chroma
  query path sends `where` and ownership uses collection `get(where=...)`.
- **Authoritative routing:** `tenant_id`, `namespace` and `workspace_id` are
  provider-owned; reserved metadata keys are rejected by the canonical
  document contract and cannot override routing.
- **Identity:** `ADD IDs == QUERY.vector_id == OWNERSHIP IDs == DELETE input
  IDs`; logical IDs remained the only ID domain.
- **Source ownership:** exact scoped `list_source_record_ids` used Chroma
  `get(where=...)`, never similarity, top-k or basename lookup.
- **Same basename:** source A and source B used `same.txt` in different
  directories; replacing/deleting A preserved B.
- **Replacement:** canonical `IngestPipeline` passed version 1, changed
  version 2, fewer-chunk version 3 and repeated version 3. Stale IDs were
  removed and B remained intact.
- **Scoped delete:** one exact scope was deleted while other namespace,
  workspace, tenant and same-basename source records remained.
- **Metadata:** non-routing `group` filtering passed against the real server;
  routing fields remained authoritative.
- **Reconstruction:** content, identity, provenance, tenant, namespace,
  workspace and user metadata reconstructed correctly without provider
  metadata leakage.
- **Failure semantics:** wrong scope and malformed records failed closed;
  incompatible dimension failed explicitly; query, ownership and delete
  backend failures propagated; no post-open failure became a skip.
- **Soak:** bounded qualification evidence only; no universal capacity or SLO
  claim.

## Offline regression

Exact targeted command:

```text
uv run pytest tests/unit/integrations/providers/vector_store/test_chroma_ownership.py tests/unit/integrations/providers/vector_store/test_qdrant_chroma.py tests/unit/rag/vectorstore/test_source_ownership_contract.py tests/unit/rag/vectorstore/test_real_backend_harness_skip_semantics.py tests/unit/rag/vectorstore/test_vectorstore_cross_tenant_isolation.py tests/integration/rag/test_source_scoped_reingest_qualification.py tests/integration/rag/test_namespace_workspace_isolation_qualification.py -q
```

Result: `55 passed, 1 skipped`. The single skip is the existing PgVector case
that explicitly requires its Docker-backed runtime.

Additional validation:

- Ruff changed Python: PASS.
- Compose config: PASS.
- Relative documentation links: PASS.
- Mojibake check: PASS.
- `git diff --check`: PASS.

## Cleanup and limitations

Both runs removed all qualification-owned records and collections. The
Chroma client exposes no close method in the pinned client; collection deletion
and post-delete collection verification were performed for every opened
client.

This evidence is limited to the repository-owned Chroma 1.4.1 service,
HTTP/server topology, local environment and tested native provider path. It
makes no universal network, durability, capacity, latency or deployment claim.
Neo4j live GraphRAG qualification and publication-generation fencing remain
unresolved.

## Decision

All mandatory R2 live gates and targeted offline regressions passed:

`Chroma = QUALIFIED_OFFLINE_CONTRACT + LIVE_QUALIFIED`

The global status remains:

`PRODUCTION_QUALIFIED_WITH_LIMITATIONS`
