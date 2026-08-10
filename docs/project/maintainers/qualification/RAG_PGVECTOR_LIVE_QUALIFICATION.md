# RAG-LIVE-15A-R2 — PgVector Live Qualification

**Status:** `READY_FOR_REVIEW`
**Provider status:** `QUALIFIED_OFFLINE_CONTRACT + LIVE_QUALIFIED`
**Global status:** `PRODUCTION_QUALIFIED_WITH_LIMITATIONS`
**Qualification date:** 2026-08-10

This is an append-only live qualification record. It does not rewrite the
historical RAG-PROD-13 evidence.

## Repository and environment

- Start HEAD/origin: `b529c896` (`development`).
- Required ancestor: `cec5355e3fdb0110e6e07f9b1cf3b99c791eab89`.
- Qualification test commit: `c3fe1ecf`
  (`test(rag): live-qualify pgvector lifecycle`).
- Branch: `development`; no branch, worktree, detached HEAD, rebase, reset,
  stash, clean or history rewrite operation was used.
- Docker compose: `infra/docker/postgresql/docker-compose.yml`.
- Started service: `pgvector` only.
- Image: `pgvector/pgvector:0.8.0-pg16`.
- PostgreSQL: 16, repository-owned qualification database on `localhost:5433`.
- `vector` extension: installed and confirmed by provider health.
- Explicit dimension: `4`.
- Provider: production `create_pgvector_vector_store` path, native
  `PgVectorRagStore`.
- InMemory fallback: absent; no fallback path was used.
- Qualification credentials: repository-local compose credentials; not
  reproduced in this record.

## Exact commands and live runs

Environment was configured with the repository-local qualification DSN and
`INTERGRAX_PGVECTOR_DIMENSION=4`. Both runs used:

```text
uv run pytest tests/integration/rag/vectorstore/test_pgvector_live_qualification.py -q -s
```

### Live run 1 — PASS

- Run identifier: `3f8fb8cd26ac41a882ec1621b40bbd12`
- Scope matrix: PASS
- Identity, ownership, replacement, same-basename and delete gates: PASS
- Soak: 50 records, 5 query rounds, p95 `1.11 ms`, threshold `5000.00 ms`
- Cleanup: PASS

### Live run 2 — PASS

- Run identifier: `44251fc2e09445cf9554821404d47c00`
- Scope matrix: PASS
- Identity, ownership, replacement, same-basename and delete gates: PASS
- Soak: 50 records, 5 query rounds, p95 `1.07 ms`, threshold `5000.00 ms`
- Cleanup: PASS

Both runs opened the provider before executing lifecycle assertions. Runtime,
SQL, ownership and contract failures were treated as test failures, not
environment skips.

## Gate evidence

- **Isolation:** tenant A and tenant B; tenant A namespace A/B; workspace A/B;
  combined reverse scope; adversarial higher-similarity foreign records were
  never returned.
- **Identity:** `ADD IDs == QUERY.vector_id == OWNERSHIP IDs == DELETE INPUT
  IDs`; logical IDs remained canonical and database row IDs did not escape.
- **Source ownership:** exact `tenant_id`, `namespace`, `workspace_id` and
  `provenance.source_id`; enumeration was scoped SQL ownership, not
  similarity, top-k or basename lookup.
- **Same basename:** source A and source B used identical `same.txt` metadata;
  replacing and deleting A preserved B.
- **Replacement:** canonical `IngestPipeline` lifecycle passed v1, changed
  v2, fewer-chunk v3 and repeated v3; stale IDs were removed and B remained.
- **Scoped delete:** deletion in one exact scope preserved another namespace,
  workspace, combined scope and tenant.
- **Metadata filter:** non-routing `group` filter passed; routing remained
  provider/server authoritative and reserved routing metadata was rejected.
- **Failure semantics:** dimension mismatch, wrong scope, incompatible input,
  isolated SQL failure and ownership failure failed closed or raised
  explicitly.
- **Soak:** bounded qualification evidence only; it is not a universal capacity
  or production SLO claim.

## Offline regression

Exact command:

```text
uv run pytest tests/unit/rag/vectorstore/test_source_ownership_contract.py tests/unit/rag/vectorstore/test_real_backend_harness_skip_semantics.py tests/unit/rag/vectorstore/test_vectorstore_cross_tenant_isolation.py tests/integration/rag/test_source_scoped_reingest_qualification.py tests/integration/rag/test_namespace_workspace_isolation_qualification.py -q
```

Result: `35 passed, 1 skipped`. The single skip is the existing unit contract
case that explicitly requires the Docker-backed PgVector runtime.

Additional validation:

- Ruff changed Python: PASS.
- Compose config: PASS; Docker `pgvector` container healthy.
- `git diff --check`: PASS.
- Relative documentation links: PASS.
- Mojibake check: PASS.

## Cleanup and limitations

Both live runs removed all qualification-owned records and closed their
provider connections. Qualification state was generated under unique tenant
and source identifiers, leaving the database rerunnable.

This evidence is limited to the repository-owned Docker image, local
PostgreSQL configuration, explicit dimension and tested native provider path.
It makes no universal network, durability, capacity, latency or deployment
claim. Chroma live qualification and Neo4j live GraphRAG qualification remain
unresolved. `RAG-LIVE-15B` is not started.

## Decision

All mandatory R2 live gates and targeted offline regressions passed:

`PgVector = QUALIFIED_OFFLINE_CONTRACT + LIVE_QUALIFIED`

The global status remains:

`PRODUCTION_QUALIFIED_WITH_LIMITATIONS`
