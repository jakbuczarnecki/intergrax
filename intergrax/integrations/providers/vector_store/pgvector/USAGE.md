# Pgvector (pgvector)

Category: `vector_store`

**Operator guide:** [`docs/project/technical/guides/RAG_OPERATOR_GUIDE.md`](../../../../../docs/project/technical/guides/RAG_OPERATOR_GUIDE.md)

## Single public entrypoint

- **`PgvectorVectorStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `PgvectorVectorStoreIntegration`.
- Contract factory: `create_pgvector_vector_store_integration()`.

## Native runtime

The production PgVector path is fail-closed. It requires:

- `INTERGRAX_PGVECTOR_DSN` (or `INTERGRAX_PGVECTOR_CONNECTION_STRING`);
- `INTERGRAX_PGVECTOR_DIMENSION`, matching the embedding provider;
- the `integrations-pgvector` extra (`psycopg[binary]` and `pgvector`);
- PostgreSQL with the `vector` extension.

The provider creates/verifies a native `vector(N)` column, executes cosine
ranking in PostgreSQL, and applies tenant/namespace/workspace predicates in
SQL. It never falls back to `InMemoryVectorStore`; use the explicit `inmemory`
provider for local or unit-test memory storage.

Health probe: `SELECT 1` plus `pg_extension` check for `vector` (see
`PgVectorRagStore.health()`).

The repository-owned qualification service is available independently with:

```text
docker compose -f infra/docker/postgresql/docker-compose.yml up pgvector
```

Its DSN is:
`postgresql://intergrax_pgvector:intergrax_pgvector@localhost:5433/intergrax_pgvector`.

Existing JSONB `intergrax_pgvector` tables are treated as incompatible legacy
schemas. They are not migrated or dropped automatically.

Provider status: `QUALIFIED_OFFLINE_CONTRACT + LIVE_QUALIFIED` (RAG-LIVE-15A-R2).
Live evidence is environment-specific; see
[`RAG_PGVECTOR_LIVE_QUALIFICATION.md`](../../../../../docs/project/maintainers/qualification/RAG_PGVECTOR_LIVE_QUALIFICATION.md).
