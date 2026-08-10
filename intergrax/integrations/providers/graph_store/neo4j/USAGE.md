# Neo4J (neo4j)

Category: `graph_store`

**Operator guide:** [`docs/project/technical/guides/RAG_OPERATOR_GUIDE.md`](../../../../../docs/project/technical/guides/RAG_OPERATOR_GUIDE.md)

## Single public entrypoint

- **`Neo4jGraphStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `Neo4jGraphStoreIntegration`.
- Contract factory: `create_neo4j_graph_store_integration()`.

## Configuration

Environment prefix `INTERGRAX_NEO4J`:

| Variable | Purpose |
|---|---|
| `INTERGRAX_NEO4J_URL` | Bolt URI (default `bolt://localhost:7687`) |
| `INTERGRAX_NEO4J_USER` | Database user |
| `INTERGRAX_NEO4J_PASSWORD` | Database password |
| `INTERGRAX_NEO4J_TIMEOUT` | Optional connection timeout (seconds) |

Enable the RAG graph backend with `INTERGRAX_RAG_GRAPH_STORE=neo4j` (or
`RagProfile.graph_store_backend`).

## Health and connectivity

Opening the production path calls `driver.verify_connectivity()`. Runtime
health uses the integration client probe (`Neo4jGraphStore.health()`).

Repository qualification baseline (not a universal production topology):

- Server: `neo4j:5.26-community` (`infra/docker/neo4j/docker-compose.yml`)
- Driver: `neo4j==5.28.4` (`neo4j>=5.26,<6`)

## Qualification status

| Surface | Status |
|---|---|
| GraphRAG baseline | `LIVE_QUALIFIED_BASELINE` (RAG-LIVE-15C-R2) |
| Publication-generation fencing | `LIVE_QUALIFIED` (RAG-LIVE-15D-R2) |

Legacy Neo4j graph schema is not silently migrated or reinterpreted. Deployment,
recovery and incident procedures:
[`RAG_OPERATOR_GUIDE.md`](../../../../../docs/project/technical/guides/RAG_OPERATOR_GUIDE.md).
