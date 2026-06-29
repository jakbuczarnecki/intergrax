# Neo4J (neo4j)

Category: `graph_store`

## Single public entrypoint

- **`Neo4jGraphStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `Neo4jGraphStoreIntegration`.
- Contract factory: `create_neo4j_graph_store_integration()`.
