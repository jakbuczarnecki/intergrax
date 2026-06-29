# Falkordb (falkordb)

Category: `graph_store`

## Single public entrypoint

- **`FalkordbGraphStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `FalkordbGraphStoreIntegration`.
- Contract factory: `create_falkordb_graph_store_integration()`.
