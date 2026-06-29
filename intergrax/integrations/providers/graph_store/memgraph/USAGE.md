# Memgraph (memgraph)

Category: `graph_store`

## Single public entrypoint

- **`MemgraphGraphStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `MemgraphGraphStoreIntegration`.
- Contract factory: `create_memgraph_graph_store_integration()`.
