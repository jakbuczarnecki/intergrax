# Lancedb (lancedb)

Category: `vector_store`

## Single public entrypoint

- **`LancedbVectorStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `LancedbVectorStoreIntegration`.
- Contract factory: `create_lancedb_vector_store_integration()`.
