# Inmemory (inmemory)

Category: `vector_store`

## Single public entrypoint

- **`InmemoryVectorStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `InmemoryVectorStoreIntegration`.
- Contract factory: `create_inmemory_vector_store_integration()`.
