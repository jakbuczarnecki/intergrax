# Pgvector (pgvector)

Category: `vector_store`

## Single public entrypoint

- **`PgvectorVectorStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `PgvectorVectorStoreIntegration`.
- Contract factory: `create_pgvector_vector_store_integration()`.
