# Typesense (typesense)

Category: `vector_store`

## Single public entrypoint

- **`TypesenseVectorStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `TypesenseVectorStoreIntegration`.
- Contract factory: `create_typesense_vector_store_integration()`.
