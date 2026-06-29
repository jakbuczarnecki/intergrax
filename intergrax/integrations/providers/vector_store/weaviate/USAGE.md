# Weaviate (weaviate)

Category: `vector_store`

## Single public entrypoint

- **`WeaviateVectorStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `WeaviateVectorStoreIntegration`.
- Contract factory: `create_weaviate_vector_store_integration()`.
