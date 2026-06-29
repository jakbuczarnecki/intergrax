# Milvus (milvus)

Category: `vector_store`

## Single public entrypoint

- **`MilvusVectorStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `MilvusVectorStoreIntegration`.
- Contract factory: `create_milvus_vector_store_integration()`.
