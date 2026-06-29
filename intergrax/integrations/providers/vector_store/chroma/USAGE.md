# Chroma (chroma)

Category: `vector_store`

## Single public entrypoint

- **`ChromaVectorStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `ChromaVectorStoreIntegration`.
- Contract factory: `create_chroma_vector_store_integration()`.
