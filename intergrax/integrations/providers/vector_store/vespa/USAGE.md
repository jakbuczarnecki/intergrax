# Vespa (vespa)

Category: `vector_store`

## Single public entrypoint

- **`VespaVectorStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `VespaVectorStoreIntegration`.
- Contract factory: `create_vespa_vector_store_integration()`.
