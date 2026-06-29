# Motherduck (motherduck)

Category: `relational_store`

## Single public entrypoint

- **`MotherduckRelationalStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `MotherduckRelationalStoreIntegration`.
- Contract factory: `create_motherduck_relational_store_integration()`.
