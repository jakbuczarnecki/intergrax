# Neon (neon)

Category: `relational_store`

## Single public entrypoint

- **`NeonRelationalStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `NeonRelationalStoreIntegration`.
- Contract factory: `create_neon_relational_store_integration()`.
