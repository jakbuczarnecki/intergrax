# Timescaledb (timescaledb)

Category: `relational_store`

## Single public entrypoint

- **`TimescaledbRelationalStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `TimescaledbRelationalStoreIntegration`.
- Contract factory: `create_timescaledb_relational_store_integration()`.
