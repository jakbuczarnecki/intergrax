# Duckdb (duckdb)

Category: `relational_store`

## Single public entrypoint

- **`DuckdbRelationalStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `DuckdbRelationalStoreIntegration`.
- Contract factory: `create_duckdb_relational_store_integration()`.
