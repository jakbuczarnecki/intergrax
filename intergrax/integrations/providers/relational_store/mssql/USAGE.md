# Mssql (mssql)

Category: `relational_store`

## Single public entrypoint

- **`MssqlRelationalStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `MssqlRelationalStoreIntegration`.
- Contract factory: `create_mssql_relational_store_integration()`.
