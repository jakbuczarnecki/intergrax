# Sqlite (sqlite)

Category: `relational_store`

## Single public entrypoint

- **`SqliteRelationalStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `SqliteRelationalStoreIntegration`.
- Contract factory: `create_sqlite_relational_store_integration()`.
