# Postgresql (postgresql)

Category: `relational_store`

## Single public entrypoint

- **`PostgresqlRelationalStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `PostgresqlRelationalStoreIntegration`.
- Contract factory: `create_postgresql_relational_store_integration()`.
