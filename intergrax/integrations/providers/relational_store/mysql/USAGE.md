# Mysql (mysql)

Category: `relational_store`

## Single public entrypoint

- **`MysqlRelationalStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `MysqlRelationalStoreIntegration`.
- Contract factory: `create_mysql_relational_store_integration()`.
