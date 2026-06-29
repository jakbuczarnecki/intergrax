# Cloud Sql (cloud_sql)

Category: `relational_store`

## Single public entrypoint

- **`CloudSqlRelationalStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `CloudSqlRelationalStoreIntegration`.
- Contract factory: `create_cloud_sql_relational_store_integration()`.
