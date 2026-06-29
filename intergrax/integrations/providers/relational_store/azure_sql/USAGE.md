# Azure Sql (azure_sql)

Category: `relational_store`

## Single public entrypoint

- **`AzureSqlRelationalStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `AzureSqlRelationalStoreIntegration`.
- Contract factory: `create_azure_sql_relational_store_integration()`.
