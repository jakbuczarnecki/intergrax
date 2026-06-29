# Databricks (databricks)

Category: `relational_store`

## Single public entrypoint

- **`DatabricksRelationalStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `DatabricksRelationalStoreIntegration`.
- Contract factory: `create_databricks_relational_store_integration()`.
