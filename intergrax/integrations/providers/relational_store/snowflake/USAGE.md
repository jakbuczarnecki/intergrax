# Snowflake (snowflake)

Category: `relational_store`

## Single public entrypoint

- **`SnowflakeRelationalStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `SnowflakeRelationalStoreIntegration`.
- Contract factory: `create_snowflake_relational_store_integration()`.
