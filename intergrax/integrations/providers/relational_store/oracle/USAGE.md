# Oracle (oracle)

Category: `relational_store`

## Single public entrypoint

- **`OracleRelationalStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `OracleRelationalStoreIntegration`.
- Contract factory: `create_oracle_relational_store_integration()`.
