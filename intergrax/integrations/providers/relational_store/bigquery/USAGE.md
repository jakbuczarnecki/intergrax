# Bigquery (bigquery)

Category: `relational_store`

## Single public entrypoint

- **`BigqueryRelationalStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `BigqueryRelationalStoreIntegration`.
- Contract factory: `create_bigquery_relational_store_integration()`.
