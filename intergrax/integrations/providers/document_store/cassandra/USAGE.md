# Cassandra (cassandra)

Category: `document_store`

## Single public entrypoint

- **`CassandraDocumentStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `CassandraDocumentStoreIntegration`.
- Contract factory: `create_cassandra_document_store_integration()`.
