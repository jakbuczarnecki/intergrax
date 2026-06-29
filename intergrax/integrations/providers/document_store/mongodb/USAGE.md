# Mongodb (mongodb)

Category: `document_store`

## Single public entrypoint

- **`MongodbDocumentStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `MongodbDocumentStoreIntegration`.
- Contract factory: `create_mongodb_document_store_integration()`.
