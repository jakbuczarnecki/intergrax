# Dynamodb (dynamodb)

Category: `document_store`

## Single public entrypoint

- **`DynamodbDocumentStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `DynamodbDocumentStoreIntegration`.
- Contract factory: `create_dynamodb_document_store_integration()`.
