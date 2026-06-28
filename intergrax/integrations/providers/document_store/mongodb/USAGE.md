# Mongodb (mongodb)

Category: `document_store`

## Legacy facade

- `create_mongodb_integration()` remains backward-compatible.

## Contract-based integration

- `MongodbDocumentStoreIntegration` derives from the category-specific contract.
- Factory: `create_mongodb_document_store_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
