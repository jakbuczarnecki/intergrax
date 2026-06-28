# S3 (s3)

Category: `object_storage`

## Legacy facade

- `create_s3_integration()` remains backward-compatible.

## Contract-based integration

- `S3ObjectStorageIntegration` derives from the category-specific contract.
- Factory: `create_s3_object_storage_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
