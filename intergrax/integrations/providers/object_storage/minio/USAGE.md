# Minio (minio)

Category: `object_storage`

## Legacy facade

- `create_minio_object_storage()` remains backward-compatible.

## Contract-based integration

- `MinioObjectStorageIntegration` derives from the category-specific contract.
- Factory: `create_minio_object_storage_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
