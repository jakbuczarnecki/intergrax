# Gcs (gcs)

Category: `object_storage`

## Single public entrypoint

- **`GcsObjectStorageIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `GcsObjectStorageIntegration`.
- Contract factory: `create_gcs_object_storage_integration()`.
