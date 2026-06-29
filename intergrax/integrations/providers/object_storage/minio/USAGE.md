# Minio (minio)

Category: `object_storage`

## Single public entrypoint

- **`MinioObjectStorageIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `MinioObjectStorageIntegration`.
- Contract factory: `create_minio_object_storage_integration()`.
