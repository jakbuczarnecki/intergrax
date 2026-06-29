# S3 (s3)

Category: `object_storage`

## Single public entrypoint

- **`S3ObjectStorageIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `S3ObjectStorageIntegration`.
- Contract factory: `create_s3_object_storage_integration()`.
