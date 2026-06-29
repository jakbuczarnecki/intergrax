# Filesystem (filesystem)

Category: `object_storage`

## Single public entrypoint

- **`FilesystemObjectStorageIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `FilesystemObjectStorageIntegration`.
- Contract factory: `create_filesystem_object_storage_integration()`.
