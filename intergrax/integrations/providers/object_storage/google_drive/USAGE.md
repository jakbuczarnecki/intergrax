# Google Drive (google_drive)

Category: `object_storage`

## Single public entrypoint

- **`GoogleDriveObjectStorageIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `GoogleDriveObjectStorageIntegration`.
- Contract factory: `create_google_drive_object_storage_integration()`.
