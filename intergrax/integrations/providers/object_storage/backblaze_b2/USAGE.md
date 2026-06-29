# Backblaze B2 (backblaze_b2)

Category: `object_storage`

## Single public entrypoint

- **`BackblazeB2ObjectStorageIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `BackblazeB2ObjectStorageIntegration`.
- Contract factory: `create_backblaze_b2_object_storage_integration()`.
