# Cloudflare R2 (cloudflare_r2)

Category: `object_storage`

## Single public entrypoint

- **`CloudflareR2ObjectStorageIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `CloudflareR2ObjectStorageIntegration`.
- Contract factory: `create_cloudflare_r2_object_storage_integration()`.
