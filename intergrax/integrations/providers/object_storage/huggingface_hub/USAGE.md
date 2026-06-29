# Huggingface Hub (huggingface_hub)

Category: `object_storage`

## Single public entrypoint

- **`HuggingfaceHubObjectStorageIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `HuggingfaceHubObjectStorageIntegration`.
- Contract factory: `create_huggingface_hub_object_storage_integration()`.
