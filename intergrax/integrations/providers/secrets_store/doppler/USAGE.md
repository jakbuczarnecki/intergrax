# Doppler (doppler)

Category: `secrets_store`

## Single public entrypoint

- **`DopplerSecretsStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `DopplerSecretsStoreIntegration`.
- Contract factory: `create_doppler_secrets_store_integration()`.
