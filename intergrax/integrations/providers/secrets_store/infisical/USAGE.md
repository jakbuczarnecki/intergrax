# Infisical (infisical)

Category: `secrets_store`

## Single public entrypoint

- **`InfisicalSecretsStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `InfisicalSecretsStoreIntegration`.
- Contract factory: `create_infisical_secrets_store_integration()`.
