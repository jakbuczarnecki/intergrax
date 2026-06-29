# Gcp Secret Manager (gcp_secret_manager)

Category: `secrets_store`

## Single public entrypoint

- **`GcpSecretManagerSecretsStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `GcpSecretManagerSecretsStoreIntegration`.
- Contract factory: `create_gcp_secret_manager_secrets_store_integration()`.
