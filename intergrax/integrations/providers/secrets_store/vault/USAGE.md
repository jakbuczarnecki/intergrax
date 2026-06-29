# Vault (vault)

Category: `secrets_store`

## Single public entrypoint

- **`VaultSecretsStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `VaultSecretsStoreIntegration`.
- Contract factory: `create_vault_secrets_store_integration()`.
