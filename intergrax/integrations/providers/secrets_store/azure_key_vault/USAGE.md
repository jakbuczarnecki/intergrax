# Azure Key Vault (azure_key_vault)

Category: `secrets_store`

## Single public entrypoint

- **`AzureKeyVaultSecretsStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `AzureKeyVaultSecretsStoreIntegration`.
- Contract factory: `create_azure_key_vault_secrets_store_integration()`.
