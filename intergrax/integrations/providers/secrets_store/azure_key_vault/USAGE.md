# Azure Key Vault (azure_key_vault)

Category: `secrets_store`

## Legacy facade

- `create_azure_key_vault_secrets_store()` remains backward-compatible.

## Contract-based integration

- `AzureKeyVaultSecretsStoreIntegration` derives from the category-specific contract.
- Factory: `create_azure_key_vault_secrets_store_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
