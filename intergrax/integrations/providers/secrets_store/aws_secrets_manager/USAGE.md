# Aws Secrets Manager (aws_secrets_manager)

Category: `secrets_store`

## Legacy facade

- `create_aws_secrets_manager_secrets_store()` remains backward-compatible.

## Contract-based integration

- `AwsSecretsManagerSecretsStoreIntegration` derives from the category-specific contract.
- Factory: `create_aws_secrets_manager_secrets_store_integration()`.
- Disabled by default (`enabled=False`).
- No vendor SDK or network I/O in the contract adapter.
- Injectable `{prefix}Client` required when `enabled=True`.

## Registry

- `register.py` remains legacy-compatible.
- Registry v2 / contract registry wiring deferred.
