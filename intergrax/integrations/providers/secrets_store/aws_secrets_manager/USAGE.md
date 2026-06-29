# Aws Secrets Manager (aws_secrets_manager)

Category: `secrets_store`

## Single public entrypoint

- **`AwsSecretsManagerSecretsStoreIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `AwsSecretsManagerSecretsStoreIntegration`.
- Contract factory: `create_aws_secrets_manager_secrets_store_integration()`.
