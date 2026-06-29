# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.utils.lazy_export import export_from_bundle

__all__ = [
    "AWS_SECRETS_MANAGER_SECRETS_STORE_PROVIDER_ID",
    "AwsSecretsManagerSecretsStoreIntegration",
    "AwsSecretsManagerSecretsStoreIntegrationConfig",
    "AwsSecretsManagerSecretsStoreClient",
    "create_aws_secrets_manager_secrets_store",
    "create_aws_secrets_manager_secrets_store_integration",
    "register_aws_secrets_manager_integration",
]

_BUNDLE_EXPORTS = frozenset(
    {
        "create_aws_secrets_manager_secrets_store",
        "create_aws_secrets_manager_secrets_store_integration",
    }
)

_INTEGRATION_EXPORTS = frozenset(
    {
        "AWS_SECRETS_MANAGER_SECRETS_STORE_PROVIDER_ID",
        "AwsSecretsManagerSecretsStoreIntegration",
        "AwsSecretsManagerSecretsStoreIntegrationConfig",
        "AwsSecretsManagerSecretsStoreClient",
    }
)


_CONTRACT_INTEGRATION_EXPORTS = frozenset(
    {
        "AWS_SECRETS_MANAGER_SECRETS_STORE_PROVIDER_ID",
        "AwsSecretsManagerSecretsStoreIntegration",
        "AwsSecretsManagerSecretsStoreIntegrationConfig",
        "AwsSecretsManagerSecretsStoreClient",
    }
)

def __getattr__(name: str):
    if name == "register_aws_secrets_manager_integration":
        from intergrax.integrations.providers.secrets_store.aws_secrets_manager.register import register_aws_secrets_manager_integration

        return register_aws_secrets_manager_integration
    if name in _BUNDLE_EXPORTS:
        from intergrax.integrations.providers.secrets_store.aws_secrets_manager import bundle as _bundle

        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)
    if name in _INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.secrets_store.aws_secrets_manager import integration as _integration

        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)
    if name in _CONTRACT_INTEGRATION_EXPORTS:
        from intergrax.integrations.providers.secrets_store.aws_secrets_manager import integration as _integration

        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
