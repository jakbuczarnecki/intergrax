# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p5.factories import create_aws_secrets_manager_secrets_store as _legacy_create_aws_secrets_manager_secrets_store

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.secrets_store.aws_secrets_manager.integration import (
    AWS_SECRETS_MANAGER_SECRETS_STORE_PROVIDER_ID,
    AwsSecretsManagerSecretsStoreIntegration,
    AwsSecretsManagerSecretsStoreIntegrationConfig,
    AwsSecretsManagerSecretsStoreClient,
)

__all__ = [
    "create_aws_secrets_manager_secrets_store",
    "create_aws_secrets_manager_secrets_store_integration",
]


def create_aws_secrets_manager_secrets_store_integration(
    *,
    client: AwsSecretsManagerSecretsStoreClient | None = None,
    enabled: bool = False,
) -> AwsSecretsManagerSecretsStoreIntegration:
    """
    Build a contract-based Aws Secrets Manager secrets store integration.

    The legacy facade (create_aws_secrets_manager_secrets_store) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Aws Secrets Manager secrets store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return AwsSecretsManagerSecretsStoreIntegration.from_client(client, enabled=enabled)
    return AwsSecretsManagerSecretsStoreIntegration.for_provider(
        provider_id=AWS_SECRETS_MANAGER_SECRETS_STORE_PROVIDER_ID,
        display_name="Aws Secrets Manager",
        config=AwsSecretsManagerSecretsStoreIntegrationConfig(enabled=enabled),
    )


def create_aws_secrets_manager_secrets_store(**kwargs: object) -> AwsSecretsManagerSecretsStoreIntegration:
    """Compatibility shim — constructs AwsSecretsManagerSecretsStoreIntegration from legacy runtime."""
    runtime = _legacy_create_aws_secrets_manager_secrets_store(**kwargs)
    if isinstance(runtime, AwsSecretsManagerSecretsStoreIntegration):
        return runtime
    return AwsSecretsManagerSecretsStoreIntegration.from_runtime(runtime)
