# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p7.factories import create_infisical_secrets_store

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.secrets_store.infisical.integration import (
    INFISICAL_SECRETS_STORE_PROVIDER_ID,
    InfisicalSecretsStoreIntegration,
    InfisicalSecretsStoreIntegrationConfig,
    InfisicalSecretsStoreClient,
)

__all__ = [
    "create_infisical_secrets_store",
    "create_infisical_secrets_store_integration",
]


def create_infisical_secrets_store_integration(
    *,
    client: InfisicalSecretsStoreClient | None = None,
    enabled: bool = False,
) -> InfisicalSecretsStoreIntegration:
    """
    Build a contract-based Infisical secrets store integration.

    The legacy facade (create_infisical_secrets_store) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Infisical secrets store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return InfisicalSecretsStoreIntegration.from_client(client, enabled=enabled)
    return InfisicalSecretsStoreIntegration.for_provider(
        provider_id=INFISICAL_SECRETS_STORE_PROVIDER_ID,
        display_name="Infisical",
        config=InfisicalSecretsStoreIntegrationConfig(enabled=enabled),
    )
