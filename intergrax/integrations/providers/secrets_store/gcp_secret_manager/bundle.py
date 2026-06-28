# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p5.factories import create_gcp_secret_manager_secrets_store

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.secrets_store.gcp_secret_manager.integration import (
    GCP_SECRET_MANAGER_SECRETS_STORE_PROVIDER_ID,
    GcpSecretManagerSecretsStoreIntegration,
    GcpSecretManagerSecretsStoreIntegrationConfig,
    GcpSecretManagerSecretsStoreClient,
)

__all__ = [
    "create_gcp_secret_manager_secrets_store",
    "create_gcp_secret_manager_secrets_store_integration",
]


def create_gcp_secret_manager_secrets_store_integration(
    *,
    client: GcpSecretManagerSecretsStoreClient | None = None,
    enabled: bool = False,
) -> GcpSecretManagerSecretsStoreIntegration:
    """
    Build a contract-based Gcp Secret Manager secrets store integration.

    The legacy facade (create_gcp_secret_manager_secrets_store) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Gcp Secret Manager secrets store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return GcpSecretManagerSecretsStoreIntegration.from_client(client, enabled=enabled)
    return GcpSecretManagerSecretsStoreIntegration.for_provider(
        provider_id=GCP_SECRET_MANAGER_SECRETS_STORE_PROVIDER_ID,
        display_name="Gcp Secret Manager",
        config=GcpSecretManagerSecretsStoreIntegrationConfig(enabled=enabled),
    )
