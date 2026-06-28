# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p7.factories import create_neon_relational_store

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.relational_store.neon.integration import (
    NEON_RELATIONAL_STORE_PROVIDER_ID,
    NeonRelationalStoreIntegration,
    NeonRelationalStoreIntegrationConfig,
    NeonRelationalStoreClient,
)

__all__ = [
    "create_neon_relational_store",
    "create_neon_relational_store_integration",
]


def create_neon_relational_store_integration(
    *,
    client: NeonRelationalStoreClient | None = None,
    enabled: bool = False,
) -> NeonRelationalStoreIntegration:
    """
    Build a contract-based Neon relational store integration.

    The legacy facade (create_neon_relational_store) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Neon relational store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return NeonRelationalStoreIntegration.from_client(client, enabled=enabled)
    return NeonRelationalStoreIntegration.for_provider(
        provider_id=NEON_RELATIONAL_STORE_PROVIDER_ID,
        display_name="Neon",
        config=NeonRelationalStoreIntegrationConfig(enabled=enabled),
    )
