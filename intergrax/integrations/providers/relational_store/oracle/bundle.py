# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p2.factories import create_oracle_relational_store as _legacy_create_oracle_relational_store

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.relational_store.oracle.integration import (
    ORACLE_RELATIONAL_STORE_PROVIDER_ID,
    OracleRelationalStoreIntegration,
    OracleRelationalStoreIntegrationConfig,
    OracleRelationalStoreClient,
)

__all__ = [
    "create_oracle_relational_store",
    "create_oracle_relational_store_integration",
]


def create_oracle_relational_store_integration(
    *,
    client: OracleRelationalStoreClient | None = None,
    enabled: bool = False,
) -> OracleRelationalStoreIntegration:
    """
    Build a contract-based Oracle relational store integration.

    The legacy facade (create_oracle_relational_store) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Oracle relational store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return OracleRelationalStoreIntegration.from_client(client, enabled=enabled)
    return OracleRelationalStoreIntegration.for_provider(
        provider_id=ORACLE_RELATIONAL_STORE_PROVIDER_ID,
        display_name="Oracle",
        config=OracleRelationalStoreIntegrationConfig(enabled=enabled),
    )


def create_oracle_relational_store(**kwargs: object) -> OracleRelationalStoreIntegration:
    """Compatibility shim — constructs OracleRelationalStoreIntegration from legacy runtime."""
    runtime = _legacy_create_oracle_relational_store(**kwargs)
    if isinstance(runtime, OracleRelationalStoreIntegration):
        return runtime
    return OracleRelationalStoreIntegration.from_runtime(runtime)
