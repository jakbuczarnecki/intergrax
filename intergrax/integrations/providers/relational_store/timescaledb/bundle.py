# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p5.factories import create_timescaledb_relational_store as _legacy_create_timescaledb_relational_store

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.relational_store.timescaledb.integration import (
    TIMESCALEDB_RELATIONAL_STORE_PROVIDER_ID,
    TimescaledbRelationalStoreIntegration,
    TimescaledbRelationalStoreIntegrationConfig,
    TimescaledbRelationalStoreClient,
)

__all__ = [
    "create_timescaledb_relational_store",
    "create_timescaledb_relational_store_integration",
]


def create_timescaledb_relational_store_integration(
    *,
    client: TimescaledbRelationalStoreClient | None = None,
    enabled: bool = False,
) -> TimescaledbRelationalStoreIntegration:
    """
    Build a contract-based Timescaledb relational store integration.

    The legacy facade (create_timescaledb_relational_store) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Timescaledb relational store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return TimescaledbRelationalStoreIntegration.from_client(client, enabled=enabled)
    return TimescaledbRelationalStoreIntegration.for_provider(
        provider_id=TIMESCALEDB_RELATIONAL_STORE_PROVIDER_ID,
        display_name="Timescaledb",
        config=TimescaledbRelationalStoreIntegrationConfig(enabled=enabled),
    )


def create_timescaledb_relational_store(**kwargs: object) -> TimescaledbRelationalStoreIntegration:
    """Compatibility shim — constructs TimescaledbRelationalStoreIntegration from legacy runtime."""
    runtime = _legacy_create_timescaledb_relational_store(**kwargs)
    if isinstance(runtime, TimescaledbRelationalStoreIntegration):
        return runtime
    return TimescaledbRelationalStoreIntegration.from_runtime(runtime)
