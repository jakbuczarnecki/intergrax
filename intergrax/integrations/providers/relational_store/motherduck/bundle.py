# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations._shared.p8.factories import create_motherduck_relational_store as _legacy_create_motherduck_relational_store

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.relational_store.motherduck.integration import (
    MOTHERDUCK_RELATIONAL_STORE_PROVIDER_ID,
    MotherduckRelationalStoreIntegration,
    MotherduckRelationalStoreIntegrationConfig,
    MotherduckRelationalStoreClient,
)

__all__ = [
    "create_motherduck_relational_store",
    "create_motherduck_relational_store_integration",
]


def create_motherduck_relational_store_integration(
    *,
    client: MotherduckRelationalStoreClient | None = None,
    enabled: bool = False,
) -> MotherduckRelationalStoreIntegration:
    """
    Build a contract-based Motherduck relational store integration.

    The legacy facade (create_motherduck_relational_store) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Motherduck relational store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return MotherduckRelationalStoreIntegration.from_client(client, enabled=enabled)
    return MotherduckRelationalStoreIntegration.for_provider(
        provider_id=MOTHERDUCK_RELATIONAL_STORE_PROVIDER_ID,
        display_name="Motherduck",
        config=MotherduckRelationalStoreIntegrationConfig(enabled=enabled),
    )


def create_motherduck_relational_store(**kwargs: object) -> MotherduckRelationalStoreIntegration:
    """Compatibility shim — constructs MotherduckRelationalStoreIntegration from legacy runtime."""
    runtime = _legacy_create_motherduck_relational_store(**kwargs)
    if isinstance(runtime, MotherduckRelationalStoreIntegration):
        return runtime
    return MotherduckRelationalStoreIntegration.from_client(runtime)
