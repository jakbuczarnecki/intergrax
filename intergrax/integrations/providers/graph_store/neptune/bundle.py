# © Artur Czarnecki. All rights reserved.

from intergrax.integrations._shared.p5.factories import create_neptune_graph_store as _legacy_create_neptune_graph_store

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.graph_store.neptune.integration import (
    NEPTUNE_GRAPH_STORE_PROVIDER_ID,
    NeptuneGraphStoreClient,
    NeptuneGraphStoreIntegration,
    NeptuneGraphStoreIntegrationConfig,
)

__all__ = [
    "create_neptune_graph_store",
    "create_neptune_graph_store_integration",
]


def create_neptune_graph_store_integration(
    *,
    client: NeptuneGraphStoreClient | None = None,
    enabled: bool = False,
) -> NeptuneGraphStoreIntegration:
    """
    Build a contract-based Neptune graph store integration.

    The legacy facade (create_neptune_graph_store) is unchanged.
    Client must be injected explicitly when enabled=True; disabled by default.
    """
    if enabled and client is None:
        raise IntegrationConfigurationError(
            "Neptune graph store integration requires an injected client when enabled=True",
        )
    if client is not None:
        return NeptuneGraphStoreIntegration.from_client(client, enabled=enabled)
    return NeptuneGraphStoreIntegration.for_provider(
        provider_id=NEPTUNE_GRAPH_STORE_PROVIDER_ID,
        display_name="Neptune",
        config=NeptuneGraphStoreIntegrationConfig(enabled=enabled),
    )


def create_neptune_graph_store(**kwargs: object) -> NeptuneGraphStoreIntegration:
    """Compatibility shim — constructs NeptuneGraphStoreIntegration from legacy runtime."""
    runtime = _legacy_create_neptune_graph_store(**kwargs)
    if isinstance(runtime, NeptuneGraphStoreIntegration):
        return runtime
    return NeptuneGraphStoreIntegration.from_client(runtime)
