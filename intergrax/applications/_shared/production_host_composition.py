# © Artur Czarnecki. All rights reserved.

"""Production host registry projection composition (AC-3).

Host startup resolves the traffic-serving ``MaterializedRegistryProjection`` from the
same AP-9/AP-10 store instances populated by activation/projection. Manifest-only
registry assembly and startup-time reprojection are forbidden.
"""

from __future__ import annotations

from intergrax.agent_distribution.stores import ApplicationEnvironmentServingStore
from intergrax.applications._shared.active_registry_projection import resolve_active_registry_projection
from intergrax.applications._shared.harness_registry_authority import HarnessHostRegistryAuthorityError
from intergrax.applications._shared.production_agent_platform_runtime import AgentPlatformRuntimeStores
from intergrax.applications._shared.registry_projection import (
    MaterializedRegistryProjection,
    RuntimeRegistryProjectionStore,
)


def bootstrap_production_registry_projection(
    *,
    application_id: str,
    application_environment_id: str,
    stores: AgentPlatformRuntimeStores | None = None,
    serving_store: ApplicationEnvironmentServingStore | None = None,
    projection_store: RuntimeRegistryProjectionStore | None = None,
) -> MaterializedRegistryProjection:
    """Resolve production registry authority from canonical AP-9/AP-10 stores."""
    if stores is not None:
        serving_store = stores.serving_store
        projection_store = stores.registry_projection_store
    if serving_store is None or projection_store is None:
        raise HarnessHostRegistryAuthorityError(
            "production host composition requires ApplicationEnvironmentServingStore and "
            "RuntimeRegistryProjectionStore; manifest-only registry assembly is forbidden"
        )
    return resolve_active_registry_projection(
        application_id=application_id,
        application_environment_id=application_environment_id,
        serving_store=serving_store,
        projection_store=projection_store,
    )


__all__ = ["bootstrap_production_registry_projection"]
