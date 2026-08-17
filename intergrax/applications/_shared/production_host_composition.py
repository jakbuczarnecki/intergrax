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


class StrictProductionAsgiPlaceholder:
    """Import-safe module-level ASGI symbol for STRICT apps without lifecycle activation.

    Production serving requires an explicit ``ProductionProcessComposition`` that has
    already passed through AP lifecycle activation. Use ``create_<app>_process_app`` or
    ``create_app(process_composition=...)`` from a process root instead of ``module:app``.
    """

    def __init__(self, *, application_package: str) -> None:
        self._application_package = application_package

    async def __call__(self, scope: object, receive: object, send: object) -> None:
        del scope, receive, send
        raise HarnessHostRegistryAuthorityError(
            f"{self._application_package}.host.main:app cannot serve STRICT production "
            "without an activated ProductionProcessComposition; use "
            f"create_app(process_composition=...) with uvicorn --factory or build the "
            f"host via create_{self._application_package.rsplit('.', 1)[-1]}_process_app "
            "from a process root that owns lifecycle activation"
        )


__all__ = [
    "StrictProductionAsgiPlaceholder",
    "bootstrap_production_registry_projection",
]
