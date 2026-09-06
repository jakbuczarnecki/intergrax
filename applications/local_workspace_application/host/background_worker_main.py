# © Artur Czarnecki. All rights reserved.

"""LKW Kafka background worker process entrypoint (LKW.4E)."""

from __future__ import annotations

import logging
import sys

from intergrax.applications._shared.production_host_composition import (
    bootstrap_production_registry_projection,
)
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
    create_reference_production_process_composition,
)
from intergrax.applications._shared.registry_projection import MaterializedRegistryProjection
from intergrax.applications._shared.registry_projection_input_bundle import (
    reference_admission_mutation_id,
)
from intergrax.applications._shared.reference_production_governance_wiring import (
    wire_governed_reference_production_launcher,
)
from intergrax.applications._shared.reference_runtime_materialization import (
    prepare_reference_runtime_materialization,
)
from local_workspace_application.host.background_worker_factory import (
    build_local_workspace_background_worker_wiring,
)
from local_workspace_application.host.environment_profile import (
    build_local_workspace_environment_profile,
)
from local_workspace_application.host.message_bus_wiring import local_workspace_message_bus_enabled
from local_workspace_application.host.reference_lifecycle_input import (
    build_local_workspace_reference_lifecycle_input,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST

logger = logging.getLogger(__name__)


def activate_local_workspace_reference_production_authority(
    settings: LocalWorkspaceBackendSettings | None = None,
) -> tuple[ProductionProcessComposition, MaterializedRegistryProjection]:
    """Deploy/activate reference production lifecycle and resolve registry projection."""
    resolved_settings = settings or LocalWorkspaceBackendSettings.from_env()
    composition = create_reference_production_process_composition()
    projection_input, activation_request = build_local_workspace_reference_lifecycle_input(
        resolved_settings,
    )
    env = build_local_workspace_environment_profile(resolved_settings)
    launcher, governance = wire_governed_reference_production_launcher(composition, env)
    prepare_reference_runtime_materialization(
        composition.agent_platform_runtime.stores,
        projection_input,
        artifact_locator=activation_request.artifact_locator,
    )
    launcher.deploy_and_activate(
        projection_input,
        activation_request,
        principal=governance.principal,
        admission_mutation_id=reference_admission_mutation_id(
            projection_input.runtime_revision.runtime_revision_id
        ),
    )
    registry_projection = bootstrap_production_registry_projection(
        application_id=LOCAL_WORKSPACE_APPLICATION_MANIFEST.app_id,
        application_environment_id=env.profile_id,
        stores=composition.agent_platform_runtime.stores,
    )
    return composition, registry_projection


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not local_workspace_message_bus_enabled():
        logger.error("LOCAL_WORKSPACE_ENABLE_MESSAGE_BUS must be true for the background worker")
        return 1

    settings = LocalWorkspaceBackendSettings.from_env()
    _, registry_projection = activate_local_workspace_reference_production_authority(
        settings,
    )
    wiring = build_local_workspace_background_worker_wiring(
        manifest=LOCAL_WORKSPACE_APPLICATION_MANIFEST,
        registry_projection=registry_projection,
        settings=settings,
    )
    logger.info("Starting LKW Kafka background worker for lkw.background_ingest.v1")
    wiring.worker.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
