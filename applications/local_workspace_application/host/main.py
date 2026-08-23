# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI

from intergrax.applications._shared.harness_registry_authority import HarnessHostRegistryAuthorityError
from intergrax.applications._shared.production_host_composition import (
    StrictProductionAsgiPlaceholder,
    bootstrap_production_registry_projection,
)
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
    create_reference_production_process_composition,
)
from intergrax.applications._shared.registry_projection_input_bundle import (
    reference_admission_mutation_id,
)
from intergrax.applications._shared.reference_production_governance_wiring import (
    wire_governed_reference_production_launcher,
)
from local_workspace_application.host.reference_lifecycle_input import (
    build_local_workspace_reference_lifecycle_input,
)
from local_workspace_application.host.environment_profile import (
    build_local_workspace_environment_profile,
)
from local_workspace_application.host.factory import create_local_workspace_backend_app
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST

load_dotenv()

_STRICT_RUN_MESSAGE = (
    "Local Workspace STRICT production requires explicit lifecycle deploy/activate before serving. "
    "Use run_reference_production() or wire ReferenceProductionLifecycleLauncher with "
    "build_local_workspace_reference_lifecycle_input(), then "
    "create_local_workspace_process_app(process_composition=...) with the same composition."
)


def create_local_workspace_process_app(
    *,
    process_composition: ProductionProcessComposition,
    settings: LocalWorkspaceBackendSettings | None = None,
) -> FastAPI:
    """Build the Local Workspace STRICT production host from an activated composition."""
    resolved_settings = settings or LocalWorkspaceBackendSettings.from_env()
    env = build_local_workspace_environment_profile(resolved_settings)
    return create_local_workspace_backend_app(
        registry_projection=bootstrap_production_registry_projection(
            application_id=LOCAL_WORKSPACE_APPLICATION_MANIFEST.app_id,
            application_environment_id=env.profile_id,
            stores=process_composition.agent_platform_runtime.stores,
        ),
        settings=resolved_settings,
    )


def create_app(
    *,
    process_composition: ProductionProcessComposition | None = None,
    settings: LocalWorkspaceBackendSettings | None = None,
) -> FastAPI:
    """Uvicorn factory entrypoint; requires an activated process composition."""
    if process_composition is None:
        raise HarnessHostRegistryAuthorityError(_STRICT_RUN_MESSAGE)
    return create_local_workspace_process_app(
        process_composition=process_composition,
        settings=settings,
    )


app = StrictProductionAsgiPlaceholder(application_package="local_workspace_application")


def run_reference_production() -> None:
    """Explicit reference path: lifecycle deploy/activate then serve on one composition."""
    import uvicorn

    composition = create_reference_production_process_composition()
    projection_input, activation_request = build_local_workspace_reference_lifecycle_input()
    resolved_settings = LocalWorkspaceBackendSettings.from_env()
    env = build_local_workspace_environment_profile(resolved_settings)
    launcher, governance = wire_governed_reference_production_launcher(composition, env)
    launcher.deploy_and_activate(
        projection_input,
        activation_request,
        principal=governance.principal,
        admission_mutation_id=reference_admission_mutation_id(
            projection_input.runtime_revision.runtime_revision_id
        ),
    )
    host = os.environ.get("LOCAL_WORKSPACE_BACKEND_HOST", "0.0.0.0")
    port = int(os.environ.get("LOCAL_WORKSPACE_BACKEND_PORT", "8020"))
    uvicorn.run(
        create_local_workspace_process_app(process_composition=composition),
        host=host,
        port=port,
        reload=False,
    )


def run() -> None:
    import sys

    if os.environ.get("LOCAL_WORKSPACE_REFERENCE_PRODUCTION", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        run_reference_production()
        return
    print(_STRICT_RUN_MESSAGE, file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    run()
