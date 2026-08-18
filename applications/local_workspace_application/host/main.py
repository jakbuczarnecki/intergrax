# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI

from intergrax.applications._shared.harness_registry_authority import HarnessHostRegistryAuthorityError
from intergrax.applications._shared.production_host_composition import (
    StrictProductionAsgiPlaceholder,
    bootstrap_production_registry_projection,
)
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
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
    "Wire ReferenceProductionLifecycleLauncher with explicit lifecycle input, then "
    "create_local_workspace_process_app(process_composition=...) or use hosted foreground "
    "with an activated composition."
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


def run() -> None:
    import sys

    print(_STRICT_RUN_MESSAGE, file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    run()
