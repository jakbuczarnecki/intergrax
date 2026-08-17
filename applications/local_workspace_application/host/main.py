# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI

from intergrax.applications._shared.production_host_composition import (
    StrictProductionAsgiPlaceholder,
    bootstrap_production_registry_projection,
)
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
    create_reference_production_process_composition,
)
from local_workspace_application.host.environment_profile import (
    build_local_workspace_environment_profile,
)
from local_workspace_application.host.factory import create_local_workspace_backend_app
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST

load_dotenv()


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
    """Uvicorn factory entrypoint; owns one reference process composition when omitted."""
    composition = (
        process_composition
        if process_composition is not None
        else create_reference_production_process_composition()
    )
    return create_local_workspace_process_app(
        process_composition=composition,
        settings=settings,
    )


app = StrictProductionAsgiPlaceholder(application_package="local_workspace_application")


def run() -> None:
    import uvicorn

    composition = create_reference_production_process_composition()
    host = os.environ.get("LOCAL_WORKSPACE_BACKEND_HOST", "127.0.0.1")
    port = int(os.environ.get("LOCAL_WORKSPACE_BACKEND_PORT", "8020"))
    reload = os.environ.get("LOCAL_WORKSPACE_BACKEND_RELOAD", "").lower() in {"1", "true", "yes"}
    if reload:
        uvicorn.run(
            "local_workspace_application.host.main:create_app",
            factory=True,
            host=host,
            port=port,
            reload=True,
        )
        return
    uvicorn.run(
        create_local_workspace_process_app(process_composition=composition),
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    run()
