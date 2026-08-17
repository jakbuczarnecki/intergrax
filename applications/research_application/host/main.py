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
from research_application.host.factory import create_research_backend_app
from research_application.host.wiring import build_research_environment_profile
from research_application.manifest import RESEARCH_APPLICATION_MANIFEST

load_dotenv()


def create_research_process_app(
    *,
    process_composition: ProductionProcessComposition,
) -> FastAPI:
    """Build the Research STRICT production host from an activated process composition."""
    manifest = RESEARCH_APPLICATION_MANIFEST
    env = manifest.environment or build_research_environment_profile()
    return create_research_backend_app(
        registry_projection=bootstrap_production_registry_projection(
            application_id=manifest.app_id,
            application_environment_id=env.profile_id,
            stores=process_composition.agent_platform_runtime.stores,
        ),
    )


def create_app(
    *,
    process_composition: ProductionProcessComposition | None = None,
) -> FastAPI:
    """Uvicorn factory entrypoint; owns one reference process composition when omitted."""
    composition = (
        process_composition
        if process_composition is not None
        else create_reference_production_process_composition()
    )
    return create_research_process_app(process_composition=composition)


app = StrictProductionAsgiPlaceholder(application_package="research_application")


def run() -> None:
    import uvicorn

    composition = create_reference_production_process_composition()
    host = os.environ.get("RESEARCH_BACKEND_HOST", "0.0.0.0")
    port = int(os.environ.get("RESEARCH_BACKEND_PORT", "8010"))
    reload = os.environ.get("RESEARCH_BACKEND_RELOAD", "").lower() in {"1", "true", "yes"}
    if reload:
        uvicorn.run(
            "research_application.host.main:create_app",
            factory=True,
            host=host,
            port=port,
            reload=True,
        )
        return
    uvicorn.run(
        create_research_process_app(process_composition=composition),
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    run()
