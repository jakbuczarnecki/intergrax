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
from dispute_sim_application.host.environment_profile import build_dispute_sim_environment_profile
from dispute_sim_application.host.factory import create_dispute_sim_backend_app
from dispute_sim_application.manifest import build_dispute_sim_manifest

load_dotenv()


def create_dispute_sim_process_app(
    *,
    process_composition: ProductionProcessComposition,
) -> FastAPI:
    """Build the Dispute Sim STRICT production host from an activated process composition."""
    manifest = build_dispute_sim_manifest()
    env = manifest.environment or build_dispute_sim_environment_profile()
    return create_dispute_sim_backend_app(
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
    return create_dispute_sim_process_app(process_composition=composition)


app = StrictProductionAsgiPlaceholder(application_package="dispute_sim_application")


def run() -> None:
    import uvicorn

    composition = create_reference_production_process_composition()
    host = os.environ.get("DISPUTE_SIM_BACKEND_HOST", "127.0.0.1")
    port = int(os.environ.get("DISPUTE_SIM_BACKEND_PORT", "8020"))
    reload = os.environ.get("DISPUTE_SIM_BACKEND_RELOAD", "").lower() in {"1", "true", "yes"}
    if reload:
        uvicorn.run(
            "dispute_sim_application.host.main:create_app",
            factory=True,
            host=host,
            port=port,
            reload=True,
        )
        return
    uvicorn.run(
        create_dispute_sim_process_app(process_composition=composition),
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    run()
