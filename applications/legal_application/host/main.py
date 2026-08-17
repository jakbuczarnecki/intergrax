# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ASGI entrypoint for the Legal backend host."""

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
from legal_application.host.factory import create_legal_backend_app
from legal_application.host.settings import LegalBackendSettings
from legal_application.host.wiring import build_legal_environment_profile, build_legal_manifest

# Load `.env` when present (does not override existing process env).
load_dotenv()


def create_legal_process_app(
    *,
    process_composition: ProductionProcessComposition,
    settings: LegalBackendSettings | None = None,
) -> FastAPI:
    """Build the Legal STRICT production host from an activated process composition."""
    resolved_settings = settings or LegalBackendSettings.from_env()
    manifest = build_legal_manifest(resolved_settings)
    env = manifest.environment or build_legal_environment_profile(resolved_settings)
    return create_legal_backend_app(
        registry_projection=bootstrap_production_registry_projection(
            application_id=manifest.app_id,
            application_environment_id=env.profile_id,
            stores=process_composition.agent_platform_runtime.stores,
        ),
        settings=resolved_settings,
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
    return create_legal_process_app(process_composition=composition)


app = StrictProductionAsgiPlaceholder(application_package="legal_application")


def run() -> None:
    """CLI entry using uvicorn when installed."""
    import uvicorn

    composition = create_reference_production_process_composition()
    host = os.environ.get("LEGAL_BACKEND_HOST", "0.0.0.0")
    port = int(os.environ.get("LEGAL_BACKEND_PORT", "8000"))
    reload = os.environ.get("LEGAL_BACKEND_RELOAD", "").strip().lower() in {"1", "true", "yes"}
    if reload:
        uvicorn.run(
            "legal_application.host.main:create_app",
            factory=True,
            host=host,
            port=port,
            reload=True,
        )
        return
    uvicorn.run(
        create_legal_process_app(process_composition=composition),
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    run()
