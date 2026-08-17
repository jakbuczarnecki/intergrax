# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ASGI entrypoint for the Legal backend host."""

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
from legal_application.host.factory import create_legal_backend_app
from legal_application.host.settings import LegalBackendSettings
from legal_application.host.wiring import build_legal_environment_profile, build_legal_manifest

# Load `.env` when present (does not override existing process env).
load_dotenv()

_STRICT_RUN_MESSAGE = (
    "Legal STRICT production requires explicit lifecycle deploy/activate before serving. "
    "Wire ReferenceProductionLifecycleLauncher with explicit RegistryProjectionInputBundle "
    "and ActivateRuntimeRevisionRequest, then create_legal_process_app(process_composition=...)."
)


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
    settings: LegalBackendSettings | None = None,
) -> FastAPI:
    """Uvicorn factory entrypoint; requires an activated process composition."""
    if process_composition is None:
        raise HarnessHostRegistryAuthorityError(_STRICT_RUN_MESSAGE)
    return create_legal_process_app(
        process_composition=process_composition,
        settings=settings,
    )


app = StrictProductionAsgiPlaceholder(application_package="legal_application")


def run() -> None:
    """CLI entry using uvicorn when installed."""
    import sys

    print(_STRICT_RUN_MESSAGE, file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    run()
