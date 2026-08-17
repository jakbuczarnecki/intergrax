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
from governed_contractor_application.host.environment_profile import (
    build_governed_contractor_environment_profile,
)
from governed_contractor_application.host.factory import create_governed_contractor_backend_app
from governed_contractor_application.manifest import build_governed_contractor_manifest

load_dotenv()

_STRICT_RUN_MESSAGE = (
    "Governed Contractor STRICT production requires explicit lifecycle deploy/activate "
    "before serving. Wire ReferenceProductionLifecycleLauncher with explicit lifecycle "
    "input, then create_governed_contractor_process_app(process_composition=...)."
)


def create_governed_contractor_process_app(
    *,
    process_composition: ProductionProcessComposition,
) -> FastAPI:
    """Build the Governed Contractor STRICT host from an activated process composition."""
    manifest = build_governed_contractor_manifest()
    env = manifest.environment or build_governed_contractor_environment_profile()
    return create_governed_contractor_backend_app(
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
    """Uvicorn factory entrypoint; requires an activated process composition."""
    if process_composition is None:
        raise HarnessHostRegistryAuthorityError(_STRICT_RUN_MESSAGE)
    return create_governed_contractor_process_app(process_composition=process_composition)


app = StrictProductionAsgiPlaceholder(application_package="governed_contractor_application")


def run() -> None:
    import sys

    print(_STRICT_RUN_MESSAGE, file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    run()
