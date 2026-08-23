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
from research_application.host.factory import create_research_backend_app
from research_application.host.reference_lifecycle_input import build_research_reference_lifecycle_input
from research_application.host.wiring import build_research_environment_profile
from research_application.manifest import RESEARCH_APPLICATION_MANIFEST

load_dotenv()

_STRICT_RUN_MESSAGE = (
    "Research STRICT production requires explicit lifecycle deploy/activate before serving. "
    "Use run_reference_production() or wire ReferenceProductionLifecycleLauncher with "
    "build_research_reference_lifecycle_input(), then create_research_process_app("
    "process_composition=...) with the same composition."
)


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
    """Uvicorn factory entrypoint; requires an activated process composition."""
    if process_composition is None:
        raise HarnessHostRegistryAuthorityError(_STRICT_RUN_MESSAGE)
    return create_research_process_app(process_composition=process_composition)


app = StrictProductionAsgiPlaceholder(application_package="research_application")


def run_reference_production() -> None:
    """Explicit reference path: lifecycle deploy/activate then serve on one composition."""
    import uvicorn

    composition = create_reference_production_process_composition()
    projection_input, activation_request = build_research_reference_lifecycle_input()
    manifest = RESEARCH_APPLICATION_MANIFEST
    env = manifest.environment or build_research_environment_profile()
    launcher, governance = wire_governed_reference_production_launcher(composition, env)
    launcher.deploy_and_activate(
        projection_input,
        activation_request,
        principal=governance.principal,
        admission_mutation_id=reference_admission_mutation_id(
            projection_input.runtime_revision.runtime_revision_id
        ),
    )
    host = os.environ.get("RESEARCH_BACKEND_HOST", "0.0.0.0")
    port = int(os.environ.get("RESEARCH_BACKEND_PORT", "8010"))
    uvicorn.run(
        create_research_process_app(process_composition=composition),
        host=host,
        port=port,
        reload=False,
    )


def run() -> None:
    import sys

    if os.environ.get("RESEARCH_REFERENCE_PRODUCTION", "").strip().lower() in {"1", "true", "yes"}:
        run_reference_production()
        return
    print(_STRICT_RUN_MESSAGE, file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    run()
