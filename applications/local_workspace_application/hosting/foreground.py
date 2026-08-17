# © Artur Czarnecki. All rights reserved.

"""LKW foreground hosted application facade (APP-HOST-8C)."""

from __future__ import annotations

from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
    create_reference_production_process_composition,
)
from intergrax.hosting import (
    HostedApplicationSupervisorResult,
    run_hosted_application,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.hosting.profile import (
    build_local_workspace_hosted_profile,
)


def run_local_workspace_hosted_application(
    *,
    process_composition: ProductionProcessComposition | None = None,
    settings: LocalWorkspaceBackendSettings | None = None,
) -> HostedApplicationSupervisorResult:
    """Build one LKW hosted profile and run it via the platform foreground runner."""
    composition = (
        process_composition
        if process_composition is not None
        else create_reference_production_process_composition()
    )
    profile = build_local_workspace_hosted_profile(
        process_composition=composition,
        settings=settings,
    )
    return run_hosted_application(profile)
