# © Artur Czarnecki. All rights reserved.

"""LKW foreground hosted application facade (APP-HOST-8C)."""

from __future__ import annotations

from intergrax.applications._shared.harness_registry_authority import HarnessHostRegistryAuthorityError
from intergrax.applications._shared.production_process_composition import (
    ProductionProcessComposition,
)
from intergrax.hosting import (
    HostedApplicationSupervisorResult,
    run_hosted_application,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.hosting.profile import (
    build_local_workspace_hosted_profile,
)


_STRICT_HOSTED_MESSAGE = (
    "LKW hosted foreground requires an activated ProductionProcessComposition. "
    "Pass process_composition= with lifecycle deploy/activate completed."
)


def run_local_workspace_hosted_application(
    *,
    process_composition: ProductionProcessComposition | None = None,
    settings: LocalWorkspaceBackendSettings | None = None,
) -> HostedApplicationSupervisorResult:
    """Build one LKW hosted profile and run it via the platform foreground runner."""
    if process_composition is None:
        raise HarnessHostRegistryAuthorityError(_STRICT_HOSTED_MESSAGE)
    profile = build_local_workspace_hosted_profile(
        process_composition=process_composition,
        settings=settings,
    )
    return run_hosted_application(profile)
