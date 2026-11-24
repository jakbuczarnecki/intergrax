# © Artur Czarnecki. All rights reserved.

"""GCA Application Hosting adoption surface (Package B strict production)."""

from __future__ import annotations

from intergrax.applications._shared.strict_production_serving_launcher import (
    StrictProductionServingSession,
)
from intergrax.hosting import HostedApplicationSupervisorResult, run_hosted_application

from governed_contractor_application.host.governed_contractor_host_runtime_composition import (
    GovernedContractorHostRuntimeComposition,
)
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from governed_contractor_application.hosting.profile import build_governed_contractor_hosted_profile


def run_governed_contractor_hosted_strict_production_application(
    *,
    session: StrictProductionServingSession,
    settings: GovernedContractorBackendSettings | None = None,
    host_runtime: GovernedContractorHostRuntimeComposition | None = None,
) -> HostedApplicationSupervisorResult:
    """Build one GCA hosted profile for an opened strict session and run the platform runner."""
    profile = build_governed_contractor_hosted_profile(
        session=session,
        settings=settings,
        host_runtime=host_runtime,
    )
    return run_hosted_application(profile)


__all__ = [
    "build_governed_contractor_hosted_profile",
    "run_governed_contractor_hosted_strict_production_application",
]
