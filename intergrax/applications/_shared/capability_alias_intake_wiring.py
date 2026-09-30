# © Artur Czarnecki. All rights reserved.

"""Wire capability alias middleware on harness hosts (APP-EVOL-3)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.host_orchestration_application_wiring_target import (
    HostOrchestrationApplicationWiringTarget,
)


def apply_capability_alias_wiring(
    nexus: HostOrchestrationApplicationWiringTarget,
    *,
    environment: ApplicationEnvironmentProfile,
) -> None:
    """Attach alias redirect middleware when the environment declares aliases."""
    if not environment.capability_governance_profile.aliases:
        return
    from intergrax.applications._shared.application_host_wiring import _attach_middleware
    from intergrax.applications._shared.capability_alias_middleware import CapabilityAliasMiddleware

    _attach_middleware(nexus, CapabilityAliasMiddleware(environment=environment))
