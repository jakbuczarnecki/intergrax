# © Artur Czarnecki. All rights reserved.

"""Wire capability alias middleware on harness hosts (APP-EVOL-3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
    from intergrax.runtime.nexus.nexus_loop import NexusLoop


def apply_capability_alias_wiring(
    nexus: NexusLoop,
    *,
    environment: ApplicationEnvironmentProfile,
) -> None:
    """Attach alias redirect middleware when the environment declares aliases."""
    if not environment.capability_governance_profile.aliases:
        return
    from intergrax.applications._shared.application_host_wiring import _attach_middleware
    from intergrax.applications._shared.capability_alias_middleware import CapabilityAliasMiddleware

    _attach_middleware(nexus, CapabilityAliasMiddleware(environment=environment))
