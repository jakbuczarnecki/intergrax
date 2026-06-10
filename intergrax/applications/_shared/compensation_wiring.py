# © Artur Czarnecki. All rights reserved.

"""Product compensation flow wiring (AUDIT-IDEAL-22.1)."""

from __future__ import annotations

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.reliability.compensation import CompensationFlow, CompensationStep


async def _noop_compensation_handler(_step_id: str, _context: dict) -> None:
    return None


def resolve_compensation_flow(
    env: ApplicationEnvironmentProfile,
) -> CompensationFlow | None:
    """Return ordered compensation handlers for product side-effect paths."""
    if env.application_profile is not ApplicationProfile.PRODUCT:
        return None
    if not env.reliability_profile.compensation_enabled:
        return None
    return CompensationFlow(
        steps=[
            CompensationStep(step_id="side_effect", handler_id="product.noop_rollback"),
        ],
        handlers={"product.noop_rollback": _noop_compensation_handler},
    )
