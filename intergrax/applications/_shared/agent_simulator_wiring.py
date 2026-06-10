# © Artur Czarnecki. All rights reserved.

"""Agent simulator HTTP wiring for product hosts (AUDIT-IDEAL-27.3)."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import APIRouter

from intergrax.applications._shared.mvp_evolution_routes import create_mvp_evolution_router
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


@dataclass(frozen=True, slots=True)
class AgentSimulatorWiring:
    enabled: bool
    router: APIRouter | None


def resolve_agent_simulator_wiring(env: ApplicationEnvironmentProfile) -> AgentSimulatorWiring:
    """Mount MVP evolution simulate/replay routes on product hosts when enabled."""
    if env.application_profile is not ApplicationProfile.PRODUCT:
        return AgentSimulatorWiring(enabled=False, router=None)
    if not env.features.agent_simulator_enabled:
        return AgentSimulatorWiring(enabled=False, router=None)
    return AgentSimulatorWiring(
        enabled=True,
        router=create_mvp_evolution_router(enabled=True),
    )
