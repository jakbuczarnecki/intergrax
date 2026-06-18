# © Artur Czarnecki. All rights reserved.

"""Replay environment HTTP wiring for product hosts (AUDIT-IDEAL-27.2)."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import APIRouter

from intergrax.applications._shared.replay_routes import create_replay_router
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


@dataclass(frozen=True, slots=True)
class ReplayEnvironmentWiring:
    enabled: bool
    router: APIRouter | None


def resolve_replay_environment_wiring(
    env: ApplicationEnvironmentProfile,
) -> ReplayEnvironmentWiring:
    """Mount harness replay routes on product hosts when enabled."""
    if env.application_profile is not ApplicationProfile.PRODUCT:
        return ReplayEnvironmentWiring(enabled=False, router=None)
    if not env.features.replay_environment_enabled:
        return ReplayEnvironmentWiring(enabled=False, router=None)
    return ReplayEnvironmentWiring(
        enabled=True,
        router=create_replay_router(enabled=True),
    )
