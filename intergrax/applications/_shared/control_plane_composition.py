# © Artur Czarnecki. All rights reserved.

"""Shared control-plane composition invariants (GR-12 CLA-04 mandatory wiring)."""

from __future__ import annotations

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)

CONTROL_PLANE_COMPOSITION_BLOCKED_MISSING_BOUNDARY = (
    "CONTROL_PLANE_COMPOSITION_BLOCKED_MISSING_BOUNDARY"
)


class ControlPlaneCompositionError(ValueError):
    """Host composition rejected — consequential control-plane route without authority."""

    def __init__(self, blocker_code: str, message: str) -> None:
        super().__init__(message)
        self.blocker_code = blocker_code


def product_consequential_capacity_mutations_enabled(
    env: ApplicationEnvironmentProfile,
) -> bool:
    return (
        env.application_profile is ApplicationProfile.PRODUCT
        and env.scaling_profile.production_adapters_enabled
    )


def product_consequential_task_control_enabled(
    env: ApplicationEnvironmentProfile,
    *,
    task_control_routes_enabled: bool,
) -> bool:
    return env.application_profile is ApplicationProfile.PRODUCT and task_control_routes_enabled


def require_control_plane_mutation_boundary(
    boundary: ControlPlaneMutationAuthorizationBoundary | None,
    *,
    blocker_code: str,
    message: str,
) -> ControlPlaneMutationAuthorizationBoundary:
    if boundary is None:
        raise ControlPlaneCompositionError(blocker_code, message)
    return boundary


__all__ = [
    "CONTROL_PLANE_COMPOSITION_BLOCKED_MISSING_BOUNDARY",
    "ControlPlaneCompositionError",
    "product_consequential_capacity_mutations_enabled",
    "product_consequential_task_control_enabled",
    "require_control_plane_mutation_boundary",
]
