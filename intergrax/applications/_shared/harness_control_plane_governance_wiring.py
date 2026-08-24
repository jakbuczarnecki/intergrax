# © Artur Czarnecki. All rights reserved.

"""Canonical harness-host control-plane governance composition (TASK-CPM-1B)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications._shared.harness_control_plane_policy_wiring import (
    build_harness_control_plane_mutation_boundary,
)
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)
from intergrax.runtime.governance.control_plane_mutation_approval import (
    ApprovalConsumingControlPlaneMutationEvaluator,
    ControlPlaneMutationApprovalCoordinator,
)


@dataclass(frozen=True, slots=True)
class HarnessControlPlaneGovernance:
    """Shared host-level control-plane mutation authority for Tier-3 harness hosts."""

    mutation_authorization_boundary: ControlPlaneMutationAuthorizationBoundary | None
    approval_coordinator: ControlPlaneMutationApprovalCoordinator | None = None


def _wrap_mutation_boundary(
    mutation_boundary: ControlPlaneMutationAuthorizationBoundary | None,
    approval_coordinator: ControlPlaneMutationApprovalCoordinator | None,
) -> ControlPlaneMutationAuthorizationBoundary | None:
    if mutation_boundary is None:
        return None
    if approval_coordinator is None:
        return mutation_boundary
    return ControlPlaneMutationAuthorizationBoundary(
        evaluator=ApprovalConsumingControlPlaneMutationEvaluator(
            inner=mutation_boundary.evaluator,
            coordinator=approval_coordinator,
        ),
    )


def build_harness_control_plane_governance(
    env: ApplicationEnvironmentProfile,
    *,
    mutation_authorization_boundary: ControlPlaneMutationAuthorizationBoundary | None = None,
    approval_coordinator: ControlPlaneMutationApprovalCoordinator | None = None,
) -> HarnessControlPlaneGovernance:
    """Compose one canonical host control-plane boundary from platform governance."""
    resolved_boundary = mutation_authorization_boundary
    if resolved_boundary is None and env.application_profile is ApplicationProfile.PRODUCT:
        resolved_boundary = build_harness_control_plane_mutation_boundary(env)
    resolved_coordinator = approval_coordinator
    if resolved_boundary is not None and resolved_coordinator is None:
        resolved_coordinator = ControlPlaneMutationApprovalCoordinator()
    return HarnessControlPlaneGovernance(
        mutation_authorization_boundary=_wrap_mutation_boundary(
            resolved_boundary,
            resolved_coordinator,
        ),
        approval_coordinator=resolved_coordinator,
    )


def resolve_harness_task_control_mutation_boundary(
    governance: HarnessControlPlaneGovernance,
) -> ControlPlaneMutationAuthorizationBoundary | None:
    """Return the shared host mutation boundary for governed task-control cancel."""
    return governance.mutation_authorization_boundary


__all__ = [
    "HarnessControlPlaneGovernance",
    "build_harness_control_plane_governance",
    "resolve_harness_task_control_mutation_boundary",
]
