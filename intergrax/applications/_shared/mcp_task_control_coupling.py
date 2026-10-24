# © Artur Czarnecki. All rights reserved.

"""Resolve harness task-control mutation boundary for MCP host coupling."""

from __future__ import annotations

from typing import TYPE_CHECKING

from intergrax.applications._shared.harness_control_plane_governance_wiring import (
    build_harness_control_plane_governance,
    resolve_harness_task_control_mutation_boundary,
)
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)

if TYPE_CHECKING:
    from intergrax.applications._shared.harness_host_runtime import HarnessHostRuntime


def resolve_mcp_coupling_task_control_mutation_boundary(
    *,
    runtime: HarnessHostRuntime | None = None,
    env: ApplicationEnvironmentProfile | None = None,
) -> ControlPlaneMutationAuthorizationBoundary | None:
    """Same boundary authority as harness HTTP task-control cancel routes."""
    if runtime is not None and runtime.control_plane_governance is not None:
        return resolve_harness_task_control_mutation_boundary(runtime.control_plane_governance)
    if env is not None and env.application_profile is ApplicationProfile.PRODUCT:
        return resolve_harness_task_control_mutation_boundary(
            build_harness_control_plane_governance(env),
        )
    return None


__all__ = ["resolve_mcp_coupling_task_control_mutation_boundary"]
