# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared composition-root host task execution wiring for application hosts."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications._shared.profile_resolution.execution_admission import (
    EffectiveProfileExecutionPinningDependencies,
    build_effective_profile_revision_admission,
)
from intergrax.runtime.execution.nexus_host_execution import (
    build_host_task_execution,
    build_nexus_host_task_terminal_publisher,
)
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.orchestration_capabilities import orchestration_capabilities_from_triggers

__all__ = [
    "build_environment_host_task_execution",
    "build_host_task_execution",
    "build_nexus_host_task_terminal_publisher",
]


def build_environment_host_task_execution(
    nexus_loop: NexusLoop,
    env: ApplicationEnvironmentProfile,
    *,
    pinning_dependencies: EffectiveProfileExecutionPinningDependencies | None = None,
):
    """Build canonical host task execution from environment orchestration profile."""
    graph_spec = env.graph_spec
    revision_admission = (
        build_effective_profile_revision_admission(pinning_dependencies)
        if pinning_dependencies is not None
        else None
    )
    return build_host_task_execution(
        nexus_loop,
        orchestration_triggers=orchestration_capabilities_from_triggers(
            graph_spec.trigger_capabilities if graph_spec is not None else None,
        ),
        pipeline_capability_suffix=(
            graph_spec.pipeline_capability_suffix if graph_spec is not None else ".pipeline"
        ),
        revision_admission=revision_admission,
    )
