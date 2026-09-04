# © Artur Czarnecki. All rights reserved.

"""LKW composition-root canonical host task execution wiring."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications._shared.host_task_execution_wiring import build_host_task_execution
from intergrax.runtime.execution.host_task import HostTaskExecution
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.orchestration_capabilities import orchestration_capabilities_from_triggers


def build_lkw_host_task_execution(
    nexus_loop: NexusLoop,
    env: ApplicationEnvironmentProfile,
) -> HostTaskExecution:
    graph_spec = env.graph_spec
    return build_host_task_execution(
        nexus_loop,
        orchestration_triggers=orchestration_capabilities_from_triggers(
            graph_spec.trigger_capabilities if graph_spec is not None else None,
        ),
        pipeline_capability_suffix=(
            graph_spec.pipeline_capability_suffix if graph_spec is not None else ".pipeline"
        ),
    )
