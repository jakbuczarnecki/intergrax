# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared composition-root host task execution wiring for application hosts."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.execution.host_task import HostTaskExecution
from intergrax.runtime.execution.orchestration import OrchestrationExecutor
from intergrax.runtime.nexus.agent_router import AgentRouter
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.orchestration_capabilities import orchestration_capabilities_from_triggers


def build_host_task_execution(
    nexus_loop: NexusLoop,
    *,
    orchestration_triggers: frozenset[str],
    pipeline_capability_suffix: str = ".pipeline",
) -> HostTaskExecution:
    """Internal composition builder: extract canonical execution dependencies from Nexus."""
    return HostTaskExecution(
        _agent_engine=nexus_loop.agent_engine,
        _agent_router=AgentRouter(
            nexus_loop.registry,
            event_bus=nexus_loop.event_bus,
        ),
        _orchestration_executor=OrchestrationExecutor(nexus_loop),
        _orchestration_triggers=orchestration_triggers,
        _pipeline_capability_suffix=pipeline_capability_suffix,
        _ledger_factory=nexus_loop.execution_budget_ledger_factory,
        _run_budget=nexus_loop.run_budget,
    )


def build_environment_host_task_execution(
    nexus_loop: NexusLoop,
    env: ApplicationEnvironmentProfile,
) -> HostTaskExecution:
    """Build canonical host task execution from environment orchestration profile."""
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
