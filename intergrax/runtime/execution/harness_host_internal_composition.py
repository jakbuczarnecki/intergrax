# © Artur Czarnecki. All rights reserved.

"""Assemble harness host internal capabilities from private orchestration backend (EE only)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.agent_execution_result import AgentExecutionResult
from intergrax.contracts.host_orchestration_application_wiring_target import (
    HostOrchestrationPluginBootstrapTarget,
)
from intergrax.runtime.decision_flow import DecisionFlowGate
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.execution._orchestration_backend_access import (
    orchestration_backend_for_execution_engine,
)
from intergrax.runtime.execution.environment_orchestration_materialization import (
    EnvironmentOrchestrationMaterialization,
)
from intergrax.runtime.execution.execution_terminal.service import ExecutionTerminalService
from intergrax.runtime.hooks.nexus_lifecycle_hooks import NexusLifecycleHookCoordinator
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline


@dataclass(frozen=True, slots=True)
class HarnessHostPluginSurface:
    """Narrow plugin registration inputs assembled from the orchestration backend."""

    event_bus: RuntimeEventBus


@dataclass(frozen=True, slots=True)
class HarnessHostInternalCapabilities:
    """Private orchestration capabilities assembled from the materialized backend."""

    execution_terminal: ExecutionTerminalService
    event_bus: RuntimeEventBus
    decision_flow_gate: DecisionFlowGate[AgentExecutionResult] | None
    middleware_pipeline: MiddlewarePipeline
    lifecycle_hook_coordinator: NexusLifecycleHookCoordinator
    plugin_surface: HarnessHostPluginSurface
    runtime_event_persistence: RuntimeEventPersistence | None
    plugin_bootstrap_target: HostOrchestrationPluginBootstrapTarget


def build_harness_host_internal_composition_from_materialization(
    materialization: EnvironmentOrchestrationMaterialization,
) -> HarnessHostInternalCapabilities:
    nexus_loop = orchestration_backend_for_execution_engine(materialization)
    middleware = nexus_loop.middleware
    return HarnessHostInternalCapabilities(
        execution_terminal=nexus_loop.execution_terminal,
        event_bus=nexus_loop.event_bus,
        decision_flow_gate=nexus_loop.peek_decision_flow_gate(),
        middleware_pipeline=middleware,
        lifecycle_hook_coordinator=nexus_loop._lifecycle_hooks,  # noqa: SLF001
        plugin_surface=HarnessHostPluginSurface(
            event_bus=nexus_loop.event_bus,
        ),
        runtime_event_persistence=nexus_loop.runtime_event_store,
        plugin_bootstrap_target=nexus_loop,
    )


__all__ = [
    "HarnessHostInternalCapabilities",
    "HarnessHostPluginSurface",
    "build_harness_host_internal_composition_from_materialization",
]
