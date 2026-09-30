# © Artur Czarnecki. All rights reserved.

"""Assemble harness host internal composition from private orchestration backend (EE only)."""

from __future__ import annotations

from intergrax.applications._shared.harness_host_composition import (
    HarnessHostInternalComposition,
    HarnessHostPluginRegistrationSurface,
)
from intergrax.runtime.execution._orchestration_backend_access import (
    orchestration_backend_for_execution_engine,
)
from intergrax.runtime.execution.environment_orchestration_materialization import (
    EnvironmentOrchestrationMaterialization,
)


def build_harness_host_internal_composition_from_materialization(
    materialization: EnvironmentOrchestrationMaterialization,
) -> HarnessHostInternalComposition:
    nexus_loop = orchestration_backend_for_execution_engine(materialization)
    middleware = nexus_loop.middleware
    return HarnessHostInternalComposition(
        execution_terminal=nexus_loop.execution_terminal,
        event_bus=nexus_loop.event_bus,
        decision_flow_gate=nexus_loop.peek_decision_flow_gate(),
        middleware_pipeline=middleware,
        lifecycle_hook_coordinator=nexus_loop._lifecycle_hooks,  # noqa: SLF001
        plugin_surface=HarnessHostPluginRegistrationSurface(
            event_bus=nexus_loop.event_bus,
            hook_registry=middleware.hooks,
            policy_engine=nexus_loop.policy_engine,
        ),
        runtime_event_persistence=nexus_loop.runtime_event_store,
        plugin_bootstrap_target=nexus_loop,
    )


__all__ = ["build_harness_host_internal_composition_from_materialization"]
