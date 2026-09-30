# © Artur Czarnecki. All rights reserved.

"""Post-materialization harness host wiring inside Execution Engine owner zone."""

from __future__ import annotations

from intergrax.contracts.host_orchestration_application_wiring_target import (
    HostOrchestrationApplicationWiringTarget,
    HostOrchestrationPluginBootstrapTarget,
)
from intergrax.runtime.execution._orchestration_backend_access import (
    orchestration_backend_for_execution_engine,
)
from intergrax.runtime.execution.environment_orchestration_materialization import (
    EnvironmentOrchestrationMaterialization,
)


def orchestration_plugin_bootstrap_target_from_materialization(
    materialization: EnvironmentOrchestrationMaterialization,
) -> HostOrchestrationPluginBootstrapTarget:
    return orchestration_backend_for_execution_engine(materialization)


def orchestration_application_wiring_target_from_materialization(
    materialization: EnvironmentOrchestrationMaterialization,
) -> HostOrchestrationApplicationWiringTarget:
    return orchestration_backend_for_execution_engine(materialization)


__all__ = [
    "orchestration_application_wiring_target_from_materialization",
    "orchestration_plugin_bootstrap_target_from_materialization",
]
