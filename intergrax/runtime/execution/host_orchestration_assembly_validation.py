# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Host orchestration assembly validation inside Execution Engine owner zone."""

from __future__ import annotations

from intergrax.contracts.host_orchestration_wiring_port import (
    HostOrchestrationApplicationWiringPort,
)
from intergrax.runtime.execution._orchestration_backend_access import (
    orchestration_backend_for_execution_engine,
)
from intergrax.runtime.execution.environment_orchestration_materialization import (
    EnvironmentOrchestrationMaterialization,
)


def assert_host_orchestration_application_assembly(
    materialization: EnvironmentOrchestrationMaterialization,
    *,
    wiring: HostOrchestrationApplicationWiringPort,
) -> None:
    """Validate application assembly against the materialized orchestration host."""
    host = orchestration_backend_for_execution_engine(materialization)
    wiring.assert_assembly(host)


def apply_host_orchestration_application_runtime_wiring(
    materialization: EnvironmentOrchestrationMaterialization,
    *,
    wiring: HostOrchestrationApplicationWiringPort,
) -> None:
    """Subscribe observability and reliability governance on the private backend."""
    host = orchestration_backend_for_execution_engine(materialization)
    wiring.apply_runtime_wiring(host, event_bus=materialization.event_bus)


__all__ = [
    "apply_host_orchestration_application_runtime_wiring",
    "assert_host_orchestration_application_assembly",
]
