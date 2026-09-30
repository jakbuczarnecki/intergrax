# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Scenario-local diagnostic wiring against neutral orchestration host target."""

from __future__ import annotations

from intergrax.applications._shared.diagnostic_runtime_wiring import (
    wire_terminal_execution_diagnostics,
)
from intergrax.applications._shared.environment_wiring import ApplicationEnvironmentWiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.execution.environment_orchestration_materialization import (
    EnvironmentOrchestrationMaterialization,
)
from intergrax.runtime.execution._orchestration_backend_access import (
    orchestration_backend_for_execution_engine,
)
from intergrax.runtime.nexus.observability_wiring import NexusObservabilityStores

from intergrax.applications._shared.diagnostic_assembly_resolver import DiagnosticWiring
from intergrax.applications._shared.scenario_runtime_profiles import ScenarioRuntimeMode


def wire_scenario_terminal_execution_diagnostics(
    *,
    materialization: EnvironmentOrchestrationMaterialization,
    env: ApplicationEnvironmentProfile,
    env_wiring: ApplicationEnvironmentWiring,
    observability: NexusObservabilityStores,
    scenario_runtime_mode: ScenarioRuntimeMode | None = None,
) -> DiagnosticWiring:
    host = orchestration_backend_for_execution_engine(materialization)
    return wire_terminal_execution_diagnostics(
        env=env,
        env_wiring=env_wiring,
        observability=observability,
        nexus_loop=host,  # type: ignore[arg-type]
        scenario_runtime_mode=scenario_runtime_mode,
    )


__all__ = ["wire_scenario_terminal_execution_diagnostics"]
