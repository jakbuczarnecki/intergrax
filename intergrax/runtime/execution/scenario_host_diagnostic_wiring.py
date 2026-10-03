# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Scenario-local diagnostic wiring against neutral orchestration host target."""

from __future__ import annotations

from intergrax.applications._shared.diagnostic_runtime_wiring import (
    wire_terminal_execution_diagnostics,
)
from intergrax.applications._shared.environment_wiring import ApplicationEnvironmentWiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.host_observability_stores import HostObservabilityStores
from intergrax.runtime.execution.environment_orchestration_materialization import (
    EnvironmentOrchestrationMaterialization,
)
from intergrax.runtime.execution.harness_host_orchestration_wiring import (
    orchestration_application_wiring_target_from_materialization,
)
from intergrax.runtime.nexus.observability_wiring import NexusObservabilityStores

from intergrax.applications._shared.diagnostic_assembly_resolver import DiagnosticWiring
from intergrax.applications._shared.scenario_runtime_profiles import ScenarioRuntimeMode


def _as_host_observability_stores(
    observability: NexusObservabilityStores,
) -> HostObservabilityStores:
    return HostObservabilityStores(
        trace_store=observability.trace_store,
        runtime_event_store=observability.runtime_event_store,
        trace_db_path=observability.trace_db_path,
        runtime_events_db_path=observability.runtime_events_db_path,
    )


def wire_scenario_terminal_execution_diagnostics(
    *,
    materialization: EnvironmentOrchestrationMaterialization,
    env: ApplicationEnvironmentProfile,
    env_wiring: ApplicationEnvironmentWiring,
    observability: NexusObservabilityStores,
    scenario_runtime_mode: ScenarioRuntimeMode | None = None,
) -> DiagnosticWiring:
    orchestration_host = orchestration_application_wiring_target_from_materialization(
        materialization,
    )
    return wire_terminal_execution_diagnostics(
        env=env,
        env_wiring=env_wiring,
        observability=_as_host_observability_stores(observability),
        orchestration_host=orchestration_host,
        scenario_runtime_mode=scenario_runtime_mode,
    )


__all__ = ["wire_scenario_terminal_execution_diagnostics"]
