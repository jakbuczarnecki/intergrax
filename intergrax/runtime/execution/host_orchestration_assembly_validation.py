# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Host orchestration assembly validation inside Execution Engine owner zone."""

from __future__ import annotations

from intergrax.applications._shared.guardrail_assembly_resolver import (
    assert_guardrail_assembly_valid,
)
from intergrax.applications._shared.guardrail_wiring import ApplicationGuardrailWiring
from intergrax.applications._shared.observability_wiring import (
    wire_observability_event_subscriptions,
)
from intergrax.applications._shared.reliability_wiring import (
    apply_reliability_governance_wiring,
)
from intergrax.applications._shared.security_assembly_resolver import (
    assert_security_assembly_valid,
)
from intergrax.applications._shared.security_wiring import ApplicationSecurityWiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.execution.environment_orchestration_materialization import (
    EnvironmentOrchestrationMaterialization,
)
from intergrax.runtime.execution._orchestration_backend_access import (
    orchestration_backend_for_execution_engine,
)


def assert_host_orchestration_application_assembly(
    materialization: EnvironmentOrchestrationMaterialization,
    *,
    env: ApplicationEnvironmentProfile,
    security_wiring: ApplicationSecurityWiring,
    guardrail_wiring: ApplicationGuardrailWiring,
) -> None:
    """Validate Tier-3 assembly against the materialized orchestration host."""
    host = orchestration_backend_for_execution_engine(materialization)
    assert_security_assembly_valid(
        security_wiring,
        env,
        orchestration_host=host,
    )
    assert_guardrail_assembly_valid(
        guardrail_wiring,
        env,
        orchestration_host=host,
    )


def apply_host_orchestration_application_runtime_wiring(
    materialization: EnvironmentOrchestrationMaterialization,
    *,
    env: ApplicationEnvironmentProfile,
) -> None:
    """Subscribe observability and reliability governance on the private backend."""
    host = orchestration_backend_for_execution_engine(materialization)
    wire_observability_event_subscriptions(materialization.event_bus, env.observability_profile)
    apply_reliability_governance_wiring(host, env)


__all__ = [
    "apply_host_orchestration_application_runtime_wiring",
    "assert_host_orchestration_application_assembly",
]
