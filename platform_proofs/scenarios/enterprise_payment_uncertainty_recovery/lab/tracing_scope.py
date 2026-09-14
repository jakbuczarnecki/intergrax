"""Lab-only trace scope minting — outside scenario application layer (SCENARIO-PLATFORM-6A)."""

from __future__ import annotations

from intergrax.contracts.execution_identity import mint_execution_id, mint_run_id
from intergrax.runtime.events.w3c_trace_context import generate_trace_id

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.tracing.recorder import (
    ScenarioExecutionTraceScope,
)


def mint_lab_scenario_execution_trace_scope(
    *,
    correlation_id: str,
    scenario_id: str,
    variant_id: str,
) -> ScenarioExecutionTraceScope:
    return ScenarioExecutionTraceScope.from_fields(
        trace_id=generate_trace_id(),
        correlation_id=correlation_id,
        execution_id=mint_execution_id(),
        scenario_id=scenario_id,
        variant_id=variant_id,
        run_id=mint_run_id(),
    )
