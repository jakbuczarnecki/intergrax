"""Scenario application execution entry."""

from __future__ import annotations

from intergrax.applications._shared.scenario_runtime_baseline import (
    ScenarioExecutionRequest,
    ScenarioRuntimeComposition,
    ScenarioRuntimeExecutionResult,
    execute_scenario_task,
)

from platform_proofs.scenarios.indirect_prompt_injection.application.runtime_composition import build_scenario_runtime


async def execute_scenario(
    *,
    tenant_id: str,
    message: str,
    composition: ScenarioRuntimeComposition | None = None,
) -> ScenarioRuntimeExecutionResult:
    """Execute one scenario task through the platform scenario runtime facade."""
    runtime = composition or build_scenario_runtime(tenant_id=tenant_id)
    return await execute_scenario_task(
        runtime,
        ScenarioExecutionRequest(tenant_id=tenant_id, message=message),
    )
