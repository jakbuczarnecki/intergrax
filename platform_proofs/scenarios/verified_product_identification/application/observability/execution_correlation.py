"""Execution correlation helpers for VPI platform observability (P1B)."""

from __future__ import annotations

from intergrax.contracts.application_execution_stage_signal import ApplicationExecutionCorrelation
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)

from platform_proofs.scenarios.verified_product_identification.application.observability.contracts import (
    ProductIdentificationRunId,
)


def build_vpi_application_execution_correlation(
    *,
    tenant_id: str,
    scenario_run_id: ProductIdentificationRunId,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    execution_id: ExecutionId,
) -> ApplicationExecutionCorrelation:
    """
    Bind canonical platform execution identity to one VPI pipeline run id.

    Platform ids must be supplied by runtime composition or Execution Engine wiring.
    """
    return ApplicationExecutionCorrelation(
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        scenario_execution_correlation_id=scenario_run_id.value,
    )


def mint_lab_vpi_application_execution_correlation(
    *,
    tenant_id: str,
    scenario_run_id: ProductIdentificationRunId,
) -> ApplicationExecutionCorrelation:
    """
    Lab / unit-test helper when Execution Engine correlation is not yet attached.

    Production paths must not rely on this helper — supply ids from governed runtime.
    """
    return build_vpi_application_execution_correlation(
        tenant_id=tenant_id,
        scenario_run_id=scenario_run_id,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
