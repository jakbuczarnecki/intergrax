"""Lab-only VPI execution correlation minting — outside scenario application layer."""

from __future__ import annotations

from intergrax.contracts.application_execution_stage_signal import ApplicationExecutionCorrelation
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)

from platform_proofs.scenarios.verified_product_identification.application.observability.execution_correlation import (
    build_vpi_application_execution_correlation,
)
from platform_proofs.scenarios.verified_product_identification.application.observability.contracts import (
    ProductIdentificationRunId,
)


def mint_lab_vpi_application_execution_correlation(
    *,
    tenant_id: str,
    scenario_run_id: ProductIdentificationRunId,
) -> ApplicationExecutionCorrelation:
    return build_vpi_application_execution_correlation(
        tenant_id=tenant_id,
        scenario_run_id=scenario_run_id,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
