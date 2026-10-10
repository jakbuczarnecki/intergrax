# © Artur Czarnecki. All rights reserved.

"""Build candidate recovery staging from canonical active execution identity (TRACE-X-P5-R2-P4-R2-R1)."""

from __future__ import annotations

from datetime import UTC, datetime

from intergrax.contracts.execution_identity import (
    ExecutionId,
    require_active_execution_id,
    require_active_execution_identity,
    validate_execution_id,
)
from intergrax.integrations.contracts.execution_integration_configuration_requirement_recovery_staging_source import (
    ExecutionIntegrationConfigurationRequirementRecoveryStagingSourcePort,
)
from intergrax.integrations.contracts.execution_integration_configuration_pin_record import (
    ExecutionIntegrationConfigurationRequirementRecoveryStaging,
)


class ActiveExecutionIdentityRequirementRecoveryStagingSource(
    ExecutionIntegrationConfigurationRequirementRecoveryStagingSourcePort,
):
    """Runtime/application composition — no Integrations persistence lookup."""

    def candidate_staging_for_configured_invocation(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
    ) -> ExecutionIntegrationConfigurationRequirementRecoveryStaging:
        _ = tenant_id
        validated_execution_id = validate_execution_id(execution_id)
        active_execution_id = require_active_execution_id()
        if active_execution_id != validated_execution_id:
            raise ValueError("active execution_id does not match configured invocation")
        run_id, attempt_id = require_active_execution_identity()
        from intergrax.contracts.execution_identity import peek_active_execution_task_id

        task_id = peek_active_execution_task_id()
        if task_id is None:
            raise RuntimeError("active TaskId required for requirement recovery staging")
        prepared_at = datetime.now(UTC)
        return ExecutionIntegrationConfigurationRequirementRecoveryStaging(
            requirement_boundary_prepared_at=prepared_at,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
        )


__all__ = ["ActiveExecutionIdentityRequirementRecoveryStagingSource"]
