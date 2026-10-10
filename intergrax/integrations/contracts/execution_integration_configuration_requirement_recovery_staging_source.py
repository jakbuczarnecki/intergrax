# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Candidate requirement recovery staging at configured invocation boundary (TRACE-X-P5-R2-P4-R2-R1)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.execution_identity import ExecutionId
from intergrax.integrations.contracts.execution_integration_configuration_pin_record import (
    ExecutionIntegrationConfigurationRequirementRecoveryStaging,
)


@runtime_checkable
class ExecutionIntegrationConfigurationRequirementRecoveryStagingSourcePort(Protocol):
    """Supplies candidate staging before first business I/O (reconcile-at-pin reuses durable rows)."""

    def candidate_staging_for_configured_invocation(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
    ) -> ExecutionIntegrationConfigurationRequirementRecoveryStaging: ...


__all__ = ["ExecutionIntegrationConfigurationRequirementRecoveryStagingSourcePort"]
