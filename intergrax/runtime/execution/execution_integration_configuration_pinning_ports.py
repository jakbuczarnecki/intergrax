# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution-runtime hook for configured adoption provenance pin (TRACE-X-P5-R2-P3)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.execution_identity import ExecutionId
from intergrax.integrations.contracts.execution_integration_configuration import (
    ExecutionIntegrationConfigurationAdoption,
)


@runtime_checkable
class ExecutionIntegrationConfigurationExecutionPinningPort(Protocol):
    """Pin configured adoption after canonical ExecutionId admission — before provider use."""

    def pin_configured_adoption_for_execution(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
        adoption: ExecutionIntegrationConfigurationAdoption,
    ) -> None:
        ...


__all__ = ["ExecutionIntegrationConfigurationExecutionPinningPort"]
