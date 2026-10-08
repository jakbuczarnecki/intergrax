# © Artur Czarnecki. All rights reserved.

"""Integrations pinning-store adapter for neutral provenance reads (TRACE-X-P5-R2-P2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_identity import ExecutionId, validate_execution_id
from intergrax.contracts.execution_integration_configuration_provenance import (
    ExecutionIntegrationConfigurationProvenance,
    ExecutionIntegrationConfigurationProvenanceReader,
    require_tenant_id_for_integration_configuration_provenance,
)
from intergrax.integrations.contracts.execution_integration_configuration_pinning import (
    ExecutionIntegrationConfigurationPinningStore,
)


@dataclass(frozen=True, slots=True)
class PinningStoreExecutionIntegrationConfigurationProvenanceReader(
    ExecutionIntegrationConfigurationProvenanceReader,
):
    """Read-only projection of durable integration configuration provenance pins."""

    pinning_store: ExecutionIntegrationConfigurationPinningStore

    def read_all(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
    ) -> tuple[ExecutionIntegrationConfigurationProvenance, ...]:
        tenant = require_tenant_id_for_integration_configuration_provenance(tenant_id)
        validated_execution_id = validate_execution_id(execution_id)
        records = self.pinning_store.read_all(
            tenant_id=tenant,
            execution_id=validated_execution_id,
        )
        filtered = tuple(
            record
            for record in records
            if record.tenant_id == tenant
            and validate_execution_id(record.execution_id) == validated_execution_id
        )
        return filtered


__all__ = ["PinningStoreExecutionIntegrationConfigurationProvenanceReader"]
