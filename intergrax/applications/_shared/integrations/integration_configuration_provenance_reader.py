# © Artur Czarnecki. All rights reserved.

"""Integrations pinning-store adapter for neutral provenance reads (TRACE-X-P5-R2-P2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.execution_identity import ExecutionId, validate_execution_id
from intergrax.contracts.execution_integration_configuration_provenance import (
    ExecutionIntegrationConfigurationProvenance,
    ExecutionIntegrationConfigurationProvenanceReader,
    require_tenant_id_for_integration_configuration_provenance,
    validate_execution_integration_configuration_provenance_record,
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
        validated: list[ExecutionIntegrationConfigurationProvenance] = []
        for record in records:
            validate_execution_integration_configuration_provenance_record(
                record,
                expected_tenant_id=tenant,
                expected_execution_id=validated_execution_id,
            )
            validated.append(record)
        return tuple(validated)


def resolve_pinning_store_integration_configuration_provenance_reader(
    *,
    kv_store: object | None = None,
    document_store: object | None = None,
) -> PinningStoreExecutionIntegrationConfigurationProvenanceReader | None:
    """Same durable pinning backend as P2/P3 writes — optional for diagnostic reconstruction."""
    from intergrax.applications._shared.integrations.persistence import (
        wire_execution_integration_configuration_pinning_store,
    )
    from intergrax.distributed.contracts.kv_store import DistributedKVStore
    from intergrax.integrations.contracts.document_store import ConditionalDocumentStore

    if kv_store is not None:
        if not isinstance(kv_store, DistributedKVStore):
            return None
        pinning = wire_execution_integration_configuration_pinning_store(kv_store=kv_store)
        return PinningStoreExecutionIntegrationConfigurationProvenanceReader(pinning)
    if isinstance(document_store, ConditionalDocumentStore):
        pinning = wire_execution_integration_configuration_pinning_store(
            document_store=document_store,
        )
        return PinningStoreExecutionIntegrationConfigurationProvenanceReader(pinning)
    return None


__all__ = [
    "PinningStoreExecutionIntegrationConfigurationProvenanceReader",
    "resolve_pinning_store_integration_configuration_provenance_reader",
]
