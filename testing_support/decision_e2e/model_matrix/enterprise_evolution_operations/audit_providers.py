# © Artur Czarnecki. All rights reserved.

"""Evolution operations audit metadata providers (DS-E2E-15J-L14)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from testing_support.decision_e2e.model_matrix.enterprise_evolution_operations.contracts import (
    ENTERPRISE_EVOLUTION_OPERATIONS_TASK_ID,
    ENTERPRISE_EVOLUTION_OPERATIONS_VERSION,
    EvolutionOperationRequest,
    EvolutionOperationStatus,
    EvolutionOperationsAuditMetadata,
)

_STANDARD_OPERATIONS_AUDIT_PROVIDER_ID = "standard_evolution_operations_audit"
_STANDARD_OPERATIONS_AUDIT_PROVIDER_VERSION = "1"


@dataclass(frozen=True, slots=True)
class StandardEvolutionOperationsAuditProvider:
    @property
    def provider_id(self) -> str:
        return _STANDARD_OPERATIONS_AUDIT_PROVIDER_ID

    @property
    def provider_version(self) -> str:
        return _STANDARD_OPERATIONS_AUDIT_PROVIDER_VERSION

    def build_audit(
        self,
        request: EvolutionOperationRequest,
        *,
        outcome_status: EvolutionOperationStatus,
        operations_provider_id: str,
        operations_provider_version: str,
        health_provider_id: str,
        health_provider_version: str,
        executed_at: datetime,
        outcome_summary: str,
    ) -> EvolutionOperationsAuditMetadata:
        reference = request.adaptation_reference
        return EvolutionOperationsAuditMetadata(
            operations_task_id=ENTERPRISE_EVOLUTION_OPERATIONS_TASK_ID,
            operations_layer_version=ENTERPRISE_EVOLUTION_OPERATIONS_VERSION,
            adaptation_id=reference.adaptation_id,
            adaptation_version=reference.version,
            adaptation_reference=reference,
            operations_provider_id=operations_provider_id,
            operations_provider_version=operations_provider_version,
            health_provider_id=health_provider_id,
            health_provider_version=health_provider_version,
            operation_type=request.operation_type,
            outcome_status=outcome_status,
            executed_at=executed_at,
            outcome_summary=outcome_summary,
        )


def default_evolution_operations_audit_provider() -> (
    StandardEvolutionOperationsAuditProvider
):
    return StandardEvolutionOperationsAuditProvider()


__all__ = [
    "StandardEvolutionOperationsAuditProvider",
    "default_evolution_operations_audit_provider",
]
