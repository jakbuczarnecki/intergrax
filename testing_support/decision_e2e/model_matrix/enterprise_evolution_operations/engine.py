# © Artur Czarnecki. All rights reserved.

"""Enterprise evolution operations orchestration via injected plugins (DS-E2E-15J-L14)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

from testing_support.decision_e2e.model_matrix.enterprise_evolution_operations.contracts import (
    EvolutionHealthObservation,
    EvolutionOperationRecord,
    EvolutionOperationRequest,
    EvolutionOperationResult,
    EvolutionOperationStatus,
    EvolutionOperationType,
    EvolutionOperationalEventKind,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_operations.protocol import (
    EnterpriseEvolutionOperationsProvider,
    EvolutionHealthObservationProvider,
    EvolutionOperationOutcome,
    EvolutionOperationsAuditProvider,
)

_EVENT_BY_OPERATION: dict[EvolutionOperationType, EvolutionOperationalEventKind] = {
    EvolutionOperationType.OBSERVE: EvolutionOperationalEventKind.OBSERVED,
    EvolutionOperationType.PAUSE: EvolutionOperationalEventKind.PAUSED,
    EvolutionOperationType.RESUME: EvolutionOperationalEventKind.RESUMED,
    EvolutionOperationType.MARK_REVIEW_REQUIRED: EvolutionOperationalEventKind.REVIEWED,
    EvolutionOperationType.DISABLE: EvolutionOperationalEventKind.DISABLED,
}


def _constraint_gate(
    request: EvolutionOperationRequest,
) -> EvolutionOperationOutcome | None:
    if not request.constraints:
        return EvolutionOperationOutcome(
            operational_status=EvolutionOperationStatus.FAILED,
            summary="Operation rejected: no constraints defined for controlled operations.",
        )
    allowed: set[EvolutionOperationType] = set()
    for item in request.constraints:
        allowed.update(item.allowed_operation_types)
    if request.operation_type not in allowed:
        return EvolutionOperationOutcome(
            operational_status=EvolutionOperationStatus.FAILED,
            summary=(
                f"Operation rejected: {request.operation_type.value} "
                "not permitted by operational constraints."
            ),
        )
    return None


def _build_record(
    request: EvolutionOperationRequest,
    *,
    outcome: EvolutionOperationOutcome,
    executed_at: datetime,
) -> EvolutionOperationRecord:
    metadata = request.request_metadata
    return EvolutionOperationRecord(
        record_id=str(uuid4()),
        adaptation_id=request.adaptation_reference.adaptation_id,
        version=request.adaptation_reference.version,
        event_kind=_EVENT_BY_OPERATION[request.operation_type],
        operational_status=outcome.operational_status,
        occurred_at=executed_at,
        operator_identity=metadata.operator_identity,
        summary=outcome.summary,
    )


@dataclass
class EnterpriseEvolutionOperationsEngine:
    operations_provider: EnterpriseEvolutionOperationsProvider
    health_observation_provider: EvolutionHealthObservationProvider
    audit_provider: EvolutionOperationsAuditProvider
    operation_history: list[EvolutionOperationRecord] = field(default_factory=list)

    def operate(
        self,
        request: EvolutionOperationRequest,
        *,
        executed_at: datetime | None = None,
    ) -> EvolutionOperationResult:
        stamp = executed_at or datetime.now(tz=UTC)
        gate = _constraint_gate(request)
        if gate is not None:
            return self._finalize(request, outcome=gate, executed_at=stamp)
        provider_outcome = self.operations_provider.operate(request)
        return self._finalize(request, outcome=provider_outcome, executed_at=stamp)

    def observe_health(
        self,
        request: EvolutionOperationRequest,
        *,
        observed_at: datetime | None = None,
    ) -> EvolutionHealthObservation:
        _ = observed_at
        status = self.operations_provider.current_status(request.adaptation_reference)
        return self.health_observation_provider.observe(
            request.adaptation_reference,
            operational_status=status,
        )

    def history_for_adaptation(
        self, adaptation_id: str
    ) -> tuple[EvolutionOperationRecord, ...]:
        return tuple(
            item
            for item in self.operation_history
            if item.adaptation_id == adaptation_id
        )

    def _finalize(
        self,
        request: EvolutionOperationRequest,
        *,
        outcome: EvolutionOperationOutcome,
        executed_at: datetime,
    ) -> EvolutionOperationResult:
        record = _build_record(request, outcome=outcome, executed_at=executed_at)
        self.operation_history.append(record)
        health = self.health_observation_provider.observe(
            request.adaptation_reference,
            operational_status=outcome.operational_status,
        )
        audit = self.audit_provider.build_audit(
            request,
            outcome_status=outcome.operational_status,
            operations_provider_id=self.operations_provider.provider_id,
            operations_provider_version=self.operations_provider.provider_version,
            health_provider_id=self.health_observation_provider.provider_id,
            health_provider_version=self.health_observation_provider.provider_version,
            executed_at=executed_at,
            outcome_summary=outcome.summary,
        )
        return EvolutionOperationResult(
            operational_status=outcome.operational_status,
            provider_id=self.operations_provider.provider_id,
            provider_version=self.operations_provider.provider_version,
            operation_record=record,
            health_observation=health,
            audit_metadata=audit,
            adaptation_reference=request.adaptation_reference,
            operation_type=request.operation_type,
        )


def default_enterprise_evolution_operations_engine(
    *,
    operations_provider: EnterpriseEvolutionOperationsProvider | None = None,
    health_observation_provider: EvolutionHealthObservationProvider | None = None,
    audit_provider: EvolutionOperationsAuditProvider | None = None,
) -> EnterpriseEvolutionOperationsEngine:
    if audit_provider is None:
        from testing_support.decision_e2e.model_matrix.enterprise_evolution_operations.audit_providers import (
            default_evolution_operations_audit_provider,
        )

        audit_provider = default_evolution_operations_audit_provider()
    if health_observation_provider is None:
        from testing_support.decision_e2e.model_matrix.enterprise_evolution_operations.health_observation_providers import (
            default_evolution_health_observation_provider,
        )

        health_observation_provider = default_evolution_health_observation_provider()
    if operations_provider is None:
        from testing_support.decision_e2e.model_matrix.enterprise_evolution_operations.operations_providers import (
            DefaultEnterpriseEvolutionOperationsProvider,
        )

        operations_provider = DefaultEnterpriseEvolutionOperationsProvider()
    return EnterpriseEvolutionOperationsEngine(
        operations_provider=operations_provider,
        health_observation_provider=health_observation_provider,
        audit_provider=audit_provider,
    )


__all__ = [
    "EnterpriseEvolutionOperationsEngine",
    "default_enterprise_evolution_operations_engine",
]
