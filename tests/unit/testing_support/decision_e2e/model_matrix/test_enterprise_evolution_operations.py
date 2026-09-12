# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from testing_support.decision_e2e.model_matrix.enterprise_evolution_operations import (
    AdaptationOperationalReference,
    EnterpriseEvolutionOperationsEngine,
    EvolutionOperationConstraint,
    EvolutionOperationOutcome,
    EvolutionOperationRequest,
    EvolutionOperationRequestMetadata,
    EvolutionOperationStatus,
    EvolutionOperationType,
    EvolutionOperationalEventKind,
    default_enterprise_evolution_operations_engine,
    default_evolution_health_observation_provider,
    default_evolution_operations_audit_provider,
)


def _stamp() -> datetime:
    return datetime(2026, 9, 12, 19, 0, 0, tzinfo=UTC)


def _reference() -> AdaptationOperationalReference:
    return AdaptationOperationalReference(
        adaptation_id="adapt-1",
        version="1",
        applied_change_reference="adapted:adapt-1:v1:prop-routing-1",
        scope_id="scope-routing-shadow",
        proposal_id="prop-routing-1",
        controlled_evolution_record_id="evo-rec-1",
    )


def _constraints() -> tuple[EvolutionOperationConstraint, ...]:
    return (
        EvolutionOperationConstraint(
            constraint_id="ops-cstr-1",
            allowed_operation_types=(
                EvolutionOperationType.OBSERVE,
                EvolutionOperationType.PAUSE,
                EvolutionOperationType.RESUME,
                EvolutionOperationType.MARK_REVIEW_REQUIRED,
                EvolutionOperationType.DISABLE,
            ),
        ),
    )


def _request(
    *,
    operation_type: EvolutionOperationType = EvolutionOperationType.OBSERVE,
) -> EvolutionOperationRequest:
    return EvolutionOperationRequest(
        adaptation_reference=_reference(),
        operation_type=operation_type,
        constraints=_constraints(),
        request_metadata=EvolutionOperationRequestMetadata(
            operator_identity="ops-analyst-1",
            reason_code="routine_check",
        ),
    )


def test_default_operations_provider_observes_active_adaptation() -> None:
    engine = default_enterprise_evolution_operations_engine()
    result = engine.operate(_request(), executed_at=_stamp())
    assert result.operational_status is EvolutionOperationStatus.ACTIVE
    assert result.provider_id == "default_enterprise_evolution_operations"
    assert result.health_observation.quality_indicator == "stable"
    assert result.health_observation.risk_indicator == "low"


@dataclass
class CustomEvolutionOperationsProvider:
    @property
    def provider_id(self) -> str:
        return "custom_evolution_operations"

    @property
    def provider_version(self) -> str:
        return "7"

    def current_status(
        self, adaptation_reference: AdaptationOperationalReference
    ) -> EvolutionOperationStatus:
        return EvolutionOperationStatus.ACTIVE

    def operate(self, request: EvolutionOperationRequest) -> EvolutionOperationOutcome:
        return EvolutionOperationOutcome(
            operational_status=EvolutionOperationStatus.PAUSED,
            summary=f"Custom pause for {request.adaptation_reference.adaptation_id}.",
        )


def test_operations_provider_swap_without_engine_change() -> None:
    engine = EnterpriseEvolutionOperationsEngine(
        operations_provider=CustomEvolutionOperationsProvider(),
        health_observation_provider=default_evolution_health_observation_provider(),
        audit_provider=default_evolution_operations_audit_provider(),
    )
    result = engine.operate(
        _request(operation_type=EvolutionOperationType.PAUSE),
        executed_at=_stamp(),
    )
    assert result.provider_id == "custom_evolution_operations"
    assert result.provider_version == "7"
    assert result.operational_status is EvolutionOperationStatus.PAUSED


def test_operation_history_record_created() -> None:
    engine = default_enterprise_evolution_operations_engine()
    result = engine.operate(_request(), executed_at=_stamp())
    history = engine.history_for_adaptation("adapt-1")
    assert len(history) == 1
    assert history[0].record_id == result.operation_record.record_id
    assert history[0].event_kind is EvolutionOperationalEventKind.OBSERVED
    assert history[0].adaptation_id == "adapt-1"
    assert history[0].operator_identity == "ops-analyst-1"


def test_health_observation_provider_returns_observation() -> None:
    engine = default_enterprise_evolution_operations_engine()
    observation = engine.observe_health(_request(), observed_at=_stamp())
    assert observation.adaptation_id == "adapt-1"
    assert observation.version == "1"
    assert observation.operational_status is EvolutionOperationStatus.ACTIVE
    assert observation.quality_indicator == "stable"


def test_operations_audit_records_reference_provider_and_timestamp() -> None:
    engine = default_enterprise_evolution_operations_engine()
    result = engine.operate(_request(), executed_at=_stamp())
    audit = result.audit_metadata
    assert audit.adaptation_id == "adapt-1"
    assert audit.adaptation_version == "1"
    assert audit.adaptation_reference.proposal_id == "prop-routing-1"
    assert audit.operations_provider_id == "default_enterprise_evolution_operations"
    assert audit.outcome_status is EvolutionOperationStatus.ACTIVE
    assert audit.executed_at == _stamp()
