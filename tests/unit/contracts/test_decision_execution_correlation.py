# © Artur Czarnecki. All rights reserved.

from datetime import UTC, datetime

import pytest

from intergrax.contracts.decision_execution_correlation import (
    DecisionExecutionCorrelationIntegrityError,
    DecisionExecutionCorrelationKind,
    DecisionExecutionCorrelationRecord,
    correlation_record_from_decision_identity,
    validate_correlation_tenant_scope,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_AWARE = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)


def _identity(tenant_id: str = "tenant-a") -> DecisionIdentity:
    return DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="qualification", subject="r4"),
        tenant_id=tenant_id,
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )


@pytest.mark.unit
@pytest.mark.gate
def test_correlation_record_from_decision_identity_maps_execution_attempt() -> None:
    identity = _identity()
    record = correlation_record_from_decision_identity(
        identity,
        correlation_kind=DecisionExecutionCorrelationKind.DECISION_BOUND_EXECUTION,
        created_at=_AWARE,
    )
    assert record.decision_attempt_id == identity.execution.attempt_id
    assert record.decision_id == identity.decision_id
    assert record.tenant_id == identity.tenant_id


@pytest.mark.unit
@pytest.mark.gate
def test_validate_correlation_tenant_scope_rejects_cross_tenant() -> None:
    record = correlation_record_from_decision_identity(
        _identity(tenant_id="tenant-a"),
        correlation_kind=DecisionExecutionCorrelationKind.DECISION_BOUND_EXECUTION,
        created_at=_AWARE,
    )
    with pytest.raises(DecisionExecutionCorrelationIntegrityError):
        validate_correlation_tenant_scope(record, tenant_id="tenant-b")


@pytest.mark.unit
@pytest.mark.gate
def test_correlation_record_requires_timezone_aware_created_at() -> None:
    identity = _identity()
    with pytest.raises(ValueError):
        DecisionExecutionCorrelationRecord(
            tenant_id=identity.tenant_id,
            task_id=identity.execution.task_id,
            run_id=identity.execution.run_id,
            decision_id=identity.decision_id,
            decision_attempt_id=identity.execution.attempt_id,
            execution_id=identity.execution.execution_id,
            correlation_kind=DecisionExecutionCorrelationKind.DECISION_BOUND_EXECUTION,
            created_at=datetime(2026, 9, 11, 12, 0),
        )
