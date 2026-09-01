# © Artur Czarnecki. All rights reserved.

from dataclasses import FrozenInstanceError, fields
from typing import get_type_hints

import pytest

from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    DecisionVersion,
    initial_decision_version,
    mint_decision_id,
    next_decision_version,
)
from intergrax.contracts.decision_lifecycle import DecisionLifecycleStage
from intergrax.contracts.decision_resolution import (
    AuthoritativeResolutionRecord,
    DecisionResolution,
    validate_authoritative_resolution_record,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)

_CANONICAL_RESOLUTIONS = (
    DecisionResolution.ACCEPTED,
    DecisionResolution.REJECTED,
    DecisionResolution.UNRESOLVED,
)

_FORBIDDEN_EXECUTION_RESOLUTION_NAMES = frozenset(
    {
        "FAILED",
        "ERROR",
        "CANCELLED",
        "CANCELED",
        "TIMED_OUT",
        "TIMEOUT",
        "BUDGET_EXHAUSTED",
        "PROVIDER_ERROR",
    },
)

_FORBIDDEN_LIFECYCLE_STAGE_NAMES = frozenset(
    {
        "PROPOSAL",
        "DELIBERATION",
        "VERIFICATION",
        "REVISION",
        "ADJUDICATION",
        "RESOLUTION",
        "FINALIZATION",
        "TERMINAL",
    },
)

_FORBIDDEN_ARTIFACT_FIELD_NAMES = frozenset(
    {
        "artifact",
        "accepted_artifact",
        "candidate",
    },
)


def _execution_lineage() -> DecisionExecutionLineage:
    return DecisionExecutionLineage(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _identity(
    *,
    tenant_id: str = "tenant-a",
    namespace: str = "incident",
    subject: str = "incident-123",
    version: DecisionVersion | None = None,
    decision_id: str | None = None,
) -> DecisionIdentity:
    return DecisionIdentity(
        decision_id=mint_decision_id() if decision_id is None else decision_id,
        version=version or initial_decision_version(),
        scope=DecisionScope(namespace=namespace, subject=subject),
        tenant_id=tenant_id,
        execution=_execution_lineage(),
    )


@pytest.mark.unit
@pytest.mark.gate
def test_decision_resolution_exact_set() -> None:
    assert tuple(DecisionResolution) == _CANONICAL_RESOLUTIONS


@pytest.mark.unit
@pytest.mark.gate
def test_decision_resolution_excludes_execution_failure_values() -> None:
    member_names = frozenset(resolution.name for resolution in DecisionResolution)
    assert member_names.isdisjoint(_FORBIDDEN_EXECUTION_RESOLUTION_NAMES)


@pytest.mark.unit
@pytest.mark.gate
def test_decision_resolution_excludes_lifecycle_stage_values() -> None:
    member_names = frozenset(resolution.name for resolution in DecisionResolution)
    assert member_names.isdisjoint(_FORBIDDEN_LIFECYCLE_STAGE_NAMES)


@pytest.mark.unit
@pytest.mark.gate
def test_authoritative_resolution_record_rejected() -> None:
    identity = _identity()
    record = AuthoritativeResolutionRecord(
        identity=identity,
        resolution=DecisionResolution.REJECTED,
    )
    assert record.identity is identity
    assert record.resolution is DecisionResolution.REJECTED
    assert validate_authoritative_resolution_record(record) is record


@pytest.mark.unit
@pytest.mark.gate
def test_authoritative_resolution_record_unresolved() -> None:
    identity = _identity()
    record = AuthoritativeResolutionRecord(
        identity=identity,
        resolution=DecisionResolution.UNRESOLVED,
    )
    assert record.resolution is DecisionResolution.UNRESOLVED


@pytest.mark.unit
@pytest.mark.gate
def test_authoritative_resolution_record_rejects_accepted() -> None:
    with pytest.raises(ValueError, match="cannot represent ACCEPTED"):
        AuthoritativeResolutionRecord(
            identity=_identity(),
            resolution=DecisionResolution.ACCEPTED,
        )


@pytest.mark.unit
@pytest.mark.gate
def test_authoritative_resolution_record_is_immutable() -> None:
    record = AuthoritativeResolutionRecord(
        identity=_identity(),
        resolution=DecisionResolution.REJECTED,
    )
    with pytest.raises(FrozenInstanceError):
        setattr(record, "resolution", DecisionResolution.UNRESOLVED)


@pytest.mark.unit
@pytest.mark.gate
def test_authoritative_resolution_record_strict_field_type_contract() -> None:
    hints = get_type_hints(AuthoritativeResolutionRecord)
    assert hints["identity"] is DecisionIdentity
    assert hints["resolution"] is DecisionResolution


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize(
    "wrong_type",
    [str, bool, int, type(None)],
)
def test_authoritative_resolution_record_resolution_field_is_not(
    wrong_type: type,
) -> None:
    hints = get_type_hints(AuthoritativeResolutionRecord)
    assert hints["resolution"] is not wrong_type


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize(
    "wrong_type",
    [str, bool, type(None)],
)
def test_authoritative_resolution_record_identity_field_is_not(
    wrong_type: type,
) -> None:
    hints = get_type_hints(AuthoritativeResolutionRecord)
    assert hints["identity"] is not wrong_type


@pytest.mark.unit
@pytest.mark.gate
def test_decision_resolution_runtime_guard_requires_enum_instance_not_value_alias() -> None:
    assert DecisionResolution.REJECTED.value == "rejected"
    assert type(DecisionResolution.REJECTED) is DecisionResolution
    assert type("rejected") is str


@pytest.mark.unit
@pytest.mark.gate
def test_validate_authoritative_resolution_record_strict_parameter_type_contract() -> None:
    hints = get_type_hints(validate_authoritative_resolution_record)
    assert hints["record"] is AuthoritativeResolutionRecord


@pytest.mark.unit
@pytest.mark.gate
def test_authoritative_resolution_record_has_no_artifact_fields() -> None:
    field_names = frozenset(field.name for field in fields(AuthoritativeResolutionRecord))
    assert field_names.isdisjoint(_FORBIDDEN_ARTIFACT_FIELD_NAMES)


@pytest.mark.unit
@pytest.mark.gate
def test_authoritative_resolution_records_differ_by_identity_context() -> None:
    base = _identity()
    different_tenant = AuthoritativeResolutionRecord(
        identity=_identity(tenant_id="tenant-b"),
        resolution=DecisionResolution.REJECTED,
    )
    different_scope = AuthoritativeResolutionRecord(
        identity=_identity(namespace="policy", subject="policy-9"),
        resolution=DecisionResolution.REJECTED,
    )
    different_version = AuthoritativeResolutionRecord(
        identity=_identity(version=next_decision_version(initial_decision_version())),
        resolution=DecisionResolution.REJECTED,
    )
    different_decision_id = AuthoritativeResolutionRecord(
        identity=_identity(decision_id=mint_decision_id()),
        resolution=DecisionResolution.REJECTED,
    )
    same = AuthoritativeResolutionRecord(
        identity=base,
        resolution=DecisionResolution.REJECTED,
    )
    assert different_tenant != same
    assert different_scope != same
    assert different_version != same
    assert different_decision_id != same


@pytest.mark.unit
@pytest.mark.gate
def test_decision_resolution_values_are_stable_strings() -> None:
    assert DecisionResolution.ACCEPTED.value == "accepted"
    assert DecisionResolution.REJECTED.value == "rejected"
    assert DecisionResolution.UNRESOLVED.value == "unresolved"


@pytest.mark.unit
@pytest.mark.gate
def test_lifecycle_stage_resolution_name_is_not_decision_resolution_member() -> None:
    assert DecisionLifecycleStage.RESOLUTION.name not in {
        resolution.name for resolution in DecisionResolution
    }
