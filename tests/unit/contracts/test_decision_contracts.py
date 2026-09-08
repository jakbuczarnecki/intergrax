# © Artur Czarnecki. All rights reserved.

"""MP-4B — Decision domain contract tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from intergrax.contracts.collaborative_work import WorkArtifactVersionRef
from intergrax.contracts.decision import (
    SCHEMA_DECISION_OUTCOME_V1,
    SCHEMA_DECISION_V1,
    Decision,
    DecisionContractInvariantError,
    DecisionIdentityInvariantError,
    DecisionLifecycleState,
    DecisionLifecycleTransitionError,
    DecisionOutcome,
    DecisionOutcomeDisposition,
    DecisionProvenanceInvariantError,
    DecisionReferences,
    DecisionScopeInvariantError,
    mint_decision_id,
    validate_decision_identity,
    validate_decision_provenance,
    validate_decision_scope,
    validate_decision_transition,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_provenance import ExecutionProvenanceRef

_UTC = timezone.utc


def _execution() -> ExecutionProvenanceRef:
    return ExecutionProvenanceRef(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _decision(**overrides: object) -> Decision:
    payload = {
        "decision_id": str(mint_decision_id()),
        "tenant_id": "tenant-1",
        "workspace_id": "workspace-1",
        "created_by_principal_id": "principal-1",
        "created_at": datetime(2026, 9, 8, 12, 0, tzinfo=_UTC),
        "lifecycle_state": DecisionLifecycleState.DRAFT,
    }
    payload.update(overrides)
    return Decision.model_validate(payload)


def _artifact_version_ref(**overrides: object) -> WorkArtifactVersionRef:
    payload = {
        "tenant_id": "tenant-1",
        "workspace_id": "workspace-1",
        "work_item_id": "work-item-1",
        "work_artifact_id": "artifact-1",
        "work_artifact_version_id": "artifact-version-1",
    }
    payload.update(overrides)
    return WorkArtifactVersionRef.model_validate(payload)


@pytest.mark.unit
def test_decision_creation_valid() -> None:
    decision = _decision(execution=_execution())
    assert decision.schema_version == SCHEMA_DECISION_V1
    assert decision.lifecycle_state is DecisionLifecycleState.DRAFT


@pytest.mark.unit
@pytest.mark.parametrize(
    "field_name",
    ["decision_id", "tenant_id", "workspace_id", "created_by_principal_id"],
)
def test_decision_rejects_empty_identifiers(field_name: str) -> None:
    with pytest.raises(ValidationError):
        _decision(**{field_name: "   "})


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    [
        "",
        "task_" + "a" * 32,
        "run_" + "a" * 32,
        "attempt_" + "a" * 32,
        "exec_" + "a" * 32,
        "decision_tooshort",
    ],
)
def test_decision_rejects_invalid_decision_id(value: str) -> None:
    with pytest.raises((ValidationError, DecisionIdentityInvariantError)):
        _decision(decision_id=value)


@pytest.mark.unit
def test_validate_decision_identity_accepts_valid_fields() -> None:
    decision_id = mint_decision_id()
    validated = validate_decision_identity(
        decision_id=decision_id,
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        created_by_principal_id="principal-1",
    )
    assert validated == decision_id


@pytest.mark.unit
def test_validate_decision_scope_rejects_mismatched_reference_scope() -> None:
    with pytest.raises(DecisionScopeInvariantError):
        validate_decision_scope(
            tenant_id="tenant-1",
            workspace_id="workspace-1",
            reference_tenant_id="tenant-2",
        )


@pytest.mark.unit
def test_decision_is_frozen_and_rejects_extra_fields() -> None:
    decision = _decision()
    with pytest.raises(ValidationError):
        decision.tenant_id = "other"
    with pytest.raises(ValidationError):
        Decision.model_validate(
            {
                "decision_id": str(mint_decision_id()),
                "tenant_id": "tenant-1",
                "workspace_id": "workspace-1",
                "created_by_principal_id": "principal-1",
                "created_at": datetime(2026, 9, 8, 12, 0, tzinfo=_UTC),
                "approval_status": "approved",
            },
        )


@pytest.mark.unit
def test_decision_json_round_trip_preserves_fields() -> None:
    decision = _decision(
        execution=_execution(),
        references=DecisionReferences(
            work_item_id="work-item-1",
            work_artifact_version_ref=_artifact_version_ref(),
        ),
    )
    restored = Decision.from_json(decision.to_json())
    assert restored == decision


@pytest.mark.unit
@pytest.mark.parametrize(
    ("from_state", "to_state"),
    [
        (DecisionLifecycleState.DRAFT, DecisionLifecycleState.PROPOSED),
        (DecisionLifecycleState.PROPOSED, DecisionLifecycleState.FINALIZED),
        (DecisionLifecycleState.FINALIZED, DecisionLifecycleState.SUPERSEDED),
        (DecisionLifecycleState.DRAFT, DecisionLifecycleState.CANCELLED),
    ],
)
def test_validate_decision_transition_accepts_valid_transitions(
    from_state: DecisionLifecycleState,
    to_state: DecisionLifecycleState,
) -> None:
    transition = validate_decision_transition(from_state=from_state, to_state=to_state)
    assert transition.from_state is from_state
    assert transition.to_state is to_state


@pytest.mark.unit
def test_validate_decision_transition_rejects_invalid_transition() -> None:
    with pytest.raises(DecisionLifecycleTransitionError):
        validate_decision_transition(
            from_state=DecisionLifecycleState.DRAFT,
            to_state=DecisionLifecycleState.FINALIZED,
        )


@pytest.mark.unit
def test_decision_outcome_is_immutable_and_separate_from_lifecycle() -> None:
    outcome = DecisionOutcome(
        decision_id=str(mint_decision_id()),
        outcome_version=1,
        disposition=DecisionOutcomeDisposition.RECORDED,
        recorded_at=datetime(2026, 9, 8, 12, 30, tzinfo=_UTC),
        summary="Selected remediation path A",
    )
    assert outcome.schema_version == SCHEMA_DECISION_OUTCOME_V1
    assert "lifecycle" not in DecisionOutcome.model_fields
    with pytest.raises(ValidationError):
        outcome.outcome_version = 2
    restored = DecisionOutcome.from_json(outcome.to_json())
    assert restored == outcome


@pytest.mark.unit
def test_decision_outcome_rejects_approval_semantics_fields() -> None:
    with pytest.raises(ValidationError):
        DecisionOutcome.model_validate(
            {
                "decision_id": str(mint_decision_id()),
                "outcome_version": 1,
                "disposition": "approved",
                "recorded_at": datetime(2026, 9, 8, 12, 30, tzinfo=_UTC),
            },
        )


@pytest.mark.unit
def test_decision_accepts_execution_provenance_reference() -> None:
    decision = _decision(execution=_execution())
    validate_decision_provenance(decision.execution)


@pytest.mark.unit
def test_decision_rejects_invalid_execution_reference() -> None:
    with pytest.raises((ValidationError, DecisionProvenanceInvariantError)):
        _decision(
            execution={
                "task_id": str(mint_task_id()),
                "run_id": str(mint_run_id()),
                "attempt_id": str(mint_attempt_id()),
            },
        )


@pytest.mark.unit
def test_decision_rejects_mismatched_artifact_reference_scope() -> None:
    with pytest.raises(DecisionScopeInvariantError):
        _decision(
            references=DecisionReferences(
                work_artifact_version_ref=_artifact_version_ref(tenant_id="tenant-2"),
            ),
        )


@pytest.mark.unit
def test_decision_is_not_work_artifact_substitution() -> None:
    decision_fields = set(Decision.model_fields)
    forbidden = {
        "artifact_status",
        "approval_status",
        "execution_status",
        "governance_status",
        "current_version_id",
        "work_artifact_id",
        "content_ref",
        "task_state",
        "runtime_state",
    }
    assert not decision_fields.intersection(forbidden)


@pytest.mark.unit
def test_decision_outcome_is_not_approval_substitution() -> None:
    outcome_fields = set(DecisionOutcome.model_fields)
    forbidden = {"approved", "rejected", "human_reviewed", "approval_status"}
    assert not outcome_fields.intersection(forbidden)


@pytest.mark.unit
def test_decision_identity_is_not_execution_identity_substitution() -> None:
    with pytest.raises(DecisionIdentityInvariantError):
        validate_decision_identity(
            decision_id=mint_task_id(),
            tenant_id="tenant-1",
            workspace_id="workspace-1",
            created_by_principal_id="principal-1",
        )


@pytest.mark.unit
def test_decision_contract_errors_are_typed_domain_exceptions() -> None:
    assert issubclass(DecisionScopeInvariantError, DecisionContractInvariantError)
    assert issubclass(DecisionLifecycleTransitionError, DecisionContractInvariantError)
    assert not issubclass(DecisionContractInvariantError, ValueError)
