# © Artur Czarnecki. All rights reserved.

"""MP-4C — Approval domain contract tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from intergrax.contracts.approval import (
    SCHEMA_APPROVAL_OUTCOME_V1,
    SCHEMA_APPROVAL_REQUEST_V1,
    SCHEMA_HUMAN_APPROVAL_ACTION_V1,
    ApprovalContractInvariantError,
    ApprovalIdentityInvariantError,
    ApprovalLifecycleState,
    ApprovalLifecycleTransitionError,
    ApprovalOutcome,
    ApprovalOutcomeDisposition,
    ApprovalReferenceInvariantError,
    ApprovalReferences,
    ApprovalRequest,
    ApprovalScopeInvariantError,
    HumanApprovalAction,
    HumanApprovalActionType,
    mint_approval_id,
    validate_approval_identity,
    validate_approval_references,
    validate_approval_scope,
    validate_approval_transition,
)
from intergrax.contracts.collaborative_work import WorkArtifactVersionRef
from intergrax.contracts.decision import mint_decision_id
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


def _approval_request(**overrides: object) -> ApprovalRequest:
    payload = {
        "approval_id": str(mint_approval_id()),
        "tenant_id": "tenant-1",
        "workspace_id": "workspace-1",
        "decision_id": str(mint_decision_id()),
        "requested_by_principal_id": "principal-1",
        "requested_at": datetime(2026, 9, 8, 12, 0, tzinfo=_UTC),
        "lifecycle_state": ApprovalLifecycleState.REQUESTED,
    }
    payload.update(overrides)
    return ApprovalRequest.model_validate(payload)


@pytest.mark.unit
def test_approval_id_valid() -> None:
    approval_id = mint_approval_id()
    validated = validate_approval_identity(
        approval_id=approval_id,
        tenant_id="tenant-1",
        workspace_id="workspace-1",
        decision_id=mint_decision_id(),
        requested_by_principal_id="principal-1",
    )
    assert validated == approval_id


@pytest.mark.unit
@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "approval_tooshort",
        str(mint_decision_id()),
        str(mint_task_id()),
        str(mint_run_id()),
        str(mint_attempt_id()),
        str(mint_execution_id()),
    ],
)
def test_approval_id_rejects_invalid_values(value: str) -> None:
    with pytest.raises((ValidationError, ApprovalIdentityInvariantError)):
        _approval_request(approval_id=value)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("from_state", "to_state"),
    [
        (ApprovalLifecycleState.REQUESTED, ApprovalLifecycleState.ASSIGNED),
        (ApprovalLifecycleState.ASSIGNED, ApprovalLifecycleState.IN_REVIEW),
        (ApprovalLifecycleState.IN_REVIEW, ApprovalLifecycleState.APPROVED),
        (ApprovalLifecycleState.IN_REVIEW, ApprovalLifecycleState.REJECTED),
        (ApprovalLifecycleState.REQUESTED, ApprovalLifecycleState.CANCELLED),
        (ApprovalLifecycleState.ASSIGNED, ApprovalLifecycleState.EXPIRED),
    ],
)
def test_validate_approval_transition_accepts_valid_transitions(
    from_state: ApprovalLifecycleState,
    to_state: ApprovalLifecycleState,
) -> None:
    transition = validate_approval_transition(from_state=from_state, to_state=to_state)
    assert transition.from_state is from_state
    assert transition.to_state is to_state


@pytest.mark.unit
def test_validate_approval_transition_rejects_invalid_transition() -> None:
    with pytest.raises(ApprovalLifecycleTransitionError):
        validate_approval_transition(
            from_state=ApprovalLifecycleState.REQUESTED,
            to_state=ApprovalLifecycleState.APPROVED,
        )


@pytest.mark.unit
def test_approval_request_is_frozen_and_rejects_extra_fields() -> None:
    request = _approval_request()
    with pytest.raises(ValidationError):
        request.tenant_id = "other"
    with pytest.raises(ValidationError):
        ApprovalRequest.model_validate(
            {
                "approval_id": str(mint_approval_id()),
                "tenant_id": "tenant-1",
                "workspace_id": "workspace-1",
                "decision_id": str(mint_decision_id()),
                "requested_by_principal_id": "principal-1",
                "requested_at": datetime(2026, 9, 8, 12, 0, tzinfo=_UTC),
                "unexpected": True,
            },
        )


@pytest.mark.unit
def test_approval_request_json_round_trip_preserves_fields() -> None:
    request = _approval_request(
        references=ApprovalReferences(
            work_item_id="work-item-1",
            work_artifact_version_ref=_artifact_version_ref(),
            execution=_execution(),
        ),
    )
    restored = ApprovalRequest.from_json(request.to_json())
    assert restored == request


@pytest.mark.unit
def test_approval_outcome_is_immutable_and_separate_from_lifecycle() -> None:
    outcome = ApprovalOutcome(
        approval_id=str(mint_approval_id()),
        disposition=ApprovalOutcomeDisposition.APPROVED,
        recorded_at=datetime(2026, 9, 8, 12, 30, tzinfo=_UTC),
        summary="Approved after review",
    )
    assert outcome.schema_version == SCHEMA_APPROVAL_OUTCOME_V1
    assert "lifecycle_state" not in ApprovalOutcome.model_fields
    with pytest.raises(ValidationError):
        outcome.disposition = ApprovalOutcomeDisposition.REJECTED
    restored = ApprovalOutcome.from_json(outcome.to_json())
    assert restored == outcome


@pytest.mark.unit
def test_human_approval_action_preserves_acting_principal() -> None:
    action = HumanApprovalAction(
        approval_id=str(mint_approval_id()),
        acting_principal_id="principal-reviewer-1",
        action=HumanApprovalActionType.CONFIRM_APPROVAL,
        timestamp=datetime(2026, 9, 8, 13, 0, tzinfo=_UTC),
        comment_reference="comment-ref-1",
    )
    assert action.schema_version == SCHEMA_HUMAN_APPROVAL_ACTION_V1
    assert action.acting_principal_id == "principal-reviewer-1"
    assert action.action is HumanApprovalActionType.CONFIRM_APPROVAL
    assert action.action is not ApprovalLifecycleState.APPROVED
    restored = HumanApprovalAction.from_json(action.to_json())
    assert restored == action


@pytest.mark.unit
def test_validate_approval_scope_rejects_mismatched_reference_scope() -> None:
    with pytest.raises(ApprovalScopeInvariantError):
        validate_approval_scope(
            tenant_id="tenant-1",
            workspace_id="workspace-1",
            decision_id=mint_decision_id(),
            reference_tenant_id="tenant-2",
        )


@pytest.mark.unit
def test_approval_request_rejects_mismatched_artifact_reference_scope() -> None:
    with pytest.raises(ApprovalScopeInvariantError):
        _approval_request(
            references=ApprovalReferences(
                work_artifact_version_ref=_artifact_version_ref(tenant_id="tenant-2"),
            ),
        )


@pytest.mark.unit
def test_validate_approval_references_rejects_invalid_execution_reference() -> None:
    with pytest.raises(ApprovalReferenceInvariantError):
        validate_approval_references(
            execution={
                "task_id": str(mint_task_id()),
                "run_id": str(mint_run_id()),
                "attempt_id": str(mint_attempt_id()),
            },
        )


@pytest.mark.unit
def test_approval_is_not_decision_substitution() -> None:
    request_fields = set(ApprovalRequest.model_fields)
    forbidden = {
        "lifecycle_state_decision",
        "decision_outcome",
        "decision_lifecycle_state",
        "created_by_principal_id",
        "created_at",
    }
    assert "decision_id" in request_fields
    assert not request_fields.intersection(forbidden)


@pytest.mark.unit
def test_approval_is_not_work_artifact_substitution() -> None:
    request_fields = set(ApprovalRequest.model_fields)
    forbidden = {
        "artifact_status",
        "work_artifact_id",
        "content_ref",
        "current_version_id",
        "payload",
    }
    assert not request_fields.intersection(forbidden)


@pytest.mark.unit
def test_approval_is_not_execution_substitution() -> None:
    request_fields = set(ApprovalRequest.model_fields)
    forbidden = {
        "task_state",
        "execution_status",
        "runtime_state",
        "task_id",
        "run_id",
        "attempt_id",
        "execution_id",
    }
    assert not request_fields.intersection(forbidden)


@pytest.mark.unit
def test_approval_request_creation_valid() -> None:
    request = _approval_request()
    assert request.schema_version == SCHEMA_APPROVAL_REQUEST_V1
    assert request.lifecycle_state is ApprovalLifecycleState.REQUESTED


@pytest.mark.unit
def test_approval_contract_errors_are_typed_domain_exceptions() -> None:
    assert issubclass(ApprovalScopeInvariantError, ApprovalContractInvariantError)
    assert issubclass(ApprovalLifecycleTransitionError, ApprovalContractInvariantError)
    assert not issubclass(ApprovalContractInvariantError, ValueError)
