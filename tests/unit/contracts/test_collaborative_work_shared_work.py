# © Artur Czarnecki. All rights reserved.

"""COLLAB-WORK-2A — MP-2 Shared Work contract and lifecycle tests."""

from __future__ import annotations

import ast
import importlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from intergrax.contracts.collaborative_work import (
    SCHEMA_ASSIGNMENT_TRANSITION_REQUEST_V1,
    SCHEMA_ASSIGNMENT_V1,
    SCHEMA_WORK_ITEM_TRANSITION_REQUEST_V1,
    SCHEMA_WORK_ITEM_V1,
    Assignment,
    AssignmentState,
    AssignmentTransitionRequest,
    CollaborativeWorkLifecycleError,
    WorkItem,
    WorkItemState,
    WorkItemTransitionRequest,
    apply_assignment_transition,
    apply_work_item_transition,
    is_work_item_reopen_transition,
    validate_assignment_state_transition,
    validate_work_item_state_transition,
    work_item_resource_scope,
)
from intergrax.runtime.task.task_state import TaskState

_UTC = timezone.utc
_NOW = datetime(2026, 9, 6, 12, 0, tzinfo=_UTC)
_LATER = _NOW + timedelta(minutes=5)

_CANONICAL_WORK_ITEM_STATES = (
    WorkItemState.OPEN,
    WorkItemState.ACTIVE,
    WorkItemState.COMPLETED,
    WorkItemState.CANCELLED,
)

_CANONICAL_ASSIGNMENT_STATES = (
    AssignmentState.ACTIVE,
    AssignmentState.REVOKED,
    AssignmentState.COMPLETED,
)

_FORBIDDEN_EXECUTION_STATE_NAMES = frozenset(
    {
        "RUNNING",
        "FAILED",
        "RETRYING",
        "WAITING_FOR_HUMAN",
        "BLOCKED",
    },
)

_ALLOWED_WORK_ITEM_TRANSITIONS = {
    (WorkItemState.OPEN, WorkItemState.ACTIVE),
    (WorkItemState.OPEN, WorkItemState.CANCELLED),
    (WorkItemState.ACTIVE, WorkItemState.COMPLETED),
    (WorkItemState.ACTIVE, WorkItemState.CANCELLED),
    (WorkItemState.COMPLETED, WorkItemState.ACTIVE),
    (WorkItemState.CANCELLED, WorkItemState.ACTIVE),
}

_ALLOWED_ASSIGNMENT_TRANSITIONS = {
    (AssignmentState.ACTIVE, AssignmentState.REVOKED),
    (AssignmentState.ACTIVE, AssignmentState.COMPLETED),
}


def _work_item(**overrides: object) -> WorkItem:
    payload = {
        "work_item_id": "work-item-1",
        "tenant_id": "tenant-1",
        "workspace_id": "workspace-1",
        "created_by_principal_id": "principal-creator",
        "state": WorkItemState.OPEN,
        "revision": 0,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    payload.update(overrides)
    return WorkItem.model_validate(payload)


def _assignment(**overrides: object) -> Assignment:
    payload = {
        "assignment_id": "assignment-1",
        "tenant_id": "tenant-1",
        "workspace_id": "workspace-1",
        "work_item_id": "work-item-1",
        "principal_id": "principal-a",
        "created_by_principal_id": "principal-creator",
        "state": AssignmentState.ACTIVE,
        "revision": 0,
    }
    payload.update(overrides)
    return Assignment.model_validate(payload)


def _work_item_transition(**overrides: object) -> WorkItemTransitionRequest:
    payload = {
        "tenant_id": "tenant-1",
        "workspace_id": "workspace-1",
        "work_item_id": "work-item-1",
        "expected_revision": 0,
        "target_state": WorkItemState.ACTIVE,
        "acting_principal_id": "principal-actor",
        "idempotency_key": "idem-work-item-1",
    }
    payload.update(overrides)
    return WorkItemTransitionRequest.model_validate(payload)


def _assignment_transition(**overrides: object) -> AssignmentTransitionRequest:
    payload = {
        "tenant_id": "tenant-1",
        "workspace_id": "workspace-1",
        "assignment_id": "assignment-1",
        "work_item_id": "work-item-1",
        "expected_revision": 0,
        "target_state": AssignmentState.REVOKED,
        "acting_principal_id": "principal-actor",
        "idempotency_key": "idem-assignment-1",
    }
    payload.update(overrides)
    return AssignmentTransitionRequest.model_validate(payload)


@pytest.mark.unit
def test_work_item_schema_version_is_stable() -> None:
    work_item = _work_item()
    assert work_item.schema_version == SCHEMA_WORK_ITEM_V1


@pytest.mark.unit
def test_work_item_valid_construction_and_normalization() -> None:
    work_item = _work_item(
        work_item_id="  work-item-1  ",
        title="  Shared task  ",
        description="  Details  ",
    )
    assert work_item.work_item_id == "work-item-1"
    assert work_item.title == "Shared task"
    assert work_item.description == "Details"


@pytest.mark.unit
def test_work_item_is_frozen_and_rejects_extra_fields() -> None:
    work_item = _work_item()
    with pytest.raises(ValidationError):
        work_item.work_item_id = "other"
    with pytest.raises(ValidationError):
        _work_item(task_id="task-1")


@pytest.mark.unit
@pytest.mark.parametrize(
    "field_name",
    ["work_item_id", "tenant_id", "workspace_id", "created_by_principal_id"],
)
def test_work_item_rejects_empty_ids(field_name: str) -> None:
    with pytest.raises(ValidationError):
        _work_item(**{field_name: "   "})


@pytest.mark.unit
def test_work_item_revision_validation() -> None:
    assert _work_item(revision=3).revision == 3
    with pytest.raises(ValidationError):
        _work_item(revision=-1)


@pytest.mark.unit
def test_work_item_timestamp_validation() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        _work_item(created_at=datetime(2026, 9, 6, 12, 0))
    with pytest.raises(ValidationError, match="updated_at must be greater"):
        _work_item(updated_at=_NOW - timedelta(seconds=1))


@pytest.mark.unit
def test_assignment_schema_version_is_stable() -> None:
    assignment = _assignment()
    assert assignment.schema_version == SCHEMA_ASSIGNMENT_V1


@pytest.mark.unit
def test_assignment_valid_construction() -> None:
    assignment = _assignment()
    assert assignment.work_item_id == "work-item-1"
    assert assignment.principal_id == "principal-a"


@pytest.mark.unit
def test_multiple_assignments_per_work_item_are_supported() -> None:
    first = _assignment(assignment_id="assignment-1", principal_id="principal-a")
    second = _assignment(assignment_id="assignment-2", principal_id="principal-b")
    assert first.work_item_id == second.work_item_id
    assert first.principal_id != second.principal_id


@pytest.mark.unit
@pytest.mark.parametrize(
    "field_name",
    [
        "assignment_id",
        "tenant_id",
        "workspace_id",
        "work_item_id",
        "principal_id",
        "created_by_principal_id",
    ],
)
def test_assignment_rejects_empty_ids(field_name: str) -> None:
    with pytest.raises(ValidationError):
        _assignment(**{field_name: ""})


@pytest.mark.unit
def test_assignment_revision_validation_and_immutability() -> None:
    assignment = _assignment(revision=2)
    assert assignment.revision == 2
    with pytest.raises(ValidationError):
        _assignment(revision=-1)
    with pytest.raises(ValidationError):
        assignment.revision = 3


@pytest.mark.unit
def test_transition_request_schema_versions_are_stable() -> None:
    assert (
        _work_item_transition().schema_version == SCHEMA_WORK_ITEM_TRANSITION_REQUEST_V1
    )
    assert (
        _assignment_transition().schema_version == SCHEMA_ASSIGNMENT_TRANSITION_REQUEST_V1
    )


@pytest.mark.unit
def test_work_item_state_vocabulary_is_conservative() -> None:
    assert tuple(WorkItemState) == _CANONICAL_WORK_ITEM_STATES
    assert frozenset(state.name for state in WorkItemState).isdisjoint(
        _FORBIDDEN_EXECUTION_STATE_NAMES,
    )


@pytest.mark.unit
def test_assignment_state_vocabulary_is_collaborative_only() -> None:
    assert tuple(AssignmentState) == _CANONICAL_ASSIGNMENT_STATES
    assert frozenset(state.name for state in AssignmentState).isdisjoint(
        {"RUNNING", "FAILED", "RETRYING"},
    )


@pytest.mark.unit
@pytest.mark.parametrize("from_state,to_state", sorted(_ALLOWED_WORK_ITEM_TRANSITIONS))
def test_allowed_work_item_transitions(from_state: WorkItemState, to_state: WorkItemState) -> None:
    transition = validate_work_item_state_transition(from_state=from_state, to_state=to_state)
    assert transition.from_state is from_state
    assert transition.to_state is to_state


@pytest.mark.unit
@pytest.mark.parametrize(
    "from_state,to_state",
    [
        (WorkItemState.OPEN, WorkItemState.COMPLETED),
        (WorkItemState.OPEN, WorkItemState.OPEN),
        (WorkItemState.ACTIVE, WorkItemState.OPEN),
        (WorkItemState.COMPLETED, WorkItemState.COMPLETED),
        (WorkItemState.CANCELLED, WorkItemState.CANCELLED),
        (WorkItemState.COMPLETED, WorkItemState.CANCELLED),
    ],
)
def test_invalid_work_item_transitions_rejected(
    from_state: WorkItemState,
    to_state: WorkItemState,
) -> None:
    with pytest.raises(CollaborativeWorkLifecycleError):
        validate_work_item_state_transition(from_state=from_state, to_state=to_state)


@pytest.mark.unit
def test_work_item_reopen_semantics_are_explicit() -> None:
    completed_reopen = validate_work_item_state_transition(
        from_state=WorkItemState.COMPLETED,
        to_state=WorkItemState.ACTIVE,
    )
    cancelled_reopen = validate_work_item_state_transition(
        from_state=WorkItemState.CANCELLED,
        to_state=WorkItemState.ACTIVE,
    )
    assert is_work_item_reopen_transition(completed_reopen)
    assert is_work_item_reopen_transition(cancelled_reopen)

    activate = validate_work_item_state_transition(
        from_state=WorkItemState.OPEN,
        to_state=WorkItemState.ACTIVE,
    )
    assert not is_work_item_reopen_transition(activate)


@pytest.mark.unit
@pytest.mark.parametrize("from_state,to_state", sorted(_ALLOWED_ASSIGNMENT_TRANSITIONS))
def test_allowed_assignment_transitions(
    from_state: AssignmentState,
    to_state: AssignmentState,
) -> None:
    transition = validate_assignment_state_transition(from_state=from_state, to_state=to_state)
    assert transition.from_state is from_state
    assert transition.to_state is to_state


@pytest.mark.unit
@pytest.mark.parametrize(
    "from_state,to_state",
    [
        (AssignmentState.ACTIVE, AssignmentState.ACTIVE),
        (AssignmentState.REVOKED, AssignmentState.ACTIVE),
        (AssignmentState.COMPLETED, AssignmentState.ACTIVE),
        (AssignmentState.REVOKED, AssignmentState.COMPLETED),
    ],
)
def test_invalid_assignment_transitions_rejected(
    from_state: AssignmentState,
    to_state: AssignmentState,
) -> None:
    with pytest.raises(CollaborativeWorkLifecycleError):
        validate_assignment_state_transition(from_state=from_state, to_state=to_state)


@pytest.mark.unit
def test_apply_work_item_transition_increments_revision_and_updates_state() -> None:
    work_item = _work_item(state=WorkItemState.OPEN, revision=0)
    request = _work_item_transition(
        expected_revision=0,
        target_state=WorkItemState.ACTIVE,
    )
    updated = apply_work_item_transition(work_item, request, updated_at=_LATER)
    assert updated.state is WorkItemState.ACTIVE
    assert updated.revision == 1
    assert updated.updated_at == _LATER


@pytest.mark.unit
def test_apply_work_item_transition_rejects_stale_revision() -> None:
    work_item = _work_item(revision=1)
    request = _work_item_transition(expected_revision=0)
    with pytest.raises(CollaborativeWorkLifecycleError, match="expected_revision"):
        apply_work_item_transition(work_item, request, updated_at=_LATER)


@pytest.mark.unit
def test_apply_work_item_transition_rejects_scope_mismatch() -> None:
    work_item = _work_item()
    request = _work_item_transition(work_item_id="other-work-item")
    with pytest.raises(CollaborativeWorkLifecycleError, match="work_item_id"):
        apply_work_item_transition(work_item, request, updated_at=_LATER)


@pytest.mark.unit
def test_apply_assignment_transition_updates_state_and_revision() -> None:
    assignment = _assignment(revision=0)
    request = _assignment_transition(
        expected_revision=0,
        target_state=AssignmentState.COMPLETED,
    )
    updated = apply_assignment_transition(assignment, request, updated_at=_LATER)
    assert updated.state is AssignmentState.COMPLETED
    assert updated.revision == 1
    assert updated.updated_at == _LATER


@pytest.mark.unit
def test_work_item_resource_scope_convention() -> None:
    assert work_item_resource_scope(work_item_id="work-item-1") == "work_item:work-item-1"
    with pytest.raises(ValueError):
        work_item_resource_scope(work_item_id="   ")


@pytest.mark.unit
def test_work_item_state_is_not_task_state() -> None:
    assert WorkItemState is not TaskState
    assert {state.value for state in WorkItemState}.isdisjoint(
        {
            TaskState.RUNNING.value,
            TaskState.FAILED.value,
            TaskState.WAITING_FOR_HUMAN.value,
            TaskState.CREATED.value,
        },
    )


@pytest.mark.unit
def test_work_item_has_no_execution_identity_fields() -> None:
    field_names = set(WorkItem.model_fields)
    forbidden = {"task_id", "run_id", "attempt_id", "assignee_id", "metadata", "payload"}
    assert forbidden.isdisjoint(field_names)


@pytest.mark.unit
def test_assignment_has_no_execution_identity_fields() -> None:
    field_names = set(Assignment.model_fields)
    forbidden = {"task_id", "run_id", "attempt_id", "agent_id", "metadata", "payload"}
    assert forbidden.isdisjoint(field_names)


@pytest.mark.unit
def test_collaborative_work_contract_module_has_no_nexus_task_imports() -> None:
    module = importlib.import_module("intergrax.contracts.collaborative_work")
    assert module.__file__ is not None
    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    joined = "\n".join(imported)
    assert "runtime.task" not in joined
    assert "TaskState" not in joined
    assert "AgentAssignment" not in joined


@pytest.mark.unit
def test_work_item_does_not_subclass_task() -> None:
    from pydantic import BaseModel as PydanticBaseModel
    from intergrax.runtime.task.task import Task

    assert issubclass(WorkItem, PydanticBaseModel)
    assert not issubclass(WorkItem, Task)
