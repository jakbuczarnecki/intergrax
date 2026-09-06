# © Artur Czarnecki. All rights reserved.

"""Repository tests for Collaborative Work WorkItem and Assignment (COLLAB-WORK-2B)."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAssignmentRepository,
    InMemoryWorkItemRepository,
)
from intergrax.collaborative_work.repository import (
    AssignmentAlreadyExists,
    AssignmentIdempotencyConflict,
    AssignmentNotFound,
    AssignmentRepository,
    AssignmentRevisionConflict,
    AssignmentScopeKey,
    CreateAssignmentCommand,
    CreateWorkItemCommand,
    INITIAL_RECORD_REVISION,
    UpdateAssignmentCommand,
    UpdateWorkItemCommand,
    WorkItemAlreadyExists,
    WorkItemIdempotencyConflict,
    WorkItemNotFound,
    WorkItemRepository,
    WorkItemRevisionConflict,
    WorkItemScopeKey,
)
from intergrax.contracts.collaborative_work import AssignmentState, WorkItemState

pytestmark = pytest.mark.unit

_TENANT_A = "tenant-a"
_TENANT_B = "tenant-b"
_WORKSPACE_A = "workspace-a"
_WORKSPACE_B = "workspace-b"
_CREATED_AT = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
_UPDATED_AT = datetime(2026, 1, 1, 12, 30, tzinfo=UTC)


def _work_item_repo() -> InMemoryWorkItemRepository:
    return InMemoryWorkItemRepository()


def _assignment_repo() -> InMemoryAssignmentRepository:
    return InMemoryAssignmentRepository()


def _create_work_item_command(**overrides: object) -> CreateWorkItemCommand:
    payload = {
        "tenant_id": _TENANT_A,
        "workspace_id": _WORKSPACE_A,
        "work_item_id": "work-item-1",
        "created_by_principal_id": "principal-creator",
        "state": WorkItemState.OPEN,
        "created_at": _CREATED_AT,
        "updated_at": _CREATED_AT,
        "title": "Title",
        "description": "Description",
    }
    payload.update(overrides)
    return CreateWorkItemCommand(**payload)


def _create_assignment_command(**overrides: object) -> CreateAssignmentCommand:
    payload = {
        "tenant_id": _TENANT_A,
        "workspace_id": _WORKSPACE_A,
        "assignment_id": "assignment-1",
        "work_item_id": "work-item-1",
        "principal_id": "principal-1",
        "created_by_principal_id": "principal-creator",
        "state": AssignmentState.ACTIVE,
        "created_at": _CREATED_AT,
        "updated_at": _CREATED_AT,
    }
    payload.update(overrides)
    return CreateAssignmentCommand(**payload)


@pytest.mark.parametrize(
    "repo_factory, protocol_type",
    [
        (_work_item_repo, WorkItemRepository),
        (_assignment_repo, AssignmentRepository),
    ],
)
def test_repository_protocol_is_satisfied(
    repo_factory: object,
    protocol_type: type[object],
) -> None:
    repo = repo_factory()
    assert isinstance(repo, protocol_type)


def test_work_item_create_and_get() -> None:
    repo = _work_item_repo()
    created = repo.create(_create_work_item_command())
    assert created.revision == INITIAL_RECORD_REVISION
    assert created.work_item_id == "work-item-1"
    assert created.tenant_id == _TENANT_A
    assert created.workspace_id == _WORKSPACE_A
    loaded = repo.get(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        work_item_id="work-item-1",
    )
    assert loaded == created


def test_work_item_create_normalizes_values() -> None:
    repo = _work_item_repo()
    created = repo.create(
        _create_work_item_command(
            work_item_id="  work-item-1  ",
            title="  Title  ",
        )
    )
    assert created.work_item_id == "work-item-1"
    assert created.title == "Title"


def test_work_item_duplicate_create_raises() -> None:
    repo = _work_item_repo()
    repo.create(_create_work_item_command())
    with pytest.raises(WorkItemAlreadyExists):
        repo.create(_create_work_item_command())


def test_work_item_scoped_isolation_read() -> None:
    repo = _work_item_repo()
    repo.create(_create_work_item_command())
    assert (
        repo.get(
            tenant_id=_TENANT_B,
            workspace_id=_WORKSPACE_A,
            work_item_id="work-item-1",
        )
        is None
    )
    assert (
        repo.get(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_B,
            work_item_id="work-item-1",
        )
        is None
    )


def test_work_item_update_success() -> None:
    repo = _work_item_repo()
    created = repo.create(_create_work_item_command())
    updated = repo.update(
        UpdateWorkItemCommand(
            scope=WorkItemScopeKey(
                tenant_id=_TENANT_A,
                workspace_id=_WORKSPACE_A,
                work_item_id="work-item-1",
            ),
            expected_revision=created.revision,
            state=WorkItemState.ACTIVE,
            updated_at=_UPDATED_AT,
            title="Updated title",
            description="Updated description",
        )
    )
    assert updated.revision == created.revision + 1
    assert updated.state is WorkItemState.ACTIVE
    assert updated.work_item_id == created.work_item_id
    assert updated.tenant_id == created.tenant_id
    assert updated.workspace_id == created.workspace_id
    assert updated.created_by_principal_id == created.created_by_principal_id
    assert updated.created_at == created.created_at
    assert updated.title == "Updated title"


def test_work_item_stale_revision_conflict_preserves_state() -> None:
    repo = _work_item_repo()
    created = repo.create(_create_work_item_command())
    repo.update(
        UpdateWorkItemCommand(
            scope=WorkItemScopeKey(
                tenant_id=_TENANT_A,
                workspace_id=_WORKSPACE_A,
                work_item_id="work-item-1",
            ),
            expected_revision=created.revision,
            state=WorkItemState.ACTIVE,
            updated_at=_UPDATED_AT,
        )
    )
    with pytest.raises(WorkItemRevisionConflict):
        repo.update(
            UpdateWorkItemCommand(
                scope=WorkItemScopeKey(
                    tenant_id=_TENANT_A,
                    workspace_id=_WORKSPACE_A,
                    work_item_id="work-item-1",
                ),
                expected_revision=created.revision,
                state=WorkItemState.CANCELLED,
                updated_at=_UPDATED_AT + timedelta(minutes=1),
            )
        )
    current = repo.get(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        work_item_id="work-item-1",
    )
    assert current is not None
    assert current.revision == created.revision + 1
    assert current.state is WorkItemState.ACTIVE


def test_work_item_cross_scope_update_is_not_found() -> None:
    repo = _work_item_repo()
    created = repo.create(_create_work_item_command())
    with pytest.raises(WorkItemNotFound):
        repo.update(
            UpdateWorkItemCommand(
                scope=WorkItemScopeKey(
                    tenant_id=_TENANT_A,
                    workspace_id=_WORKSPACE_B,
                    work_item_id="work-item-1",
                ),
                expected_revision=created.revision,
                state=WorkItemState.ACTIVE,
                updated_at=_UPDATED_AT,
            )
        )


def test_work_item_idempotency_replay_and_conflict() -> None:
    repo = _work_item_repo()
    command = _create_work_item_command(idempotency_key="work-item-idem")
    first = repo.create(command)
    second = repo.create(command)
    assert second == first
    with pytest.raises(WorkItemIdempotencyConflict):
        repo.create(
            _create_work_item_command(
                idempotency_key="work-item-idem",
                title="Different title",
            )
        )


def test_work_item_idempotency_replay_after_update_returns_original_create() -> None:
    repo = _work_item_repo()
    command = _create_work_item_command(idempotency_key="work-item-idem-delayed")
    created = repo.create(command)
    assert created.title == "Title"
    assert created.revision == INITIAL_RECORD_REVISION

    updated = repo.update(
        UpdateWorkItemCommand(
            scope=WorkItemScopeKey(
                tenant_id=_TENANT_A,
                workspace_id=_WORKSPACE_A,
                work_item_id="work-item-1",
            ),
            expected_revision=created.revision,
            state=WorkItemState.ACTIVE,
            updated_at=_UPDATED_AT,
            title="Updated title",
        )
    )
    assert updated.revision == created.revision + 1

    replayed = repo.create(command)
    assert replayed == created
    assert replayed.title == "Title"
    assert replayed.revision == INITIAL_RECORD_REVISION

    current = repo.get(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        work_item_id="work-item-1",
    )
    assert current == updated


def test_work_item_idempotency_key_allowed_in_different_scope() -> None:
    repo = _work_item_repo()
    command_a = _create_work_item_command(idempotency_key="shared-idem")
    command_b = _create_work_item_command(
        tenant_id=_TENANT_B,
        work_item_id="work-item-2",
        idempotency_key="shared-idem",
    )
    first = repo.create(command_a)
    second = repo.create(command_b)
    assert first.work_item_id == "work-item-1"
    assert second.work_item_id == "work-item-2"


def test_work_item_concurrent_update_one_wins() -> None:
    repo = _work_item_repo()
    created = repo.create(_create_work_item_command())
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def attempt() -> None:
        try:
            barrier.wait(timeout=5)
            repo.update(
                UpdateWorkItemCommand(
                    scope=WorkItemScopeKey(
                        tenant_id=_TENANT_A,
                        workspace_id=_WORKSPACE_A,
                        work_item_id="work-item-1",
                    ),
                    expected_revision=created.revision,
                    state=WorkItemState.ACTIVE,
                    updated_at=_UPDATED_AT,
                )
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=attempt), threading.Thread(target=attempt)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(errors) == 1
    assert isinstance(errors[0], WorkItemRevisionConflict)
    final = repo.get(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        work_item_id="work-item-1",
    )
    assert final is not None
    assert final.revision == created.revision + 1


def test_assignment_create_and_get() -> None:
    repo = _assignment_repo()
    created = repo.create(_create_assignment_command())
    assert created.revision == INITIAL_RECORD_REVISION
    loaded = repo.get(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        assignment_id="assignment-1",
    )
    assert loaded == created


def test_assignment_duplicate_create_raises() -> None:
    repo = _assignment_repo()
    repo.create(_create_assignment_command())
    with pytest.raises(AssignmentAlreadyExists):
        repo.create(_create_assignment_command())


def test_assignment_scoped_isolation_read() -> None:
    repo = _assignment_repo()
    repo.create(_create_assignment_command())
    assert (
        repo.get(
            tenant_id=_TENANT_B,
            workspace_id=_WORKSPACE_A,
            assignment_id="assignment-1",
        )
        is None
    )
    assert (
        repo.get(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_B,
            assignment_id="assignment-1",
        )
        is None
    )


def test_assignment_multiple_for_same_work_item_and_principals() -> None:
    repo = _assignment_repo()
    first = repo.create(_create_assignment_command(assignment_id="assignment-1"))
    second = repo.create(
        _create_assignment_command(
            assignment_id="assignment-2",
            principal_id="principal-2",
        )
    )
    third = repo.create(
        _create_assignment_command(
            assignment_id="assignment-3",
            principal_id="principal-1",
        )
    )
    assert first.work_item_id == second.work_item_id == third.work_item_id == "work-item-1"
    assert first.principal_id == third.principal_id == "principal-1"
    assert second.principal_id == "principal-2"


def test_assignment_update_preserves_identity() -> None:
    repo = _assignment_repo()
    created = repo.create(_create_assignment_command())
    updated = repo.update(
        UpdateAssignmentCommand(
            scope=AssignmentScopeKey(
                tenant_id=_TENANT_A,
                workspace_id=_WORKSPACE_A,
                assignment_id="assignment-1",
            ),
            expected_revision=created.revision,
            state=AssignmentState.COMPLETED,
            updated_at=_UPDATED_AT,
        )
    )
    assert updated.revision == created.revision + 1
    assert updated.assignment_id == created.assignment_id
    assert updated.work_item_id == created.work_item_id
    assert updated.principal_id == created.principal_id
    assert updated.created_by_principal_id == created.created_by_principal_id
    assert updated.created_at == created.created_at
    assert updated.state is AssignmentState.COMPLETED


def test_assignment_stale_revision_conflict_preserves_state() -> None:
    repo = _assignment_repo()
    created = repo.create(_create_assignment_command())
    repo.update(
        UpdateAssignmentCommand(
            scope=AssignmentScopeKey(
                tenant_id=_TENANT_A,
                workspace_id=_WORKSPACE_A,
                assignment_id="assignment-1",
            ),
            expected_revision=created.revision,
            state=AssignmentState.COMPLETED,
            updated_at=_UPDATED_AT,
        )
    )
    with pytest.raises(AssignmentRevisionConflict):
        repo.update(
            UpdateAssignmentCommand(
                scope=AssignmentScopeKey(
                    tenant_id=_TENANT_A,
                    workspace_id=_WORKSPACE_A,
                    assignment_id="assignment-1",
                ),
                expected_revision=created.revision,
                state=AssignmentState.REVOKED,
                updated_at=_UPDATED_AT + timedelta(minutes=1),
            )
        )


def test_assignment_cross_scope_update_is_not_found() -> None:
    repo = _assignment_repo()
    created = repo.create(_create_assignment_command())
    with pytest.raises(AssignmentNotFound):
        repo.update(
            UpdateAssignmentCommand(
                scope=AssignmentScopeKey(
                    tenant_id=_TENANT_A,
                    workspace_id=_WORKSPACE_B,
                    assignment_id="assignment-1",
                ),
                expected_revision=created.revision,
                state=AssignmentState.REVOKED,
                updated_at=_UPDATED_AT,
            )
        )


def test_assignment_idempotency_replay_and_conflict() -> None:
    repo = _assignment_repo()
    command = _create_assignment_command(idempotency_key="assignment-idem")
    first = repo.create(command)
    second = repo.create(command)
    assert second == first
    with pytest.raises(AssignmentIdempotencyConflict):
        repo.create(
            _create_assignment_command(
                idempotency_key="assignment-idem",
                principal_id="principal-other",
            )
        )


def test_assignment_idempotency_replay_after_update_returns_original_create() -> None:
    repo = _assignment_repo()
    command = _create_assignment_command(idempotency_key="assignment-idem-delayed")
    created = repo.create(command)
    assert created.state is AssignmentState.ACTIVE
    assert created.revision == INITIAL_RECORD_REVISION

    updated = repo.update(
        UpdateAssignmentCommand(
            scope=AssignmentScopeKey(
                tenant_id=_TENANT_A,
                workspace_id=_WORKSPACE_A,
                assignment_id="assignment-1",
            ),
            expected_revision=created.revision,
            state=AssignmentState.COMPLETED,
            updated_at=_UPDATED_AT,
        )
    )
    assert updated.revision == created.revision + 1

    replayed = repo.create(command)
    assert replayed == created
    assert replayed.state is AssignmentState.ACTIVE
    assert replayed.revision == INITIAL_RECORD_REVISION


def test_assignment_idempotency_key_allowed_in_different_scope() -> None:
    repo = _assignment_repo()
    command_a = _create_assignment_command(idempotency_key="shared-idem")
    command_b = _create_assignment_command(
        tenant_id=_TENANT_B,
        assignment_id="assignment-2",
        idempotency_key="shared-idem",
    )
    first = repo.create(command_a)
    second = repo.create(command_b)
    assert first.assignment_id == "assignment-1"
    assert second.assignment_id == "assignment-2"
