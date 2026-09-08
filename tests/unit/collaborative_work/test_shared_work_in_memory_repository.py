# © Artur Czarnecki. All rights reserved.

"""Repository contract tests for Collaborative Work WorkItem and Assignment."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAssignmentRepository,
    InMemoryWorkItemRepository,
)
from intergrax.collaborative_work.persistence import open_sqlite_collaborative_work_repositories
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


@pytest.fixture(params=("memory", "sqlite"))
def work_item_repo(request: pytest.FixtureRequest, tmp_path: Path):
    if request.param == "memory":
        yield InMemoryWorkItemRepository()
        return
    bundle = open_sqlite_collaborative_work_repositories(str(tmp_path / "work-item.sqlite"))
    try:
        yield bundle.work_item
    finally:
        bundle.close()


@pytest.fixture(params=("memory", "sqlite"))
def assignment_repo(request: pytest.FixtureRequest, tmp_path: Path):
    if request.param == "memory":
        yield InMemoryAssignmentRepository()
        return
    bundle = open_sqlite_collaborative_work_repositories(str(tmp_path / "assignment.sqlite"))
    try:
        yield bundle.assignment
    finally:
        bundle.close()


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
        (InMemoryWorkItemRepository, WorkItemRepository),
        (InMemoryAssignmentRepository, AssignmentRepository),
    ],
)
def test_repository_protocol_is_satisfied(
    repo_factory: type[object],
    protocol_type: type[object],
) -> None:
    repo = repo_factory()
    assert isinstance(repo, protocol_type)


def test_sqlite_bundle_exposes_shared_work_ports(tmp_path: Path) -> None:
    bundle = open_sqlite_collaborative_work_repositories(str(tmp_path / "bundle.sqlite"))
    try:
        assert isinstance(bundle.work_item, WorkItemRepository)
        assert isinstance(bundle.assignment, AssignmentRepository)
        assert bundle.work_item.capabilities.durable is True
        assert bundle.work_item.capabilities.reference_only is False
        assert bundle.assignment.capabilities.durable is True
        assert bundle.assignment.capabilities.reference_only is False
        assert bundle.work_item.capabilities.backend_id == "collaborative_work.sqlite"
    finally:
        bundle.close()


def test_work_item_create_and_get(work_item_repo: WorkItemRepository) -> None:
    created = work_item_repo.create(_create_work_item_command())
    assert created.revision == INITIAL_RECORD_REVISION
    assert created.work_item_id == "work-item-1"
    assert created.tenant_id == _TENANT_A
    assert created.workspace_id == _WORKSPACE_A
    loaded = work_item_repo.get(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        work_item_id="work-item-1",
    )
    assert loaded == created


def test_work_item_create_normalizes_values(work_item_repo: WorkItemRepository) -> None:
    created = work_item_repo.create(
        _create_work_item_command(
            work_item_id="  work-item-1  ",
            title="  Title  ",
        )
    )
    assert created.work_item_id == "work-item-1"
    assert created.title == "Title"


def test_work_item_duplicate_create_raises(work_item_repo: WorkItemRepository) -> None:
    work_item_repo.create(_create_work_item_command())
    with pytest.raises(WorkItemAlreadyExists):
        work_item_repo.create(_create_work_item_command())


def test_work_item_scoped_isolation_read(work_item_repo: WorkItemRepository) -> None:
    work_item_repo.create(_create_work_item_command())
    assert (
        work_item_repo.get(
            tenant_id=_TENANT_B,
            workspace_id=_WORKSPACE_A,
            work_item_id="work-item-1",
        )
        is None
    )
    assert (
        work_item_repo.get(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_B,
            work_item_id="work-item-1",
        )
        is None
    )


def test_work_item_update_success(work_item_repo: WorkItemRepository) -> None:
    created = work_item_repo.create(_create_work_item_command())
    updated = work_item_repo.update(
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


def test_work_item_stale_revision_conflict_preserves_state(work_item_repo: WorkItemRepository) -> None:
    created = work_item_repo.create(_create_work_item_command())
    work_item_repo.update(
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
        work_item_repo.update(
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
    current = work_item_repo.get(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        work_item_id="work-item-1",
    )
    assert current is not None
    assert current.revision == created.revision + 1
    assert current.state is WorkItemState.ACTIVE


def test_work_item_cross_scope_update_is_not_found(work_item_repo: WorkItemRepository) -> None:
    created = work_item_repo.create(_create_work_item_command())
    with pytest.raises(WorkItemNotFound):
        work_item_repo.update(
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


def test_work_item_idempotency_replay_and_conflict(work_item_repo: WorkItemRepository) -> None:
    command = _create_work_item_command(idempotency_key="work-item-idem")
    first = work_item_repo.create(command)
    second = work_item_repo.create(command)
    assert second == first
    with pytest.raises(WorkItemIdempotencyConflict):
        work_item_repo.create(
            _create_work_item_command(
                idempotency_key="work-item-idem",
                title="Different title",
            )
        )


def test_work_item_idempotency_replay_after_update_returns_original_create(
    work_item_repo: WorkItemRepository,
) -> None:
    command = _create_work_item_command(idempotency_key="work-item-idem-delayed")
    created = work_item_repo.create(command)
    assert created.title == "Title"
    assert created.revision == INITIAL_RECORD_REVISION

    updated = work_item_repo.update(
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

    replayed = work_item_repo.create(command)
    assert replayed == created
    assert replayed.title == "Title"
    assert replayed.revision == INITIAL_RECORD_REVISION

    current = work_item_repo.get(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        work_item_id="work-item-1",
    )
    assert current == updated


def test_work_item_idempotency_key_allowed_in_different_scope(work_item_repo: WorkItemRepository) -> None:
    command_a = _create_work_item_command(idempotency_key="shared-idem")
    command_b = _create_work_item_command(
        tenant_id=_TENANT_B,
        work_item_id="work-item-2",
        idempotency_key="shared-idem",
    )
    first = work_item_repo.create(command_a)
    second = work_item_repo.create(command_b)
    assert first.work_item_id == "work-item-1"
    assert second.work_item_id == "work-item-2"


def test_work_item_concurrent_update_one_wins(work_item_repo: WorkItemRepository) -> None:
    created = work_item_repo.create(_create_work_item_command())
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def attempt() -> None:
        try:
            barrier.wait(timeout=5)
            work_item_repo.update(
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
    final = work_item_repo.get(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        work_item_id="work-item-1",
    )
    assert final is not None
    assert final.revision == created.revision + 1


def test_assignment_create_and_get(assignment_repo: AssignmentRepository) -> None:
    created = assignment_repo.create(_create_assignment_command())
    assert created.revision == INITIAL_RECORD_REVISION
    loaded = assignment_repo.get(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        assignment_id="assignment-1",
    )
    assert loaded == created


def test_assignment_duplicate_create_raises(assignment_repo: AssignmentRepository) -> None:
    assignment_repo.create(_create_assignment_command())
    with pytest.raises(AssignmentAlreadyExists):
        assignment_repo.create(_create_assignment_command())


def test_assignment_scoped_isolation_read(assignment_repo: AssignmentRepository) -> None:
    assignment_repo.create(_create_assignment_command())
    assert (
        assignment_repo.get(
            tenant_id=_TENANT_B,
            workspace_id=_WORKSPACE_A,
            assignment_id="assignment-1",
        )
        is None
    )
    assert (
        assignment_repo.get(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_B,
            assignment_id="assignment-1",
        )
        is None
    )


def test_assignment_multiple_for_same_work_item_and_principals(
    assignment_repo: AssignmentRepository,
) -> None:
    first = assignment_repo.create(_create_assignment_command(assignment_id="assignment-1"))
    second = assignment_repo.create(
        _create_assignment_command(
            assignment_id="assignment-2",
            principal_id="principal-2",
        )
    )
    third = assignment_repo.create(
        _create_assignment_command(
            assignment_id="assignment-3",
            principal_id="principal-1",
        )
    )
    assert first.work_item_id == second.work_item_id == third.work_item_id == "work-item-1"
    assert first.principal_id == third.principal_id == "principal-1"
    assert second.principal_id == "principal-2"


def test_assignment_update_preserves_identity(assignment_repo: AssignmentRepository) -> None:
    created = assignment_repo.create(_create_assignment_command())
    updated = assignment_repo.update(
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


def test_assignment_stale_revision_conflict_preserves_state(assignment_repo: AssignmentRepository) -> None:
    created = assignment_repo.create(_create_assignment_command())
    assignment_repo.update(
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
        assignment_repo.update(
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


def test_assignment_cross_scope_update_is_not_found(assignment_repo: AssignmentRepository) -> None:
    created = assignment_repo.create(_create_assignment_command())
    with pytest.raises(AssignmentNotFound):
        assignment_repo.update(
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


def test_assignment_idempotency_replay_and_conflict(assignment_repo: AssignmentRepository) -> None:
    command = _create_assignment_command(idempotency_key="assignment-idem")
    first = assignment_repo.create(command)
    second = assignment_repo.create(command)
    assert second == first
    with pytest.raises(AssignmentIdempotencyConflict):
        assignment_repo.create(
            _create_assignment_command(
                idempotency_key="assignment-idem",
                principal_id="principal-other",
            )
        )


def test_assignment_idempotency_replay_after_update_returns_original_create(
    assignment_repo: AssignmentRepository,
) -> None:
    command = _create_assignment_command(idempotency_key="assignment-idem-delayed")
    created = assignment_repo.create(command)
    assert created.state is AssignmentState.ACTIVE
    assert created.revision == INITIAL_RECORD_REVISION

    updated = assignment_repo.update(
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

    replayed = assignment_repo.create(command)
    assert replayed == created
    assert replayed.state is AssignmentState.ACTIVE
    assert replayed.revision == INITIAL_RECORD_REVISION


def test_assignment_idempotency_key_allowed_in_different_scope(assignment_repo: AssignmentRepository) -> None:
    command_a = _create_assignment_command(idempotency_key="shared-idem")
    command_b = _create_assignment_command(
        tenant_id=_TENANT_B,
        assignment_id="assignment-2",
        idempotency_key="shared-idem",
    )
    first = assignment_repo.create(command_a)
    second = assignment_repo.create(command_b)
    assert first.assignment_id == "assignment-1"
    assert second.assignment_id == "assignment-2"
