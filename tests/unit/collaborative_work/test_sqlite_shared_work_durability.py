# © Artur Czarnecki. All rights reserved.

"""SQLite durability tests for Collaborative Work MP-2 (COLLAB-WORK-2D)."""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from intergrax.collaborative_work.persistence import open_sqlite_collaborative_work_repositories
from intergrax.collaborative_work.repository import (
    CreateAssignmentCommand,
    CreateWorkItemCommand,
    CreateWorkspaceMembershipCommand,
    INITIAL_RECORD_REVISION,
    UpdateAssignmentCommand,
    UpdateWorkItemCommand,
    WorkItemRevisionConflict,
    WorkItemScopeKey,
    AssignmentScopeKey,
)
from intergrax.collaborative_work.serialization import (
    assignment_from_json,
    assignment_to_json,
    work_item_from_json,
    work_item_to_json,
    workspace_membership_to_json,
)
from intergrax.contracts.collaborative_work import (
    Assignment,
    AssignmentState,
    MembershipStatus,
    WorkItem,
    WorkItemState,
    WorkspaceMembership,
    WorkspaceMembershipRole,
)

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_WORKSPACE = "workspace-a"
_CREATED_AT = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
_UPDATED_AT = datetime(2026, 1, 1, 12, 30, tzinfo=UTC)


def _work_item_command(**overrides: object) -> CreateWorkItemCommand:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
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


def _assignment_command(**overrides: object) -> CreateAssignmentCommand:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
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


def _membership_record() -> WorkspaceMembership:
    return WorkspaceMembership(
        membership_id="membership-legacy",
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id="principal-legacy",
        role=WorkspaceMembershipRole.MEMBER,
        status=MembershipStatus.ACTIVE,
        revision=0,
    )


def _seed_mp1_legacy_database(db_path: Path) -> None:
    connection = sqlite3.connect(str(db_path))
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(
        """
        CREATE TABLE workspace_memberships (
            tenant_id TEXT NOT NULL,
            workspace_id TEXT NOT NULL,
            membership_id TEXT NOT NULL,
            principal_id TEXT NOT NULL,
            record_json TEXT NOT NULL,
            revision INTEGER NOT NULL,
            PRIMARY KEY (tenant_id, workspace_id, membership_id),
            UNIQUE (tenant_id, workspace_id, principal_id)
        );
        CREATE TABLE authority_delegations (
            tenant_id TEXT NOT NULL,
            workspace_id TEXT NOT NULL,
            delegation_id TEXT NOT NULL,
            record_json TEXT NOT NULL,
            revision INTEGER NOT NULL,
            PRIMARY KEY (tenant_id, workspace_id, delegation_id)
        );
        CREATE TABLE collaborative_idempotency (
            tenant_id TEXT NOT NULL,
            workspace_id TEXT NOT NULL,
            entity_kind TEXT NOT NULL,
            idempotency_key TEXT NOT NULL,
            semantic_fingerprint TEXT NOT NULL,
            result_json TEXT NOT NULL,
            PRIMARY KEY (tenant_id, workspace_id, entity_kind, idempotency_key)
        );
        """
    )
    membership = _membership_record()
    connection.execute(
        """
        INSERT INTO workspace_memberships (
            tenant_id, workspace_id, membership_id, principal_id, record_json, revision
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            membership.tenant_id,
            membership.workspace_id,
            membership.membership_id,
            membership.principal_id,
            workspace_membership_to_json(membership),
            membership.revision,
        ),
    )
    connection.execute(
        """
        INSERT INTO collaborative_idempotency (
            tenant_id, workspace_id, entity_kind, idempotency_key,
            semantic_fingerprint, result_json
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            _TENANT,
            _WORKSPACE,
            "workspace_membership",
            "legacy-idem",
            "legacy-fingerprint",
            workspace_membership_to_json(membership),
        ),
    )
    connection.commit()
    connection.close()


def test_work_item_restart_durability(tmp_path: Path) -> None:
    db_path = str(tmp_path / "work-item-restart.sqlite")
    bundle = open_sqlite_collaborative_work_repositories(db_path)
    created = bundle.work_item.create(_work_item_command())
    updated = bundle.work_item.update(
        UpdateWorkItemCommand(
            scope=WorkItemScopeKey(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                work_item_id="work-item-1",
            ),
            expected_revision=created.revision,
            state=WorkItemState.ACTIVE,
            updated_at=_UPDATED_AT,
            title="Updated",
            description="Updated description",
        )
    )
    bundle.close()

    reopened = open_sqlite_collaborative_work_repositories(db_path)
    try:
        loaded = reopened.work_item.get(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            work_item_id="work-item-1",
        )
        assert loaded == updated
    finally:
        reopened.close()


def test_assignment_restart_durability(tmp_path: Path) -> None:
    db_path = str(tmp_path / "assignment-restart.sqlite")
    bundle = open_sqlite_collaborative_work_repositories(db_path)
    created = bundle.assignment.create(_assignment_command())
    updated = bundle.assignment.update(
        UpdateAssignmentCommand(
            scope=AssignmentScopeKey(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                assignment_id="assignment-1",
            ),
            expected_revision=created.revision,
            state=AssignmentState.COMPLETED,
            updated_at=_UPDATED_AT,
        )
    )
    bundle.close()

    reopened = open_sqlite_collaborative_work_repositories(db_path)
    try:
        loaded = reopened.assignment.get(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            assignment_id="assignment-1",
        )
        assert loaded == updated
    finally:
        reopened.close()


def test_work_item_idempotency_survives_restart(tmp_path: Path) -> None:
    db_path = str(tmp_path / "work-item-idem.sqlite")
    command = _work_item_command(idempotency_key="restart-idem")
    bundle = open_sqlite_collaborative_work_repositories(db_path)
    created = bundle.work_item.create(command)
    bundle.work_item.update(
        UpdateWorkItemCommand(
            scope=WorkItemScopeKey(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                work_item_id="work-item-1",
            ),
            expected_revision=created.revision,
            state=WorkItemState.ACTIVE,
            updated_at=_UPDATED_AT,
            title="Updated",
        )
    )
    bundle.close()

    reopened = open_sqlite_collaborative_work_repositories(db_path)
    try:
        replayed = reopened.work_item.create(command)
        assert replayed == created
        assert replayed.revision == INITIAL_RECORD_REVISION
        assert replayed.title == "Title"
    finally:
        reopened.close()


def test_assignment_idempotency_survives_restart(tmp_path: Path) -> None:
    db_path = str(tmp_path / "assignment-idem.sqlite")
    command = _assignment_command(idempotency_key="restart-idem")
    bundle = open_sqlite_collaborative_work_repositories(db_path)
    created = bundle.assignment.create(command)
    bundle.assignment.update(
        UpdateAssignmentCommand(
            scope=AssignmentScopeKey(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                assignment_id="assignment-1",
            ),
            expected_revision=created.revision,
            state=AssignmentState.COMPLETED,
            updated_at=_UPDATED_AT,
        )
    )
    bundle.close()

    reopened = open_sqlite_collaborative_work_repositories(db_path)
    try:
        replayed = reopened.assignment.create(command)
        assert replayed == created
        assert replayed.revision == INITIAL_RECORD_REVISION
        assert replayed.state is AssignmentState.ACTIVE
    finally:
        reopened.close()


def test_work_item_two_connection_concurrent_update(tmp_path: Path) -> None:
    db_path = str(tmp_path / "concurrency.sqlite")
    bundle_a = open_sqlite_collaborative_work_repositories(db_path)
    bundle_b = open_sqlite_collaborative_work_repositories(db_path)
    try:
        created = bundle_a.work_item.create(_work_item_command())
        read_a = bundle_a.work_item.get(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            work_item_id="work-item-1",
        )
        read_b = bundle_b.work_item.get(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            work_item_id="work-item-1",
        )
        assert read_a is not None and read_b is not None
        assert read_a.revision == read_b.revision == created.revision

        errors: list[BaseException] = []
        barrier = threading.Barrier(2)

        def attempt(bundle: object) -> None:
            try:
                barrier.wait(timeout=5)
                bundle.work_item.update(  # type: ignore[attr-defined]
                    UpdateWorkItemCommand(
                        scope=WorkItemScopeKey(
                            tenant_id=_TENANT,
                            workspace_id=_WORKSPACE,
                            work_item_id="work-item-1",
                        ),
                        expected_revision=created.revision,
                        state=WorkItemState.ACTIVE,
                        updated_at=_UPDATED_AT,
                    )
                )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=attempt, args=(bundle_a,)),
            threading.Thread(target=attempt, args=(bundle_b,)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(errors) == 1
        assert isinstance(errors[0], WorkItemRevisionConflict)
        final = bundle_a.work_item.get(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            work_item_id="work-item-1",
        )
        assert final is not None
        assert final.revision == created.revision + 1
    finally:
        bundle_a.close()
        bundle_b.close()


def test_mp2_schema_additive_preserves_mp1_data(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy-mp1.sqlite"
    _seed_mp1_legacy_database(db_path)

    bundle = open_sqlite_collaborative_work_repositories(str(db_path))
    try:
        membership = bundle.membership.get(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id="membership-legacy",
        )
        assert membership is not None
        assert membership.principal_id == "principal-legacy"

        created = bundle.work_item.create(_work_item_command(work_item_id="mp2-work-item"))
        loaded = bundle.work_item.get(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            work_item_id="mp2-work-item",
        )
        assert loaded == created
    finally:
        bundle.close()


def test_closed_store_rejects_mp2_operations(tmp_path: Path) -> None:
    bundle = open_sqlite_collaborative_work_repositories(str(tmp_path / "closed.sqlite"))
    bundle.close()
    with pytest.raises(RuntimeError, match="closed"):
        bundle.work_item.create(_work_item_command())
    with pytest.raises(RuntimeError, match="closed"):
        bundle.assignment.create(_assignment_command())


def test_work_item_serialization_round_trip() -> None:
    record = WorkItem(
        work_item_id="work-item-1",
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        created_by_principal_id="principal-creator",
        state=WorkItemState.ACTIVE,
        revision=2,
        created_at=_CREATED_AT,
        updated_at=_UPDATED_AT,
        title="Title",
        description="Description",
    )
    restored = work_item_from_json(work_item_to_json(record))
    assert restored == record
    assert restored.created_at.tzinfo is not None
    assert restored.updated_at.tzinfo is not None


def test_assignment_serialization_round_trip() -> None:
    record = Assignment(
        assignment_id="assignment-1",
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        work_item_id="work-item-1",
        principal_id="principal-1",
        created_by_principal_id="principal-creator",
        state=AssignmentState.COMPLETED,
        revision=1,
        created_at=_CREATED_AT,
        updated_at=_UPDATED_AT,
    )
    restored = assignment_from_json(assignment_to_json(record))
    assert restored == record
    assert restored.created_at is not None and restored.created_at.tzinfo is not None
    assert restored.updated_at is not None and restored.updated_at.tzinfo is not None
