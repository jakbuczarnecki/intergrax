# © Artur Czarnecki. All rights reserved.

"""COLLAB-WORK-1H-R3 — SQLite canonical membership schema migration tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from intergrax.collaborative_work.persistence import open_sqlite_collaborative_work_repositories
from intergrax.collaborative_work.repository import (
    CreateWorkspaceMembershipCommand,
    WorkspaceMembershipAlreadyExists,
)
from intergrax.collaborative_work.sqlite_repository import (
    WorkspaceMembershipSchemaMigrationError,
)
from intergrax.contracts.collaborative_work import (
    MembershipStatus,
    WorkspaceMembership,
    WorkspaceMembershipRole,
)

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_WORKSPACE = "workspace-a"
_PRINCIPAL = "principal-acting"
_MEMBERSHIP_1 = "membership-1"
_MEMBERSHIP_2 = "membership-2"
_REBUILD_TABLE = "workspace_memberships_rebuild"


def _membership_record(
    *,
    membership_id: str,
    principal_id: str = _PRINCIPAL,
) -> WorkspaceMembership:
    return WorkspaceMembership(
        membership_id=membership_id,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=principal_id,
        role=WorkspaceMembershipRole.MEMBER,
        status=MembershipStatus.ACTIVE,
        revision=0,
    )


def _create_legacy_memberships_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE workspace_memberships (
            tenant_id TEXT NOT NULL,
            workspace_id TEXT NOT NULL,
            membership_id TEXT NOT NULL,
            record_json TEXT NOT NULL,
            revision INTEGER NOT NULL,
            PRIMARY KEY (tenant_id, workspace_id, membership_id)
        )
        """
    )


def _insert_legacy_membership(
    connection: sqlite3.Connection,
    record: WorkspaceMembership,
) -> None:
    connection.execute(
        """
        INSERT INTO workspace_memberships (
            tenant_id, workspace_id, membership_id, record_json, revision
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            record.tenant_id,
            record.workspace_id,
            record.membership_id,
            record.model_dump_json(),
            record.revision,
        ),
    )


def _open_legacy_db(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    return connection


def _principal_column_notnull(connection: sqlite3.Connection) -> int | None:
    for column in connection.execute("PRAGMA table_info(workspace_memberships)"):
        if column["name"] == "principal_id":
            return int(column["notnull"])
    return None


def _has_unique_principal_constraint(connection: sqlite3.Connection) -> bool:
    for index in connection.execute("PRAGMA index_list(workspace_memberships)"):
        if int(index["unique"]) != 1:
            continue
        index_name = str(index["name"]).replace('"', '""')
        columns = [
            str(info["name"])
            for info in connection.execute(f'PRAGMA index_info("{index_name}")')
        ]
        if columns == ["tenant_id", "workspace_id", "principal_id"]:
            return True
    return False


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row["name"])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }


def test_legacy_sqlite_membership_migration_reaches_canonical_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy-membership.sqlite"
    record = _membership_record(membership_id=_MEMBERSHIP_1)
    setup = _open_legacy_db(db_path)
    try:
        _create_legacy_memberships_table(setup)
        _insert_legacy_membership(setup, record)
        setup.commit()
    finally:
        setup.close()

    bundle = open_sqlite_collaborative_work_repositories(str(db_path))
    try:
        loaded = bundle.membership.get_for_principal(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            principal_id=_PRINCIPAL,
        )
        assert loaded == record
        connection = bundle.store.transaction()
        assert _principal_column_notnull(connection) == 1
        assert _has_unique_principal_constraint(connection)
        assert _REBUILD_TABLE not in _table_names(connection)
        with pytest.raises(WorkspaceMembershipAlreadyExists):
            bundle.membership.create(
                CreateWorkspaceMembershipCommand(
                    tenant_id=_TENANT,
                    workspace_id=_WORKSPACE,
                    membership_id=_MEMBERSHIP_2,
                    principal_id=_PRINCIPAL,
                    role=WorkspaceMembershipRole.MEMBER,
                    status=MembershipStatus.ACTIVE,
                )
            )
    finally:
        bundle.close()

    inspect = _open_legacy_db(db_path)
    try:
        assert _principal_column_notnull(inspect) == 1
        assert _has_unique_principal_constraint(inspect)
        with pytest.raises(sqlite3.IntegrityError):
            inspect.execute(
                """
                INSERT INTO workspace_memberships (
                    tenant_id, workspace_id, membership_id, principal_id,
                    record_json, revision
                ) VALUES (?, ?, ?, NULL, ?, ?)
                """,
                (_TENANT, _WORKSPACE, "membership-null", "{}", 0),
            )
        inspect.rollback()
        with pytest.raises(sqlite3.IntegrityError):
            inspect.execute(
                """
                INSERT INTO workspace_memberships (
                    tenant_id, workspace_id, membership_id, principal_id,
                    record_json, revision
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    _TENANT,
                    _WORKSPACE,
                    _MEMBERSHIP_2,
                    _PRINCIPAL,
                    "{}",
                    0,
                ),
            )
        inspect.rollback()
    finally:
        inspect.close()

    reopened = open_sqlite_collaborative_work_repositories(str(db_path))
    try:
        loaded_again = reopened.membership.get_for_principal(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            principal_id=_PRINCIPAL,
        )
        assert loaded_again == record
        connection = reopened.store.transaction()
        assert _principal_column_notnull(connection) == 1
        assert _has_unique_principal_constraint(connection)
    finally:
        reopened.close()


def test_nullable_principal_id_column_is_rebuilt_to_not_null(tmp_path: Path) -> None:
    db_path = tmp_path / "nullable-membership.sqlite"
    record = _membership_record(membership_id=_MEMBERSHIP_1)
    setup = _open_legacy_db(db_path)
    try:
        _create_legacy_memberships_table(setup)
        _insert_legacy_membership(setup, record)
        setup.execute("ALTER TABLE workspace_memberships ADD COLUMN principal_id TEXT")
        setup.commit()
    finally:
        setup.close()

    bundle = open_sqlite_collaborative_work_repositories(str(db_path))
    try:
        loaded = bundle.membership.get_for_principal(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            principal_id=_PRINCIPAL,
        )
        assert loaded == record
        assert _principal_column_notnull(bundle.store.transaction()) == 1
        assert _has_unique_principal_constraint(bundle.store.transaction())
    finally:
        bundle.close()


def test_duplicate_legacy_principal_membership_fails_and_preserves_original(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "duplicate-membership.sqlite"
    first = _membership_record(membership_id=_MEMBERSHIP_1)
    second = _membership_record(membership_id=_MEMBERSHIP_2)
    setup = _open_legacy_db(db_path)
    try:
        _create_legacy_memberships_table(setup)
        _insert_legacy_membership(setup, first)
        _insert_legacy_membership(setup, second)
        setup.commit()
        original_rows = setup.execute(
            """
            SELECT membership_id, record_json, revision
            FROM workspace_memberships
            ORDER BY membership_id
            """
        ).fetchall()
        original_payload = [
            (row["membership_id"], row["record_json"], int(row["revision"]))
            for row in original_rows
        ]
    finally:
        setup.close()

    with pytest.raises(WorkspaceMembershipSchemaMigrationError, match="duplicate canonical"):
        open_sqlite_collaborative_work_repositories(str(db_path))

    verify = _open_legacy_db(db_path)
    try:
        assert _principal_column_notnull(verify) is None
        assert _REBUILD_TABLE not in _table_names(verify)
        preserved = [
            (row["membership_id"], row["record_json"], int(row["revision"]))
            for row in verify.execute(
                """
                SELECT membership_id, record_json, revision
                FROM workspace_memberships
                ORDER BY membership_id
                """
            )
        ]
        assert preserved == original_payload
    finally:
        verify.close()
