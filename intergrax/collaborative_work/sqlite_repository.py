# © Artur Czarnecki. All rights reserved.

"""Durable SQL adapter for Collaborative Work authoritative state (COLLAB-WORK-1H)."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from intergrax.collaborative_work.repository import (
    AssignmentAlreadyExists,
    AssignmentIdempotencyConflict,
    AssignmentNotFound,
    AssignmentRevisionConflict,
    AuthorityDelegationAlreadyExists,
    AuthorityDelegationIdempotencyConflict,
    AuthorityDelegationNotFound,
    AuthorityDelegationRevisionConflict,
    CollaborativeOperationPolicyProfileAlreadyExists,
    CollaborativeOperationPolicyProfileIdempotencyConflict,
    CollaborativeOperationPolicyProfileNotFound,
    CollaborativeOperationPolicyProfileRevisionConflict,
    CollaborativePolicyRuleAlreadyExists,
    CollaborativePolicyRuleIdempotencyConflict,
    CollaborativePolicyRuleNotFound,
    CollaborativePolicyRuleRevisionConflict,
    CollaborativeWorkRepositoryCapabilities,
    CreateAssignmentCommand,
    CreateAuthorityDelegationCommand,
    CreateCollaborativeOperationPolicyProfileCommand,
    CreateCollaborativePolicyRuleCommand,
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkItemCommand,
    CreateWorkspaceMembershipCommand,
    INITIAL_RECORD_REVISION,
    PrincipalAuthorityGrantAlreadyExists,
    PrincipalAuthorityGrantIdempotencyConflict,
    PrincipalAuthorityGrantNotFound,
    PrincipalAuthorityGrantRevisionConflict,
    UpdateAssignmentCommand,
    UpdateAuthorityDelegationCommand,
    UpdateCollaborativeOperationPolicyProfileCommand,
    UpdateCollaborativePolicyRuleCommand,
    UpdatePrincipalAuthorityGrantCommand,
    UpdateWorkItemCommand,
    UpdateWorkspaceMembershipCommand,
    WorkItemAlreadyExists,
    WorkItemIdempotencyConflict,
    WorkItemNotFound,
    WorkItemRevisionConflict,
    WorkspaceMembershipAlreadyExists,
    WorkspaceMembershipIdempotencyConflict,
    WorkspaceMembershipNotFound,
    WorkspaceMembershipRevisionConflict,
)
from intergrax.collaborative_work.serialization import (
    assignment_from_json,
    assignment_to_json,
    authority_delegation_from_json,
    authority_delegation_to_json,
    collaborative_policy_rule_from_json,
    collaborative_policy_rule_to_json,
    operation_policy_profile_from_json,
    operation_policy_profile_to_json,
    principal_authority_grant_from_json,
    principal_authority_grant_to_json,
    work_item_from_json,
    work_item_to_json,
    workspace_membership_from_json,
    workspace_membership_to_json,
)
from intergrax.contracts.collaborative_work import (
    Assignment,
    AuthorityDelegation,
    CollaborativeOperationPolicyProfile,
    CollaborativePolicyRule,
    PolicyCompositionLayer,
    PrincipalAuthorityGrant,
    WorkItem,
    WorkspaceMembership,
)

_CLOSED_ERROR = "Collaborative Work repository store is closed"
_BACKEND_ID = "collaborative_work.sqlite"
_CAPABILITIES = CollaborativeWorkRepositoryCapabilities(
    backend_id=_BACKEND_ID,
    durable=True,
    reference_only=False,
)
_WORKSPACE_MEMBERSHIPS_REBUILD_TABLE = "workspace_memberships_rebuild"
_WORKSPACE_MEMBERSHIPS_TABLE_BODY = """
                    tenant_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    membership_id TEXT NOT NULL,
                    principal_id TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    PRIMARY KEY (tenant_id, workspace_id, membership_id),
                    UNIQUE (tenant_id, workspace_id, principal_id)
"""

TRecord = TypeVar("TRecord")


class WorkspaceMembershipSchemaMigrationError(RuntimeError):
    """Raised when legacy workspace_memberships cannot be rebuilt to canonical schema."""


class SQLiteCollaborativeWorkStore:
    """Shared SQLite connection and schema for Collaborative Work repositories."""

    def __init__(self, db_path: str) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            str(path),
            timeout=30.0,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 30000")
        self._lock = threading.RLock()
        self._closed = False
        try:
            self._initialize_schema()
        except Exception:
            try:
                self._connection.close()
            except Exception:
                pass
            raise

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._connection.close()

    def transaction(self) -> sqlite3.Connection:
        self._ensure_open()
        return self._connection

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(_CLOSED_ERROR)

    def _initialize_schema(self) -> None:
        with self._lock:
            self._ensure_open()
            self._connection.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS workspace_memberships (
                    {_WORKSPACE_MEMBERSHIPS_TABLE_BODY}
                );

                CREATE TABLE IF NOT EXISTS authority_delegations (
                    tenant_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    delegation_id TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    PRIMARY KEY (tenant_id, workspace_id, delegation_id)
                );

                CREATE TABLE IF NOT EXISTS principal_authority_grants (
                    tenant_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    authority_grant_id TEXT NOT NULL,
                    principal_id TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    PRIMARY KEY (tenant_id, workspace_id, authority_grant_id),
                    UNIQUE (tenant_id, workspace_id, principal_id)
                );

                CREATE TABLE IF NOT EXISTS collaborative_policy_rules (
                    tenant_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    policy_rule_id TEXT NOT NULL,
                    layer TEXT NOT NULL,
                    authority_scope TEXT NOT NULL,
                    resource_scope TEXT,
                    resource_scope_key TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    PRIMARY KEY (tenant_id, workspace_id, policy_rule_id),
                    UNIQUE (
                        tenant_id,
                        workspace_id,
                        layer,
                        authority_scope,
                        resource_scope_key
                    )
                );

                CREATE TABLE IF NOT EXISTS collaborative_operation_policy_profiles (
                    tenant_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    operation_id TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    PRIMARY KEY (tenant_id, workspace_id, operation_id)
                );

                CREATE TABLE IF NOT EXISTS collaborative_idempotency (
                    tenant_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    entity_kind TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    semantic_fingerprint TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, workspace_id, entity_kind, idempotency_key)
                );

                CREATE TABLE IF NOT EXISTS work_items (
                    tenant_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    work_item_id TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    PRIMARY KEY (tenant_id, workspace_id, work_item_id)
                );

                CREATE TABLE IF NOT EXISTS assignments (
                    tenant_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    assignment_id TEXT NOT NULL,
                    work_item_id TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    PRIMARY KEY (tenant_id, workspace_id, assignment_id)
                );

                CREATE INDEX IF NOT EXISTS idx_assignments_work_item
                    ON assignments (tenant_id, workspace_id, work_item_id);
                """
            )
            self._migrate_workspace_memberships_principal_column()

    def _migrate_workspace_memberships_principal_column(self) -> None:
        columns = self._connection.execute("PRAGMA table_info(workspace_memberships)").fetchall()
        if not columns:
            return
        if not self._workspace_memberships_requires_canonical_rebuild(columns):
            return
        self._rebuild_workspace_memberships_canonical_schema()

    def _workspace_memberships_requires_canonical_rebuild(
        self,
        columns: list[sqlite3.Row],
    ) -> bool:
        for column in columns:
            if column["name"] == "principal_id":
                return int(column["notnull"]) != 1
        return True

    def _rebuild_workspace_memberships_canonical_schema(self) -> None:
        previous_isolation = self._connection.isolation_level
        self._connection.isolation_level = None
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._connection.execute(
                    f"DROP TABLE IF EXISTS {_WORKSPACE_MEMBERSHIPS_REBUILD_TABLE}"
                )
                self._connection.execute(
                    f"""
                    CREATE TABLE {_WORKSPACE_MEMBERSHIPS_REBUILD_TABLE} (
                        {_WORKSPACE_MEMBERSHIPS_TABLE_BODY}
                    )
                    """
                )
                source_rows = self._connection.execute(
                    """
                    SELECT tenant_id, workspace_id, membership_id, record_json, revision
                    FROM workspace_memberships
                    """
                ).fetchall()
                migrated_rows = self._canonical_membership_rows(source_rows)
                self._connection.executemany(
                    f"""
                    INSERT INTO {_WORKSPACE_MEMBERSHIPS_REBUILD_TABLE} (
                        tenant_id, workspace_id, membership_id, principal_id,
                        record_json, revision
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    migrated_rows,
                )
                inserted = self._connection.execute(
                    f"SELECT COUNT(*) AS row_count FROM {_WORKSPACE_MEMBERSHIPS_REBUILD_TABLE}"
                ).fetchone()
                if inserted is None or int(inserted["row_count"]) != len(source_rows):
                    raise WorkspaceMembershipSchemaMigrationError(
                        "workspace_memberships migration failed: row count mismatch"
                    )
                self._connection.execute("DROP TABLE workspace_memberships")
                self._connection.execute(
                    "ALTER TABLE "
                    f"{_WORKSPACE_MEMBERSHIPS_REBUILD_TABLE} "
                    "RENAME TO workspace_memberships"
                )
                self._connection.execute("COMMIT")
            except Exception as exc:
                self._connection.execute("ROLLBACK")
                if isinstance(exc, WorkspaceMembershipSchemaMigrationError):
                    raise
                raise WorkspaceMembershipSchemaMigrationError(
                    "workspace_memberships migration failed: canonical rebuild could not complete"
                ) from exc
        finally:
            self._connection.isolation_level = previous_isolation

    def _canonical_membership_rows(
        self,
        source_rows: list[sqlite3.Row],
    ) -> list[tuple[str, str, str, str, str, int]]:
        migrated_rows: list[tuple[str, str, str, str, str, int]] = []
        seen_principals: set[tuple[str, str, str]] = set()
        for row in source_rows:
            try:
                membership = workspace_membership_from_json(row["record_json"])
            except Exception as exc:
                raise WorkspaceMembershipSchemaMigrationError(
                    "workspace_memberships migration failed: record_json could not be deserialized"
                ) from exc
            tenant_id = membership.tenant_id.strip()
            workspace_id = membership.workspace_id.strip()
            membership_id = membership.membership_id.strip()
            principal_id = membership.principal_id.strip()
            if tenant_id != row["tenant_id"] or workspace_id != row["workspace_id"]:
                raise WorkspaceMembershipSchemaMigrationError(
                    "workspace_memberships migration failed: record identity does not match table keys"
                )
            if membership_id != row["membership_id"]:
                raise WorkspaceMembershipSchemaMigrationError(
                    "workspace_memberships migration failed: record identity does not match table keys"
                )
            if principal_id == "":
                raise WorkspaceMembershipSchemaMigrationError(
                    "workspace_memberships migration failed: canonical principal_id is empty"
                )
            principal_key = (tenant_id, workspace_id, principal_id)
            if principal_key in seen_principals:
                raise WorkspaceMembershipSchemaMigrationError(
                    "workspace_memberships migration failed: duplicate canonical principal membership"
                )
            seen_principals.add(principal_key)
            migrated_rows.append(
                (
                    tenant_id,
                    workspace_id,
                    membership_id,
                    principal_id,
                    row["record_json"],
                    int(row["revision"]),
                )
            )
        return migrated_rows


def _scope_matches_tenant_workspace(
    record_tenant: str,
    record_workspace: str,
    *,
    tenant_id: str,
    workspace_id: str,
) -> bool:
    return record_tenant == tenant_id.strip() and record_workspace == workspace_id.strip()


def _resource_scope_key(resource_scope: str | None) -> str:
    return "" if resource_scope is None else resource_scope.strip()


class _IdempotencyMixin:
    _store: SQLiteCollaborativeWorkStore
    _entity_kind: str

    def _load_idempotency(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        idempotency_key: str,
        decode: Callable[[str], TRecord],
    ) -> tuple[str, TRecord] | None:
        row = self._store.transaction().execute(
            """
            SELECT semantic_fingerprint, result_json
            FROM collaborative_idempotency
            WHERE tenant_id = ? AND workspace_id = ? AND entity_kind = ? AND idempotency_key = ?
            """,
            (tenant_id.strip(), workspace_id.strip(), self._entity_kind, idempotency_key.strip()),
        ).fetchone()
        if row is None:
            return None
        return row["semantic_fingerprint"], decode(row["result_json"])

    def _store_idempotency(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        idempotency_key: str,
        fingerprint: str,
        result_json: str,
    ) -> None:
        self._store.transaction().execute(
            """
            INSERT INTO collaborative_idempotency (
                tenant_id, workspace_id, entity_kind, idempotency_key,
                semantic_fingerprint, result_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                tenant_id.strip(),
                workspace_id.strip(),
                self._entity_kind,
                idempotency_key.strip(),
                fingerprint,
                result_json,
            ),
        )


class SQLiteWorkspaceMembershipRepository(_IdempotencyMixin):
    _entity_kind = "workspace_membership"

    def __init__(self, store: SQLiteCollaborativeWorkStore) -> None:
        self._store = store

    @property
    def capabilities(self) -> CollaborativeWorkRepositoryCapabilities:
        return _CAPABILITIES

    def create(self, command: CreateWorkspaceMembershipCommand) -> WorkspaceMembership:
        with self._store._lock:
            self._store._ensure_open()
            self._store.transaction().execute("BEGIN IMMEDIATE")
            try:
                if command.idempotency_key is not None:
                    replay = self._replay_create(command)
                    if replay is not None:
                        self._store.transaction().commit()
                        return replay

                existing = self._get_in_transaction(
                    tenant_id=command.tenant_id,
                    workspace_id=command.workspace_id,
                    membership_id=command.membership_id,
                )
                if existing is not None:
                    raise WorkspaceMembershipAlreadyExists("workspace membership already exists")

                principal_existing = self._get_for_principal_in_transaction(
                    tenant_id=command.tenant_id,
                    workspace_id=command.workspace_id,
                    principal_id=command.principal_id,
                )
                if principal_existing is not None:
                    raise WorkspaceMembershipAlreadyExists("workspace membership already exists")

                record = WorkspaceMembership(
                    membership_id=command.membership_id,
                    tenant_id=command.tenant_id,
                    workspace_id=command.workspace_id,
                    principal_id=command.principal_id,
                    role=command.role,
                    status=command.status,
                    revision=INITIAL_RECORD_REVISION,
                )
                result_json = workspace_membership_to_json(record)
                self._store.transaction().execute(
                    """
                    INSERT INTO workspace_memberships (
                        tenant_id, workspace_id, membership_id, principal_id, record_json, revision
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.tenant_id.strip(),
                        record.workspace_id.strip(),
                        record.membership_id.strip(),
                        record.principal_id.strip(),
                        result_json,
                        record.revision,
                    ),
                )
                if command.idempotency_key is not None:
                    self._store_idempotency(
                        tenant_id=command.tenant_id,
                        workspace_id=command.workspace_id,
                        idempotency_key=command.idempotency_key,
                        fingerprint=command.semantic_fingerprint(),
                        result_json=result_json,
                    )
                self._store.transaction().commit()
                return record
            except sqlite3.IntegrityError as exc:
                self._store.transaction().rollback()
                raise WorkspaceMembershipAlreadyExists("workspace membership already exists") from exc
            except Exception:
                self._store.transaction().rollback()
                raise

    def get(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        membership_id: str,
    ) -> WorkspaceMembership | None:
        with self._store._lock:
            self._store._ensure_open()
            return self._get_in_transaction(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                membership_id=membership_id,
            )

    def get_for_principal(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        principal_id: str,
    ) -> WorkspaceMembership | None:
        with self._store._lock:
            self._store._ensure_open()
            return self._get_for_principal_in_transaction(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                principal_id=principal_id,
            )

    def update(self, command: UpdateWorkspaceMembershipCommand) -> WorkspaceMembership:
        with self._store._lock:
            self._store._ensure_open()
            self._store.transaction().execute("BEGIN IMMEDIATE")
            try:
                current = self._get_in_transaction(
                    tenant_id=command.scope.tenant_id,
                    workspace_id=command.scope.workspace_id,
                    membership_id=command.scope.membership_id,
                )
                if current is None:
                    raise WorkspaceMembershipNotFound("workspace membership was not found")
                if current.revision != command.expected_revision:
                    raise WorkspaceMembershipRevisionConflict("workspace membership revision conflict")

                replacement = WorkspaceMembership(
                    membership_id=current.membership_id,
                    tenant_id=current.tenant_id,
                    workspace_id=current.workspace_id,
                    principal_id=current.principal_id,
                    role=command.role,
                    status=command.status,
                    revision=current.revision + 1,
                )
                updated = self._store.transaction().execute(
                    """
                    UPDATE workspace_memberships
                    SET record_json = ?, revision = ?
                    WHERE tenant_id = ? AND workspace_id = ? AND membership_id = ?
                      AND revision = ?
                    """,
                    (
                        workspace_membership_to_json(replacement),
                        replacement.revision,
                        replacement.tenant_id.strip(),
                        replacement.workspace_id.strip(),
                        replacement.membership_id.strip(),
                        command.expected_revision,
                    ),
                )
                if updated.rowcount != 1:
                    raise WorkspaceMembershipRevisionConflict("workspace membership revision conflict")
                self._store.transaction().commit()
                return replacement
            except (WorkspaceMembershipNotFound, WorkspaceMembershipRevisionConflict):
                self._store.transaction().rollback()
                raise
            except Exception:
                self._store.transaction().rollback()
                raise

    def _get_in_transaction(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        membership_id: str,
    ) -> WorkspaceMembership | None:
        row = self._store.transaction().execute(
            """
            SELECT record_json FROM workspace_memberships
            WHERE tenant_id = ? AND workspace_id = ? AND membership_id = ?
            """,
            (tenant_id.strip(), workspace_id.strip(), membership_id.strip()),
        ).fetchone()
        if row is None:
            return None
        record = workspace_membership_from_json(row["record_json"])
        if not _scope_matches_tenant_workspace(
            record.tenant_id,
            record.workspace_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        ):
            return None
        return record

    def _get_for_principal_in_transaction(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        principal_id: str,
    ) -> WorkspaceMembership | None:
        row = self._store.transaction().execute(
            """
            SELECT record_json FROM workspace_memberships
            WHERE tenant_id = ? AND workspace_id = ? AND principal_id = ?
            """,
            (tenant_id.strip(), workspace_id.strip(), principal_id.strip()),
        ).fetchone()
        if row is None:
            return None
        record = workspace_membership_from_json(row["record_json"])
        if not _scope_matches_tenant_workspace(
            record.tenant_id,
            record.workspace_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        ):
            return None
        if record.principal_id != principal_id.strip():
            return None
        return record

    def _replay_create(self, command: CreateWorkspaceMembershipCommand) -> WorkspaceMembership | None:
        assert command.idempotency_key is not None
        loaded = self._load_idempotency(
            tenant_id=command.tenant_id,
            workspace_id=command.workspace_id,
            idempotency_key=command.idempotency_key,
            decode=workspace_membership_from_json,
        )
        if loaded is None:
            return None
        fingerprint, record = loaded
        if fingerprint != command.semantic_fingerprint():
            raise WorkspaceMembershipIdempotencyConflict(
                "workspace membership idempotency key conflict"
            )
        return record


class SQLiteAuthorityDelegationRepository(_IdempotencyMixin):
    _entity_kind = "authority_delegation"

    def __init__(self, store: SQLiteCollaborativeWorkStore) -> None:
        self._store = store

    @property
    def capabilities(self) -> CollaborativeWorkRepositoryCapabilities:
        return _CAPABILITIES

    def create(self, command: CreateAuthorityDelegationCommand) -> AuthorityDelegation:
        with self._store._lock:
            self._store._ensure_open()
            self._store.transaction().execute("BEGIN IMMEDIATE")
            try:
                if command.idempotency_key is not None:
                    replay = self._replay_create(command)
                    if replay is not None:
                        self._store.transaction().commit()
                        return replay

                existing = self._get_in_transaction(
                    tenant_id=command.tenant_id,
                    workspace_id=command.workspace_id,
                    delegation_id=command.delegation_id,
                )
                if existing is not None:
                    raise AuthorityDelegationAlreadyExists("authority delegation already exists")

                record = AuthorityDelegation(
                    delegation_id=command.delegation_id,
                    tenant_id=command.tenant_id,
                    workspace_id=command.workspace_id,
                    delegator_principal_id=command.delegator_principal_id,
                    delegate_principal_id=command.delegate_principal_id,
                    authority_scopes=command.authority_scopes,
                    resource_scope=command.resource_scope,
                    valid_from=command.valid_from,
                    valid_until=command.valid_until,
                    status=command.status,
                    revision=INITIAL_RECORD_REVISION,
                )
                result_json = authority_delegation_to_json(record)
                self._store.transaction().execute(
                    """
                    INSERT INTO authority_delegations (
                        tenant_id, workspace_id, delegation_id, record_json, revision
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        record.tenant_id.strip(),
                        record.workspace_id.strip(),
                        record.delegation_id.strip(),
                        result_json,
                        record.revision,
                    ),
                )
                if command.idempotency_key is not None:
                    self._store_idempotency(
                        tenant_id=command.tenant_id,
                        workspace_id=command.workspace_id,
                        idempotency_key=command.idempotency_key,
                        fingerprint=command.semantic_fingerprint(),
                        result_json=result_json,
                    )
                self._store.transaction().commit()
                return record
            except sqlite3.IntegrityError as exc:
                self._store.transaction().rollback()
                raise AuthorityDelegationAlreadyExists("authority delegation already exists") from exc
            except Exception:
                self._store.transaction().rollback()
                raise

    def get(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        delegation_id: str,
    ) -> AuthorityDelegation | None:
        with self._store._lock:
            self._store._ensure_open()
            return self._get_in_transaction(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                delegation_id=delegation_id,
            )

    def update(self, command: UpdateAuthorityDelegationCommand) -> AuthorityDelegation:
        with self._store._lock:
            self._store._ensure_open()
            self._store.transaction().execute("BEGIN IMMEDIATE")
            try:
                current = self._get_in_transaction(
                    tenant_id=command.scope.tenant_id,
                    workspace_id=command.scope.workspace_id,
                    delegation_id=command.scope.delegation_id,
                )
                if current is None:
                    raise AuthorityDelegationNotFound("authority delegation was not found")
                if current.revision != command.expected_revision:
                    raise AuthorityDelegationRevisionConflict("authority delegation revision conflict")

                replacement = AuthorityDelegation(
                    delegation_id=current.delegation_id,
                    tenant_id=current.tenant_id,
                    workspace_id=current.workspace_id,
                    delegator_principal_id=current.delegator_principal_id,
                    delegate_principal_id=current.delegate_principal_id,
                    authority_scopes=command.authority_scopes,
                    resource_scope=command.resource_scope,
                    valid_from=command.valid_from,
                    valid_until=command.valid_until,
                    status=command.status,
                    revision=current.revision + 1,
                )
                updated = self._store.transaction().execute(
                    """
                    UPDATE authority_delegations
                    SET record_json = ?, revision = ?
                    WHERE tenant_id = ? AND workspace_id = ? AND delegation_id = ?
                      AND revision = ?
                    """,
                    (
                        authority_delegation_to_json(replacement),
                        replacement.revision,
                        replacement.tenant_id.strip(),
                        replacement.workspace_id.strip(),
                        replacement.delegation_id.strip(),
                        command.expected_revision,
                    ),
                )
                if updated.rowcount != 1:
                    raise AuthorityDelegationRevisionConflict("authority delegation revision conflict")
                self._store.transaction().commit()
                return replacement
            except (AuthorityDelegationNotFound, AuthorityDelegationRevisionConflict):
                self._store.transaction().rollback()
                raise
            except Exception:
                self._store.transaction().rollback()
                raise

    def _get_in_transaction(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        delegation_id: str,
    ) -> AuthorityDelegation | None:
        row = self._store.transaction().execute(
            """
            SELECT record_json FROM authority_delegations
            WHERE tenant_id = ? AND workspace_id = ? AND delegation_id = ?
            """,
            (tenant_id.strip(), workspace_id.strip(), delegation_id.strip()),
        ).fetchone()
        if row is None:
            return None
        record = authority_delegation_from_json(row["record_json"])
        if not _scope_matches_tenant_workspace(
            record.tenant_id,
            record.workspace_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        ):
            return None
        return record

    def _replay_create(self, command: CreateAuthorityDelegationCommand) -> AuthorityDelegation | None:
        assert command.idempotency_key is not None
        loaded = self._load_idempotency(
            tenant_id=command.tenant_id,
            workspace_id=command.workspace_id,
            idempotency_key=command.idempotency_key,
            decode=authority_delegation_from_json,
        )
        if loaded is None:
            return None
        fingerprint, record = loaded
        if fingerprint != command.semantic_fingerprint():
            raise AuthorityDelegationIdempotencyConflict(
                "authority delegation idempotency key conflict"
            )
        return record


class SQLitePrincipalAuthorityRepository(_IdempotencyMixin):
    _entity_kind = "principal_authority_grant"

    def __init__(self, store: SQLiteCollaborativeWorkStore) -> None:
        self._store = store

    @property
    def capabilities(self) -> CollaborativeWorkRepositoryCapabilities:
        return _CAPABILITIES

    def create(self, command: CreatePrincipalAuthorityGrantCommand) -> PrincipalAuthorityGrant:
        with self._store._lock:
            self._store._ensure_open()
            self._store.transaction().execute("BEGIN IMMEDIATE")
            try:
                if command.idempotency_key is not None:
                    replay = self._replay_create(command)
                    if replay is not None:
                        self._store.transaction().commit()
                        return replay

                existing = self._get_in_transaction(
                    tenant_id=command.tenant_id,
                    workspace_id=command.workspace_id,
                    authority_grant_id=command.authority_grant_id,
                )
                if existing is not None:
                    raise PrincipalAuthorityGrantAlreadyExists(
                        "principal authority grant already exists"
                    )
                principal_existing = self._get_for_principal_in_transaction(
                    tenant_id=command.tenant_id,
                    workspace_id=command.workspace_id,
                    principal_id=command.principal_id,
                )
                if principal_existing is not None:
                    raise PrincipalAuthorityGrantAlreadyExists(
                        "principal already has an authority grant in this workspace scope"
                    )

                record = PrincipalAuthorityGrant(
                    authority_grant_id=command.authority_grant_id,
                    tenant_id=command.tenant_id,
                    workspace_id=command.workspace_id,
                    principal_id=command.principal_id,
                    authority_scopes=command.authority_scopes,
                    status=command.status,
                    revision=INITIAL_RECORD_REVISION,
                )
                result_json = principal_authority_grant_to_json(record)
                self._store.transaction().execute(
                    """
                    INSERT INTO principal_authority_grants (
                        tenant_id, workspace_id, authority_grant_id, principal_id,
                        record_json, revision
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.tenant_id.strip(),
                        record.workspace_id.strip(),
                        record.authority_grant_id.strip(),
                        record.principal_id.strip(),
                        result_json,
                        record.revision,
                    ),
                )
                if command.idempotency_key is not None:
                    self._store_idempotency(
                        tenant_id=command.tenant_id,
                        workspace_id=command.workspace_id,
                        idempotency_key=command.idempotency_key,
                        fingerprint=command.semantic_fingerprint(),
                        result_json=result_json,
                    )
                self._store.transaction().commit()
                return record
            except sqlite3.IntegrityError as exc:
                self._store.transaction().rollback()
                raise PrincipalAuthorityGrantAlreadyExists(
                    "principal authority grant already exists"
                ) from exc
            except Exception:
                self._store.transaction().rollback()
                raise

    def get(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        authority_grant_id: str,
    ) -> PrincipalAuthorityGrant | None:
        with self._store._lock:
            self._store._ensure_open()
            return self._get_in_transaction(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                authority_grant_id=authority_grant_id,
            )

    def get_for_principal(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        principal_id: str,
    ) -> PrincipalAuthorityGrant | None:
        with self._store._lock:
            self._store._ensure_open()
            return self._get_for_principal_in_transaction(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                principal_id=principal_id,
            )

    def update(self, command: UpdatePrincipalAuthorityGrantCommand) -> PrincipalAuthorityGrant:
        with self._store._lock:
            self._store._ensure_open()
            self._store.transaction().execute("BEGIN IMMEDIATE")
            try:
                current = self._get_in_transaction(
                    tenant_id=command.scope.tenant_id,
                    workspace_id=command.scope.workspace_id,
                    authority_grant_id=command.scope.authority_grant_id,
                )
                if current is None:
                    raise PrincipalAuthorityGrantNotFound("principal authority grant was not found")
                if current.revision != command.expected_revision:
                    raise PrincipalAuthorityGrantRevisionConflict(
                        "principal authority grant revision conflict"
                    )

                replacement = PrincipalAuthorityGrant(
                    authority_grant_id=current.authority_grant_id,
                    tenant_id=current.tenant_id,
                    workspace_id=current.workspace_id,
                    principal_id=current.principal_id,
                    authority_scopes=command.authority_scopes,
                    status=command.status,
                    revision=current.revision + 1,
                )
                updated = self._store.transaction().execute(
                    """
                    UPDATE principal_authority_grants
                    SET record_json = ?, revision = ?
                    WHERE tenant_id = ? AND workspace_id = ? AND authority_grant_id = ?
                      AND revision = ?
                    """,
                    (
                        principal_authority_grant_to_json(replacement),
                        replacement.revision,
                        replacement.tenant_id.strip(),
                        replacement.workspace_id.strip(),
                        replacement.authority_grant_id.strip(),
                        command.expected_revision,
                    ),
                )
                if updated.rowcount != 1:
                    raise PrincipalAuthorityGrantRevisionConflict(
                        "principal authority grant revision conflict"
                    )
                self._store.transaction().commit()
                return replacement
            except (PrincipalAuthorityGrantNotFound, PrincipalAuthorityGrantRevisionConflict):
                self._store.transaction().rollback()
                raise
            except Exception:
                self._store.transaction().rollback()
                raise

    def _get_in_transaction(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        authority_grant_id: str,
    ) -> PrincipalAuthorityGrant | None:
        row = self._store.transaction().execute(
            """
            SELECT record_json FROM principal_authority_grants
            WHERE tenant_id = ? AND workspace_id = ? AND authority_grant_id = ?
            """,
            (tenant_id.strip(), workspace_id.strip(), authority_grant_id.strip()),
        ).fetchone()
        if row is None:
            return None
        record = principal_authority_grant_from_json(row["record_json"])
        if not _scope_matches_tenant_workspace(
            record.tenant_id,
            record.workspace_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        ):
            return None
        return record

    def _get_for_principal_in_transaction(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        principal_id: str,
    ) -> PrincipalAuthorityGrant | None:
        row = self._store.transaction().execute(
            """
            SELECT record_json FROM principal_authority_grants
            WHERE tenant_id = ? AND workspace_id = ? AND principal_id = ?
            """,
            (tenant_id.strip(), workspace_id.strip(), principal_id.strip()),
        ).fetchone()
        if row is None:
            return None
        record = principal_authority_grant_from_json(row["record_json"])
        if record.principal_id != principal_id.strip():
            return None
        if not _scope_matches_tenant_workspace(
            record.tenant_id,
            record.workspace_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        ):
            return None
        return record

    def _replay_create(
        self,
        command: CreatePrincipalAuthorityGrantCommand,
    ) -> PrincipalAuthorityGrant | None:
        assert command.idempotency_key is not None
        loaded = self._load_idempotency(
            tenant_id=command.tenant_id,
            workspace_id=command.workspace_id,
            idempotency_key=command.idempotency_key,
            decode=principal_authority_grant_from_json,
        )
        if loaded is None:
            return None
        fingerprint, record = loaded
        if fingerprint != command.semantic_fingerprint():
            raise PrincipalAuthorityGrantIdempotencyConflict(
                "principal authority grant idempotency key conflict"
            )
        return record


class SQLiteCollaborativePolicyRepository(_IdempotencyMixin):
    _entity_kind = "collaborative_policy_rule"

    def __init__(self, store: SQLiteCollaborativeWorkStore) -> None:
        self._store = store

    @property
    def capabilities(self) -> CollaborativeWorkRepositoryCapabilities:
        return _CAPABILITIES

    def create(self, command: CreateCollaborativePolicyRuleCommand) -> CollaborativePolicyRule:
        with self._store._lock:
            self._store._ensure_open()
            self._store.transaction().execute("BEGIN IMMEDIATE")
            try:
                if command.idempotency_key is not None:
                    replay = self._replay_create(command)
                    if replay is not None:
                        self._store.transaction().commit()
                        return replay

                existing = self._get_in_transaction(
                    tenant_id=command.tenant_id,
                    workspace_id=command.workspace_id,
                    policy_rule_id=command.policy_rule_id,
                )
                if existing is not None:
                    raise CollaborativePolicyRuleAlreadyExists(
                        "collaborative policy rule already exists"
                    )
                exact_existing = self._get_effective_in_transaction(
                    tenant_id=command.tenant_id,
                    workspace_id=command.workspace_id,
                    layer=command.layer,
                    authority_scope=command.authority_scope,
                    resource_scope=command.resource_scope,
                )
                if exact_existing is not None:
                    raise CollaborativePolicyRuleAlreadyExists(
                        "exact collaborative policy key already exists"
                    )

                record = CollaborativePolicyRule(
                    policy_rule_id=command.policy_rule_id,
                    tenant_id=command.tenant_id,
                    workspace_id=command.workspace_id,
                    layer=command.layer,
                    authority_scope=command.authority_scope,
                    action=command.action,
                    resource_scope=command.resource_scope,
                    status=command.status,
                    revision=INITIAL_RECORD_REVISION,
                )
                result_json = collaborative_policy_rule_to_json(record)
                self._store.transaction().execute(
                    """
                    INSERT INTO collaborative_policy_rules (
                        tenant_id, workspace_id, policy_rule_id, layer, authority_scope,
                        resource_scope, resource_scope_key, record_json, revision
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.tenant_id.strip(),
                        record.workspace_id.strip(),
                        record.policy_rule_id.strip(),
                        record.layer.value,
                        record.authority_scope.strip(),
                        record.resource_scope,
                        _resource_scope_key(record.resource_scope),
                        result_json,
                        record.revision,
                    ),
                )
                if command.idempotency_key is not None:
                    self._store_idempotency(
                        tenant_id=command.tenant_id,
                        workspace_id=command.workspace_id,
                        idempotency_key=command.idempotency_key,
                        fingerprint=command.semantic_fingerprint(),
                        result_json=result_json,
                    )
                self._store.transaction().commit()
                return record
            except sqlite3.IntegrityError as exc:
                self._store.transaction().rollback()
                raise CollaborativePolicyRuleAlreadyExists(
                    "collaborative policy rule already exists"
                ) from exc
            except Exception:
                self._store.transaction().rollback()
                raise

    def get(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        policy_rule_id: str,
    ) -> CollaborativePolicyRule | None:
        with self._store._lock:
            self._store._ensure_open()
            return self._get_in_transaction(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                policy_rule_id=policy_rule_id,
            )

    def get_effective_rule(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        layer: PolicyCompositionLayer,
        authority_scope: str,
        resource_scope: str | None = None,
    ) -> CollaborativePolicyRule | None:
        with self._store._lock:
            self._store._ensure_open()
            return self._get_effective_in_transaction(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                layer=layer,
                authority_scope=authority_scope,
                resource_scope=resource_scope,
            )

    def update(self, command: UpdateCollaborativePolicyRuleCommand) -> CollaborativePolicyRule:
        with self._store._lock:
            self._store._ensure_open()
            self._store.transaction().execute("BEGIN IMMEDIATE")
            try:
                current = self._get_in_transaction(
                    tenant_id=command.scope.tenant_id,
                    workspace_id=command.scope.workspace_id,
                    policy_rule_id=command.scope.policy_rule_id,
                )
                if current is None:
                    raise CollaborativePolicyRuleNotFound("collaborative policy rule was not found")
                if current.revision != command.expected_revision:
                    raise CollaborativePolicyRuleRevisionConflict(
                        "collaborative policy rule revision conflict"
                    )

                replacement = CollaborativePolicyRule(
                    policy_rule_id=current.policy_rule_id,
                    tenant_id=current.tenant_id,
                    workspace_id=current.workspace_id,
                    layer=current.layer,
                    authority_scope=current.authority_scope,
                    action=command.action,
                    resource_scope=current.resource_scope,
                    status=command.status,
                    revision=current.revision + 1,
                )
                updated = self._store.transaction().execute(
                    """
                    UPDATE collaborative_policy_rules
                    SET record_json = ?, revision = ?
                    WHERE tenant_id = ? AND workspace_id = ? AND policy_rule_id = ?
                      AND revision = ?
                    """,
                    (
                        collaborative_policy_rule_to_json(replacement),
                        replacement.revision,
                        replacement.tenant_id.strip(),
                        replacement.workspace_id.strip(),
                        replacement.policy_rule_id.strip(),
                        command.expected_revision,
                    ),
                )
                if updated.rowcount != 1:
                    raise CollaborativePolicyRuleRevisionConflict(
                        "collaborative policy rule revision conflict"
                    )
                self._store.transaction().commit()
                return replacement
            except (CollaborativePolicyRuleNotFound, CollaborativePolicyRuleRevisionConflict):
                self._store.transaction().rollback()
                raise
            except Exception:
                self._store.transaction().rollback()
                raise

    def _get_in_transaction(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        policy_rule_id: str,
    ) -> CollaborativePolicyRule | None:
        row = self._store.transaction().execute(
            """
            SELECT record_json FROM collaborative_policy_rules
            WHERE tenant_id = ? AND workspace_id = ? AND policy_rule_id = ?
            """,
            (tenant_id.strip(), workspace_id.strip(), policy_rule_id.strip()),
        ).fetchone()
        if row is None:
            return None
        record = collaborative_policy_rule_from_json(row["record_json"])
        if not _scope_matches_tenant_workspace(
            record.tenant_id,
            record.workspace_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        ):
            return None
        return record

    def _get_effective_in_transaction(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        layer: PolicyCompositionLayer,
        authority_scope: str,
        resource_scope: str | None,
    ) -> CollaborativePolicyRule | None:
        normalized_resource = resource_scope if resource_scope is not None else None
        row = self._store.transaction().execute(
            """
            SELECT record_json FROM collaborative_policy_rules
            WHERE tenant_id = ? AND workspace_id = ? AND layer = ?
              AND authority_scope = ? AND resource_scope_key = ?
            """,
            (
                tenant_id.strip(),
                workspace_id.strip(),
                layer.value,
                authority_scope.strip(),
                _resource_scope_key(normalized_resource),
            ),
        ).fetchone()
        if row is None:
            return None
        record = collaborative_policy_rule_from_json(row["record_json"])
        if not _scope_matches_tenant_workspace(
            record.tenant_id,
            record.workspace_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        ):
            return None
        return record

    def _replay_create(
        self,
        command: CreateCollaborativePolicyRuleCommand,
    ) -> CollaborativePolicyRule | None:
        assert command.idempotency_key is not None
        loaded = self._load_idempotency(
            tenant_id=command.tenant_id,
            workspace_id=command.workspace_id,
            idempotency_key=command.idempotency_key,
            decode=collaborative_policy_rule_from_json,
        )
        if loaded is None:
            return None
        fingerprint, record = loaded
        if fingerprint != command.semantic_fingerprint():
            raise CollaborativePolicyRuleIdempotencyConflict(
                "collaborative policy rule idempotency key conflict"
            )
        return record


class SQLiteCollaborativeOperationPolicyProfileRepository(_IdempotencyMixin):
    _entity_kind = "operation_policy_profile"

    def __init__(self, store: SQLiteCollaborativeWorkStore) -> None:
        self._store = store

    @property
    def capabilities(self) -> CollaborativeWorkRepositoryCapabilities:
        return _CAPABILITIES

    def create(
        self,
        command: CreateCollaborativeOperationPolicyProfileCommand,
    ) -> CollaborativeOperationPolicyProfile:
        with self._store._lock:
            self._store._ensure_open()
            self._store.transaction().execute("BEGIN IMMEDIATE")
            try:
                if command.idempotency_key is not None:
                    replay = self._replay_create(command)
                    if replay is not None:
                        self._store.transaction().commit()
                        return replay

                existing = self._get_in_transaction(
                    tenant_id=command.tenant_id,
                    workspace_id=command.workspace_id,
                    operation_id=command.operation_id,
                )
                if existing is not None:
                    raise CollaborativeOperationPolicyProfileAlreadyExists(
                        "operation policy profile already exists"
                    )

                record = CollaborativeOperationPolicyProfile(
                    operation_id=command.operation_id,
                    tenant_id=command.tenant_id,
                    workspace_id=command.workspace_id,
                    authority_scope=command.authority_scope,
                    workspace_policy_applicability=command.workspace_policy_applicability,
                    resource_policy_applicability=command.resource_policy_applicability,
                    runtime_policy_applicability=command.runtime_policy_applicability,
                    resource_requirement=command.resource_requirement,
                    meaningful_side_effect_requirement=command.meaningful_side_effect_requirement,
                    status=command.status,
                    revision=INITIAL_RECORD_REVISION,
                )
                result_json = operation_policy_profile_to_json(record)
                self._store.transaction().execute(
                    """
                    INSERT INTO collaborative_operation_policy_profiles (
                        tenant_id, workspace_id, operation_id, record_json, revision
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        record.tenant_id.strip(),
                        record.workspace_id.strip(),
                        record.operation_id.strip(),
                        result_json,
                        record.revision,
                    ),
                )
                if command.idempotency_key is not None:
                    self._store_idempotency(
                        tenant_id=command.tenant_id,
                        workspace_id=command.workspace_id,
                        idempotency_key=command.idempotency_key,
                        fingerprint=command.semantic_fingerprint(),
                        result_json=result_json,
                    )
                self._store.transaction().commit()
                return record
            except sqlite3.IntegrityError as exc:
                self._store.transaction().rollback()
                raise CollaborativeOperationPolicyProfileAlreadyExists(
                    "operation policy profile already exists"
                ) from exc
            except Exception:
                self._store.transaction().rollback()
                raise

    def get_for_operation(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        operation_id: str,
    ) -> CollaborativeOperationPolicyProfile | None:
        with self._store._lock:
            self._store._ensure_open()
            return self._get_in_transaction(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                operation_id=operation_id,
            )

    def update(
        self,
        command: UpdateCollaborativeOperationPolicyProfileCommand,
    ) -> CollaborativeOperationPolicyProfile:
        with self._store._lock:
            self._store._ensure_open()
            self._store.transaction().execute("BEGIN IMMEDIATE")
            try:
                current = self._get_in_transaction(
                    tenant_id=command.scope.tenant_id,
                    workspace_id=command.scope.workspace_id,
                    operation_id=command.scope.operation_id,
                )
                if current is None:
                    raise CollaborativeOperationPolicyProfileNotFound(
                        "operation policy profile was not found"
                    )
                if current.revision != command.expected_revision:
                    raise CollaborativeOperationPolicyProfileRevisionConflict(
                        "operation policy profile revision conflict"
                    )

                replacement = CollaborativeOperationPolicyProfile(
                    operation_id=current.operation_id,
                    tenant_id=current.tenant_id,
                    workspace_id=current.workspace_id,
                    authority_scope=command.authority_scope,
                    workspace_policy_applicability=command.workspace_policy_applicability,
                    resource_policy_applicability=command.resource_policy_applicability,
                    runtime_policy_applicability=command.runtime_policy_applicability,
                    resource_requirement=command.resource_requirement,
                    meaningful_side_effect_requirement=command.meaningful_side_effect_requirement,
                    status=command.status,
                    revision=current.revision + 1,
                )
                updated = self._store.transaction().execute(
                    """
                    UPDATE collaborative_operation_policy_profiles
                    SET record_json = ?, revision = ?
                    WHERE tenant_id = ? AND workspace_id = ? AND operation_id = ?
                      AND revision = ?
                    """,
                    (
                        operation_policy_profile_to_json(replacement),
                        replacement.revision,
                        replacement.tenant_id.strip(),
                        replacement.workspace_id.strip(),
                        replacement.operation_id.strip(),
                        command.expected_revision,
                    ),
                )
                if updated.rowcount != 1:
                    raise CollaborativeOperationPolicyProfileRevisionConflict(
                        "operation policy profile revision conflict"
                    )
                self._store.transaction().commit()
                return replacement
            except (
                CollaborativeOperationPolicyProfileNotFound,
                CollaborativeOperationPolicyProfileRevisionConflict,
            ):
                self._store.transaction().rollback()
                raise
            except Exception:
                self._store.transaction().rollback()
                raise

    def _get_in_transaction(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        operation_id: str,
    ) -> CollaborativeOperationPolicyProfile | None:
        row = self._store.transaction().execute(
            """
            SELECT record_json FROM collaborative_operation_policy_profiles
            WHERE tenant_id = ? AND workspace_id = ? AND operation_id = ?
            """,
            (tenant_id.strip(), workspace_id.strip(), operation_id.strip()),
        ).fetchone()
        if row is None:
            return None
        record = operation_policy_profile_from_json(row["record_json"])
        if not _scope_matches_tenant_workspace(
            record.tenant_id,
            record.workspace_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        ):
            return None
        return record

    def _replay_create(
        self,
        command: CreateCollaborativeOperationPolicyProfileCommand,
    ) -> CollaborativeOperationPolicyProfile | None:
        assert command.idempotency_key is not None
        loaded = self._load_idempotency(
            tenant_id=command.tenant_id,
            workspace_id=command.workspace_id,
            idempotency_key=command.idempotency_key,
            decode=operation_policy_profile_from_json,
        )
        if loaded is None:
            return None
        fingerprint, record = loaded
        if fingerprint != command.semantic_fingerprint():
            raise CollaborativeOperationPolicyProfileIdempotencyConflict(
                "operation policy profile idempotency key conflict"
            )
        return record


class SQLiteWorkItemRepository(_IdempotencyMixin):
    _entity_kind = "work_item"

    def __init__(self, store: SQLiteCollaborativeWorkStore) -> None:
        self._store = store

    @property
    def capabilities(self) -> CollaborativeWorkRepositoryCapabilities:
        return _CAPABILITIES

    def create(self, command: CreateWorkItemCommand) -> WorkItem:
        with self._store._lock:
            self._store._ensure_open()
            self._store.transaction().execute("BEGIN IMMEDIATE")
            try:
                if command.idempotency_key is not None:
                    replay = self._replay_create(command)
                    if replay is not None:
                        self._store.transaction().commit()
                        return replay

                existing = self._get_in_transaction(
                    tenant_id=command.tenant_id,
                    workspace_id=command.workspace_id,
                    work_item_id=command.work_item_id,
                )
                if existing is not None:
                    raise WorkItemAlreadyExists("work item already exists")

                record = WorkItem(
                    work_item_id=command.work_item_id,
                    tenant_id=command.tenant_id,
                    workspace_id=command.workspace_id,
                    created_by_principal_id=command.created_by_principal_id,
                    state=command.state,
                    revision=INITIAL_RECORD_REVISION,
                    created_at=command.created_at,
                    updated_at=command.updated_at,
                    title=command.title,
                    description=command.description,
                )
                result_json = work_item_to_json(record)
                self._store.transaction().execute(
                    """
                    INSERT INTO work_items (
                        tenant_id, workspace_id, work_item_id, record_json, revision
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        record.tenant_id.strip(),
                        record.workspace_id.strip(),
                        record.work_item_id.strip(),
                        result_json,
                        record.revision,
                    ),
                )
                if command.idempotency_key is not None:
                    self._store_idempotency(
                        tenant_id=command.tenant_id,
                        workspace_id=command.workspace_id,
                        idempotency_key=command.idempotency_key,
                        fingerprint=command.semantic_fingerprint(),
                        result_json=result_json,
                    )
                self._store.transaction().commit()
                return record
            except sqlite3.IntegrityError as exc:
                self._store.transaction().rollback()
                raise WorkItemAlreadyExists("work item already exists") from exc
            except Exception:
                self._store.transaction().rollback()
                raise

    def get(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        work_item_id: str,
    ) -> WorkItem | None:
        with self._store._lock:
            self._store._ensure_open()
            return self._get_in_transaction(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                work_item_id=work_item_id,
            )

    def update(self, command: UpdateWorkItemCommand) -> WorkItem:
        with self._store._lock:
            self._store._ensure_open()
            self._store.transaction().execute("BEGIN IMMEDIATE")
            try:
                current = self._get_in_transaction(
                    tenant_id=command.scope.tenant_id,
                    workspace_id=command.scope.workspace_id,
                    work_item_id=command.scope.work_item_id,
                )
                if current is None:
                    raise WorkItemNotFound("work item was not found")
                if current.revision != command.expected_revision:
                    raise WorkItemRevisionConflict("work item revision conflict")

                replacement = WorkItem(
                    work_item_id=current.work_item_id,
                    tenant_id=current.tenant_id,
                    workspace_id=current.workspace_id,
                    created_by_principal_id=current.created_by_principal_id,
                    state=command.state,
                    revision=current.revision + 1,
                    created_at=current.created_at,
                    updated_at=command.updated_at,
                    title=command.title,
                    description=command.description,
                )
                updated = self._store.transaction().execute(
                    """
                    UPDATE work_items
                    SET record_json = ?, revision = ?
                    WHERE tenant_id = ? AND workspace_id = ? AND work_item_id = ?
                      AND revision = ?
                    """,
                    (
                        work_item_to_json(replacement),
                        replacement.revision,
                        replacement.tenant_id.strip(),
                        replacement.workspace_id.strip(),
                        replacement.work_item_id.strip(),
                        command.expected_revision,
                    ),
                )
                if updated.rowcount != 1:
                    raise WorkItemRevisionConflict("work item revision conflict")
                self._store.transaction().commit()
                return replacement
            except (WorkItemNotFound, WorkItemRevisionConflict):
                self._store.transaction().rollback()
                raise
            except Exception:
                self._store.transaction().rollback()
                raise

    def _get_in_transaction(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        work_item_id: str,
    ) -> WorkItem | None:
        row = self._store.transaction().execute(
            """
            SELECT record_json FROM work_items
            WHERE tenant_id = ? AND workspace_id = ? AND work_item_id = ?
            """,
            (tenant_id.strip(), workspace_id.strip(), work_item_id.strip()),
        ).fetchone()
        if row is None:
            return None
        record = work_item_from_json(row["record_json"])
        if not _scope_matches_tenant_workspace(
            record.tenant_id,
            record.workspace_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        ):
            return None
        if record.work_item_id.strip() != work_item_id.strip():
            return None
        return record

    def _replay_create(self, command: CreateWorkItemCommand) -> WorkItem | None:
        assert command.idempotency_key is not None
        loaded = self._load_idempotency(
            tenant_id=command.tenant_id,
            workspace_id=command.workspace_id,
            idempotency_key=command.idempotency_key,
            decode=work_item_from_json,
        )
        if loaded is None:
            return None
        fingerprint, record = loaded
        if fingerprint != command.semantic_fingerprint():
            raise WorkItemIdempotencyConflict("work item idempotency key conflict")
        return record


class SQLiteAssignmentRepository(_IdempotencyMixin):
    _entity_kind = "assignment"

    def __init__(self, store: SQLiteCollaborativeWorkStore) -> None:
        self._store = store

    @property
    def capabilities(self) -> CollaborativeWorkRepositoryCapabilities:
        return _CAPABILITIES

    def create(self, command: CreateAssignmentCommand) -> Assignment:
        with self._store._lock:
            self._store._ensure_open()
            self._store.transaction().execute("BEGIN IMMEDIATE")
            try:
                if command.idempotency_key is not None:
                    replay = self._replay_create(command)
                    if replay is not None:
                        self._store.transaction().commit()
                        return replay

                existing = self._get_in_transaction(
                    tenant_id=command.tenant_id,
                    workspace_id=command.workspace_id,
                    assignment_id=command.assignment_id,
                )
                if existing is not None:
                    raise AssignmentAlreadyExists("assignment already exists")

                record = Assignment(
                    assignment_id=command.assignment_id,
                    tenant_id=command.tenant_id,
                    workspace_id=command.workspace_id,
                    work_item_id=command.work_item_id,
                    principal_id=command.principal_id,
                    created_by_principal_id=command.created_by_principal_id,
                    state=command.state,
                    revision=INITIAL_RECORD_REVISION,
                    created_at=command.created_at,
                    updated_at=command.updated_at,
                )
                result_json = assignment_to_json(record)
                self._store.transaction().execute(
                    """
                    INSERT INTO assignments (
                        tenant_id, workspace_id, assignment_id, work_item_id,
                        record_json, revision
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.tenant_id.strip(),
                        record.workspace_id.strip(),
                        record.assignment_id.strip(),
                        record.work_item_id.strip(),
                        result_json,
                        record.revision,
                    ),
                )
                if command.idempotency_key is not None:
                    self._store_idempotency(
                        tenant_id=command.tenant_id,
                        workspace_id=command.workspace_id,
                        idempotency_key=command.idempotency_key,
                        fingerprint=command.semantic_fingerprint(),
                        result_json=result_json,
                    )
                self._store.transaction().commit()
                return record
            except sqlite3.IntegrityError as exc:
                self._store.transaction().rollback()
                raise AssignmentAlreadyExists("assignment already exists") from exc
            except Exception:
                self._store.transaction().rollback()
                raise

    def get(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        assignment_id: str,
    ) -> Assignment | None:
        with self._store._lock:
            self._store._ensure_open()
            return self._get_in_transaction(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                assignment_id=assignment_id,
            )

    def update(self, command: UpdateAssignmentCommand) -> Assignment:
        with self._store._lock:
            self._store._ensure_open()
            self._store.transaction().execute("BEGIN IMMEDIATE")
            try:
                current = self._get_in_transaction(
                    tenant_id=command.scope.tenant_id,
                    workspace_id=command.scope.workspace_id,
                    assignment_id=command.scope.assignment_id,
                )
                if current is None:
                    raise AssignmentNotFound("assignment was not found")
                if current.revision != command.expected_revision:
                    raise AssignmentRevisionConflict("assignment revision conflict")

                replacement = Assignment(
                    assignment_id=current.assignment_id,
                    tenant_id=current.tenant_id,
                    workspace_id=current.workspace_id,
                    work_item_id=current.work_item_id,
                    principal_id=current.principal_id,
                    created_by_principal_id=current.created_by_principal_id,
                    state=command.state,
                    revision=current.revision + 1,
                    created_at=current.created_at,
                    updated_at=command.updated_at,
                )
                updated = self._store.transaction().execute(
                    """
                    UPDATE assignments
                    SET record_json = ?, revision = ?
                    WHERE tenant_id = ? AND workspace_id = ? AND assignment_id = ?
                      AND revision = ?
                    """,
                    (
                        assignment_to_json(replacement),
                        replacement.revision,
                        replacement.tenant_id.strip(),
                        replacement.workspace_id.strip(),
                        replacement.assignment_id.strip(),
                        command.expected_revision,
                    ),
                )
                if updated.rowcount != 1:
                    raise AssignmentRevisionConflict("assignment revision conflict")
                self._store.transaction().commit()
                return replacement
            except (AssignmentNotFound, AssignmentRevisionConflict):
                self._store.transaction().rollback()
                raise
            except Exception:
                self._store.transaction().rollback()
                raise

    def _get_in_transaction(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        assignment_id: str,
    ) -> Assignment | None:
        row = self._store.transaction().execute(
            """
            SELECT record_json FROM assignments
            WHERE tenant_id = ? AND workspace_id = ? AND assignment_id = ?
            """,
            (tenant_id.strip(), workspace_id.strip(), assignment_id.strip()),
        ).fetchone()
        if row is None:
            return None
        record = assignment_from_json(row["record_json"])
        if not _scope_matches_tenant_workspace(
            record.tenant_id,
            record.workspace_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        ):
            return None
        if record.assignment_id.strip() != assignment_id.strip():
            return None
        return record

    def _replay_create(self, command: CreateAssignmentCommand) -> Assignment | None:
        assert command.idempotency_key is not None
        loaded = self._load_idempotency(
            tenant_id=command.tenant_id,
            workspace_id=command.workspace_id,
            idempotency_key=command.idempotency_key,
            decode=assignment_from_json,
        )
        if loaded is None:
            return None
        fingerprint, record = loaded
        if fingerprint != command.semantic_fingerprint():
            raise AssignmentIdempotencyConflict("assignment idempotency key conflict")
        return record
