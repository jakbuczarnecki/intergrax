# © Artur Czarnecki. All rights reserved.

"""Durable SQL adapter for Collaborative Work authoritative state (COLLAB-WORK-1H)."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from intergrax.collaborative_work.repository import (
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
    CreateAuthorityDelegationCommand,
    CreateCollaborativeOperationPolicyProfileCommand,
    CreateCollaborativePolicyRuleCommand,
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkspaceMembershipCommand,
    INITIAL_RECORD_REVISION,
    PrincipalAuthorityGrantAlreadyExists,
    PrincipalAuthorityGrantIdempotencyConflict,
    PrincipalAuthorityGrantNotFound,
    PrincipalAuthorityGrantRevisionConflict,
    UpdateAuthorityDelegationCommand,
    UpdateCollaborativeOperationPolicyProfileCommand,
    UpdateCollaborativePolicyRuleCommand,
    UpdatePrincipalAuthorityGrantCommand,
    UpdateWorkspaceMembershipCommand,
    WorkspaceMembershipAlreadyExists,
    WorkspaceMembershipIdempotencyConflict,
    WorkspaceMembershipNotFound,
    WorkspaceMembershipRevisionConflict,
)
from intergrax.collaborative_work.serialization import (
    authority_delegation_from_json,
    authority_delegation_to_json,
    collaborative_policy_rule_from_json,
    collaborative_policy_rule_to_json,
    operation_policy_profile_from_json,
    operation_policy_profile_to_json,
    principal_authority_grant_from_json,
    principal_authority_grant_to_json,
    workspace_membership_from_json,
    workspace_membership_to_json,
)
from intergrax.contracts.collaborative_work import (
    AuthorityDelegation,
    CollaborativeOperationPolicyProfile,
    CollaborativePolicyRule,
    PolicyCompositionLayer,
    PrincipalAuthorityGrant,
    WorkspaceMembership,
)

_CLOSED_ERROR = "Collaborative Work repository store is closed"
_BACKEND_ID = "collaborative_work.sqlite"
_CAPABILITIES = CollaborativeWorkRepositoryCapabilities(
    backend_id=_BACKEND_ID,
    durable=True,
    reference_only=False,
)

TRecord = TypeVar("TRecord")


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
        self._initialize_schema()

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
                """
                CREATE TABLE IF NOT EXISTS workspace_memberships (
                    tenant_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    membership_id TEXT NOT NULL,
                    principal_id TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    PRIMARY KEY (tenant_id, workspace_id, membership_id),
                    UNIQUE (tenant_id, workspace_id, principal_id)
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
                """
            )
            self._migrate_workspace_memberships_principal_column()

    def _migrate_workspace_memberships_principal_column(self) -> None:
        rows = self._connection.execute("PRAGMA table_info(workspace_memberships)").fetchall()
        if not rows:
            return
        column_names = {row[1] for row in rows}
        if "principal_id" in column_names:
            return
        self._connection.execute(
            "ALTER TABLE workspace_memberships ADD COLUMN principal_id TEXT"
        )
        for row in self._connection.execute(
            """
            SELECT tenant_id, workspace_id, membership_id, record_json
            FROM workspace_memberships
            """
        ):
            membership = workspace_membership_from_json(row["record_json"])
            self._connection.execute(
                """
                UPDATE workspace_memberships
                SET principal_id = ?
                WHERE tenant_id = ? AND workspace_id = ? AND membership_id = ?
                """,
                (
                    membership.principal_id.strip(),
                    row["tenant_id"],
                    row["workspace_id"],
                    row["membership_id"],
                ),
            )
        self._connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_workspace_memberships_principal
            ON workspace_memberships (tenant_id, workspace_id, principal_id)
            """
        )
        self._connection.commit()


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
