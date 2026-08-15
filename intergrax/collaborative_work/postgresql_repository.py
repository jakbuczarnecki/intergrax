# © Artur Czarnecki. All rights reserved.

"""Production PostgreSQL adapter for Collaborative Work authoritative state (COLLAB-WORK-1J)."""

from __future__ import annotations

import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any, TypeVar

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.relational_store.postgresql.config import (
    PostgreSQLIntegrationConfig,
    validate_schema_identifier,
)
from intergrax.integrations.providers.relational_store.postgresql.session import (
    PostgreSQLConnectionProvider,
    PostgreSQLIsolationLevel,
    PostgreSQLSession,
    is_postgresql_unique_violation,
)
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
_BACKEND_ID = "collaborative_work.postgresql"
_CAPABILITIES = CollaborativeWorkRepositoryCapabilities(
    backend_id=_BACKEND_ID,
    durable=True,
    reference_only=False,
)
_ISOLATION_LEVEL = PostgreSQLIsolationLevel.READ_COMMITTED

TRecord = TypeVar("TRecord")


class PostgreSQLCollaborativeWorkStore:
    """Shared PostgreSQL schema and connection factory for Collaborative Work repositories."""

    def __init__(
        self,
        config: PostgreSQLIntegrationConfig,
        *,
        connection_factory: Callable[[], Any] | None = None,
        schema_name: str | None = None,
    ) -> None:
        resolved_schema = validate_schema_identifier(
            (schema_name or config.tenant_schema or "public").strip()
        )
        self._config = config
        self._schema_name = resolved_schema
        self._provider = PostgreSQLConnectionProvider(
            config,
            connection_factory=connection_factory,
            tenant_schema=resolved_schema,
        )
        self._closed = False
        self._init_lock = threading.Lock()
        self._schema_ready = False
        try:
            with self._provider.connection() as session:
                self._initialize_schema(session)
                session.commit()
        except IntegrationConfigurationError:
            self._closed = True
            raise
        except Exception as exc:
            self._closed = True
            raise IntegrationConfigurationError(
                "PostgreSQL Collaborative Work store could not be opened"
            ) from exc

    @property
    def config(self) -> PostgreSQLIntegrationConfig:
        return self._config

    @property
    def schema_name(self) -> str:
        return self._schema_name

    def close(self) -> None:
        with self._init_lock:
            if self._closed:
                return
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(_CLOSED_ERROR)

    @contextmanager
    def transaction(self) -> Generator[PostgreSQLSession, None, None]:
        self._ensure_open()
        with self._provider.transaction(isolation_level=_ISOLATION_LEVEL) as session:
            yield session

    def _initialize_schema(self, session: PostgreSQLSession) -> None:
        with self._init_lock:
            if self._schema_ready:
                return
            self._provider.ensure_schema_exists(session, self._schema_name)
            session.execute(
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

                CREATE INDEX IF NOT EXISTS idx_cw_memberships_principal
                    ON workspace_memberships (tenant_id, workspace_id, principal_id);
                CREATE INDEX IF NOT EXISTS idx_cw_delegations_id
                    ON authority_delegations (tenant_id, workspace_id, delegation_id);
                CREATE INDEX IF NOT EXISTS idx_cw_principal_authority_principal
                    ON principal_authority_grants (tenant_id, workspace_id, principal_id);
                CREATE INDEX IF NOT EXISTS idx_cw_policy_exact_key
                    ON collaborative_policy_rules (
                        tenant_id, workspace_id, layer, authority_scope, resource_scope_key
                    );
                CREATE INDEX IF NOT EXISTS idx_cw_operation_profiles
                    ON collaborative_operation_policy_profiles (tenant_id, workspace_id, operation_id);
                CREATE INDEX IF NOT EXISTS idx_cw_idempotency_lookup
                    ON collaborative_idempotency (tenant_id, workspace_id, entity_kind, idempotency_key);
                """
            )
            self._schema_ready = True


def _unique_violation(exc: BaseException) -> bool:
    return is_postgresql_unique_violation(exc)


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
    _store: PostgreSQLCollaborativeWorkStore
    _entity_kind: str

    def _load_idempotency(
        self,
        conn: Any,
        *,
        tenant_id: str,
        workspace_id: str,
        idempotency_key: str,
        decode: Callable[[str], TRecord],
    ) -> tuple[str, TRecord] | None:
        row = conn.execute(
            """
            SELECT semantic_fingerprint, result_json
            FROM collaborative_idempotency
            WHERE tenant_id = %s AND workspace_id = %s AND entity_kind = %s AND idempotency_key = %s
            """,
            (tenant_id.strip(), workspace_id.strip(), self._entity_kind, idempotency_key.strip()),
        ).fetchone()
        if row is None:
            return None
        return row["semantic_fingerprint"], decode(row["result_json"])

    def _store_idempotency(
        self,
        conn: Any,
        *,
        tenant_id: str,
        workspace_id: str,
        idempotency_key: str,
        fingerprint: str,
        result_json: str,
    ) -> None:
        conn.execute(
            """
            INSERT INTO collaborative_idempotency (
                tenant_id, workspace_id, entity_kind, idempotency_key,
                semantic_fingerprint, result_json
            ) VALUES (%s, %s, %s, %s, %s, %s)
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


class PostgreSQLWorkspaceMembershipRepository(_IdempotencyMixin):
    _entity_kind = "workspace_membership"

    def __init__(self, store: PostgreSQLCollaborativeWorkStore) -> None:
        self._store = store

    @property
    def capabilities(self) -> CollaborativeWorkRepositoryCapabilities:
        return _CAPABILITIES

    def create(self, command: CreateWorkspaceMembershipCommand) -> WorkspaceMembership:
        with self._store.transaction() as conn:
            try:
                if command.idempotency_key is not None:
                    replay = self._replay_create(conn, command)
                    if replay is not None:
                        return replay

                existing = self._get_in_transaction(
                    conn,
                    tenant_id=command.tenant_id,
                    workspace_id=command.workspace_id,
                    membership_id=command.membership_id,
                )
                if existing is not None:
                    raise WorkspaceMembershipAlreadyExists("workspace membership already exists")

                principal_existing = self._get_for_principal_in_transaction(
                    conn,
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
                conn.execute(
                    """
                    INSERT INTO workspace_memberships (
                        tenant_id, workspace_id, membership_id, principal_id, record_json, revision
                    ) VALUES (%s, %s, %s, %s, %s, %s)
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
                        conn,
                        tenant_id=command.tenant_id,
                        workspace_id=command.workspace_id,
                        idempotency_key=command.idempotency_key,
                        fingerprint=command.semantic_fingerprint(),
                        result_json=result_json,
                    )
                return record
            except Exception as exc:
                if _unique_violation(exc):
                    if command.idempotency_key is not None:
                        with self._store.transaction() as replay_conn:
                            replay = self._replay_create(replay_conn, command)
                            if replay is not None:
                                return replay
                    raise WorkspaceMembershipAlreadyExists("workspace membership already exists") from exc
                raise

    def get(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        membership_id: str,
    ) -> WorkspaceMembership | None:
        with self._store.transaction() as conn:
            return self._get_in_transaction(
                conn,
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
        with self._store.transaction() as conn:
            return self._get_for_principal_in_transaction(
                conn,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                principal_id=principal_id,
            )

    def update(self, command: UpdateWorkspaceMembershipCommand) -> WorkspaceMembership:
        with self._store.transaction() as conn:
            try:
                current = self._get_in_transaction(
                    conn,
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
                updated = conn.execute(
                    """
                    UPDATE workspace_memberships
                    SET record_json = %s, revision = %s
                    WHERE tenant_id = %s AND workspace_id = %s AND membership_id = %s
                      AND revision = %s
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
                return replacement
            except (WorkspaceMembershipNotFound, WorkspaceMembershipRevisionConflict):
                raise
            except Exception:
                raise

    def _get_in_transaction(
        self,
        conn: Any,
        *,
        tenant_id: str,
        workspace_id: str,
        membership_id: str,
    ) -> WorkspaceMembership | None:
        row = conn.execute(
            """
            SELECT record_json FROM workspace_memberships
            WHERE tenant_id = %s AND workspace_id = %s AND membership_id = %s
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
        conn: Any,
        *,
        tenant_id: str,
        workspace_id: str,
        principal_id: str,
    ) -> WorkspaceMembership | None:
        row = conn.execute(
            """
            SELECT record_json FROM workspace_memberships
            WHERE tenant_id = %s AND workspace_id = %s AND principal_id = %s
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

    def _replay_create(self, conn: Any, command: CreateWorkspaceMembershipCommand) -> WorkspaceMembership | None:
        assert command.idempotency_key is not None
        loaded = self._load_idempotency(
            conn,
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


class PostgreSQLAuthorityDelegationRepository(_IdempotencyMixin):
    _entity_kind = "authority_delegation"

    def __init__(self, store: PostgreSQLCollaborativeWorkStore) -> None:
        self._store = store

    @property
    def capabilities(self) -> CollaborativeWorkRepositoryCapabilities:
        return _CAPABILITIES

    def create(self, command: CreateAuthorityDelegationCommand) -> AuthorityDelegation:
        with self._store.transaction() as conn:
            try:
                if command.idempotency_key is not None:
                    replay = self._replay_create(conn, command)
                    if replay is not None:
                        return replay

                existing = self._get_in_transaction(
                    conn,
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
                conn.execute(
                    """
                    INSERT INTO authority_delegations (
                        tenant_id, workspace_id, delegation_id, record_json, revision
                    ) VALUES (%s, %s, %s, %s, %s)
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
                        conn,
                        tenant_id=command.tenant_id,
                        workspace_id=command.workspace_id,
                        idempotency_key=command.idempotency_key,
                        fingerprint=command.semantic_fingerprint(),
                        result_json=result_json,
                    )
                return record
            except Exception as exc:
                if _unique_violation(exc):
                    if command.idempotency_key is not None:
                        with self._store.transaction() as replay_conn:
                            replay = self._replay_create(replay_conn, command)
                            if replay is not None:
                                return replay
                    raise AuthorityDelegationAlreadyExists("authority delegation already exists") from exc
                raise

    def get(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        delegation_id: str,
    ) -> AuthorityDelegation | None:
        with self._store.transaction() as conn:
            return self._get_in_transaction(
                conn,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                delegation_id=delegation_id,
            )

    def update(self, command: UpdateAuthorityDelegationCommand) -> AuthorityDelegation:
        with self._store.transaction() as conn:
            try:
                current = self._get_in_transaction(
                    conn,
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
                updated = conn.execute(
                    """
                    UPDATE authority_delegations
                    SET record_json = %s, revision = %s
                    WHERE tenant_id = %s AND workspace_id = %s AND delegation_id = %s
                      AND revision = %s
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
                return replacement
            except (AuthorityDelegationNotFound, AuthorityDelegationRevisionConflict):
                raise
            except Exception:
                raise

    def _get_in_transaction(
        self,
        conn: Any,
        *,
        tenant_id: str,
        workspace_id: str,
        delegation_id: str,
    ) -> AuthorityDelegation | None:
        row = conn.execute(
            """
            SELECT record_json FROM authority_delegations
            WHERE tenant_id = %s AND workspace_id = %s AND delegation_id = %s
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

    def _replay_create(self, conn: Any, command: CreateAuthorityDelegationCommand) -> AuthorityDelegation | None:
        assert command.idempotency_key is not None
        loaded = self._load_idempotency(
            conn,
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


class PostgreSQLPrincipalAuthorityRepository(_IdempotencyMixin):
    _entity_kind = "principal_authority_grant"

    def __init__(self, store: PostgreSQLCollaborativeWorkStore) -> None:
        self._store = store

    @property
    def capabilities(self) -> CollaborativeWorkRepositoryCapabilities:
        return _CAPABILITIES

    def create(self, command: CreatePrincipalAuthorityGrantCommand) -> PrincipalAuthorityGrant:
        with self._store.transaction() as conn:
            try:
                if command.idempotency_key is not None:
                    replay = self._replay_create(conn, command)
                    if replay is not None:
                        return replay

                existing = self._get_in_transaction(
                    conn,
                    tenant_id=command.tenant_id,
                    workspace_id=command.workspace_id,
                    authority_grant_id=command.authority_grant_id,
                )
                if existing is not None:
                    raise PrincipalAuthorityGrantAlreadyExists(
                        "principal authority grant already exists"
                    )
                principal_existing = self._get_for_principal_in_transaction(
                    conn,
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
                conn.execute(
                    """
                    INSERT INTO principal_authority_grants (
                        tenant_id, workspace_id, authority_grant_id, principal_id,
                        record_json, revision
                    ) VALUES (%s, %s, %s, %s, %s, %s)
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
                        conn,
                        tenant_id=command.tenant_id,
                        workspace_id=command.workspace_id,
                        idempotency_key=command.idempotency_key,
                        fingerprint=command.semantic_fingerprint(),
                        result_json=result_json,
                    )
                return record
            except Exception as exc:
                if _unique_violation(exc):
                    if command.idempotency_key is not None:
                        with self._store.transaction() as replay_conn:
                            replay = self._replay_create(replay_conn, command)
                            if replay is not None:
                                return replay
                    raise PrincipalAuthorityGrantAlreadyExists(
                        "principal authority grant already exists"
                    ) from exc
                raise

    def get(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        authority_grant_id: str,
    ) -> PrincipalAuthorityGrant | None:
        with self._store.transaction() as conn:
            return self._get_in_transaction(
                conn,
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
        with self._store.transaction() as conn:
            return self._get_for_principal_in_transaction(
                conn,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                principal_id=principal_id,
            )

    def update(self, command: UpdatePrincipalAuthorityGrantCommand) -> PrincipalAuthorityGrant:
        with self._store.transaction() as conn:
            try:
                current = self._get_in_transaction(
                    conn,
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
                updated = conn.execute(
                    """
                    UPDATE principal_authority_grants
                    SET record_json = %s, revision = %s
                    WHERE tenant_id = %s AND workspace_id = %s AND authority_grant_id = %s
                      AND revision = %s
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
                return replacement
            except (PrincipalAuthorityGrantNotFound, PrincipalAuthorityGrantRevisionConflict):
                raise
            except Exception:
                raise

    def _get_in_transaction(
        self,
        conn: Any,
        *,
        tenant_id: str,
        workspace_id: str,
        authority_grant_id: str,
    ) -> PrincipalAuthorityGrant | None:
        row = conn.execute(
            """
            SELECT record_json FROM principal_authority_grants
            WHERE tenant_id = %s AND workspace_id = %s AND authority_grant_id = %s
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
        conn: Any,
        *,
        tenant_id: str,
        workspace_id: str,
        principal_id: str,
    ) -> PrincipalAuthorityGrant | None:
        row = conn.execute(
            """
            SELECT record_json FROM principal_authority_grants
            WHERE tenant_id = %s AND workspace_id = %s AND principal_id = %s
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
        self, conn: Any, command: CreatePrincipalAuthorityGrantCommand,
    ) -> PrincipalAuthorityGrant | None:
        assert command.idempotency_key is not None
        loaded = self._load_idempotency(
            conn,
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


class PostgreSQLCollaborativePolicyRepository(_IdempotencyMixin):
    _entity_kind = "collaborative_policy_rule"

    def __init__(self, store: PostgreSQLCollaborativeWorkStore) -> None:
        self._store = store

    @property
    def capabilities(self) -> CollaborativeWorkRepositoryCapabilities:
        return _CAPABILITIES

    def create(self, command: CreateCollaborativePolicyRuleCommand) -> CollaborativePolicyRule:
        with self._store.transaction() as conn:
            try:
                if command.idempotency_key is not None:
                    replay = self._replay_create(conn, command)
                    if replay is not None:
                        return replay

                existing = self._get_in_transaction(
                    conn,
                    tenant_id=command.tenant_id,
                    workspace_id=command.workspace_id,
                    policy_rule_id=command.policy_rule_id,
                )
                if existing is not None:
                    raise CollaborativePolicyRuleAlreadyExists(
                        "collaborative policy rule already exists"
                    )
                exact_existing = self._get_effective_in_transaction(
                    conn,
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
                conn.execute(
                    """
                    INSERT INTO collaborative_policy_rules (
                        tenant_id, workspace_id, policy_rule_id, layer, authority_scope,
                        resource_scope, resource_scope_key, record_json, revision
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                        conn,
                        tenant_id=command.tenant_id,
                        workspace_id=command.workspace_id,
                        idempotency_key=command.idempotency_key,
                        fingerprint=command.semantic_fingerprint(),
                        result_json=result_json,
                    )
                return record
            except Exception as exc:
                if _unique_violation(exc):
                    if command.idempotency_key is not None:
                        with self._store.transaction() as replay_conn:
                            replay = self._replay_create(replay_conn, command)
                            if replay is not None:
                                return replay
                    raise CollaborativePolicyRuleAlreadyExists(
                        "collaborative policy rule already exists"
                    ) from exc
                raise

    def get(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        policy_rule_id: str,
    ) -> CollaborativePolicyRule | None:
        with self._store.transaction() as conn:
            return self._get_in_transaction(
                conn,
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
        with self._store.transaction() as conn:
            return self._get_effective_in_transaction(
                conn,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                layer=layer,
                authority_scope=authority_scope,
                resource_scope=resource_scope,
            )

    def update(self, command: UpdateCollaborativePolicyRuleCommand) -> CollaborativePolicyRule:
        with self._store.transaction() as conn:
            try:
                current = self._get_in_transaction(
                    conn,
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
                updated = conn.execute(
                    """
                    UPDATE collaborative_policy_rules
                    SET record_json = %s, revision = %s
                    WHERE tenant_id = %s AND workspace_id = %s AND policy_rule_id = %s
                      AND revision = %s
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
                return replacement
            except (CollaborativePolicyRuleNotFound, CollaborativePolicyRuleRevisionConflict):
                raise
            except Exception:
                raise

    def _get_in_transaction(
        self,
        conn: Any,
        *,
        tenant_id: str,
        workspace_id: str,
        policy_rule_id: str,
    ) -> CollaborativePolicyRule | None:
        row = conn.execute(
            """
            SELECT record_json FROM collaborative_policy_rules
            WHERE tenant_id = %s AND workspace_id = %s AND policy_rule_id = %s
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
        conn: Any,
        *,
        tenant_id: str,
        workspace_id: str,
        layer: PolicyCompositionLayer,
        authority_scope: str,
        resource_scope: str | None,
    ) -> CollaborativePolicyRule | None:
        normalized_resource = resource_scope if resource_scope is not None else None
        row = conn.execute(
            """
            SELECT record_json FROM collaborative_policy_rules
            WHERE tenant_id = %s AND workspace_id = %s AND layer = %s
              AND authority_scope = %s AND resource_scope_key = %s
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
        self, conn: Any, command: CreateCollaborativePolicyRuleCommand,
    ) -> CollaborativePolicyRule | None:
        assert command.idempotency_key is not None
        loaded = self._load_idempotency(
            conn,
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


class PostgreSQLCollaborativeOperationPolicyProfileRepository(_IdempotencyMixin):
    _entity_kind = "operation_policy_profile"

    def __init__(self, store: PostgreSQLCollaborativeWorkStore) -> None:
        self._store = store

    @property
    def capabilities(self) -> CollaborativeWorkRepositoryCapabilities:
        return _CAPABILITIES

    def create(
        self,
        command: CreateCollaborativeOperationPolicyProfileCommand,
    ) -> CollaborativeOperationPolicyProfile:
        with self._store.transaction() as conn:
            try:
                if command.idempotency_key is not None:
                    replay = self._replay_create(conn, command)
                    if replay is not None:
                        return replay

                existing = self._get_in_transaction(
                    conn,
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
                conn.execute(
                    """
                    INSERT INTO collaborative_operation_policy_profiles (
                        tenant_id, workspace_id, operation_id, record_json, revision
                    ) VALUES (%s, %s, %s, %s, %s)
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
                        conn,
                        tenant_id=command.tenant_id,
                        workspace_id=command.workspace_id,
                        idempotency_key=command.idempotency_key,
                        fingerprint=command.semantic_fingerprint(),
                        result_json=result_json,
                    )
                return record
            except Exception as exc:
                if _unique_violation(exc):
                    if command.idempotency_key is not None:
                        with self._store.transaction() as replay_conn:
                            replay = self._replay_create(replay_conn, command)
                            if replay is not None:
                                return replay
                    raise CollaborativeOperationPolicyProfileAlreadyExists(
                        "operation policy profile already exists"
                    ) from exc
                raise

    def get_for_operation(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        operation_id: str,
    ) -> CollaborativeOperationPolicyProfile | None:
        with self._store.transaction() as conn:
            return self._get_in_transaction(
                conn,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                operation_id=operation_id,
            )

    def update(
        self,
        command: UpdateCollaborativeOperationPolicyProfileCommand,
    ) -> CollaborativeOperationPolicyProfile:
        with self._store.transaction() as conn:
            try:
                current = self._get_in_transaction(
                    conn,
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
                updated = conn.execute(
                    """
                    UPDATE collaborative_operation_policy_profiles
                    SET record_json = %s, revision = %s
                    WHERE tenant_id = %s AND workspace_id = %s AND operation_id = %s
                      AND revision = %s
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
                return replacement
            except (
                CollaborativeOperationPolicyProfileNotFound,
                CollaborativeOperationPolicyProfileRevisionConflict,
            ):
                raise
            except Exception:
                raise

    def _get_in_transaction(
        self,
        conn: Any,
        *,
        tenant_id: str,
        workspace_id: str,
        operation_id: str,
    ) -> CollaborativeOperationPolicyProfile | None:
        row = conn.execute(
            """
            SELECT record_json FROM collaborative_operation_policy_profiles
            WHERE tenant_id = %s AND workspace_id = %s AND operation_id = %s
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
        self, conn: Any, command: CreateCollaborativeOperationPolicyProfileCommand,
    ) -> CollaborativeOperationPolicyProfile | None:
        assert command.idempotency_key is not None
        loaded = self._load_idempotency(
            conn,
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
