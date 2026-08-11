# © Artur Czarnecki. All rights reserved.

"""In-memory reference repositories for Collaborative Work (COLLAB-WORK-1B)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TypeAlias

from intergrax.collaborative_work.repository import (
    AuthorityDelegationAlreadyExists,
    AuthorityDelegationIdempotencyConflict,
    AuthorityDelegationNotFound,
    AuthorityDelegationRevisionConflict,
    CollaborativeWorkRepositoryCapabilities,
    CreateAuthorityDelegationCommand,
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkspaceMembershipCommand,
    INITIAL_RECORD_REVISION,
    PrincipalAuthorityGrantAlreadyExists,
    PrincipalAuthorityGrantIdempotencyConflict,
    PrincipalAuthorityGrantNotFound,
    PrincipalAuthorityGrantRevisionConflict,
    UpdateAuthorityDelegationCommand,
    UpdatePrincipalAuthorityGrantCommand,
    UpdateWorkspaceMembershipCommand,
    WorkspaceMembershipAlreadyExists,
    WorkspaceMembershipIdempotencyConflict,
    WorkspaceMembershipNotFound,
    WorkspaceMembershipRevisionConflict,
)
from intergrax.contracts.collaborative_work import (
    AuthorityDelegation,
    PrincipalAuthorityGrant,
    WorkspaceMembership,
)

MembershipKey: TypeAlias = tuple[str, str, str]
DelegationKey: TypeAlias = tuple[str, str, str]
AuthorityGrantKey: TypeAlias = tuple[str, str, str]
PrincipalKey: TypeAlias = tuple[str, str, str]
IdempotencyKey: TypeAlias = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class _MembershipIdempotencyEntry:
    fingerprint: str
    original_result: WorkspaceMembership


@dataclass(frozen=True, slots=True)
class _DelegationIdempotencyEntry:
    fingerprint: str
    original_result: AuthorityDelegation


@dataclass(frozen=True, slots=True)
class _AuthorityGrantIdempotencyEntry:
    fingerprint: str
    original_result: PrincipalAuthorityGrant


class InMemoryWorkspaceMembershipRepository:
    """Process-local reference repository for tests and local development."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[MembershipKey, WorkspaceMembership] = {}
        self._idempotency: dict[IdempotencyKey, _MembershipIdempotencyEntry] = {}

    @property
    def capabilities(self) -> CollaborativeWorkRepositoryCapabilities:
        return CollaborativeWorkRepositoryCapabilities(
            backend_id="collaborative_work.membership.in_memory",
            durable=False,
            reference_only=True,
        )

    def create(self, command: CreateWorkspaceMembershipCommand) -> WorkspaceMembership:
        key = self._membership_key(
            command.tenant_id,
            command.workspace_id,
            command.membership_id,
        )
        with self._lock:
            if command.idempotency_key is not None:
                replay = self._replay_membership_create(command)
                if replay is not None:
                    return replay

            if key in self._records:
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
            self._records[key] = record
            self._store_membership_idempotency(command, record)
            return record

    def get(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        membership_id: str,
    ) -> WorkspaceMembership | None:
        key = self._membership_key(tenant_id, workspace_id, membership_id)
        with self._lock:
            record = self._records.get(key)
            if record is None:
                return None
            if not self._scope_matches(record, tenant_id=tenant_id, workspace_id=workspace_id):
                return None
            return record

    def update(self, command: UpdateWorkspaceMembershipCommand) -> WorkspaceMembership:
        key = self._membership_key(
            command.scope.tenant_id,
            command.scope.workspace_id,
            command.scope.membership_id,
        )
        with self._lock:
            current = self._records.get(key)
            if current is None or not self._scope_matches(
                current,
                tenant_id=command.scope.tenant_id,
                workspace_id=command.scope.workspace_id,
            ):
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
            self._records[key] = replacement
            return replacement

    def _replay_membership_create(
        self,
        command: CreateWorkspaceMembershipCommand,
    ) -> WorkspaceMembership | None:
        assert command.idempotency_key is not None
        entry = self._idempotency.get(
            self._idempotency_key(command.tenant_id, command.workspace_id, command.idempotency_key)
        )
        if entry is None:
            return None
        if entry.fingerprint != command.semantic_fingerprint():
            raise WorkspaceMembershipIdempotencyConflict(
                "workspace membership idempotency key conflict"
            )
        return entry.original_result

    def _store_membership_idempotency(
        self,
        command: CreateWorkspaceMembershipCommand,
        record: WorkspaceMembership,
    ) -> None:
        if command.idempotency_key is None:
            return
        self._idempotency[
            self._idempotency_key(command.tenant_id, command.workspace_id, command.idempotency_key)
        ] = _MembershipIdempotencyEntry(
            fingerprint=command.semantic_fingerprint(),
            original_result=record,
        )

    @staticmethod
    def _membership_key(tenant_id: str, workspace_id: str, membership_id: str) -> MembershipKey:
        return (tenant_id.strip(), workspace_id.strip(), membership_id.strip())

    @staticmethod
    def _idempotency_key(tenant_id: str, workspace_id: str, idempotency_key: str) -> IdempotencyKey:
        return (tenant_id.strip(), workspace_id.strip(), idempotency_key.strip())

    @staticmethod
    def _scope_matches(
        record: WorkspaceMembership,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> bool:
        return record.tenant_id == tenant_id.strip() and record.workspace_id == workspace_id.strip()


class InMemoryPrincipalAuthorityRepository:
    """Process-local reference repository for principal base-authority grants."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[AuthorityGrantKey, PrincipalAuthorityGrant] = {}
        self._principal_index: dict[PrincipalKey, AuthorityGrantKey] = {}
        self._idempotency: dict[IdempotencyKey, _AuthorityGrantIdempotencyEntry] = {}

    @property
    def capabilities(self) -> CollaborativeWorkRepositoryCapabilities:
        return CollaborativeWorkRepositoryCapabilities(
            backend_id="collaborative_work.principal_authority.in_memory",
            durable=False,
            reference_only=True,
        )

    def create(self, command: CreatePrincipalAuthorityGrantCommand) -> PrincipalAuthorityGrant:
        grant_key = self._authority_grant_key(
            command.tenant_id,
            command.workspace_id,
            command.authority_grant_id,
        )
        principal_key = self._principal_key(
            command.tenant_id,
            command.workspace_id,
            command.principal_id,
        )
        with self._lock:
            if command.idempotency_key is not None:
                replay = self._replay_authority_grant_create(command)
                if replay is not None:
                    return replay

            if grant_key in self._records:
                raise PrincipalAuthorityGrantAlreadyExists("principal authority grant already exists")
            if principal_key in self._principal_index:
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
            self._records[grant_key] = record
            self._principal_index[principal_key] = grant_key
            self._store_authority_grant_idempotency(command, record)
            return record

    def get(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        authority_grant_id: str,
    ) -> PrincipalAuthorityGrant | None:
        key = self._authority_grant_key(tenant_id, workspace_id, authority_grant_id)
        with self._lock:
            record = self._records.get(key)
            if record is None:
                return None
            if not self._scope_matches(record, tenant_id=tenant_id, workspace_id=workspace_id):
                return None
            return record

    def get_for_principal(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        principal_id: str,
    ) -> PrincipalAuthorityGrant | None:
        principal_key = self._principal_key(tenant_id, workspace_id, principal_id)
        with self._lock:
            grant_key = self._principal_index.get(principal_key)
            if grant_key is None:
                return None
            record = self._records.get(grant_key)
            if record is None:
                return None
            if not self._scope_matches(record, tenant_id=tenant_id, workspace_id=workspace_id):
                return None
            if record.principal_id != principal_id.strip():
                return None
            return record

    def update(self, command: UpdatePrincipalAuthorityGrantCommand) -> PrincipalAuthorityGrant:
        key = self._authority_grant_key(
            command.scope.tenant_id,
            command.scope.workspace_id,
            command.scope.authority_grant_id,
        )
        with self._lock:
            current = self._records.get(key)
            if current is None or not self._scope_matches(
                current,
                tenant_id=command.scope.tenant_id,
                workspace_id=command.scope.workspace_id,
            ):
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
            self._records[key] = replacement
            return replacement

    def _replay_authority_grant_create(
        self,
        command: CreatePrincipalAuthorityGrantCommand,
    ) -> PrincipalAuthorityGrant | None:
        assert command.idempotency_key is not None
        entry = self._idempotency.get(
            self._idempotency_key(command.tenant_id, command.workspace_id, command.idempotency_key)
        )
        if entry is None:
            return None
        if entry.fingerprint != command.semantic_fingerprint():
            raise PrincipalAuthorityGrantIdempotencyConflict(
                "principal authority grant idempotency key conflict"
            )
        return entry.original_result

    def _store_authority_grant_idempotency(
        self,
        command: CreatePrincipalAuthorityGrantCommand,
        record: PrincipalAuthorityGrant,
    ) -> None:
        if command.idempotency_key is None:
            return
        self._idempotency[
            self._idempotency_key(command.tenant_id, command.workspace_id, command.idempotency_key)
        ] = _AuthorityGrantIdempotencyEntry(
            fingerprint=command.semantic_fingerprint(),
            original_result=record,
        )

    @staticmethod
    def _authority_grant_key(
        tenant_id: str,
        workspace_id: str,
        authority_grant_id: str,
    ) -> AuthorityGrantKey:
        return (tenant_id.strip(), workspace_id.strip(), authority_grant_id.strip())

    @staticmethod
    def _principal_key(tenant_id: str, workspace_id: str, principal_id: str) -> PrincipalKey:
        return (tenant_id.strip(), workspace_id.strip(), principal_id.strip())

    @staticmethod
    def _idempotency_key(tenant_id: str, workspace_id: str, idempotency_key: str) -> IdempotencyKey:
        return (tenant_id.strip(), workspace_id.strip(), idempotency_key.strip())

    @staticmethod
    def _scope_matches(
        record: PrincipalAuthorityGrant,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> bool:
        return record.tenant_id == tenant_id.strip() and record.workspace_id == workspace_id.strip()


class InMemoryAuthorityDelegationRepository:
    """Process-local reference repository for tests and local development."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[DelegationKey, AuthorityDelegation] = {}
        self._idempotency: dict[IdempotencyKey, _DelegationIdempotencyEntry] = {}

    @property
    def capabilities(self) -> CollaborativeWorkRepositoryCapabilities:
        return CollaborativeWorkRepositoryCapabilities(
            backend_id="collaborative_work.delegation.in_memory",
            durable=False,
            reference_only=True,
        )

    def create(self, command: CreateAuthorityDelegationCommand) -> AuthorityDelegation:
        key = self._delegation_key(
            command.tenant_id,
            command.workspace_id,
            command.delegation_id,
        )
        with self._lock:
            if command.idempotency_key is not None:
                replay = self._replay_delegation_create(command)
                if replay is not None:
                    return replay

            if key in self._records:
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
            self._records[key] = record
            self._store_delegation_idempotency(command, record)
            return record

    def get(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        delegation_id: str,
    ) -> AuthorityDelegation | None:
        key = self._delegation_key(tenant_id, workspace_id, delegation_id)
        with self._lock:
            record = self._records.get(key)
            if record is None:
                return None
            if not self._scope_matches(record, tenant_id=tenant_id, workspace_id=workspace_id):
                return None
            return record

    def update(self, command: UpdateAuthorityDelegationCommand) -> AuthorityDelegation:
        key = self._delegation_key(
            command.scope.tenant_id,
            command.scope.workspace_id,
            command.scope.delegation_id,
        )
        with self._lock:
            current = self._records.get(key)
            if current is None or not self._scope_matches(
                current,
                tenant_id=command.scope.tenant_id,
                workspace_id=command.scope.workspace_id,
            ):
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
            self._records[key] = replacement
            return replacement

    def _replay_delegation_create(
        self,
        command: CreateAuthorityDelegationCommand,
    ) -> AuthorityDelegation | None:
        assert command.idempotency_key is not None
        entry = self._idempotency.get(
            self._idempotency_key(command.tenant_id, command.workspace_id, command.idempotency_key)
        )
        if entry is None:
            return None
        if entry.fingerprint != command.semantic_fingerprint():
            raise AuthorityDelegationIdempotencyConflict(
                "authority delegation idempotency key conflict"
            )
        return entry.original_result

    def _store_delegation_idempotency(
        self,
        command: CreateAuthorityDelegationCommand,
        record: AuthorityDelegation,
    ) -> None:
        if command.idempotency_key is None:
            return
        self._idempotency[
            self._idempotency_key(command.tenant_id, command.workspace_id, command.idempotency_key)
        ] = _DelegationIdempotencyEntry(
            fingerprint=command.semantic_fingerprint(),
            original_result=record,
        )

    @staticmethod
    def _delegation_key(tenant_id: str, workspace_id: str, delegation_id: str) -> DelegationKey:
        return (tenant_id.strip(), workspace_id.strip(), delegation_id.strip())

    @staticmethod
    def _idempotency_key(tenant_id: str, workspace_id: str, idempotency_key: str) -> IdempotencyKey:
        return (tenant_id.strip(), workspace_id.strip(), idempotency_key.strip())

    @staticmethod
    def _scope_matches(
        record: AuthorityDelegation,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> bool:
        return record.tenant_id == tenant_id.strip() and record.workspace_id == workspace_id.strip()
