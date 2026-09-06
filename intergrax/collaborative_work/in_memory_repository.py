# © Artur Czarnecki. All rights reserved.

"""In-memory reference repositories for Collaborative Work (COLLAB-WORK-1B, COLLAB-WORK-2B)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TypeAlias

from intergrax.collaborative_work.repository import (
    AssignmentAlreadyExists,
    AssignmentIdempotencyConflict,
    AssignmentNotFound,
    AssignmentRevisionConflict,
    AuthorityDelegationAlreadyExists,
    AuthorityDelegationIdempotencyConflict,
    AuthorityDelegationNotFound,
    AuthorityDelegationRevisionConflict,
    CollaborativePolicyRuleAlreadyExists,
    CollaborativePolicyRuleIdempotencyConflict,
    CollaborativePolicyRuleNotFound,
    CollaborativePolicyRuleRevisionConflict,
    CollaborativeOperationPolicyProfileAlreadyExists,
    CollaborativeOperationPolicyProfileIdempotencyConflict,
    CollaborativeOperationPolicyProfileNotFound,
    CollaborativeOperationPolicyProfileRevisionConflict,
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

MembershipKey: TypeAlias = tuple[str, str, str]
DelegationKey: TypeAlias = tuple[str, str, str]
AuthorityGrantKey: TypeAlias = tuple[str, str, str]
PrincipalKey: TypeAlias = tuple[str, str, str]
PolicyRuleKey: TypeAlias = tuple[str, str, str]
PolicyExactKey: TypeAlias = tuple[str, str, str, str, str]
OperationProfileKey: TypeAlias = tuple[str, str, str]
WorkItemKey: TypeAlias = tuple[str, str, str]
AssignmentKey: TypeAlias = tuple[str, str, str]
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


@dataclass(frozen=True, slots=True)
class _PolicyRuleIdempotencyEntry:
    fingerprint: str
    original_result: CollaborativePolicyRule


@dataclass(frozen=True, slots=True)
class _OperationProfileIdempotencyEntry:
    fingerprint: str
    original_result: CollaborativeOperationPolicyProfile


@dataclass(frozen=True, slots=True)
class _WorkItemIdempotencyEntry:
    fingerprint: str
    original_result: WorkItem


@dataclass(frozen=True, slots=True)
class _AssignmentIdempotencyEntry:
    fingerprint: str
    original_result: Assignment


class InMemoryWorkspaceMembershipRepository:
    """Process-local reference repository for tests and local development."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[MembershipKey, WorkspaceMembership] = {}
        self._principal_index: dict[PrincipalKey, MembershipKey] = {}
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
        principal_key = self._principal_key(
            command.tenant_id,
            command.workspace_id,
            command.principal_id,
        )
        with self._lock:
            if command.idempotency_key is not None:
                replay = self._replay_membership_create(command)
                if replay is not None:
                    return replay

            if key in self._records:
                raise WorkspaceMembershipAlreadyExists("workspace membership already exists")
            if principal_key in self._principal_index:
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
            self._principal_index[principal_key] = key
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

    def get_for_principal(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        principal_id: str,
    ) -> WorkspaceMembership | None:
        principal_key = self._principal_key(tenant_id, workspace_id, principal_id)
        with self._lock:
            membership_key = self._principal_index.get(principal_key)
            if membership_key is None:
                return None
            record = self._records.get(membership_key)
            if record is None:
                return None
            if not self._scope_matches(record, tenant_id=tenant_id, workspace_id=workspace_id):
                return None
            if record.principal_id != principal_id.strip():
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

    @staticmethod
    def _principal_key(tenant_id: str, workspace_id: str, principal_id: str) -> PrincipalKey:
        return (tenant_id.strip(), workspace_id.strip(), principal_id.strip())


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


class InMemoryCollaborativePolicyRepository:
    """Process-local reference repository for workspace and resource policy rules."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[PolicyRuleKey, CollaborativePolicyRule] = {}
        self._policy_key_index: dict[PolicyExactKey, PolicyRuleKey] = {}
        self._idempotency: dict[IdempotencyKey, _PolicyRuleIdempotencyEntry] = {}

    @property
    def capabilities(self) -> CollaborativeWorkRepositoryCapabilities:
        return CollaborativeWorkRepositoryCapabilities(
            backend_id="collaborative_work.policy.in_memory",
            durable=False,
            reference_only=True,
        )

    def create(self, command: CreateCollaborativePolicyRuleCommand) -> CollaborativePolicyRule:
        rule_key = self._policy_rule_key(
            command.tenant_id,
            command.workspace_id,
            command.policy_rule_id,
        )
        exact_key = self._exact_policy_key(
            tenant_id=command.tenant_id,
            workspace_id=command.workspace_id,
            layer=command.layer,
            authority_scope=command.authority_scope,
            resource_scope=command.resource_scope,
        )
        with self._lock:
            if command.idempotency_key is not None:
                replay = self._replay_policy_rule_create(command)
                if replay is not None:
                    return replay

            if rule_key in self._records:
                raise CollaborativePolicyRuleAlreadyExists("collaborative policy rule already exists")
            if exact_key in self._policy_key_index:
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
            self._records[rule_key] = record
            self._policy_key_index[exact_key] = rule_key
            self._store_policy_rule_idempotency(command, record)
            return record

    def get(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        policy_rule_id: str,
    ) -> CollaborativePolicyRule | None:
        key = self._policy_rule_key(tenant_id, workspace_id, policy_rule_id)
        with self._lock:
            record = self._records.get(key)
            if record is None:
                return None
            if not self._scope_matches(record, tenant_id=tenant_id, workspace_id=workspace_id):
                return None
            return record

    def get_effective_rule(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        layer: PolicyCompositionLayer,
        authority_scope: str,
        resource_scope: str | None = None,
    ) -> CollaborativePolicyRule | None:
        exact_key = self._exact_policy_key(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            layer=layer,
            authority_scope=authority_scope,
            resource_scope=resource_scope,
        )
        with self._lock:
            rule_key = self._policy_key_index.get(exact_key)
            if rule_key is None:
                return None
            record = self._records.get(rule_key)
            if record is None:
                return None
            if not self._scope_matches(record, tenant_id=tenant_id, workspace_id=workspace_id):
                return None
            return record

    def update(self, command: UpdateCollaborativePolicyRuleCommand) -> CollaborativePolicyRule:
        key = self._policy_rule_key(
            command.scope.tenant_id,
            command.scope.workspace_id,
            command.scope.policy_rule_id,
        )
        with self._lock:
            current = self._records.get(key)
            if current is None or not self._scope_matches(
                current,
                tenant_id=command.scope.tenant_id,
                workspace_id=command.scope.workspace_id,
            ):
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
            self._records[key] = replacement
            return replacement

    def _replay_policy_rule_create(
        self,
        command: CreateCollaborativePolicyRuleCommand,
    ) -> CollaborativePolicyRule | None:
        assert command.idempotency_key is not None
        entry = self._idempotency.get(
            self._idempotency_key(command.tenant_id, command.workspace_id, command.idempotency_key)
        )
        if entry is None:
            return None
        if entry.fingerprint != command.semantic_fingerprint():
            raise CollaborativePolicyRuleIdempotencyConflict(
                "collaborative policy rule idempotency key conflict"
            )
        return entry.original_result

    def _store_policy_rule_idempotency(
        self,
        command: CreateCollaborativePolicyRuleCommand,
        record: CollaborativePolicyRule,
    ) -> None:
        if command.idempotency_key is None:
            return
        self._idempotency[
            self._idempotency_key(command.tenant_id, command.workspace_id, command.idempotency_key)
        ] = _PolicyRuleIdempotencyEntry(
            fingerprint=command.semantic_fingerprint(),
            original_result=record,
        )

    @staticmethod
    def _policy_rule_key(tenant_id: str, workspace_id: str, policy_rule_id: str) -> PolicyRuleKey:
        return (tenant_id.strip(), workspace_id.strip(), policy_rule_id.strip())

    @staticmethod
    def _exact_policy_key(
        *,
        tenant_id: str,
        workspace_id: str,
        layer: PolicyCompositionLayer,
        authority_scope: str,
        resource_scope: str | None,
    ) -> PolicyExactKey:
        normalized_resource = "" if resource_scope is None else resource_scope.strip()
        return (
            tenant_id.strip(),
            workspace_id.strip(),
            layer.value,
            authority_scope.strip(),
            normalized_resource,
        )

    @staticmethod
    def _idempotency_key(tenant_id: str, workspace_id: str, idempotency_key: str) -> IdempotencyKey:
        return (tenant_id.strip(), workspace_id.strip(), idempotency_key.strip())

    @staticmethod
    def _scope_matches(
        record: CollaborativePolicyRule,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> bool:
        return record.tenant_id == tenant_id.strip() and record.workspace_id == workspace_id.strip()


class InMemoryCollaborativeOperationPolicyProfileRepository:
    """Process-local reference repository for operation policy profiles."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[OperationProfileKey, CollaborativeOperationPolicyProfile] = {}
        self._idempotency: dict[IdempotencyKey, _OperationProfileIdempotencyEntry] = {}

    @property
    def capabilities(self) -> CollaborativeWorkRepositoryCapabilities:
        return CollaborativeWorkRepositoryCapabilities(
            backend_id="collaborative_work.operation_profile.in_memory",
            durable=False,
            reference_only=True,
        )

    def create(
        self,
        command: CreateCollaborativeOperationPolicyProfileCommand,
    ) -> CollaborativeOperationPolicyProfile:
        key = self._operation_key(
            command.tenant_id,
            command.workspace_id,
            command.operation_id,
        )
        with self._lock:
            if command.idempotency_key is not None:
                replay = self._replay_profile_create(command)
                if replay is not None:
                    return replay

            if key in self._records:
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
            self._records[key] = record
            self._store_profile_idempotency(command, record)
            return record

    def get_for_operation(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        operation_id: str,
    ) -> CollaborativeOperationPolicyProfile | None:
        key = self._operation_key(tenant_id, workspace_id, operation_id)
        with self._lock:
            record = self._records.get(key)
            if record is None:
                return None
            if not self._scope_matches(record, tenant_id=tenant_id, workspace_id=workspace_id):
                return None
            return record

    def update(
        self,
        command: UpdateCollaborativeOperationPolicyProfileCommand,
    ) -> CollaborativeOperationPolicyProfile:
        key = self._operation_key(
            command.scope.tenant_id,
            command.scope.workspace_id,
            command.scope.operation_id,
        )
        with self._lock:
            current = self._records.get(key)
            if current is None or not self._scope_matches(
                current,
                tenant_id=command.scope.tenant_id,
                workspace_id=command.scope.workspace_id,
            ):
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
            self._records[key] = replacement
            return replacement

    def _replay_profile_create(
        self,
        command: CreateCollaborativeOperationPolicyProfileCommand,
    ) -> CollaborativeOperationPolicyProfile | None:
        assert command.idempotency_key is not None
        entry = self._idempotency.get(
            self._idempotency_key(command.tenant_id, command.workspace_id, command.idempotency_key)
        )
        if entry is None:
            return None
        if entry.fingerprint != command.semantic_fingerprint():
            raise CollaborativeOperationPolicyProfileIdempotencyConflict(
                "operation policy profile idempotency key conflict"
            )
        return entry.original_result

    def _store_profile_idempotency(
        self,
        command: CreateCollaborativeOperationPolicyProfileCommand,
        record: CollaborativeOperationPolicyProfile,
    ) -> None:
        if command.idempotency_key is None:
            return
        self._idempotency[
            self._idempotency_key(command.tenant_id, command.workspace_id, command.idempotency_key)
        ] = _OperationProfileIdempotencyEntry(
            fingerprint=command.semantic_fingerprint(),
            original_result=record,
        )

    @staticmethod
    def _operation_key(tenant_id: str, workspace_id: str, operation_id: str) -> OperationProfileKey:
        return (tenant_id.strip(), workspace_id.strip(), operation_id.strip())

    @staticmethod
    def _idempotency_key(tenant_id: str, workspace_id: str, idempotency_key: str) -> IdempotencyKey:
        return (tenant_id.strip(), workspace_id.strip(), idempotency_key.strip())

    @staticmethod
    def _scope_matches(
        record: CollaborativeOperationPolicyProfile,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> bool:
        return record.tenant_id == tenant_id.strip() and record.workspace_id == workspace_id.strip()


class InMemoryWorkItemRepository:
    """Process-local reference repository for WorkItem records."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[WorkItemKey, WorkItem] = {}
        self._idempotency: dict[IdempotencyKey, _WorkItemIdempotencyEntry] = {}

    @property
    def capabilities(self) -> CollaborativeWorkRepositoryCapabilities:
        return CollaborativeWorkRepositoryCapabilities(
            backend_id="collaborative_work.work_item.in_memory",
            durable=False,
            reference_only=True,
        )

    def create(self, command: CreateWorkItemCommand) -> WorkItem:
        key = self._work_item_key(
            command.tenant_id,
            command.workspace_id,
            command.work_item_id,
        )
        with self._lock:
            if command.idempotency_key is not None:
                replay = self._replay_work_item_create(command)
                if replay is not None:
                    return replay

            if key in self._records:
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
            self._records[key] = record
            self._store_work_item_idempotency(command, record)
            return record

    def get(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        work_item_id: str,
    ) -> WorkItem | None:
        key = self._work_item_key(tenant_id, workspace_id, work_item_id)
        with self._lock:
            record = self._records.get(key)
            if record is None:
                return None
            if not self._scope_matches(record, tenant_id=tenant_id, workspace_id=workspace_id):
                return None
            return record

    def update(self, command: UpdateWorkItemCommand) -> WorkItem:
        key = self._work_item_key(
            command.scope.tenant_id,
            command.scope.workspace_id,
            command.scope.work_item_id,
        )
        with self._lock:
            current = self._records.get(key)
            if current is None or not self._scope_matches(
                current,
                tenant_id=command.scope.tenant_id,
                workspace_id=command.scope.workspace_id,
            ):
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
            self._records[key] = replacement
            return replacement

    def _replay_work_item_create(self, command: CreateWorkItemCommand) -> WorkItem | None:
        assert command.idempotency_key is not None
        entry = self._idempotency.get(
            self._idempotency_key(command.tenant_id, command.workspace_id, command.idempotency_key)
        )
        if entry is None:
            return None
        if entry.fingerprint != command.semantic_fingerprint():
            raise WorkItemIdempotencyConflict("work item idempotency key conflict")
        return entry.original_result

    def _store_work_item_idempotency(
        self,
        command: CreateWorkItemCommand,
        record: WorkItem,
    ) -> None:
        if command.idempotency_key is None:
            return
        self._idempotency[
            self._idempotency_key(command.tenant_id, command.workspace_id, command.idempotency_key)
        ] = _WorkItemIdempotencyEntry(
            fingerprint=command.semantic_fingerprint(),
            original_result=record,
        )

    @staticmethod
    def _work_item_key(tenant_id: str, workspace_id: str, work_item_id: str) -> WorkItemKey:
        return (tenant_id.strip(), workspace_id.strip(), work_item_id.strip())

    @staticmethod
    def _idempotency_key(tenant_id: str, workspace_id: str, idempotency_key: str) -> IdempotencyKey:
        return (tenant_id.strip(), workspace_id.strip(), idempotency_key.strip())

    @staticmethod
    def _scope_matches(
        record: WorkItem,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> bool:
        return record.tenant_id == tenant_id.strip() and record.workspace_id == workspace_id.strip()


class InMemoryAssignmentRepository:
    """Process-local reference repository for Assignment records."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[AssignmentKey, Assignment] = {}
        self._idempotency: dict[IdempotencyKey, _AssignmentIdempotencyEntry] = {}

    @property
    def capabilities(self) -> CollaborativeWorkRepositoryCapabilities:
        return CollaborativeWorkRepositoryCapabilities(
            backend_id="collaborative_work.assignment.in_memory",
            durable=False,
            reference_only=True,
        )

    def create(self, command: CreateAssignmentCommand) -> Assignment:
        key = self._assignment_key(
            command.tenant_id,
            command.workspace_id,
            command.assignment_id,
        )
        with self._lock:
            if command.idempotency_key is not None:
                replay = self._replay_assignment_create(command)
                if replay is not None:
                    return replay

            if key in self._records:
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
            self._records[key] = record
            self._store_assignment_idempotency(command, record)
            return record

    def get(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        assignment_id: str,
    ) -> Assignment | None:
        key = self._assignment_key(tenant_id, workspace_id, assignment_id)
        with self._lock:
            record = self._records.get(key)
            if record is None:
                return None
            if not self._scope_matches(record, tenant_id=tenant_id, workspace_id=workspace_id):
                return None
            return record

    def update(self, command: UpdateAssignmentCommand) -> Assignment:
        key = self._assignment_key(
            command.scope.tenant_id,
            command.scope.workspace_id,
            command.scope.assignment_id,
        )
        with self._lock:
            current = self._records.get(key)
            if current is None or not self._scope_matches(
                current,
                tenant_id=command.scope.tenant_id,
                workspace_id=command.scope.workspace_id,
            ):
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
            self._records[key] = replacement
            return replacement

    def _replay_assignment_create(
        self,
        command: CreateAssignmentCommand,
    ) -> Assignment | None:
        assert command.idempotency_key is not None
        entry = self._idempotency.get(
            self._idempotency_key(command.tenant_id, command.workspace_id, command.idempotency_key)
        )
        if entry is None:
            return None
        if entry.fingerprint != command.semantic_fingerprint():
            raise AssignmentIdempotencyConflict("assignment idempotency key conflict")
        return entry.original_result

    def _store_assignment_idempotency(
        self,
        command: CreateAssignmentCommand,
        record: Assignment,
    ) -> None:
        if command.idempotency_key is None:
            return
        self._idempotency[
            self._idempotency_key(command.tenant_id, command.workspace_id, command.idempotency_key)
        ] = _AssignmentIdempotencyEntry(
            fingerprint=command.semantic_fingerprint(),
            original_result=record,
        )

    @staticmethod
    def _assignment_key(tenant_id: str, workspace_id: str, assignment_id: str) -> AssignmentKey:
        return (tenant_id.strip(), workspace_id.strip(), assignment_id.strip())

    @staticmethod
    def _idempotency_key(tenant_id: str, workspace_id: str, idempotency_key: str) -> IdempotencyKey:
        return (tenant_id.strip(), workspace_id.strip(), idempotency_key.strip())

    @staticmethod
    def _scope_matches(
        record: Assignment,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> bool:
        return record.tenant_id == tenant_id.strip() and record.workspace_id == workspace_id.strip()
