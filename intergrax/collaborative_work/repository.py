# © Artur Czarnecki. All rights reserved.

"""Collaborative Work repository contracts (COLLAB-WORK-1B, COLLAB-WORK-2B).

Provider-neutral persistence ports for authoritative Collaborative Work records,
including MP-1 membership/delegation/authority/policy state and MP-2 WorkItem and
Assignment shared-work state.

Revision ownership
------------------
The repository is authoritative for revision advancement. Callers supply
replacement semantic values and ``expected_revision`` on update; the repository
writes an immutable replacement record at ``expected_revision + 1``. Callers
must not supply arbitrary revision numbers on create or update.

Concurrency
-----------
Mutating an existing record requires ``expected_revision`` equal to the stored
revision. A mismatch raises a typed revision-conflict exception and leaves
stored state unchanged.

Isolation
---------
Every lookup and mutation requires explicit ``tenant_id`` and ``workspace_id``
scope keys in addition to the record identifier. Cross-scope access behaves as
not found.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.collaborative_work import (
    Assignment,
    AssignmentState,
    AuthorityDelegation,
    AuthorityGrantStatus,
    CollaborativeOperationPolicyProfile,
    CollaborativeOperationPolicyProfileStatus,
    CollaborativePolicyRule,
    CollaborativePolicyRuleStatus,
    DelegationStatus,
    MembershipStatus,
    OperationPolicyRequirement,
    PolicyCompositionLayer,
    PolicyLayerApplicability,
    PrincipalAuthorityGrant,
    WorkItem,
    WorkItemState,
    WorkspaceMembership,
    WorkspaceMembershipRole,
)
from intergrax.contracts.runtime_policy import PolicyAction

INITIAL_RECORD_REVISION: int = 0

_NON_EMPTY = Field(min_length=1)


class WorkspaceMembershipNotFound(Exception):
    """Membership was not found for the requested tenant/workspace scope."""


class WorkspaceMembershipAlreadyExists(Exception):
    """Membership already exists for the requested scoped identity."""


class WorkspaceMembershipRevisionConflict(Exception):
    """Optimistic revision conflict for workspace membership."""


class WorkspaceMembershipIdempotencyConflict(Exception):
    """Idempotency key replayed with a different semantic command."""


class AuthorityDelegationNotFound(Exception):
    """Delegation was not found for the requested tenant/workspace scope."""


class AuthorityDelegationAlreadyExists(Exception):
    """Delegation already exists for the requested scoped identity."""


class AuthorityDelegationRevisionConflict(Exception):
    """Optimistic revision conflict for authority delegation."""


class AuthorityDelegationIdempotencyConflict(Exception):
    """Idempotency key replayed with a different semantic command."""


class PrincipalAuthorityGrantNotFound(Exception):
    """Authority grant was not found for the requested tenant/workspace scope."""


class PrincipalAuthorityGrantAlreadyExists(Exception):
    """Authority grant already exists for the requested scoped identity."""


class PrincipalAuthorityGrantRevisionConflict(Exception):
    """Optimistic revision conflict for principal authority grant."""


class PrincipalAuthorityGrantIdempotencyConflict(Exception):
    """Idempotency key replayed with a different semantic command."""


class _RepositoryModelBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @staticmethod
    def _strip_required(value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must be non-empty")
        return cleaned

    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return cls._strip_required(value)


class CollaborativeWorkRepositoryCapabilities(_RepositoryModelBase):
    """Declared backend capabilities for collaborative work repositories."""

    backend_id: str = _NON_EMPTY
    durable: bool
    reference_only: bool

    @field_validator("backend_id")
    @classmethod
    def _validate_backend_id(cls, value: str) -> str:
        return cls._strip_required(value)


class WorkspaceMembershipScopeKey(_RepositoryModelBase):
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    membership_id: str = _NON_EMPTY

    @field_validator("tenant_id", "workspace_id", "membership_id")
    @classmethod
    def _strip_scope_fields(cls, value: str) -> str:
        return cls._strip_required(value)


class AuthorityDelegationScopeKey(_RepositoryModelBase):
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    delegation_id: str = _NON_EMPTY

    @field_validator("tenant_id", "workspace_id", "delegation_id")
    @classmethod
    def _strip_scope_fields(cls, value: str) -> str:
        return cls._strip_required(value)


class PrincipalAuthorityGrantScopeKey(_RepositoryModelBase):
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    authority_grant_id: str = _NON_EMPTY

    @field_validator("tenant_id", "workspace_id", "authority_grant_id")
    @classmethod
    def _strip_scope_fields(cls, value: str) -> str:
        return cls._strip_required(value)


class CreateWorkspaceMembershipCommand(_RepositoryModelBase):
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    membership_id: str = _NON_EMPTY
    principal_id: str = _NON_EMPTY
    role: WorkspaceMembershipRole
    status: MembershipStatus = MembershipStatus.ACTIVE
    idempotency_key: str | None = None

    @field_validator(
        "tenant_id",
        "workspace_id",
        "membership_id",
        "principal_id",
        "idempotency_key",
    )
    @classmethod
    def _strip_fields(cls, value: str | None) -> str | None:
        return cls._strip_optional(value)

    def semantic_fingerprint(self) -> str:
        payload = {
            "tenant_id": self.tenant_id,
            "workspace_id": self.workspace_id,
            "membership_id": self.membership_id,
            "principal_id": self.principal_id,
            "role": self.role.value,
            "status": self.status.value,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class CreatePrincipalAuthorityGrantCommand(_RepositoryModelBase):
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    authority_grant_id: str = _NON_EMPTY
    principal_id: str = _NON_EMPTY
    authority_scopes: tuple[str, ...]
    status: AuthorityGrantStatus = AuthorityGrantStatus.ACTIVE
    idempotency_key: str | None = None

    @field_validator("tenant_id", "workspace_id", "authority_grant_id", "principal_id")
    @classmethod
    def _strip_required_fields(cls, value: str) -> str:
        return cls._strip_required(value)

    @field_validator("idempotency_key")
    @classmethod
    def _strip_idempotency(cls, value: str | None) -> str | None:
        return cls._strip_optional(value)

    @field_validator("authority_scopes")
    @classmethod
    def _normalize_authority_scopes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("authority_scopes must be non-empty")
        return tuple(cls._strip_required(scope) for scope in value)

    def semantic_fingerprint(self) -> str:
        payload = {
            "tenant_id": self.tenant_id,
            "workspace_id": self.workspace_id,
            "authority_grant_id": self.authority_grant_id,
            "principal_id": self.principal_id,
            "authority_scopes": list(self.authority_scopes),
            "status": self.status.value,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class UpdatePrincipalAuthorityGrantCommand(_RepositoryModelBase):
    scope: PrincipalAuthorityGrantScopeKey
    expected_revision: int = Field(ge=0)
    authority_scopes: tuple[str, ...]
    status: AuthorityGrantStatus

    @field_validator("authority_scopes")
    @classmethod
    def _normalize_authority_scopes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("authority_scopes must be non-empty")
        return tuple(cls._strip_required(scope) for scope in value)


class UpdateWorkspaceMembershipCommand(_RepositoryModelBase):
    scope: WorkspaceMembershipScopeKey
    expected_revision: int = Field(ge=0)
    role: WorkspaceMembershipRole
    status: MembershipStatus


class CreateAuthorityDelegationCommand(_RepositoryModelBase):
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    delegation_id: str = _NON_EMPTY
    delegator_principal_id: str = _NON_EMPTY
    delegate_principal_id: str = _NON_EMPTY
    authority_scopes: tuple[str, ...]
    resource_scope: str | None = None
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    status: DelegationStatus = DelegationStatus.ACTIVE
    idempotency_key: str | None = None

    @field_validator(
        "tenant_id",
        "workspace_id",
        "delegation_id",
        "delegator_principal_id",
        "delegate_principal_id",
    )
    @classmethod
    def _strip_required_fields(cls, value: str) -> str:
        return cls._strip_required(value)

    @field_validator("resource_scope", "idempotency_key")
    @classmethod
    def _strip_optional_fields(cls, value: str | None) -> str | None:
        return cls._strip_optional(value)

    @field_validator("authority_scopes")
    @classmethod
    def _normalize_authority_scopes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("authority_scopes must be non-empty")
        return tuple(cls._strip_required(scope) for scope in value)

    def semantic_fingerprint(self) -> str:
        payload = {
            "tenant_id": self.tenant_id,
            "workspace_id": self.workspace_id,
            "delegation_id": self.delegation_id,
            "delegator_principal_id": self.delegator_principal_id,
            "delegate_principal_id": self.delegate_principal_id,
            "authority_scopes": list(self.authority_scopes),
            "resource_scope": self.resource_scope,
            "valid_from": self.valid_from.isoformat() if self.valid_from is not None else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until is not None else None,
            "status": self.status.value,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class UpdateAuthorityDelegationCommand(_RepositoryModelBase):
    scope: AuthorityDelegationScopeKey
    expected_revision: int = Field(ge=0)
    authority_scopes: tuple[str, ...]
    resource_scope: str | None
    valid_from: datetime | None
    valid_until: datetime | None
    status: DelegationStatus

    @field_validator("authority_scopes")
    @classmethod
    def _normalize_authority_scopes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("authority_scopes must be non-empty")
        return tuple(cls._strip_required(scope) for scope in value)

    @field_validator("resource_scope")
    @classmethod
    def _strip_resource_scope(cls, value: str | None) -> str | None:
        return cls._strip_optional(value)


@runtime_checkable
class WorkspaceMembershipRepository(Protocol):
    """Authoritative persistence port for workspace membership records."""

    @property
    def capabilities(self) -> CollaborativeWorkRepositoryCapabilities:
        """Return declared repository backend capabilities."""

    def create(self, command: CreateWorkspaceMembershipCommand) -> WorkspaceMembership:
        """Create a membership at ``INITIAL_RECORD_REVISION``."""

    def get(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        membership_id: str,
    ) -> WorkspaceMembership | None:
        """Return membership for the scoped identity or ``None``."""

    def get_for_principal(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        principal_id: str,
    ) -> WorkspaceMembership | None:
        """Return the canonical membership for a principal in workspace scope or ``None``."""

    def update(self, command: UpdateWorkspaceMembershipCommand) -> WorkspaceMembership:
        """Replace membership semantics under optimistic concurrency."""


@runtime_checkable
class AuthorityDelegationRepository(Protocol):
    """Authoritative persistence port for authority delegation records."""

    @property
    def capabilities(self) -> CollaborativeWorkRepositoryCapabilities:
        """Return declared repository backend capabilities."""

    def create(self, command: CreateAuthorityDelegationCommand) -> AuthorityDelegation:
        """Create a delegation at ``INITIAL_RECORD_REVISION``."""

    def get(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        delegation_id: str,
    ) -> AuthorityDelegation | None:
        """Return delegation for the scoped identity or ``None``."""

    def update(self, command: UpdateAuthorityDelegationCommand) -> AuthorityDelegation:
        """Replace delegation semantics under optimistic concurrency."""


@runtime_checkable
class PrincipalAuthorityRepository(Protocol):
    """Authoritative persistence port for principal base-authority grant records."""

    @property
    def capabilities(self) -> CollaborativeWorkRepositoryCapabilities:
        """Return declared repository backend capabilities."""

    def create(self, command: CreatePrincipalAuthorityGrantCommand) -> PrincipalAuthorityGrant:
        """Create an authority grant at ``INITIAL_RECORD_REVISION``."""

    def get(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        authority_grant_id: str,
    ) -> PrincipalAuthorityGrant | None:
        """Return authority grant for the scoped identity or ``None``."""

    def get_for_principal(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        principal_id: str,
    ) -> PrincipalAuthorityGrant | None:
        """Return the authoritative grant for a principal in workspace scope or ``None``."""

    def update(self, command: UpdatePrincipalAuthorityGrantCommand) -> PrincipalAuthorityGrant:
        """Replace authority grant semantics under optimistic concurrency."""


class CollaborativePolicyRuleNotFound(Exception):
    """Policy rule was not found for the requested tenant/workspace scope."""


class CollaborativePolicyRuleAlreadyExists(Exception):
    """Policy rule or exact policy key already exists in workspace scope."""


class CollaborativePolicyRuleRevisionConflict(Exception):
    """Optimistic revision conflict for collaborative policy rule."""


class CollaborativePolicyRuleIdempotencyConflict(Exception):
    """Idempotency key replayed with a different semantic command."""


class CollaborativePolicyRuleScopeKey(_RepositoryModelBase):
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    policy_rule_id: str = _NON_EMPTY

    @field_validator("tenant_id", "workspace_id", "policy_rule_id")
    @classmethod
    def _strip_scope_fields(cls, value: str) -> str:
        return cls._strip_required(value)


class CreateCollaborativePolicyRuleCommand(_RepositoryModelBase):
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    policy_rule_id: str = _NON_EMPTY
    layer: PolicyCompositionLayer
    authority_scope: str = _NON_EMPTY
    action: PolicyAction
    resource_scope: str | None = None
    status: CollaborativePolicyRuleStatus = CollaborativePolicyRuleStatus.ACTIVE
    idempotency_key: str | None = None

    @field_validator("tenant_id", "workspace_id", "policy_rule_id", "authority_scope")
    @classmethod
    def _strip_required_fields(cls, value: str) -> str:
        return cls._strip_required(value)

    @field_validator("resource_scope", "idempotency_key")
    @classmethod
    def _strip_optional_fields(cls, value: str | None) -> str | None:
        return cls._strip_optional(value)

    @model_validator(mode="after")
    def _validate_layer_resource_scope(self) -> Self:
        if self.layer not in (
            PolicyCompositionLayer.WORKSPACE_POLICY,
            PolicyCompositionLayer.RESOURCE_POLICY,
        ):
            raise ValueError("layer must be WORKSPACE_POLICY or RESOURCE_POLICY")
        if self.layer is PolicyCompositionLayer.WORKSPACE_POLICY and self.resource_scope is not None:
            raise ValueError("WORKSPACE_POLICY rules must not include resource_scope")
        if self.layer is PolicyCompositionLayer.RESOURCE_POLICY and self.resource_scope is None:
            raise ValueError("RESOURCE_POLICY rules require resource_scope")
        return self

    def semantic_fingerprint(self) -> str:
        payload = {
            "tenant_id": self.tenant_id,
            "workspace_id": self.workspace_id,
            "policy_rule_id": self.policy_rule_id,
            "layer": self.layer.value,
            "authority_scope": self.authority_scope,
            "action": self.action.value,
            "resource_scope": self.resource_scope,
            "status": self.status.value,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class UpdateCollaborativePolicyRuleCommand(_RepositoryModelBase):
    scope: CollaborativePolicyRuleScopeKey
    expected_revision: int = Field(ge=0)
    action: PolicyAction
    status: CollaborativePolicyRuleStatus


@runtime_checkable
class CollaborativePolicyRepository(Protocol):
    """Authoritative persistence port for workspace and resource policy rules."""

    @property
    def capabilities(self) -> CollaborativeWorkRepositoryCapabilities:
        """Return declared repository backend capabilities."""

    def create(self, command: CreateCollaborativePolicyRuleCommand) -> CollaborativePolicyRule:
        """Create a policy rule at ``INITIAL_RECORD_REVISION``."""

    def get(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        policy_rule_id: str,
    ) -> CollaborativePolicyRule | None:
        """Return policy rule for the scoped identity or ``None``."""

    def get_effective_rule(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        layer: PolicyCompositionLayer,
        authority_scope: str,
        resource_scope: str | None = None,
    ) -> CollaborativePolicyRule | None:
        """Return the canonical rule for an exact policy key or ``None``.

        Does not filter by status; evaluators interpret ``ACTIVE`` vs ``DISABLED``.
        """

    def update(self, command: UpdateCollaborativePolicyRuleCommand) -> CollaborativePolicyRule:
        """Replace policy rule semantics under optimistic concurrency."""


class CollaborativeOperationPolicyProfileNotFound(Exception):
    """Operation policy profile was not found for the requested tenant/workspace scope."""


class CollaborativeOperationPolicyProfileAlreadyExists(Exception):
    """Operation policy profile already exists for the requested scoped identity."""


class CollaborativeOperationPolicyProfileRevisionConflict(Exception):
    """Optimistic revision conflict for operation policy profile."""


class CollaborativeOperationPolicyProfileIdempotencyConflict(Exception):
    """Idempotency key replayed with a different semantic command."""


class CollaborativeOperationPolicyProfileScopeKey(_RepositoryModelBase):
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    operation_id: str = _NON_EMPTY

    @field_validator("tenant_id", "workspace_id", "operation_id")
    @classmethod
    def _strip_scope_fields(cls, value: str) -> str:
        return cls._strip_required(value)


class CreateCollaborativeOperationPolicyProfileCommand(_RepositoryModelBase):
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    operation_id: str = _NON_EMPTY
    authority_scope: str = _NON_EMPTY
    workspace_policy_applicability: PolicyLayerApplicability
    resource_policy_applicability: PolicyLayerApplicability
    runtime_policy_applicability: PolicyLayerApplicability
    resource_requirement: OperationPolicyRequirement
    meaningful_side_effect_requirement: OperationPolicyRequirement
    status: CollaborativeOperationPolicyProfileStatus = (
        CollaborativeOperationPolicyProfileStatus.ACTIVE
    )
    idempotency_key: str | None = None

    @field_validator("tenant_id", "workspace_id", "operation_id", "authority_scope")
    @classmethod
    def _strip_required_fields(cls, value: str) -> str:
        return cls._strip_required(value)

    @field_validator("idempotency_key")
    @classmethod
    def _strip_idempotency(cls, value: str | None) -> str | None:
        return cls._strip_optional(value)

    def semantic_fingerprint(self) -> str:
        payload = {
            "tenant_id": self.tenant_id,
            "workspace_id": self.workspace_id,
            "operation_id": self.operation_id,
            "authority_scope": self.authority_scope,
            "workspace_policy_applicability": self.workspace_policy_applicability.value,
            "resource_policy_applicability": self.resource_policy_applicability.value,
            "runtime_policy_applicability": self.runtime_policy_applicability.value,
            "resource_requirement": self.resource_requirement.value,
            "meaningful_side_effect_requirement": self.meaningful_side_effect_requirement.value,
            "status": self.status.value,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class UpdateCollaborativeOperationPolicyProfileCommand(_RepositoryModelBase):
    scope: CollaborativeOperationPolicyProfileScopeKey
    expected_revision: int = Field(ge=0)
    authority_scope: str = _NON_EMPTY
    workspace_policy_applicability: PolicyLayerApplicability
    resource_policy_applicability: PolicyLayerApplicability
    runtime_policy_applicability: PolicyLayerApplicability
    resource_requirement: OperationPolicyRequirement
    meaningful_side_effect_requirement: OperationPolicyRequirement
    status: CollaborativeOperationPolicyProfileStatus

    @field_validator("authority_scope")
    @classmethod
    def _strip_authority_scope(cls, value: str) -> str:
        return cls._strip_required(value)


@runtime_checkable
class CollaborativeOperationPolicyProfileRepository(Protocol):
    """Authoritative persistence port for operation policy profiles."""

    @property
    def capabilities(self) -> CollaborativeWorkRepositoryCapabilities:
        """Return declared repository backend capabilities."""

    def create(
        self,
        command: CreateCollaborativeOperationPolicyProfileCommand,
    ) -> CollaborativeOperationPolicyProfile:
        """Create an operation profile at ``INITIAL_RECORD_REVISION``."""

    def get_for_operation(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        operation_id: str,
    ) -> CollaborativeOperationPolicyProfile | None:
        """Return profile for the scoped operation identity or ``None``."""

    def update(
        self,
        command: UpdateCollaborativeOperationPolicyProfileCommand,
    ) -> CollaborativeOperationPolicyProfile:
        """Replace profile semantics under optimistic concurrency."""


class WorkItemNotFound(Exception):
    """WorkItem was not found for the requested tenant/workspace scope."""


class WorkItemAlreadyExists(Exception):
    """WorkItem already exists for the requested scoped identity."""


class WorkItemRevisionConflict(Exception):
    """Optimistic revision conflict for WorkItem."""


class WorkItemIdempotencyConflict(Exception):
    """Idempotency key replayed with a different semantic command."""


class AssignmentNotFound(Exception):
    """Assignment was not found for the requested tenant/workspace scope."""


class AssignmentAlreadyExists(Exception):
    """Assignment already exists for the requested scoped identity."""


class AssignmentRevisionConflict(Exception):
    """Optimistic revision conflict for Assignment."""


class AssignmentIdempotencyConflict(Exception):
    """Idempotency key replayed with a different semantic command."""


class WorkItemScopeKey(_RepositoryModelBase):
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    work_item_id: str = _NON_EMPTY

    @field_validator("tenant_id", "workspace_id", "work_item_id")
    @classmethod
    def _strip_scope_fields(cls, value: str) -> str:
        return cls._strip_required(value)


class AssignmentScopeKey(_RepositoryModelBase):
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    assignment_id: str = _NON_EMPTY

    @field_validator("tenant_id", "workspace_id", "assignment_id")
    @classmethod
    def _strip_scope_fields(cls, value: str) -> str:
        return cls._strip_required(value)


class CreateWorkItemCommand(_RepositoryModelBase):
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    work_item_id: str = _NON_EMPTY
    created_by_principal_id: str = _NON_EMPTY
    state: WorkItemState = WorkItemState.OPEN
    created_at: datetime
    updated_at: datetime
    title: str | None = None
    description: str | None = None
    idempotency_key: str | None = None

    @field_validator(
        "tenant_id",
        "workspace_id",
        "work_item_id",
        "created_by_principal_id",
        "title",
        "description",
        "idempotency_key",
    )
    @classmethod
    def _strip_fields(cls, value: str | None) -> str | None:
        return cls._strip_optional(value)

    @field_validator("created_at", "updated_at")
    @classmethod
    def _timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _validate_timestamp_order(self) -> Self:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must be greater than or equal to created_at")
        return self

    def semantic_fingerprint(self) -> str:
        payload = {
            "tenant_id": self.tenant_id,
            "workspace_id": self.workspace_id,
            "work_item_id": self.work_item_id,
            "created_by_principal_id": self.created_by_principal_id,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "title": self.title,
            "description": self.description,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class UpdateWorkItemCommand(_RepositoryModelBase):
    """Low-level WorkItem replacement under CAS.

    Authoritative lifecycle validation and authority enforcement belong to
    COLLAB-WORK-2C; this command persists replacement semantics only.
    """

    scope: WorkItemScopeKey
    expected_revision: int = Field(ge=0)
    state: WorkItemState
    updated_at: datetime
    title: str | None = None
    description: str | None = None

    @field_validator("title", "description")
    @classmethod
    def _strip_optional_fields(cls, value: str | None) -> str | None:
        return cls._strip_optional(value)

    @field_validator("updated_at")
    @classmethod
    def _timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value


class CreateAssignmentCommand(_RepositoryModelBase):
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    assignment_id: str = _NON_EMPTY
    work_item_id: str = _NON_EMPTY
    principal_id: str = _NON_EMPTY
    created_by_principal_id: str = _NON_EMPTY
    state: AssignmentState = AssignmentState.ACTIVE
    created_at: datetime | None = None
    updated_at: datetime | None = None
    idempotency_key: str | None = None

    @field_validator(
        "tenant_id",
        "workspace_id",
        "assignment_id",
        "work_item_id",
        "principal_id",
        "created_by_principal_id",
        "idempotency_key",
    )
    @classmethod
    def _strip_fields(cls, value: str | None) -> str | None:
        return cls._strip_optional(value)

    @field_validator("created_at", "updated_at")
    @classmethod
    def _timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _validate_timestamp_order(self) -> Self:
        if self.created_at is not None and self.updated_at is not None:
            if self.updated_at < self.created_at:
                raise ValueError("updated_at must be greater than or equal to created_at")
        return self

    def semantic_fingerprint(self) -> str:
        payload = {
            "tenant_id": self.tenant_id,
            "workspace_id": self.workspace_id,
            "assignment_id": self.assignment_id,
            "work_item_id": self.work_item_id,
            "principal_id": self.principal_id,
            "created_by_principal_id": self.created_by_principal_id,
            "state": self.state.value,
            "created_at": self.created_at.isoformat() if self.created_at is not None else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at is not None else None,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class UpdateAssignmentCommand(_RepositoryModelBase):
    """Low-level Assignment replacement under CAS.

    Authoritative lifecycle validation and authority enforcement belong to
    COLLAB-WORK-2C; this command persists replacement semantics only.
    """

    scope: AssignmentScopeKey
    expected_revision: int = Field(ge=0)
    state: AssignmentState
    updated_at: datetime | None = None

    @field_validator("updated_at")
    @classmethod
    def _timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value


@runtime_checkable
class WorkItemRepository(Protocol):
    """Authoritative persistence port for WorkItem records."""

    @property
    def capabilities(self) -> CollaborativeWorkRepositoryCapabilities:
        """Return declared repository backend capabilities."""

    def create(self, command: CreateWorkItemCommand) -> WorkItem:
        """Create a WorkItem at ``INITIAL_RECORD_REVISION``."""

    def get(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        work_item_id: str,
    ) -> WorkItem | None:
        """Return WorkItem for the scoped identity or ``None``."""

    def update(self, command: UpdateWorkItemCommand) -> WorkItem:
        """Replace WorkItem semantics under optimistic concurrency."""


@runtime_checkable
class AssignmentRepository(Protocol):
    """Authoritative persistence port for Assignment records."""

    @property
    def capabilities(self) -> CollaborativeWorkRepositoryCapabilities:
        """Return declared repository backend capabilities."""

    def create(self, command: CreateAssignmentCommand) -> Assignment:
        """Create an Assignment at ``INITIAL_RECORD_REVISION``."""

    def get(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        assignment_id: str,
    ) -> Assignment | None:
        """Return Assignment for the scoped identity or ``None``."""

    def update(self, command: UpdateAssignmentCommand) -> Assignment:
        """Replace Assignment semantics under optimistic concurrency."""
