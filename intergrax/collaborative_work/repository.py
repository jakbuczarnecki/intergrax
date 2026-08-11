# © Artur Czarnecki. All rights reserved.

"""Collaborative Work membership and delegation repository contracts (COLLAB-WORK-1B).

Provider-neutral persistence ports for authoritative ``WorkspaceMembership`` and
``AuthorityDelegation`` records.

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
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from intergrax.contracts.collaborative_work import (
    AuthorityDelegation,
    AuthorityGrantStatus,
    CollaborativePolicyRule,
    CollaborativePolicyRuleStatus,
    DelegationStatus,
    MembershipStatus,
    PolicyCompositionLayer,
    PrincipalAuthorityGrant,
    WorkspaceMembership,
    WorkspaceMembershipRole,
)
from intergrax.contracts.runtime_policy import PolicyAction

INITIAL_RECORD_REVISION: int = 0


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


def _require_non_empty(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be a non-empty string")
    return cleaned


def _require_non_negative_int(value: int, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return value


@dataclass(frozen=True, slots=True)
class CollaborativeWorkRepositoryCapabilities:
    """Declared backend capabilities for collaborative work repositories."""

    backend_id: str
    durable: bool
    reference_only: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backend_id",
            _require_non_empty(self.backend_id, field_name="backend_id"),
        )
        if not isinstance(self.durable, bool):
            raise ValueError("durable must be a bool")
        if not isinstance(self.reference_only, bool):
            raise ValueError("reference_only must be a bool")


@dataclass(frozen=True, slots=True)
class WorkspaceMembershipScopeKey:
    tenant_id: str
    workspace_id: str
    membership_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tenant_id",
            _require_non_empty(self.tenant_id, field_name="tenant_id"),
        )
        object.__setattr__(
            self,
            "workspace_id",
            _require_non_empty(self.workspace_id, field_name="workspace_id"),
        )
        object.__setattr__(
            self,
            "membership_id",
            _require_non_empty(self.membership_id, field_name="membership_id"),
        )


@dataclass(frozen=True, slots=True)
class AuthorityDelegationScopeKey:
    tenant_id: str
    workspace_id: str
    delegation_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tenant_id",
            _require_non_empty(self.tenant_id, field_name="tenant_id"),
        )
        object.__setattr__(
            self,
            "workspace_id",
            _require_non_empty(self.workspace_id, field_name="workspace_id"),
        )
        object.__setattr__(
            self,
            "delegation_id",
            _require_non_empty(self.delegation_id, field_name="delegation_id"),
        )


@dataclass(frozen=True, slots=True)
class PrincipalAuthorityGrantScopeKey:
    tenant_id: str
    workspace_id: str
    authority_grant_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tenant_id",
            _require_non_empty(self.tenant_id, field_name="tenant_id"),
        )
        object.__setattr__(
            self,
            "workspace_id",
            _require_non_empty(self.workspace_id, field_name="workspace_id"),
        )
        object.__setattr__(
            self,
            "authority_grant_id",
            _require_non_empty(self.authority_grant_id, field_name="authority_grant_id"),
        )


@dataclass(frozen=True, slots=True)
class CreateWorkspaceMembershipCommand:
    tenant_id: str
    workspace_id: str
    membership_id: str
    principal_id: str
    role: WorkspaceMembershipRole
    status: MembershipStatus = MembershipStatus.ACTIVE
    idempotency_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tenant_id",
            _require_non_empty(self.tenant_id, field_name="tenant_id"),
        )
        object.__setattr__(
            self,
            "workspace_id",
            _require_non_empty(self.workspace_id, field_name="workspace_id"),
        )
        object.__setattr__(
            self,
            "membership_id",
            _require_non_empty(self.membership_id, field_name="membership_id"),
        )
        object.__setattr__(
            self,
            "principal_id",
            _require_non_empty(self.principal_id, field_name="principal_id"),
        )
        if self.idempotency_key is not None:
            object.__setattr__(
                self,
                "idempotency_key",
                _require_non_empty(self.idempotency_key, field_name="idempotency_key"),
            )

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


@dataclass(frozen=True, slots=True)
class CreatePrincipalAuthorityGrantCommand:
    tenant_id: str
    workspace_id: str
    authority_grant_id: str
    principal_id: str
    authority_scopes: tuple[str, ...]
    status: AuthorityGrantStatus = AuthorityGrantStatus.ACTIVE
    idempotency_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tenant_id",
            _require_non_empty(self.tenant_id, field_name="tenant_id"),
        )
        object.__setattr__(
            self,
            "workspace_id",
            _require_non_empty(self.workspace_id, field_name="workspace_id"),
        )
        object.__setattr__(
            self,
            "authority_grant_id",
            _require_non_empty(self.authority_grant_id, field_name="authority_grant_id"),
        )
        object.__setattr__(
            self,
            "principal_id",
            _require_non_empty(self.principal_id, field_name="principal_id"),
        )
        if not self.authority_scopes:
            raise ValueError("authority_scopes must be non-empty")
        object.__setattr__(
            self,
            "authority_scopes",
            tuple(
                _require_non_empty(scope, field_name="authority_scopes")
                for scope in self.authority_scopes
            ),
        )
        if self.idempotency_key is not None:
            object.__setattr__(
                self,
                "idempotency_key",
                _require_non_empty(self.idempotency_key, field_name="idempotency_key"),
            )

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


@dataclass(frozen=True, slots=True)
class UpdatePrincipalAuthorityGrantCommand:
    scope: PrincipalAuthorityGrantScopeKey
    expected_revision: int
    authority_scopes: tuple[str, ...]
    status: AuthorityGrantStatus

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scope",
            _require_scope_key(self.scope, PrincipalAuthorityGrantScopeKey),
        )
        object.__setattr__(
            self,
            "expected_revision",
            _require_non_negative_int(self.expected_revision, field_name="expected_revision"),
        )
        if not self.authority_scopes:
            raise ValueError("authority_scopes must be non-empty")
        object.__setattr__(
            self,
            "authority_scopes",
            tuple(
                _require_non_empty(scope, field_name="authority_scopes")
                for scope in self.authority_scopes
            ),
        )


@dataclass(frozen=True, slots=True)
class UpdateWorkspaceMembershipCommand:
    scope: WorkspaceMembershipScopeKey
    expected_revision: int
    role: WorkspaceMembershipRole
    status: MembershipStatus

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", _require_scope_key(self.scope, WorkspaceMembershipScopeKey))
        object.__setattr__(
            self,
            "expected_revision",
            _require_non_negative_int(self.expected_revision, field_name="expected_revision"),
        )


@dataclass(frozen=True, slots=True)
class CreateAuthorityDelegationCommand:
    tenant_id: str
    workspace_id: str
    delegation_id: str
    delegator_principal_id: str
    delegate_principal_id: str
    authority_scopes: tuple[str, ...]
    resource_scope: str | None = None
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    status: DelegationStatus = DelegationStatus.ACTIVE
    idempotency_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tenant_id",
            _require_non_empty(self.tenant_id, field_name="tenant_id"),
        )
        object.__setattr__(
            self,
            "workspace_id",
            _require_non_empty(self.workspace_id, field_name="workspace_id"),
        )
        object.__setattr__(
            self,
            "delegation_id",
            _require_non_empty(self.delegation_id, field_name="delegation_id"),
        )
        object.__setattr__(
            self,
            "delegator_principal_id",
            _require_non_empty(self.delegator_principal_id, field_name="delegator_principal_id"),
        )
        object.__setattr__(
            self,
            "delegate_principal_id",
            _require_non_empty(self.delegate_principal_id, field_name="delegate_principal_id"),
        )
        if not self.authority_scopes:
            raise ValueError("authority_scopes must be non-empty")
        object.__setattr__(
            self,
            "authority_scopes",
            tuple(_require_non_empty(scope, field_name="authority_scopes") for scope in self.authority_scopes),
        )
        if self.resource_scope is not None:
            object.__setattr__(
                self,
                "resource_scope",
                _require_non_empty(self.resource_scope, field_name="resource_scope"),
            )
        if self.idempotency_key is not None:
            object.__setattr__(
                self,
                "idempotency_key",
                _require_non_empty(self.idempotency_key, field_name="idempotency_key"),
            )

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


@dataclass(frozen=True, slots=True)
class UpdateAuthorityDelegationCommand:
    scope: AuthorityDelegationScopeKey
    expected_revision: int
    authority_scopes: tuple[str, ...]
    resource_scope: str | None
    valid_from: datetime | None
    valid_until: datetime | None
    status: DelegationStatus

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", _require_scope_key(self.scope, AuthorityDelegationScopeKey))
        object.__setattr__(
            self,
            "expected_revision",
            _require_non_negative_int(self.expected_revision, field_name="expected_revision"),
        )
        if not self.authority_scopes:
            raise ValueError("authority_scopes must be non-empty")
        object.__setattr__(
            self,
            "authority_scopes",
            tuple(_require_non_empty(scope, field_name="authority_scopes") for scope in self.authority_scopes),
        )
        if self.resource_scope is not None:
            object.__setattr__(
                self,
                "resource_scope",
                _require_non_empty(self.resource_scope, field_name="resource_scope"),
            )


def _require_scope_key[T](value: object, expected_type: type[T]) -> T:
    if not isinstance(value, expected_type):
        raise ValueError(f"scope must be {expected_type.__name__}")
    return value


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


@dataclass(frozen=True, slots=True)
class CollaborativePolicyRuleScopeKey:
    tenant_id: str
    workspace_id: str
    policy_rule_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tenant_id",
            _require_non_empty(self.tenant_id, field_name="tenant_id"),
        )
        object.__setattr__(
            self,
            "workspace_id",
            _require_non_empty(self.workspace_id, field_name="workspace_id"),
        )
        object.__setattr__(
            self,
            "policy_rule_id",
            _require_non_empty(self.policy_rule_id, field_name="policy_rule_id"),
        )


@dataclass(frozen=True, slots=True)
class CreateCollaborativePolicyRuleCommand:
    tenant_id: str
    workspace_id: str
    policy_rule_id: str
    layer: PolicyCompositionLayer
    authority_scope: str
    action: PolicyAction
    resource_scope: str | None = None
    status: CollaborativePolicyRuleStatus = CollaborativePolicyRuleStatus.ACTIVE
    idempotency_key: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tenant_id",
            _require_non_empty(self.tenant_id, field_name="tenant_id"),
        )
        object.__setattr__(
            self,
            "workspace_id",
            _require_non_empty(self.workspace_id, field_name="workspace_id"),
        )
        object.__setattr__(
            self,
            "policy_rule_id",
            _require_non_empty(self.policy_rule_id, field_name="policy_rule_id"),
        )
        object.__setattr__(
            self,
            "authority_scope",
            _require_non_empty(self.authority_scope, field_name="authority_scope"),
        )
        if self.resource_scope is not None:
            object.__setattr__(
                self,
                "resource_scope",
                _require_non_empty(self.resource_scope, field_name="resource_scope"),
            )
        if self.idempotency_key is not None:
            object.__setattr__(
                self,
                "idempotency_key",
                _require_non_empty(self.idempotency_key, field_name="idempotency_key"),
            )
        if self.layer not in (
            PolicyCompositionLayer.WORKSPACE_POLICY,
            PolicyCompositionLayer.RESOURCE_POLICY,
        ):
            raise ValueError("layer must be WORKSPACE_POLICY or RESOURCE_POLICY")
        if self.layer is PolicyCompositionLayer.WORKSPACE_POLICY and self.resource_scope is not None:
            raise ValueError("WORKSPACE_POLICY rules must not include resource_scope")
        if self.layer is PolicyCompositionLayer.RESOURCE_POLICY and self.resource_scope is None:
            raise ValueError("RESOURCE_POLICY rules require resource_scope")

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


@dataclass(frozen=True, slots=True)
class UpdateCollaborativePolicyRuleCommand:
    scope: CollaborativePolicyRuleScopeKey
    expected_revision: int
    action: PolicyAction
    status: CollaborativePolicyRuleStatus

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "scope",
            _require_scope_key(self.scope, CollaborativePolicyRuleScopeKey),
        )
        object.__setattr__(
            self,
            "expected_revision",
            _require_non_negative_int(self.expected_revision, field_name="expected_revision"),
        )


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
