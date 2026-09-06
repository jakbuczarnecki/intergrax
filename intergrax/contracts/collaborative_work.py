# © Artur Czarnecki. All rights reserved.

"""Collaborative Work identity, authority, and shared-work contracts (MP-1 / MP-2).

Semantic source of truth for collaborative principals, explicit workspace
membership, authority delegation, effective-authority evaluation, WorkItem
lifecycle, and Assignment lifecycle. Distinct from:

- ``RequestIdentity`` / ``PrincipalType`` — run-scoped execution intake only.
- ``DelegationSpec`` — Nexus graph child-run execution delegation only.
- ``Task`` / ``TaskState`` — Nexus execution units and runtime lifecycle only.
- ``PolicyEngine`` — enforcement of resolved authority at runtime boundaries.

Effective authority intersection (resolver implementation is out of scope):

    principal authority
      ∩ WorkspaceMembership
      ∩ Delegation where applicable
      ∩ workspace policy
      ∩ resource policy
      ∩ runtime/tool policy
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectRequest
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision

SCHEMA_COLLABORATIVE_PRINCIPAL_V1: Final = "collaborative_principal.v1"
SCHEMA_WORKSPACE_MEMBERSHIP_V1: Final = "workspace_membership.v1"
SCHEMA_AUTHORITY_DELEGATION_V1: Final = "authority_delegation.v1"
SCHEMA_PRINCIPAL_AUTHORITY_GRANT_V1: Final = "principal_authority_grant.v1"
SCHEMA_EFFECTIVE_AUTHORITY_REQUEST_V1: Final = "effective_authority_request.v1"
SCHEMA_EFFECTIVE_AUTHORITY_DECISION_V1: Final = "effective_authority_decision.v1"
SCHEMA_POLICY_COMPOSITION_APPLICABILITY_V1: Final = "policy_composition_applicability.v1"
SCHEMA_POLICY_COMPOSITION_INPUT_V1: Final = "policy_composition_input.v1"
SCHEMA_POLICY_COMPOSITION_RESULT_V1: Final = "policy_composition_result.v1"
SCHEMA_COLLABORATIVE_POLICY_RULE_V1: Final = "collaborative_policy_rule.v1"
SCHEMA_COLLABORATIVE_OPERATION_POLICY_PROFILE_V1: Final = (
    "collaborative_operation_policy_profile.v1"
)
SCHEMA_COLLABORATIVE_WORK_ENFORCEMENT_REQUEST_V1: Final = (
    "collaborative_work_enforcement_request.v1"
)
SCHEMA_COLLABORATIVE_WORK_ENFORCEMENT_RESULT_V1: Final = (
    "collaborative_work_enforcement_result.v1"
)
SCHEMA_WORK_ITEM_V1: Final = "work_item.v1"
SCHEMA_ASSIGNMENT_V1: Final = "assignment.v1"
SCHEMA_WORK_ITEM_TRANSITION_REQUEST_V1: Final = "work_item_transition_request.v1"
SCHEMA_ASSIGNMENT_TRANSITION_REQUEST_V1: Final = "assignment_transition_request.v1"

_SUPPORTED_COLLABORATIVE_POLICY_ACTIONS: Final = frozenset(
    {
        PolicyAction.ALLOW,
        PolicyAction.DENY,
        PolicyAction.REQUIRE_HUMAN,
        PolicyAction.ESCALATE,
    }
)

_NON_EMPTY = Field(min_length=1)


class PrincipalKind(StrEnum):
    """Collaborative principal semantic kind — not execution intake typing."""

    HUMAN = "human"
    AGENT = "agent"
    SERVICE = "service"
    EXTERNAL_AGENT = "external_agent"


class WorkspaceMembershipRole(StrEnum):
    """Conservative membership role placeholder until RBAC taxonomy is frozen."""

    MEMBER = "member"
    ADMIN = "admin"
    OBSERVER = "observer"


class MembershipStatus(StrEnum):
    ACTIVE = "active"
    REVOKED = "revoked"
    SUSPENDED = "suspended"


class MembershipResolutionMode(StrEnum):
    """How the authority resolver resolves workspace membership."""

    LOCATOR = "locator"
    CANONICAL_PRINCIPAL = "canonical_principal"


class DelegationStatus(StrEnum):
    ACTIVE = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"


class AuthorityGrantStatus(StrEnum):
    ACTIVE = "active"
    REVOKED = "revoked"
    SUSPENDED = "suspended"


class EffectiveAuthorityDenialReason(StrEnum):
    """Fail-closed denial codes for the effective-authority boundary."""

    MISSING_ACTING_PRINCIPAL = "missing_acting_principal"
    MISSING_MEMBERSHIP = "missing_membership"
    MISSING_DELEGATION = "missing_delegation"
    INSUFFICIENT_DELEGATION_SCOPE = "insufficient_delegation_scope"
    MEMBERSHIP_NOT_ACTIVE = "membership_not_active"
    MISSING_DELEGATOR_MEMBERSHIP = "missing_delegator_membership"
    DELEGATOR_MEMBERSHIP_NOT_ACTIVE = "delegator_membership_not_active"
    DELEGATION_NOT_ACTIVE = "delegation_not_active"
    AUTHORITY_TEMPORAL_CONTEXT_UNAVAILABLE = "authority_temporal_context_unavailable"
    SCOPE_ONLY_INSUFFICIENT = "scope_only_insufficient"
    MISSING_BASE_AUTHORITY = "missing_base_authority"
    INSUFFICIENT_BASE_AUTHORITY = "insufficient_base_authority"
    BASE_AUTHORITY_NOT_ACTIVE = "base_authority_not_active"


class CollaborativePrincipal(BaseModel):
    """Stable collaborative identity — independent of a specific execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["collaborative_principal.v1"] = SCHEMA_COLLABORATIVE_PRINCIPAL_V1
    principal_id: str = _NON_EMPTY
    principal_kind: PrincipalKind
    tenant_id: str = _NON_EMPTY

    @field_validator("principal_id", "tenant_id")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class WorkspaceMembership(BaseModel):
    """Explicit workspace membership — never inferred from tenant/workspace IDs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["workspace_membership.v1"] = SCHEMA_WORKSPACE_MEMBERSHIP_V1
    membership_id: str = _NON_EMPTY
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    principal_id: str = _NON_EMPTY
    role: WorkspaceMembershipRole
    status: MembershipStatus = MembershipStatus.ACTIVE
    revision: int = Field(ge=0)

    @field_validator(
        "membership_id",
        "tenant_id",
        "workspace_id",
        "principal_id",
    )
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class AuthorityDelegation(BaseModel):
    """Authority delegation between collaborative principals — not Nexus execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["authority_delegation.v1"] = SCHEMA_AUTHORITY_DELEGATION_V1
    delegation_id: str = _NON_EMPTY
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    delegator_principal_id: str = _NON_EMPTY
    delegate_principal_id: str = _NON_EMPTY
    authority_scopes: tuple[str, ...] = Field(min_length=1)
    resource_scope: str | None = None
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    status: DelegationStatus = DelegationStatus.ACTIVE
    revision: int = Field(ge=0)

    @field_validator(
        "delegation_id",
        "tenant_id",
        "workspace_id",
        "delegator_principal_id",
        "delegate_principal_id",
        "resource_scope",
    )
    @classmethod
    def _strip_required_or_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty when provided")
        return normalized

    @field_validator("authority_scopes")
    @classmethod
    def _normalize_authority_scopes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(scope.strip() for scope in value)
        if not normalized or any(not scope for scope in normalized):
            raise ValueError("authority_scopes must contain non-empty scope values")
        return normalized

    @field_validator("valid_from", "valid_until")
    @classmethod
    def _timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("validity timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _reject_self_delegation(self) -> AuthorityDelegation:
        if self.delegator_principal_id == self.delegate_principal_id:
            raise ValueError("delegator_principal_id must differ from delegate_principal_id")
        return self

    @model_validator(mode="after")
    def _reject_invalid_validity_window(self) -> AuthorityDelegation:
        if (
            self.valid_from is not None
            and self.valid_until is not None
            and self.valid_until <= self.valid_from
        ):
            raise ValueError("valid_until must be after valid_from")
        return self


class PrincipalAuthorityGrant(BaseModel):
    """Authoritative collaborative authority scopes owned by a principal in workspace scope.

    ``WorkspaceMembershipRole`` is collaborative classification only; explicit
    ``authority_scopes`` on this record are the authoritative base-authority source.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["principal_authority_grant.v1"] = SCHEMA_PRINCIPAL_AUTHORITY_GRANT_V1
    authority_grant_id: str = _NON_EMPTY
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    principal_id: str = _NON_EMPTY
    authority_scopes: tuple[str, ...] = Field(min_length=1)
    status: AuthorityGrantStatus = AuthorityGrantStatus.ACTIVE
    revision: int = Field(ge=0)

    @field_validator(
        "authority_grant_id",
        "tenant_id",
        "workspace_id",
        "principal_id",
    )
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @field_validator("authority_scopes")
    @classmethod
    def _normalize_authority_scopes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(scope.strip() for scope in value)
        if not normalized or any(not scope for scope in normalized):
            raise ValueError("authority_scopes must contain non-empty scope values")
        return normalized


class EffectiveAuthorityRequest(BaseModel):
    """Typed input for future effective-authority resolution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["effective_authority_request.v1"] = (
        SCHEMA_EFFECTIVE_AUTHORITY_REQUEST_V1
    )
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    acting_principal_id: str = _NON_EMPTY
    requested_authority_scopes: tuple[str, ...] = Field(min_length=1)
    delegator_principal_id: str | None = None
    resource_scope: str | None = None
    membership: WorkspaceMembership | None = None
    membership_resolution_mode: MembershipResolutionMode = MembershipResolutionMode.LOCATOR
    delegation: AuthorityDelegation | None = None

    @field_validator(
        "tenant_id",
        "workspace_id",
        "acting_principal_id",
        "delegator_principal_id",
        "resource_scope",
    )
    @classmethod
    def _strip_required_or_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty when provided")
        return normalized

    @field_validator("requested_authority_scopes")
    @classmethod
    def _normalize_requested_scopes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(scope.strip() for scope in value)
        if not normalized or any(not scope for scope in normalized):
            raise ValueError("requested_authority_scopes must contain non-empty scope values")
        return normalized

    @model_validator(mode="after")
    def _validate_embedded_evidence_consistency(self) -> EffectiveAuthorityRequest:
        if (
            self.membership_resolution_mode is MembershipResolutionMode.CANONICAL_PRINCIPAL
            and self.membership is not None
        ):
            raise ValueError(
                "canonical_principal membership resolution must not include an embedded membership locator"
            )

        if self.membership is not None:
            if self.membership.tenant_id != self.tenant_id:
                raise ValueError("membership tenant_id must match request tenant_id")
            if self.membership.workspace_id != self.workspace_id:
                raise ValueError("membership workspace_id must match request workspace_id")
            if self.membership.principal_id != self.acting_principal_id:
                raise ValueError("membership principal_id must match request acting_principal_id")

        if self.delegation is not None:
            if self.delegation.tenant_id != self.tenant_id:
                raise ValueError("delegation tenant_id must match request tenant_id")
            if self.delegation.workspace_id != self.workspace_id:
                raise ValueError("delegation workspace_id must match request workspace_id")
            if self.delegation.delegate_principal_id != self.acting_principal_id:
                raise ValueError(
                    "delegation delegate_principal_id must match request acting_principal_id"
                )
            if (
                self.delegator_principal_id is not None
                and self.delegation.delegator_principal_id != self.delegator_principal_id
            ):
                raise ValueError(
                    "delegation delegator_principal_id must match request delegator_principal_id"
                )

        return self


class EffectiveAuthorityDecision(BaseModel):
    """Typed result for effective-authority evaluation — reuses ``PolicyDecision``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["effective_authority_decision.v1"] = (
        SCHEMA_EFFECTIVE_AUTHORITY_DECISION_V1
    )
    decision: PolicyDecision
    denial_reason: EffectiveAuthorityDenialReason | None = None

    @model_validator(mode="after")
    def _align_denial_reason_with_action(self) -> EffectiveAuthorityDecision:
        if self.decision.action is PolicyAction.ALLOW and self.denial_reason is not None:
            raise ValueError("denial_reason must be omitted when decision.action is allow")
        return self


def fail_closed_effective_authority_decision(
    *,
    reason: str,
    denial_reason: EffectiveAuthorityDenialReason,
    policy_rule_id: str = "collaborative_work.effective_authority.fail_closed",
) -> EffectiveAuthorityDecision:
    """Construct a mandatory fail-closed deny decision for the authority boundary."""
    return EffectiveAuthorityDecision(
        decision=PolicyDecision(
            action=PolicyAction.DENY,
            reason=reason,
            policy_rule_id=policy_rule_id,
        ),
        denial_reason=denial_reason,
    )


class PolicyCompositionLayer(StrEnum):
    """Contributing policy layer identifiers for fail-closed composition."""

    COLLABORATIVE_AUTHORITY = "collaborative_authority"
    WORKSPACE_POLICY = "workspace_policy"
    RESOURCE_POLICY = "resource_policy"
    RUNTIME_POLICY = "runtime_policy"


class PolicyLayerApplicability(StrEnum):
    """Trusted applicability state for an optional policy layer.

    Only ``NOT_APPLICABLE`` may skip a layer. ``UNKNOWN`` fails closed until future
    operation-classification code supplies a trusted determination.
    """

    REQUIRED = "required"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class PolicyCompositionApplicability(BaseModel):
    """Trusted operation context — which policy layers are mandatory for composition.

    Future operation-classification code is responsible for producing trusted
    ``NOT_APPLICABLE`` values. Absence, default, or ``UNKNOWN`` applicability fails
    closed; only explicit ``NOT_APPLICABLE`` skips a layer. When a layer is
    ``REQUIRED`` but its decision is absent, composition fails closed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["policy_composition_applicability.v1"] = (
        SCHEMA_POLICY_COMPOSITION_APPLICABILITY_V1
    )
    workspace_policy: PolicyLayerApplicability = PolicyLayerApplicability.UNKNOWN
    resource_policy: PolicyLayerApplicability = PolicyLayerApplicability.UNKNOWN
    runtime_policy: PolicyLayerApplicability = PolicyLayerApplicability.UNKNOWN


class PolicyCompositionInput(BaseModel):
    """Pre-evaluated policy decisions for fail-closed composition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["policy_composition_input.v1"] = SCHEMA_POLICY_COMPOSITION_INPUT_V1
    collaborative_authority: PolicyDecision
    workspace_policy: PolicyDecision | None = None
    resource_policy: PolicyDecision | None = None
    runtime_policy: PolicyDecision | None = None
    applicability: PolicyCompositionApplicability = Field(
        default_factory=PolicyCompositionApplicability,
    )


class PolicyCompositionResult(BaseModel):
    """Final enforcement decision with typed contributing layer provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["policy_composition_result.v1"] = SCHEMA_POLICY_COMPOSITION_RESULT_V1
    decision: PolicyDecision
    collaborative_authority: PolicyDecision
    workspace_policy: PolicyDecision | None = None
    resource_policy: PolicyDecision | None = None
    runtime_policy: PolicyDecision | None = None
    determining_layer: PolicyCompositionLayer | None = None


class CollaborativePolicyRuleStatus(StrEnum):
    """Lifecycle for authoritative collaborative workspace/resource policy rules."""

    ACTIVE = "active"
    DISABLED = "disabled"


class CollaborativePolicyRule(BaseModel):
    """Authoritative workspace or resource policy rule for collaborative authorization.

    Exact policy key uniqueness (enforced by repository):

    - ``WORKSPACE_POLICY``: ``tenant_id`` + ``workspace_id`` + ``authority_scope``
    - ``RESOURCE_POLICY``: ``tenant_id`` + ``workspace_id`` + ``resource_scope`` + ``authority_scope``

    Policy management authorization is out of scope; this contract models persistence
    and evaluation semantics only.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["collaborative_policy_rule.v1"] = SCHEMA_COLLABORATIVE_POLICY_RULE_V1
    policy_rule_id: str = _NON_EMPTY
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    layer: PolicyCompositionLayer
    authority_scope: str = _NON_EMPTY
    action: PolicyAction
    resource_scope: str | None = None
    status: CollaborativePolicyRuleStatus = CollaborativePolicyRuleStatus.ACTIVE
    revision: int = Field(ge=0)

    @field_validator("policy_rule_id", "tenant_id", "workspace_id", "authority_scope", "resource_scope")
    @classmethod
    def _strip_required_or_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty when provided")
        return normalized

    @field_validator("action")
    @classmethod
    def _reject_unsupported_actions(cls, value: PolicyAction) -> PolicyAction:
        if value is PolicyAction.MODIFY:
            raise ValueError("MODIFY is not supported for collaborative policy rules")
        if value not in _SUPPORTED_COLLABORATIVE_POLICY_ACTIONS:
            raise ValueError("unsupported collaborative policy action")
        return value

    @model_validator(mode="after")
    def _validate_layer_and_resource_scope(self) -> CollaborativePolicyRule:
        if self.layer is PolicyCompositionLayer.WORKSPACE_POLICY:
            if self.resource_scope is not None:
                raise ValueError("WORKSPACE_POLICY rules must not include resource_scope")
        elif self.layer is PolicyCompositionLayer.RESOURCE_POLICY:
            if self.resource_scope is None:
                raise ValueError("RESOURCE_POLICY rules require resource_scope")
        else:
            raise ValueError("layer must be WORKSPACE_POLICY or RESOURCE_POLICY")
        return self


class OperationPolicyRequirement(StrEnum):
    """Whether an operation requires a concrete resource or meaningful side-effect evaluation."""

    REQUIRED = "required"
    NOT_APPLICABLE = "not_applicable"


class CollaborativeOperationPolicyProfileStatus(StrEnum):
    """Lifecycle for authoritative operation policy profiles."""

    ACTIVE = "active"
    DISABLED = "disabled"


_AUTHORITATIVE_PROFILE_APPLICABILITY: Final = frozenset(
    {
        PolicyLayerApplicability.REQUIRED,
        PolicyLayerApplicability.NOT_APPLICABLE,
    }
)


class CollaborativeOperationPolicyProfile(BaseModel):
    """Authoritative operation → policy-layer classification for enforcement gating.

    Profiles classify which policy layers are required; evaluators produce decisions.
    ``UNKNOWN`` applicability is a resolution failure state and must not be authored.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["collaborative_operation_policy_profile.v1"] = (
        SCHEMA_COLLABORATIVE_OPERATION_POLICY_PROFILE_V1
    )
    operation_id: str = _NON_EMPTY
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    authority_scope: str = _NON_EMPTY
    workspace_policy_applicability: PolicyLayerApplicability
    resource_policy_applicability: PolicyLayerApplicability
    runtime_policy_applicability: PolicyLayerApplicability
    resource_requirement: OperationPolicyRequirement
    meaningful_side_effect_requirement: OperationPolicyRequirement
    status: CollaborativeOperationPolicyProfileStatus = CollaborativeOperationPolicyProfileStatus.ACTIVE
    revision: int = Field(ge=0)

    @field_validator(
        "operation_id",
        "tenant_id",
        "workspace_id",
        "authority_scope",
    )
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @field_validator(
        "workspace_policy_applicability",
        "resource_policy_applicability",
        "runtime_policy_applicability",
    )
    @classmethod
    def _reject_unknown_applicability(cls, value: PolicyLayerApplicability) -> PolicyLayerApplicability:
        if value is PolicyLayerApplicability.UNKNOWN:
            raise ValueError("UNKNOWN applicability cannot be authored in operation profiles")
        if value not in _AUTHORITATIVE_PROFILE_APPLICABILITY:
            raise ValueError("unsupported profile applicability value")
        return value

    @model_validator(mode="after")
    def _validate_internal_consistency(self) -> CollaborativeOperationPolicyProfile:
        if self.meaningful_side_effect_requirement is OperationPolicyRequirement.REQUIRED:
            if self.runtime_policy_applicability is not PolicyLayerApplicability.REQUIRED:
                raise ValueError(
                    "meaningful side-effect requirement requires runtime policy REQUIRED"
                )
        if self.resource_requirement is OperationPolicyRequirement.REQUIRED:
            if self.resource_policy_applicability is not PolicyLayerApplicability.REQUIRED:
                raise ValueError("resource requirement requires resource policy REQUIRED")
        return self


class CollaborativeWorkEnforcementRequest(BaseModel):
    """Minimal trusted input for the final enforcement gate.

    Callers identify the operation and supply locator context for authority resolution.
    They must not supply policy applicability or pre-evaluated policy decisions.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["collaborative_work_enforcement_request.v1"] = (
        SCHEMA_COLLABORATIVE_WORK_ENFORCEMENT_REQUEST_V1
    )
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    operation_id: str = _NON_EMPTY
    acting_principal_id: str = _NON_EMPTY
    delegator_principal_id: str | None = None
    resource_scope: str | None = None
    membership: WorkspaceMembership | None = None
    membership_resolution_mode: MembershipResolutionMode = MembershipResolutionMode.LOCATOR
    delegation: AuthorityDelegation | None = None
    meaningful_side_effect_request: MeaningfulSideEffectRequest | None = None

    @field_validator(
        "tenant_id",
        "workspace_id",
        "operation_id",
        "acting_principal_id",
        "delegator_principal_id",
        "resource_scope",
    )
    @classmethod
    def _strip_required_or_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty when provided")
        return normalized

    @model_validator(mode="after")
    def _validate_embedded_locators(self) -> CollaborativeWorkEnforcementRequest:
        if (
            self.membership_resolution_mode is MembershipResolutionMode.CANONICAL_PRINCIPAL
            and self.membership is not None
        ):
            raise ValueError(
                "canonical_principal membership resolution must not include an embedded membership locator"
            )

        if self.membership is not None:
            if self.membership.tenant_id != self.tenant_id:
                raise ValueError("membership tenant_id must match request tenant_id")
            if self.membership.workspace_id != self.workspace_id:
                raise ValueError("membership workspace_id must match request workspace_id")
            if self.membership.principal_id != self.acting_principal_id:
                raise ValueError("membership principal_id must match request acting_principal_id")

        if self.delegation is not None:
            if self.delegation.tenant_id != self.tenant_id:
                raise ValueError("delegation tenant_id must match request tenant_id")
            if self.delegation.workspace_id != self.workspace_id:
                raise ValueError("delegation workspace_id must match request workspace_id")
            if self.delegation.delegate_principal_id != self.acting_principal_id:
                raise ValueError(
                    "delegation delegate_principal_id must match request acting_principal_id"
                )
            if (
                self.delegator_principal_id is not None
                and self.delegation.delegator_principal_id != self.delegator_principal_id
            ):
                raise ValueError(
                    "delegation delegator_principal_id must match request delegator_principal_id"
                )

        return self


class CollaborativeWorkEnforcementResult(BaseModel):
    """Auditable enforcement gate output with profile identity and composed decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["collaborative_work_enforcement_result.v1"] = (
        SCHEMA_COLLABORATIVE_WORK_ENFORCEMENT_RESULT_V1
    )
    operation_id: str = _NON_EMPTY
    profile_revision: int | None = Field(default=None, ge=0)
    authority_scope: str | None = None
    composition: PolicyCompositionResult

    @field_validator("operation_id")
    @classmethod
    def _strip_operation_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @field_validator("authority_scope")
    @classmethod
    def _strip_optional_scope(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class CollaborativeWorkLifecycleError(ValueError):
    """Invalid collaborative WorkItem or Assignment lifecycle transition."""


class WorkItemState(StrEnum):
    """Conservative collaborative WorkItem lifecycle — not Nexus ``TaskState``."""

    OPEN = "open"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class AssignmentState(StrEnum):
    """Collaborative assignment participation lifecycle — not execution assignment."""

    ACTIVE = "active"
    REVOKED = "revoked"
    COMPLETED = "completed"


_ALLOWED_WORK_ITEM_TRANSITIONS: Final = {
    WorkItemState.OPEN: frozenset({WorkItemState.ACTIVE, WorkItemState.CANCELLED}),
    WorkItemState.ACTIVE: frozenset({WorkItemState.COMPLETED, WorkItemState.CANCELLED}),
    WorkItemState.COMPLETED: frozenset({WorkItemState.ACTIVE}),
    WorkItemState.CANCELLED: frozenset({WorkItemState.ACTIVE}),
}

_WORK_ITEM_REOPEN_TARGET: Final = WorkItemState.ACTIVE

_ALLOWED_ASSIGNMENT_TRANSITIONS: Final = {
    AssignmentState.ACTIVE: frozenset({AssignmentState.REVOKED, AssignmentState.COMPLETED}),
    AssignmentState.REVOKED: frozenset(),
    AssignmentState.COMPLETED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class WorkItemStateTransition:
    """Explicit allowed WorkItem lifecycle transition."""

    from_state: WorkItemState
    to_state: WorkItemState


@dataclass(frozen=True, slots=True)
class AssignmentStateTransition:
    """Explicit allowed Assignment lifecycle transition."""

    from_state: AssignmentState
    to_state: AssignmentState


def work_item_resource_scope(*, work_item_id: str) -> str:
    """Deterministic MP-1 resource scope convention for one WorkItem."""
    normalized = work_item_id.strip()
    if not normalized:
        raise ValueError("work_item_id must be non-empty")
    return f"work_item:{normalized}"


def validate_work_item_state_transition(
    *,
    from_state: WorkItemState,
    to_state: WorkItemState,
) -> WorkItemStateTransition:
    """Validate a deterministic collaborative WorkItem lifecycle transition."""
    if type(from_state) is not WorkItemState:
        raise TypeError("from_state must be WorkItemState")
    if type(to_state) is not WorkItemState:
        raise TypeError("to_state must be WorkItemState")
    if from_state == to_state:
        raise CollaborativeWorkLifecycleError(
            f"Unsupported WorkItem transition: {from_state.value} -> {to_state.value}",
        )
    allowed = _ALLOWED_WORK_ITEM_TRANSITIONS.get(from_state, frozenset())
    if to_state not in allowed:
        raise CollaborativeWorkLifecycleError(
            f"Unsupported WorkItem transition: {from_state.value} -> {to_state.value}",
        )
    return WorkItemStateTransition(from_state=from_state, to_state=to_state)


def validate_assignment_state_transition(
    *,
    from_state: AssignmentState,
    to_state: AssignmentState,
) -> AssignmentStateTransition:
    """Validate a deterministic collaborative Assignment lifecycle transition."""
    if type(from_state) is not AssignmentState:
        raise TypeError("from_state must be AssignmentState")
    if type(to_state) is not AssignmentState:
        raise TypeError("to_state must be AssignmentState")
    if from_state == to_state:
        raise CollaborativeWorkLifecycleError(
            f"Unsupported Assignment transition: {from_state.value} -> {to_state.value}",
        )
    allowed = _ALLOWED_ASSIGNMENT_TRANSITIONS.get(from_state, frozenset())
    if to_state not in allowed:
        raise CollaborativeWorkLifecycleError(
            f"Unsupported Assignment transition: {from_state.value} -> {to_state.value}",
        )
    return AssignmentStateTransition(from_state=from_state, to_state=to_state)


def is_work_item_reopen_transition(transition: WorkItemStateTransition) -> bool:
    """Return True when transition explicitly reopens a terminal WorkItem."""
    return (
        transition.to_state is _WORK_ITEM_REOPEN_TARGET
        and transition.from_state in {WorkItemState.COMPLETED, WorkItemState.CANCELLED}
    )


class WorkItem(BaseModel):
    """Durable collaborative work identity — distinct from Nexus ``Task``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["work_item.v1"] = SCHEMA_WORK_ITEM_V1
    work_item_id: str = _NON_EMPTY
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    created_by_principal_id: str = _NON_EMPTY
    state: WorkItemState = WorkItemState.OPEN
    revision: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime
    title: str | None = None
    description: str | None = None

    @field_validator(
        "work_item_id",
        "tenant_id",
        "workspace_id",
        "created_by_principal_id",
        "title",
        "description",
    )
    @classmethod
    def _strip_required_or_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty when provided")
        return normalized

    @field_validator("created_at", "updated_at")
    @classmethod
    def _timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _validate_timestamp_order(self) -> WorkItem:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must be greater than or equal to created_at")
        return self


class Assignment(BaseModel):
    """Collaborative WorkItem participation — distinct from runtime AgentAssignment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["assignment.v1"] = SCHEMA_ASSIGNMENT_V1
    assignment_id: str = _NON_EMPTY
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    work_item_id: str = _NON_EMPTY
    principal_id: str = _NON_EMPTY
    created_by_principal_id: str = _NON_EMPTY
    state: AssignmentState = AssignmentState.ACTIVE
    revision: int = Field(ge=0)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @field_validator(
        "assignment_id",
        "tenant_id",
        "workspace_id",
        "work_item_id",
        "principal_id",
        "created_by_principal_id",
    )
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @field_validator("created_at", "updated_at")
    @classmethod
    def _timezone_aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _validate_timestamp_order(self) -> Assignment:
        if self.created_at is not None and self.updated_at is not None:
            if self.updated_at < self.created_at:
                raise ValueError("updated_at must be greater than or equal to created_at")
        return self


class WorkItemTransitionRequest(BaseModel):
    """Typed WorkItem state mutation input for future authoritative services."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["work_item_transition_request.v1"] = (
        SCHEMA_WORK_ITEM_TRANSITION_REQUEST_V1
    )
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    work_item_id: str = _NON_EMPTY
    expected_revision: int = Field(ge=0)
    target_state: WorkItemState
    acting_principal_id: str = _NON_EMPTY
    idempotency_key: str = _NON_EMPTY

    @field_validator(
        "tenant_id",
        "workspace_id",
        "work_item_id",
        "acting_principal_id",
        "idempotency_key",
    )
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


class AssignmentTransitionRequest(BaseModel):
    """Typed Assignment state mutation input for future authoritative services."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["assignment_transition_request.v1"] = (
        SCHEMA_ASSIGNMENT_TRANSITION_REQUEST_V1
    )
    tenant_id: str = _NON_EMPTY
    workspace_id: str = _NON_EMPTY
    assignment_id: str = _NON_EMPTY
    work_item_id: str = _NON_EMPTY
    expected_revision: int = Field(ge=0)
    target_state: AssignmentState
    acting_principal_id: str = _NON_EMPTY
    idempotency_key: str = _NON_EMPTY

    @field_validator(
        "tenant_id",
        "workspace_id",
        "assignment_id",
        "work_item_id",
        "acting_principal_id",
        "idempotency_key",
    )
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized


def apply_work_item_transition(
    work_item: WorkItem,
    request: WorkItemTransitionRequest,
    *,
    updated_at: datetime,
) -> WorkItem:
    """Apply one validated WorkItem lifecycle transition without persistence."""
    if type(work_item) is not WorkItem:
        raise TypeError("work_item must be WorkItem")
    if type(request) is not WorkItemTransitionRequest:
        raise TypeError("request must be WorkItemTransitionRequest")
    if updated_at.tzinfo is None:
        raise ValueError("updated_at must be timezone-aware")

    if request.tenant_id != work_item.tenant_id:
        raise CollaborativeWorkLifecycleError("transition tenant_id must match WorkItem tenant_id")
    if request.workspace_id != work_item.workspace_id:
        raise CollaborativeWorkLifecycleError(
            "transition workspace_id must match WorkItem workspace_id",
        )
    if request.work_item_id != work_item.work_item_id:
        raise CollaborativeWorkLifecycleError(
            "transition work_item_id must match WorkItem work_item_id",
        )
    if request.expected_revision != work_item.revision:
        raise CollaborativeWorkLifecycleError(
            "expected_revision does not match current WorkItem revision",
        )
    if updated_at < work_item.updated_at:
        raise ValueError("updated_at must be greater than or equal to WorkItem updated_at")

    validate_work_item_state_transition(
        from_state=work_item.state,
        to_state=request.target_state,
    )
    return work_item.model_copy(
        update={
            "state": request.target_state,
            "revision": work_item.revision + 1,
            "updated_at": updated_at,
        },
    )


def apply_assignment_transition(
    assignment: Assignment,
    request: AssignmentTransitionRequest,
    *,
    updated_at: datetime | None = None,
) -> Assignment:
    """Apply one validated Assignment lifecycle transition without persistence."""
    if type(assignment) is not Assignment:
        raise TypeError("assignment must be Assignment")
    if type(request) is not AssignmentTransitionRequest:
        raise TypeError("request must be AssignmentTransitionRequest")
    if updated_at is not None and updated_at.tzinfo is None:
        raise ValueError("updated_at must be timezone-aware when provided")

    if request.tenant_id != assignment.tenant_id:
        raise CollaborativeWorkLifecycleError(
            "transition tenant_id must match Assignment tenant_id",
        )
    if request.workspace_id != assignment.workspace_id:
        raise CollaborativeWorkLifecycleError(
            "transition workspace_id must match Assignment workspace_id",
        )
    if request.assignment_id != assignment.assignment_id:
        raise CollaborativeWorkLifecycleError(
            "transition assignment_id must match Assignment assignment_id",
        )
    if request.work_item_id != assignment.work_item_id:
        raise CollaborativeWorkLifecycleError(
            "transition work_item_id must match Assignment work_item_id",
        )
    if request.expected_revision != assignment.revision:
        raise CollaborativeWorkLifecycleError(
            "expected_revision does not match current Assignment revision",
        )
    if (
        assignment.updated_at is not None
        and updated_at is not None
        and updated_at < assignment.updated_at
    ):
        raise ValueError("updated_at must be greater than or equal to Assignment updated_at")

    validate_assignment_state_transition(
        from_state=assignment.state,
        to_state=request.target_state,
    )
    updates: dict[str, object] = {
        "state": request.target_state,
        "revision": assignment.revision + 1,
    }
    if updated_at is not None:
        updates["updated_at"] = updated_at
    return assignment.model_copy(update=updates)
