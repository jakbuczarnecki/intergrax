# © Artur Czarnecki. All rights reserved.

"""Collaborative Work identity and authority contracts (MP-1 / COLLAB-WORK-1A).

Semantic source of truth for collaborative principals, explicit workspace
membership, authority delegation, and the effective-authority evaluation
boundary. Distinct from:

- ``RequestIdentity`` / ``PrincipalType`` — run-scoped execution intake only.
- ``DelegationSpec`` — Nexus graph child-run execution delegation only.
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
