# © Artur Czarnecki. All rights reserved.

"""Collaborative Activity scoped read authorization contracts (Multiplayer MP-6E).

Read authorization before provider query — deterministic only; no persistence.
Reuses MP-1 ``EffectiveAuthorityRequest`` / ``EffectiveAuthorityDecision``; no duplicate RBAC.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.collaborative_activity import CollaborativeActivityQuery
from intergrax.contracts.collaborative_work import (
    AuthorityDelegation,
    EffectiveAuthorityDecision,
    EffectiveAuthorityRequest,
    MembershipResolutionMode,
    WorkspaceMembership,
)

SCHEMA_COLLABORATIVE_ACTIVITY_READ_REQUEST_V1: Final = (
    "collaborative_activity_read_request.v1"
)
SCHEMA_COLLABORATIVE_ACTIVITY_READ_AUTHORIZATION_DECISION_V1: Final = (
    "collaborative_activity_read_authorization_decision.v1"
)
SCHEMA_DEFAULT_COLLABORATIVE_ACTIVITY_READ_AUTHORIZATION_POLICY_CONFIG_V1: Final = (
    "default_collaborative_activity_read_authorization_policy_config.v1"
)

COLLABORATIVE_ACTIVITY_READ_AUTHORITY_SCOPE: Final = "collaborative_work.activity.read"

DEFAULT_COLLABORATIVE_ACTIVITY_READ_AUTHORIZATION_POLICY_ID: Final = (
    "collaborative_work.collaborative_activity.read.authorization.default"
)

_NON_EMPTY = Field(min_length=1)


class CollaborativeActivityReadAuthorizationOutcome(StrEnum):
    ALLOW = "allow"
    DENY = "deny"


class CollaborativeActivityReadDenialReason(StrEnum):
    """Stable fail-closed read denial codes — generic where existence must not leak."""

    AUTHORITY_DENIED = "authority_denied"
    MISSING_AUTHORITY_RESOLUTION = "missing_authority_resolution"
    SCOPE_ISOLATION = "scope_isolation"
    POLICY_AMBIGUITY = "policy_ambiguity"


class CollaborativeActivityReadRequest(BaseModel):
    """Principal-scoped read intent — ``query`` fields are untrusted until policy authorizes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["collaborative_activity_read_request.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_READ_REQUEST_V1
    )
    query: CollaborativeActivityQuery
    acting_principal_id: str = _NON_EMPTY
    authority_request: EffectiveAuthorityRequest | None = None
    delegator_principal_id: str | None = None
    membership: WorkspaceMembership | None = None
    membership_resolution_mode: MembershipResolutionMode = MembershipResolutionMode.LOCATOR
    delegation: AuthorityDelegation | None = None

    @field_validator("acting_principal_id", "delegator_principal_id")
    @classmethod
    def _strip_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty when provided")
        return normalized

    @model_validator(mode="after")
    def _align_authority_with_query(self) -> CollaborativeActivityReadRequest:
        query = self.query
        if self.authority_request is not None:
            auth = self.authority_request
            if auth.tenant_id != query.tenant_id:
                raise ValueError("authority_request tenant_id must match query tenant_id")
            if auth.workspace_id != query.workspace_id:
                raise ValueError("authority_request workspace_id must match query workspace_id")
            if auth.acting_principal_id != self.acting_principal_id:
                raise ValueError(
                    "authority_request acting_principal_id must match acting_principal_id",
                )
        return self


class CollaborativeActivityReadAuthorizationDecision(BaseModel):
    """Policy output — ``authorized_query`` is trusted provider input on ALLOW only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["collaborative_activity_read_authorization_decision.v1"] = (
        SCHEMA_COLLABORATIVE_ACTIVITY_READ_AUTHORIZATION_DECISION_V1
    )
    outcome: CollaborativeActivityReadAuthorizationOutcome
    policy_id: str = _NON_EMPTY
    authorized_query: CollaborativeActivityQuery | None = None
    denial_reason: CollaborativeActivityReadDenialReason | None = None

    @model_validator(mode="after")
    def _align_outcome(self) -> CollaborativeActivityReadAuthorizationDecision:
        if self.outcome is CollaborativeActivityReadAuthorizationOutcome.ALLOW:
            if self.authorized_query is None:
                raise ValueError("authorized_query is required on allow")
            if self.denial_reason is not None:
                raise ValueError("denial_reason must be omitted on allow")
        else:
            if self.authorized_query is not None:
                raise ValueError("authorized_query must be omitted on deny")
        return self


class DefaultCollaborativeActivityReadAuthorizationPolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "default_collaborative_activity_read_authorization_policy_config.v1"
    ] = SCHEMA_DEFAULT_COLLABORATIVE_ACTIVITY_READ_AUTHORIZATION_POLICY_CONFIG_V1
    policy_id: str = Field(default=DEFAULT_COLLABORATIVE_ACTIVITY_READ_AUTHORIZATION_POLICY_ID)


class CollaborativeActivityReadAuthorizationPolicyInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request: CollaborativeActivityReadRequest
    effective_authority: EffectiveAuthorityDecision


@runtime_checkable
class CollaborativeActivityReadAuthorizationPolicy(Protocol):
    """Replaceable read authorization — scope may narrow, never broaden."""

    @property
    def policy_id(self) -> str: ...

    def evaluate(
        self,
        policy_input: CollaborativeActivityReadAuthorizationPolicyInput,
    ) -> CollaborativeActivityReadAuthorizationDecision: ...


def fail_closed_collaborative_activity_read_decision(
    *,
    policy_id: str,
    denial_reason: CollaborativeActivityReadDenialReason,
) -> CollaborativeActivityReadAuthorizationDecision:
    return CollaborativeActivityReadAuthorizationDecision(
        outcome=CollaborativeActivityReadAuthorizationOutcome.DENY,
        policy_id=policy_id,
        denial_reason=denial_reason,
    )


class CollaborativeActivityReadDenied(Exception):
    """Read rejected by authorization — distinct from storage failure."""

    def __init__(
        self,
        *,
        denial_reason: CollaborativeActivityReadDenialReason,
        policy_id: str,
    ) -> None:
        self.denial_reason = denial_reason
        self.policy_id = policy_id
        super().__init__(f"{policy_id}: read denied ({denial_reason.value})")


class CollaborativeActivityReadPolicyError(Exception):
    """Authorization policy evaluation failed — fail closed."""


class CollaborativeActivityCursorInvalid(Exception):
    """Opaque cursor failed validation — not an authorization token."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class CollaborativeActivityReadPersistenceError(RuntimeError):
    """Read provider failed — not an authorization denial."""
