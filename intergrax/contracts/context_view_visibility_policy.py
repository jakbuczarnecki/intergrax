# © Artur Czarnecki. All rights reserved.

"""Principal-scoped ContextView visibility policy contracts (Multiplayer MP-5C).

Deterministic eligibility only — no retrieval, composition, or storage.
Collaborative Work owns this seam; Memory / RAG / UCL / CE remain downstream.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from intergrax.contracts.collaborative_work import EffectiveAuthorityDecision
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.contracts.context_view import (
    ContextViewCategory,
    ContextViewRequest,
    ContextViewScope,
    ContextViewVisibilityClass,
)

SCHEMA_CONTEXT_VIEW_VISIBILITY_POLICY_INPUT_V1: Final = (
    "context_view_visibility_policy_input.v1"
)
SCHEMA_CONTEXT_VIEW_POLICY_DECISION_V1: Final = "context_view_policy_decision.v1"
SCHEMA_DEFAULT_CONTEXT_VIEW_VISIBILITY_POLICY_CONFIG_V1: Final = (
    "default_context_view_visibility_policy_config.v1"
)

CONTEXT_VIEW_READ_AUTHORITY_SCOPE: Final = "collaborative_work.context_view.read"
DEFAULT_CONTEXT_VIEW_VISIBILITY_POLICY_ID: Final = (
    "collaborative_work.context_view.visibility.default"
)

_NON_EMPTY = Field(min_length=1)


class ContextViewPolicyOutcome(StrEnum):
    """Visibility eligibility outcome — not governance HITL."""

    ALLOW = "allow"
    DENY = "deny"


class ContextViewPolicyDenialReason(StrEnum):
    """Typed fail-closed reasons for overall visibility denial."""

    AUTHORITY_DENIED = "authority_denied"
    MISSING_AUTHORITY_RESOLUTION = "missing_authority_resolution"
    SCOPE_ISOLATION = "scope_isolation"
    NO_ELIGIBLE_CATEGORIES = "no_eligible_categories"
    POLICY_AMBIGUITY = "policy_ambiguity"


class ContextViewCategoryDenialReason(StrEnum):
    """Per-category eligibility denial — partial eligibility may still ALLOW overall."""

    AUTHORITY_INSUFFICIENT = "authority_insufficient"
    WORK_ITEM_SCOPE_REQUIRED = "work_item_scope_required"
    DELEGATION_SCOPE_INSUFFICIENT = "delegation_scope_insufficient"
    OPERATION_SCOPE_MISMATCH = "operation_scope_mismatch"
    UNSUPPORTED_CATEGORY = "unsupported_category"
    POLICY_AMBIGUITY = "policy_ambiguity"


class ContextViewCategoryDenial(BaseModel):
    """Immutable per-category denial with typed reason."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    category: ContextViewCategory
    reason: ContextViewCategoryDenialReason


class ContextViewVisibilityPolicyInput(BaseModel):
    """Minimal semantic input for replaceable visibility policy strategies."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["context_view_visibility_policy_input.v1"] = (
        SCHEMA_CONTEXT_VIEW_VISIBILITY_POLICY_INPUT_V1
    )
    request: ContextViewRequest
    effective_authority: EffectiveAuthorityDecision
    authoritative_delegation_scopes: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def _require_resolved_authority(self) -> ContextViewVisibilityPolicyInput:
        if self.effective_authority.decision.action is PolicyAction.ALLOW:
            return self
        if self.authoritative_delegation_scopes is not None:
            raise ValueError(
                "authoritative_delegation_scopes must be omitted when authority is denied",
            )
        return self


class ContextViewPolicyDecision(BaseModel):
    """Immutable visibility eligibility decision for MP-5D composition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["context_view_policy_decision.v1"] = (
        SCHEMA_CONTEXT_VIEW_POLICY_DECISION_V1
    )
    outcome: ContextViewPolicyOutcome
    policy_id: str = _NON_EMPTY
    effective_scope: ContextViewScope
    eligible_categories: tuple[ContextViewCategory, ...] = ()
    denied_categories: tuple[ContextViewCategoryDenial, ...] = ()
    eligible_visibility_classes: tuple[ContextViewVisibilityClass, ...] = ()
    private_visibility_principal_id: str | None = None
    denial_reason: ContextViewPolicyDenialReason | None = None

    @model_validator(mode="after")
    def _align_outcome_fields(self) -> ContextViewPolicyDecision:
        if self.outcome is ContextViewPolicyOutcome.ALLOW:
            if self.denial_reason is not None:
                raise ValueError("denial_reason must be omitted when outcome is allow")
            if not self.eligible_categories:
                raise ValueError("eligible_categories required when outcome is allow")
        else:
            if self.denial_reason is None:
                raise ValueError("denial_reason required when outcome is deny")
            if self.eligible_categories:
                raise ValueError("eligible_categories must be empty when outcome is deny")
        private = ContextViewVisibilityClass.PRIVATE_TO_PRINCIPAL
        if private in self.eligible_visibility_classes:
            if not (self.private_visibility_principal_id or "").strip():
                raise ValueError(
                    "private_visibility_principal_id required when PRIVATE_TO_PRINCIPAL eligible",
                )
        elif self.private_visibility_principal_id is not None:
            raise ValueError(
                "private_visibility_principal_id must be omitted unless PRIVATE_TO_PRINCIPAL eligible",
            )
        return self


class DefaultContextViewVisibilityPolicyConfig(BaseModel):
    """Typed immutable configuration for the platform default visibility policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["default_context_view_visibility_policy_config.v1"] = (
        SCHEMA_DEFAULT_CONTEXT_VIEW_VISIBILITY_POLICY_CONFIG_V1
    )
    policy_id: str = DEFAULT_CONTEXT_VIEW_VISIBILITY_POLICY_ID
    required_authority_scope: str = CONTEXT_VIEW_READ_AUTHORITY_SCOPE
    workspace_shared_delegation_scope: str = "collaborative_work.context_view.workspace_shared"
    platform_visible_delegation_scope: str = "collaborative_work.context_view.platform_visible"


def fail_closed_context_view_policy_decision(
    *,
    policy_id: str,
    effective_scope: ContextViewScope,
    denial_reason: ContextViewPolicyDenialReason,
    denied_categories: tuple[ContextViewCategoryDenial, ...] = (),
) -> ContextViewPolicyDecision:
    """Construct a mandatory fail-closed visibility decision."""
    return ContextViewPolicyDecision(
        outcome=ContextViewPolicyOutcome.DENY,
        policy_id=policy_id,
        effective_scope=effective_scope,
        denied_categories=denied_categories,
        denial_reason=denial_reason,
    )


@runtime_checkable
class ContextViewVisibilityPolicy(Protocol):
    """Replaceable principal visibility policy — eligibility only."""

    @property
    def policy_id(self) -> str:
        """Stable policy identity for audit."""

    def evaluate(self, policy_input: ContextViewVisibilityPolicyInput) -> ContextViewPolicyDecision:
        """Determine category and visibility-class eligibility under resolved authority."""
