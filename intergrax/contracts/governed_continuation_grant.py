# © Artur Czarnecki. All rights reserved.

"""Typed execution grant for governed continuation after canonical human approval."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.validation import validate_content_digest

SCHEMA_GOVERNED_CONTINUATION_APPROVAL_GRANT_V1: Final = (
    "governed_continuation_approval_grant.v1"
)

_NON_EMPTY = Field(min_length=1)


class GovernedContinuationApprovalGrant(BaseModel):
    """Scoped authorization for one exact proposed side-effect continuation scope.

    Derived only from canonical ``HumanApprovalResolution`` plus the typed
    ``HumanRequest.governed_continuation`` correlation. Does not execute anything.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["governed_continuation_approval_grant.v1"] = (
        SCHEMA_GOVERNED_CONTINUATION_APPROVAL_GRANT_V1
    )
    grant_id: str = _NON_EMPTY
    continuation_request_id: str = _NON_EMPTY
    side_effect_scope_id: str = _NON_EMPTY
    side_effect_scope_digest: str | None = None
    task_id: str = _NON_EMPTY
    run_id: str = _NON_EMPTY
    operation_id: str = _NON_EMPTY
    resource_scope: str | None = None
    policy_rule_id: str | None = None
    pause_id: str = _NON_EMPTY
    human_request_id: str = _NON_EMPTY
    approved_at: str = _NON_EMPTY

    @field_validator(
        "grant_id",
        "continuation_request_id",
        "side_effect_scope_id",
        "task_id",
        "run_id",
        "operation_id",
        "pause_id",
        "human_request_id",
        "approved_at",
    )
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("field must be non-empty")
        return normalized

    @field_validator("resource_scope", "policy_rule_id")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("side_effect_scope_digest")
    @classmethod
    def _validate_side_effect_scope_digest(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_content_digest(value)
