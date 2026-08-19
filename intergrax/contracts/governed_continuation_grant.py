# © Artur Czarnecki. All rights reserved.

"""Typed execution grant for governed continuation after canonical human approval."""

from __future__ import annotations

from typing import Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.policy_bundle_provenance import (
    has_attested_policy_bundle_provenance,
    strip_policy_bundle_provenance_identifier,
    validate_policy_bundle_provenance_complete_or_absent,
)
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
    policy_bundle_id: str = ""
    policy_bundle_version: str = ""
    policy_bundle_digest: str = ""
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

    @field_validator(
        "policy_bundle_id",
        "policy_bundle_version",
        "policy_bundle_digest",
    )
    @classmethod
    def _strip_bundle_provenance(cls, value: str) -> str:
        return strip_policy_bundle_provenance_identifier(value)

    @model_validator(mode="after")
    def _bundle_provenance_complete_or_absent(self) -> Self:
        validate_policy_bundle_provenance_complete_or_absent(
            self.policy_bundle_id,
            self.policy_bundle_version,
            self.policy_bundle_digest,
        )
        return self

    def has_attested_policy_bundle_refs(self) -> bool:
        return has_attested_policy_bundle_provenance(
            self.policy_bundle_id,
            self.policy_bundle_version,
            self.policy_bundle_digest,
        )

    @field_validator("side_effect_scope_digest")
    @classmethod
    def _validate_side_effect_scope_digest(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_content_digest(value)
