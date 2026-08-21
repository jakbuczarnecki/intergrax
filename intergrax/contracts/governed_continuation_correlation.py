# © Artur Czarnecki. All rights reserved.

"""Typed correlation between canonical HITL and governed continuation requests."""

from __future__ import annotations

from enum import StrEnum

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.policy_action import PolicyAction
from intergrax.contracts.policy_bundle_provenance import (
    has_attested_policy_bundle_provenance,
    strip_policy_bundle_provenance_identifier,
    validate_policy_bundle_provenance_complete_or_absent,
)
from intergrax.contracts.validation import validate_content_digest

_NON_EMPTY = Field(min_length=1)


class ContinuationReason(StrEnum):
    """Why continuation is blocked — independent of commercial/domain logic."""

    QUOTE = "quote"
    SECURITY = "security"
    LEGAL = "legal"
    PROCUREMENT = "procurement"
    COMPLIANCE = "compliance"
    PUBLICATION = "publication"


class GovernedContinuationCorrelation(BaseModel):
    """Machine-readable link from HumanRequest to an exact continuation request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    continuation_request_id: str = _NON_EMPTY
    reason: ContinuationReason
    task_id: str = _NON_EMPTY
    run_id: str = _NON_EMPTY
    side_effect_scope_id: str | None = None
    side_effect_scope_digest: str | None = None
    operation_id: str = _NON_EMPTY
    policy_rule_id: str | None = None
    policy_bundle_id: str = ""
    policy_bundle_version: str = ""
    policy_bundle_digest: str = ""
    resource_scope: str | None = None
    policy_action: PolicyAction | None = None
    source_step_id: str | None = None

    @field_validator("continuation_request_id", "operation_id", "task_id", "run_id")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("field must be non-empty")
        return normalized

    @field_validator("side_effect_scope_id")
    @classmethod
    def _strip_side_effect_scope_id(cls, value: str | None) -> str | None:
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

    @field_validator("policy_rule_id", "resource_scope", "source_step_id")
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
