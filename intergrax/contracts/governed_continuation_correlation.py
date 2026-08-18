# © Artur Czarnecki. All rights reserved.

"""Typed correlation between canonical HITL and governed continuation requests."""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.policy_action import PolicyAction

_NON_EMPTY = Field(min_length=1)
_CONTENT_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


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
        normalized = value.strip()
        if not normalized:
            return None
        if not _CONTENT_DIGEST_RE.match(normalized):
            raise ValueError("digest must match sha256:<64 lowercase hex>")
        return normalized

    @field_validator("policy_rule_id", "resource_scope", "source_step_id")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None
