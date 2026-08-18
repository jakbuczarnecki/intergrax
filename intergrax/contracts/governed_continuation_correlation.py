# © Artur Czarnecki. All rights reserved.

"""Typed correlation between canonical HITL and governed continuation requests."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

_NON_EMPTY = Field(min_length=1)


class GovernedContinuationCorrelation(BaseModel):
    """Machine-readable link from HumanRequest to an exact continuation request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    continuation_request_id: str = _NON_EMPTY
    reason: str = _NON_EMPTY
    operation_id: str = _NON_EMPTY
    policy_rule_id: str | None = None
    resource_scope: str | None = None
    policy_action: str | None = None
    source_step_id: str | None = None

    @field_validator("continuation_request_id", "operation_id", "reason")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("field must be non-empty")
        return normalized

    @field_validator("policy_rule_id", "resource_scope", "source_step_id")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("reason")
    @classmethod
    def _validate_reason(cls, value: str) -> str:
        from intergrax.contracts.governed_continuation import ContinuationReason

        ContinuationReason(value)
        return value

    @field_validator("policy_action")
    @classmethod
    def _validate_policy_action(cls, value: str | None) -> str | None:
        if value is None:
            return None
        from intergrax.contracts.runtime_policy import PolicyAction

        PolicyAction(value)
        return value.strip()
