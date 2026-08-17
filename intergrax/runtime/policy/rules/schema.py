# © Artur Czarnecki. All rights reserved.

"""Declarative policy rule schema (Phase H-APP.2.4)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PolicyRuleAction(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_HITL = "require_hitl"


class DeclarativePolicyRule(BaseModel):
    """Single YAML/JSON policy rule — evaluated by typed handlers only."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    handler_id: str
    resource_kind: str = Field(description="tool | agent | capability")
    resource_id: str = "*"
    action: PolicyRuleAction = PolicyRuleAction.ALLOW
    conditions: dict[str, Any] = Field(default_factory=dict)

    @field_validator("rule_id", "handler_id")
    @classmethod
    def _normalize_identity(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("identity fields must not be empty or whitespace-only")
        return normalized
