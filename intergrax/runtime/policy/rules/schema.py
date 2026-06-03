# © Artur Czarnecki. All rights reserved.

"""Declarative policy rule schema (Phase H-APP.2.4)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PolicyRuleAction(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_HITL = "require_hitl"


class DeclarativePolicyRule(BaseModel):
    """Single YAML/JSON policy rule — evaluated by typed handlers only."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    resource_kind: str = Field(description="tool | agent | capability")
    resource_id: str = "*"
    action: PolicyRuleAction = PolicyRuleAction.ALLOW
    conditions: dict[str, Any] = Field(default_factory=dict)
