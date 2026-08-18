# © Artur Czarnecki. All rights reserved.

"""Typed configuration contract for Tool Invocation Control (Governed Execution G2C-2B)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator

from intergrax.runtime.policy.rules.schema import PolicyRuleAction

TOOL_INVOCATION_CONTROL_CONFIGURATION_CONTRACT_ID = "tool_invocation_control.v1"


class ToolInvocationControlConfig(BaseModel):
    """Immutable typed configuration for ``tool_invocation_control@1``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_id: str
    action: PolicyRuleAction

    @field_validator("tool_id")
    @classmethod
    def _normalize_tool_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("tool_id must be non-empty")
        return normalized
