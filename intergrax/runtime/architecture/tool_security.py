# © Artur Czarnecki. All rights reserved.

"""Tool injection defense contracts (Phase V-SEC.2)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ToolInvocationPolicy(BaseModel):
    allowed_tool_ids: list[str] = Field(default_factory=list)
    blocked_argument_tokens: list[str] = Field(default_factory=list)
    require_explicit_capability_match: bool = True


class ToolInvocationRequest(BaseModel):
    tool_id: str
    arguments: dict[str, str] = Field(default_factory=dict)
    capability_ids: list[str] = Field(default_factory=list)


class ToolInvocationDecision(BaseModel):
    allowed: bool
    reasons: list[str] = Field(default_factory=list)


def evaluate_tool_invocation_security(
    *,
    request: ToolInvocationRequest,
    policy: ToolInvocationPolicy,
) -> ToolInvocationDecision:
    reasons: list[str] = []
    if request.tool_id not in policy.allowed_tool_ids:
        reasons.append(f"Tool not allowed by policy: {request.tool_id}")

    for key, value in request.arguments.items():
        normalized_value = value.lower()
        for token in policy.blocked_argument_tokens:
            if token.lower() in normalized_value:
                reasons.append(
                    "Blocked token found in tool arguments: "
                    f"{key} contains `{token}`"
                )

    if policy.require_explicit_capability_match and request.tool_id not in request.capability_ids:
        reasons.append("Tool invocation missing explicit capability match")

    return ToolInvocationDecision(allowed=not reasons, reasons=reasons)
