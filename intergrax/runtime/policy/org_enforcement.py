# © Artur Czarnecki. All rights reserved.

"""Organizational policy pre-checks in harness kernel (architecture §39.4 · ACP-ORG-3)."""

from __future__ import annotations

import fnmatch

from intergrax.contracts.org_policy import OrganizationalPolicyContext
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision


def _tool_matches_pattern(tool_id: str, pattern: str) -> bool:
    if pattern == tool_id:
        return True
    return fnmatch.fnmatch(tool_id, pattern)


def evaluate_org_policy_pre(
    *,
    org: OrganizationalPolicyContext | None,
    channel: str | None,
    requested_tool_ids: list[str],
) -> PolicyDecision | None:
    """Return DENY decision when org envelope blocks channel or tool; else None."""
    if org is None:
        return None

    if channel:
        denied = set(org.channel_policy.denied_channels)
        if channel in denied:
            return PolicyDecision(
                action=PolicyAction.DENY,
                reason=f"channel {channel!r} denied by org envelope",
                policy_rule_id="org.channel.denied",
            )
        allowed = org.channel_policy.allowed_channels
        if allowed and channel not in allowed:
            return PolicyDecision(
                action=PolicyAction.DENY,
                reason=f"channel {channel!r} not in org allowlist",
                policy_rule_id="org.channel.not_allowed",
            )

    for tool_id in requested_tool_ids:
        for pattern in org.effective_tool_denies:
            if _tool_matches_pattern(tool_id, pattern):
                return PolicyDecision(
                    action=PolicyAction.DENY,
                    reason=f"tool {tool_id!r} denied by org overlay",
                    policy_rule_id="org.tool.denied",
                )

    return None


def extract_requested_tool_ids(requested_actions: list[dict[str, object]] | None) -> list[str]:
    if not requested_actions:
        return []
    tool_ids: list[str] = []
    for action in requested_actions:
        raw = action.get("tool_id")
        if isinstance(raw, str) and raw:
            tool_ids.append(raw)
    return tool_ids
