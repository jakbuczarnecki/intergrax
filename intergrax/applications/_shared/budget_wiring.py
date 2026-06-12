# © Artur Czarnecki. All rights reserved.

"""Tier-3 budget slice and reaction helpers (§43 · APP-PROD-7)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from intergrax.contracts.agent_budget import (
    AgentBudgetSlice,
    BudgetExceededReaction,
    BudgetLimitEnforcement,
    BudgetNotifyChannel,
    BudgetReactionProfile,
)

if TYPE_CHECKING:
    from intergrax.applications.contracts.manifest import ApplicationManifest

DEFAULT_PRODUCT_AGENT_TOKEN_LIMIT = 16_000


def product_budget_reaction() -> BudgetReactionProfile:
    """Default STRICT product host reaction policy — HITL on agent exceed, trace notify."""
    return BudgetReactionProfile(
        on_agent_limit_exceeded=BudgetExceededReaction.HITL,
        on_environment_limit_exceeded=BudgetExceededReaction.ABORT,
        notify_channels=[BudgetNotifyChannel.TRACE_ONLY, BudgetNotifyChannel.IN_APP],
    )


def product_agent_budget_slice(
    *,
    max_total_tokens: int = DEFAULT_PRODUCT_AGENT_TOKEN_LIMIT,
) -> AgentBudgetSlice:
    """Per-agent HARD token cap for product roster entries."""
    return AgentBudgetSlice(
        max_total_tokens=max_total_tokens,
        enforcement=BudgetLimitEnforcement.HARD,
    )


def check_manifest_budget_enforcement(
    product_id: str,
    manifest: "ApplicationManifest",
) -> list[str]:
    """Validate COST profile and per-agent budget slices for STRICT product hosts."""
    from intergrax.applications.contracts.manifest import ApplicationManifest

    _ = ApplicationManifest  # re-export guard for type checkers
    violations: list[str] = []
    env = manifest.resolved_environment()
    cost = env.cost_profile
    if not cost.budget_enforcement_enabled:
        violations.append(f"{product_id}: cost_profile.budget_enforcement_enabled must be true")
    if cost.max_total_tokens is None or cost.max_total_tokens < 1:
        violations.append(f"{product_id}: cost_profile.max_total_tokens must be set")
    if cost.budget_reaction is None:
        violations.append(f"{product_id}: cost_profile.budget_reaction must be configured")

    enabled = manifest.enabled_agents()
    if not enabled:
        violations.append(f"{product_id}: no enabled agents in manifest")
    for binding in enabled:
        label = binding.display_name()
        if binding.budget_slice is None:
            violations.append(f"{product_id}/{label}: missing AgentBinding.budget_slice")
            continue
        slice_ = binding.budget_slice
        if slice_.max_total_tokens is None or slice_.max_total_tokens < 1:
            violations.append(
                f"{product_id}/{label}: budget_slice.max_total_tokens must be a positive integer"
            )
        if slice_.enforcement is not BudgetLimitEnforcement.HARD:
            violations.append(
                f"{product_id}/{label}: budget_slice.enforcement must be HARD on STRICT product hosts"
            )
    return violations
