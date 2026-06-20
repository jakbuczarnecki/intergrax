# © Artur Czarnecki. All rights reserved.

"""Tier-0 routing snapshot refresh on ``RuntimeConfig`` (M-LLM-X.12.3 / 12.4)."""

from __future__ import annotations

from typing import Any

from intergrax.contracts.agent_budget import ResolvedBudgetLimits
from intergrax.llm_adapters.routing.context_bridge import (
    LLMRoutingRuntimeSnapshot,
    refresh_llm_routing_context,
)
from intergrax.llm_adapters.routing.contracts import RoutingContext
from intergrax.llm_adapters.routing.metering import tokens_used_from_adapter
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.config import RuntimeConfig


def resolved_budget_limits_from_run_budget(
    run_budget: RunBudget | None,
    tokens_used: int | None,
) -> ResolvedBudgetLimits | None:
    if run_budget is None or run_budget.max_total_tokens is None:
        return None
    used = tokens_used or 0
    remaining = max(0, run_budget.max_total_tokens - used)
    return ResolvedBudgetLimits(
        agent_tokens_limit=run_budget.max_total_tokens,
        agent_tokens_remaining=remaining,
        limit_source="binding",
    )


def refresh_config_routing_snapshot(
    config: RuntimeConfig,
    *,
    tenant_id: str | None = None,
    step_index: int | None = None,
    task_class: str | None = None,
    agent_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    run_id: str | None = None,
    budget_degrade_active: bool | None = None,
) -> RoutingContext | None:
    """Refresh ``llm_routing_snapshot`` on config using live adapter metering."""
    if config.llm_routing_snapshot is None and config.llm_routing_context is None:
        return config.llm_routing_context

    tokens_used = tokens_used_from_adapter(config.llm_adapter, run_id=run_id)
    usage_tracker = config.llm_usage_tracker
    if usage_tracker is not None and tokens_used == 0:
        tokens_used = int(usage_tracker.build_report().total.total_tokens)

    limits = resolved_budget_limits_from_run_budget(config.run_budget, tokens_used)
    merged_metadata = dict((config.llm_routing_snapshot or LLMRoutingRuntimeSnapshot()).metadata)
    if metadata:
        merged_metadata.update(metadata)
    if task_class is not None:
        merged_metadata.setdefault("task_class", task_class)
    if agent_id is not None:
        merged_metadata.setdefault("agent_id", agent_id)

    degrade = budget_degrade_active
    if degrade is None and config.llm_routing_snapshot is not None:
        degrade = config.llm_routing_snapshot.budget_degrade_active
    if degrade is None:
        raw = config.metadata.get("budget_degrade_active")
        degrade = bool(raw) if isinstance(raw, bool) else False

    snapshot = config.llm_routing_snapshot or LLMRoutingRuntimeSnapshot(metadata=merged_metadata)
    refreshed, context = refresh_llm_routing_context(
        snapshot,
        tenant_id=tenant_id or config.tenant_id,
        agent_id=agent_id or snapshot.agent_id,
        task_class=task_class or snapshot.task_class,
        step_index=step_index if step_index is not None else snapshot.step_index,
        budget_degrade_active=degrade,
        budget_limits=limits,
        metadata=merged_metadata,
    )
    config.llm_routing_snapshot = refreshed
    config.llm_routing_context = context
    return context
