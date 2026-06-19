# © Artur Czarnecki. All rights reserved.

"""Nexus runtime wiring for live LLM routing context (M-LLM-X.11.2)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from intergrax.contracts.agent_budget import ResolvedBudgetLimits
from intergrax.llm_adapters.routing.context_bridge import (
    LLMRoutingRuntimeSnapshot,
    refresh_llm_routing_context,
    build_routing_context_from_runtime,
)
from intergrax.llm_adapters.routing.contracts import RoutingContext
from intergrax.llm_adapters.routing.evaluating_adapter import RoutingContextProvider
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest


def resolved_budget_limits_from_run_budget(
    run_budget: RunBudget | None,
    tokens_used: int | None,
) -> ResolvedBudgetLimits | None:
    """Map Nexus ``RunBudget`` + usage into routing ``ResolvedBudgetLimits``."""
    if run_budget is None or run_budget.max_total_tokens is None:
        return None
    used = tokens_used or 0
    remaining = max(0, run_budget.max_total_tokens - used)
    return ResolvedBudgetLimits(
        agent_tokens_limit=run_budget.max_total_tokens,
        agent_tokens_remaining=remaining,
        limit_source="binding",
    )


def init_llm_routing_on_config(
    config: RuntimeConfig,
    routing_context: RoutingContext,
    request: RuntimeRequest,
) -> None:
    """Seed mutable routing snapshot on ``RuntimeConfig``."""
    metadata = dict(request.metadata)
    config.llm_routing_snapshot = LLMRoutingRuntimeSnapshot(
        task_class=routing_context.task_class,
        agent_id=routing_context.agent_id or request.agent_id,
        step_index=routing_context.step_index,
        budget_degrade_active=routing_context.budget_degrade_active,
        metadata=metadata,
    )
    config.llm_routing_context = routing_context


def make_config_routing_context_provider(config: RuntimeConfig) -> RoutingContextProvider:
    """Return provider that reads the latest snapshot from ``RuntimeConfig``."""

    def _provider() -> RoutingContext:
        snapshot = config.llm_routing_snapshot
        if snapshot is None:
            return config.llm_routing_context or RoutingContext()
        return build_routing_context_from_runtime(
            tenant_id=config.tenant_id,
            agent_id=snapshot.agent_id,
            task_class=snapshot.task_class,
            step_index=snapshot.step_index,
            budget_degrade_active=snapshot.budget_degrade_active,
            metadata=snapshot.metadata,
            budget_limits=snapshot.budget_limits,
            invocation_usage=snapshot.invocation_usage,
        )

    return _provider


def sync_llm_routing_snapshot_for_state(state: RuntimeState) -> RoutingContext:
    """Refresh routing snapshot from live Nexus ``RuntimeState`` (budget, metadata)."""
    config = state.context.config
    tokens_used: int | None = None
    if state.llm_usage_tracker is not None:
        tokens_used = state.llm_usage_tracker.build_report().total.total_tokens
    limits = resolved_budget_limits_from_run_budget(config.run_budget, tokens_used)
    metadata: dict[str, Any] = dict(state.request.metadata)
    task_class = _optional_str(metadata.get("task_class"))
    agent_id = state.request.agent_id or _optional_str(metadata.get("agent_id"))
    step_index = metadata.get("step_index")
    if not isinstance(step_index, int):
        step_index = None
    snapshot = config.llm_routing_snapshot or LLMRoutingRuntimeSnapshot(metadata=metadata)
    refreshed, context = refresh_llm_routing_context(
        snapshot,
        tenant_id=state.tenant_id,
        agent_id=agent_id,
        task_class=task_class,
        step_index=step_index,
        budget_limits=limits,
        metadata=metadata,
    )
    config.llm_routing_snapshot = refreshed
    config.llm_routing_context = context
    return context


def wire_llm_routing_observability_on_state(state: RuntimeState) -> None:
    """Attach per-evaluation trace observers to evaluating adapter and resolver."""
    state.configure_llm_tracker()


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None
