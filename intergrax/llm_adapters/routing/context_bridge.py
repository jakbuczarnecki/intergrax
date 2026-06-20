# © Artur Czarnecki. All rights reserved.

"""Auto-fill RoutingContext from Nexus / kernel / budget snapshots (M-LLM-X.10.2)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from intergrax.contracts.acp_state import AcpInvocationUsageView
from intergrax.contracts.agent_budget import ResolvedBudgetLimits
from intergrax.llm_adapters.routing.contracts import RoutingContext


@dataclass
class LLMRoutingRuntimeSnapshot:
    """Mutable routing inputs refreshed during a Nexus run (M-LLM-X.11.2)."""

    task_class: str | None = None
    agent_id: str | None = None
    step_index: int | None = None
    budget_degrade_active: bool = False
    budget_limits: ResolvedBudgetLimits | None = None
    invocation_usage: AcpInvocationUsageView | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def budget_remaining_ratio_from_limits(
    limits: ResolvedBudgetLimits | None,
) -> float | None:
    """Derive remaining budget ratio from resolved harness limits."""
    if limits is None:
        return None
    if (
        limits.agent_tokens_limit is not None
        and limits.agent_tokens_remaining is not None
        and limits.agent_tokens_limit > 0
    ):
        return limits.agent_tokens_remaining / limits.agent_tokens_limit
    if (
        limits.environment_tokens_limit is not None
        and limits.environment_tokens_remaining is not None
        and limits.environment_tokens_limit > 0
    ):
        return limits.environment_tokens_remaining / limits.environment_tokens_limit
    return None


def tokens_used_from_usage(usage: AcpInvocationUsageView | None) -> int | None:
    if usage is None:
        return None
    total = usage.agent.tokens_total
    if total <= 0:
        total = usage.environment.tokens_total
    return total if total > 0 else None


def build_routing_context_from_runtime(
    *,
    tenant_id: str | None = None,
    agent_id: str | None = None,
    task_class: str | None = None,
    budget_remaining_ratio: float | None = None,
    tokens_used: int | None = None,
    step_index: int | None = None,
    model_hint: str | None = None,
    budget_degrade_active: bool = False,
    metadata: Mapping[str, Any] | None = None,
    budget_limits: ResolvedBudgetLimits | None = None,
    invocation_usage: AcpInvocationUsageView | None = None,
) -> RoutingContext:
    """
    Build a routing snapshot from runtime fields and optional request metadata.

  When explicit values are omitted, known metadata keys and budget meters are used.
    """
    meta = dict(metadata or {})
    resolved_task_class = task_class or _meta_str(meta, "task_class", "capability")
    resolved_agent_id = agent_id or _meta_str(meta, "agent_id")
    resolved_tenant = tenant_id or _meta_str(meta, "tenant_id") or "default"
    resolved_step = step_index
    if resolved_step is None:
        raw_step = meta.get("step_index")
        if isinstance(raw_step, int):
            resolved_step = raw_step
    resolved_budget_ratio = budget_remaining_ratio
    if resolved_budget_ratio is None:
        resolved_budget_ratio = budget_remaining_ratio_from_limits(budget_limits)
    resolved_tokens = tokens_used
    if resolved_tokens is None:
        resolved_tokens = tokens_used_from_usage(invocation_usage)
    resolved_model_hint = model_hint or _meta_str(meta, "model_hint", "llm_model_hint")
    return RoutingContext(
        task_class=resolved_task_class,
        budget_remaining_ratio=resolved_budget_ratio,
        tokens_used=resolved_tokens,
        step_index=resolved_step,
        model_hint=resolved_model_hint,
        tenant_id=resolved_tenant,
        agent_id=resolved_agent_id,
        budget_degrade_active=budget_degrade_active,
    )


def refresh_llm_routing_context(
    snapshot: LLMRoutingRuntimeSnapshot,
    *,
    tenant_id: str | None = None,
    step_index: int | None = None,
    budget_degrade_active: bool | None = None,
    task_class: str | None = None,
    agent_id: str | None = None,
    budget_limits: ResolvedBudgetLimits | None = None,
    invocation_usage: AcpInvocationUsageView | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[LLMRoutingRuntimeSnapshot, RoutingContext]:
    """Update snapshot fields and return a fresh immutable ``RoutingContext``."""
    merged_metadata = dict(snapshot.metadata)
    if metadata is not None:
        merged_metadata.update(metadata)
    refreshed = LLMRoutingRuntimeSnapshot(
        task_class=task_class if task_class is not None else snapshot.task_class,
        agent_id=agent_id if agent_id is not None else snapshot.agent_id,
        step_index=step_index if step_index is not None else snapshot.step_index,
        budget_degrade_active=(
            budget_degrade_active
            if budget_degrade_active is not None
            else snapshot.budget_degrade_active
        ),
        budget_limits=budget_limits if budget_limits is not None else snapshot.budget_limits,
        invocation_usage=(
            invocation_usage if invocation_usage is not None else snapshot.invocation_usage
        ),
        metadata=merged_metadata,
    )
    context = build_routing_context_from_runtime(
        tenant_id=tenant_id,
        agent_id=refreshed.agent_id,
        task_class=refreshed.task_class,
        step_index=refreshed.step_index,
        budget_degrade_active=refreshed.budget_degrade_active,
        metadata=refreshed.metadata,
        budget_limits=refreshed.budget_limits,
        invocation_usage=refreshed.invocation_usage,
    )
    return refreshed, context


def _meta_str(meta: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
