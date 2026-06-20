# © Artur Czarnecki. All rights reserved.

"""ACP / harness helpers for live RoutingContext providers (M-LLM-X.10.2)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from intergrax.contracts.acp_state import AcpInvocationUsageView
from intergrax.contracts.agent_budget import ResolvedBudgetLimits
from intergrax.llm_adapters.routing.context_bridge import build_routing_context_from_runtime
from intergrax.llm_adapters.routing.contracts import RoutingContext


def make_acp_routing_context_provider(
    *,
    kernel_ctx_holder: list[Any],
    step_ctx_holder: list[Any],
    tenant_id: str,
    agent_id: str,
    task_class: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Callable[[], RoutingContext]:
    """Return a callable that snapshots kernel + step state for routing rules."""

    def _provider() -> RoutingContext:
        kernel = kernel_ctx_holder[0] if kernel_ctx_holder else None
        step_ctx = step_ctx_holder[0] if step_ctx_holder else None
        limits: ResolvedBudgetLimits | None = (
            kernel.resolved_budget_limits if kernel is not None else None
        )
        usage: AcpInvocationUsageView | None = (
            step_ctx.invocation_usage if step_ctx is not None else None
        )
        step_index = step_ctx.step_index if step_ctx is not None else None
        degrade = kernel.budget_degrade_active if kernel is not None else False
        return build_routing_context_from_runtime(
            tenant_id=tenant_id,
            agent_id=agent_id,
            task_class=task_class,
            step_index=step_index,
            budget_degrade_active=degrade,
            budget_limits=limits,
            invocation_usage=usage,
            metadata=metadata,
        )

    return _provider
