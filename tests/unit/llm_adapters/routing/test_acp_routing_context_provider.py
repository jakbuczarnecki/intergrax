# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.applications._shared.llm_routing_context_bridge import make_acp_routing_context_provider
from intergrax.contracts.acp_state import AcpInvocationUsageView, AcpTokenUsage
from intergrax.contracts.agent_budget import ResolvedBudgetLimits


@dataclass
class _FakeKernel:
    resolved_budget_limits: ResolvedBudgetLimits
    budget_degrade_active: bool = False


@dataclass
class _FakeStepCtx:
    step_index: int
    invocation_usage: AcpInvocationUsageView


@pytest.mark.integration
@pytest.mark.gate
def test_acp_routing_context_provider_maps_invocation_usage_tokens() -> None:
    limits = ResolvedBudgetLimits(agent_tokens_limit=1000, agent_tokens_remaining=200)
    kernel_holder = [_FakeKernel(resolved_budget_limits=limits)]
    step_holder = [
        _FakeStepCtx(
            step_index=3,
            invocation_usage=AcpInvocationUsageView(
                agent=AcpTokenUsage(tokens_total=801),
                environment=AcpTokenUsage(tokens_total=999),
            ),
        )
    ]
    provider = make_acp_routing_context_provider(
        kernel_ctx_holder=kernel_holder,
        step_ctx_holder=step_holder,
        tenant_id="tenant-a",
        agent_id="agent-b",
        task_class="lab",
    )
    context = provider()
    assert context.step_index == 3
    assert context.tokens_used == 801
    assert context.budget_remaining_ratio == pytest.approx(0.2)
