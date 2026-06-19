# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.agent_budget import ResolvedBudgetLimits
from intergrax.llm_adapters.routing.context_bridge import (
    budget_remaining_ratio_from_limits,
    build_routing_context_from_runtime,
)


@pytest.mark.unit
@pytest.mark.gate
def test_build_routing_context_from_metadata_and_budget() -> None:
    limits = ResolvedBudgetLimits(
        agent_tokens_limit=1000,
        agent_tokens_remaining=250,
    )
    context = build_routing_context_from_runtime(
        tenant_id="tenant-a",
        agent_id="agent-b",
        metadata={"task_class": "lab_routing", "step_index": 2},
        budget_limits=limits,
        budget_degrade_active=True,
    )
    assert context.tenant_id == "tenant-a"
    assert context.agent_id == "agent-b"
    assert context.task_class == "lab_routing"
    assert context.step_index == 2
    assert context.budget_remaining_ratio == pytest.approx(0.25)
    assert context.budget_degrade_active is True


@pytest.mark.unit
@pytest.mark.gate
def test_budget_remaining_ratio_from_limits_prefers_agent_slice() -> None:
    limits = ResolvedBudgetLimits(
        agent_tokens_limit=500,
        agent_tokens_remaining=100,
        environment_tokens_limit=1000,
        environment_tokens_remaining=900,
    )
    assert budget_remaining_ratio_from_limits(limits) == pytest.approx(0.2)


@pytest.mark.unit
@pytest.mark.gate
def test_refresh_llm_routing_context_updates_snapshot_fields() -> None:
    from intergrax.contracts.agent_budget import ResolvedBudgetLimits
    from intergrax.llm_adapters.routing.context_bridge import (
        LLMRoutingRuntimeSnapshot,
        refresh_llm_routing_context,
    )

    snapshot = LLMRoutingRuntimeSnapshot(task_class="initial", step_index=0)
    limits = ResolvedBudgetLimits(agent_tokens_limit=100, agent_tokens_remaining=15)
    refreshed, context = refresh_llm_routing_context(
        snapshot,
        step_index=2,
        task_class="lab_routing",
        budget_limits=limits,
    )
    assert refreshed.step_index == 2
    assert context.step_index == 2
    assert context.task_class == "lab_routing"
    assert context.budget_remaining_ratio == pytest.approx(0.15)