# © Artur Czarnecki. All rights reserved.

"""CE-4.4, CE-4.6: step-aware DefaultContextRanker."""

from __future__ import annotations

import pytest

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
    ContextFragment,
    ContextFragmentSource,
)
from intergrax.context.ranker import DefaultContextRanker
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _fragment(source: ContextFragmentSource, score: float) -> ContextFragment:
    return ContextFragment(
        fragment_id=f"{source.value}-1",
        source=source,
        source_id="s1",
        content="body",
        token_estimate=10,
        relevance_score=score,
        freshness_score=0.5,
        confidence_score=0.5,
        mandatory=False,
    )


def test_ranker_boosts_tool_output_for_tool_call_step() -> None:
    ranker = DefaultContextRanker()
    request = ContextAssemblyRequest(
        trace_id="t",
        run_id="r",
        task_id="task",
        tenant_id="tenant",
        assembly_scope="acp_step",
        objective="obj",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(),
        assembly_options=TaskContextAssemblyOptions(),
        step_kind="tool_call",
    )
    fragments = [
        _fragment(ContextFragmentSource.SESSION_HISTORY, 0.75),
        _fragment(ContextFragmentSource.TOOL_OUTPUT, 0.7),
    ]
    ranked = ranker.rank(fragments, request)
    assert ranked[0].source == ContextFragmentSource.TOOL_OUTPUT


def test_policy_gate_requires_declared_source() -> None:
    from intergrax.runtime.policy.context_assembly_policy import run_pre_context_policy_gate

    request = ContextAssemblyRequest(
        trace_id="t",
        run_id="r",
        task_id="task",
        tenant_id="tenant",
        assembly_scope="acp_step",
        objective="obj",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(),
        assembly_options=TaskContextAssemblyOptions(),
        required_sources=frozenset({ContextFragmentSource.RAG}),
    )
    result = run_pre_context_policy_gate(request, collected=())
    assert not result.allowed
