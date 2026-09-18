# © Artur Czarnecki. All rights reserved.

"""CE-02-R1C plan-native degradation consistency."""

from __future__ import annotations

import pytest

from intergrax.context.budget.degradation import DefaultContextDegradationPolicy
from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
    ContextFragment,
    ContextFragmentSource,
)
from intergrax.context.formatter import DefaultContextFormatter, merge_fragment_messages
from intergrax.context.planner import ContextPlanner
from intergrax.context.planning import ContextPlanningError
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.context_lifecycle.contracts import ModelCallExecutionScope
from intergrax.runtime.nexus.context.context_compiler_models import DegradationStepKind


def _count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _model_call_request() -> ContextAssemblyRequest:
    return ContextAssemblyRequest(
        trace_id="trace-plan-degrade",
        run_id="run-plan-degrade",
        task_id="task-plan-degrade",
        tenant_id="tenant-a",
        assembly_scope="acp_step",
        objective="plan-native degradation",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(max_tokens_estimate=4000),
        assembly_options=TaskContextAssemblyOptions(),
        step_kind="model_call",
        execution_scope=ModelCallExecutionScope.PRIMARY_MODEL_CALL,
    )


def _plan_input(
    request: ContextAssemblyRequest,
    *,
    base_messages: list[ChatMessage],
    ranked_fragments: list[ContextFragment],
) -> tuple[tuple[ChatMessage, ...], list[ChatMessage]]:
    formatter = DefaultContextFormatter()
    fragment_messages = formatter.format(ranked_fragments, request)
    messages_for_compile = merge_fragment_messages(base_messages, fragment_messages)
    return tuple(messages_for_compile), fragment_messages


def test_optional_provider_fragment_degraded_without_incomplete_plan() -> None:
    """Regression for CE-02-R1B: fragments stay aligned with messages_for_compile."""
    mandatory = ContextFragment(
        fragment_id="mandatory-frag",
        source=ContextFragmentSource.SYSTEM_INSTRUCTIONS,
        source_id="policy",
        content="MANDATORY-PLAN-NATIVE",
        token_estimate=20,
        relevance_score=1.0,
        freshness_score=1.0,
        confidence_score=1.0,
        mandatory=True,
    )
    optional = ContextFragment(
        fragment_id="optional-frag",
        source=ContextFragmentSource.RAG,
        source_id="doc",
        content="OPTIONAL-PLAN-NATIVE " + ("x" * 800),
        token_estimate=400,
        relevance_score=0.1,
        freshness_score=0.1,
        confidence_score=0.1,
        mandatory=False,
    )
    request = _model_call_request()
    messages_for_compile, fragment_messages = _plan_input(
        request,
        base_messages=[
            ChatMessage(role="system", content="base-system"),
            ChatMessage(role="user", content="user-turn"),
        ],
        ranked_fragments=[mandatory, optional],
    )
    planner = ContextPlanner(count_tokens=_count_tokens)
    plan = planner.plan(
        request,
        messages_for_compile=messages_for_compile,
        fragment_messages=fragment_messages,
        ranked_fragments=[mandatory, optional],
        session_history=None,
        resolved_global_budget_tokens=120,
        degradation_policy=DefaultContextDegradationPolicy(),
        prefer_longterm_memory=True,
        prefer_rag_when_enabled=True,
    )
    excluded_refs = {
        ref
        for group in plan.source_groups
        if group.group_id in plan.excluded_group_ids
        for ref in group.source_refs
    }
    assert "optional-frag" in excluded_refs
    assert "mandatory-frag" not in excluded_refs
    assert plan.degradation_steps


def test_incomplete_plan_when_message_removed_without_fragment() -> None:
    mandatory = ContextFragment(
        fragment_id="only-frag",
        source=ContextFragmentSource.RAG,
        source_id="doc",
        content="still-here",
        token_estimate=10,
        relevance_score=1.0,
        freshness_score=1.0,
        confidence_score=1.0,
        mandatory=False,
    )
    request = _model_call_request()
    messages_for_compile, fragment_messages = _plan_input(
        request,
        base_messages=[
            ChatMessage(role="system", content="base"),
            ChatMessage(role="user", content="ask"),
        ],
        ranked_fragments=[mandatory],
    )
    planner = ContextPlanner(count_tokens=_count_tokens)
    with pytest.raises(ContextPlanningError, match="incomplete_model_input_plan"):
        planner.plan(
            request,
            messages_for_compile=(
                ChatMessage(role="system", content="base"),
                ChatMessage(role="user", content="ask"),
            ),
            fragment_messages=fragment_messages,
            ranked_fragments=[mandatory],
            session_history=None,
            resolved_global_budget_tokens=500,
        )


def test_traceability_excluded_group_matches_fragment_identity() -> None:
    optional = ContextFragment(
        fragment_id="trace-optional",
        source=ContextFragmentSource.WEBSEARCH,
        source_id="web",
        content="WEBSEARCH:\n" + ("z" * 600),
        token_estimate=300,
        relevance_score=0.2,
        freshness_score=0.2,
        confidence_score=0.2,
        mandatory=False,
    )
    mandatory = ContextFragment(
        fragment_id="trace-mandatory",
        source=ContextFragmentSource.SYSTEM_INSTRUCTIONS,
        source_id="sys",
        content="TRACE-MANDATORY",
        token_estimate=30,
        relevance_score=1.0,
        freshness_score=1.0,
        confidence_score=1.0,
        mandatory=True,
    )
    request = _model_call_request()
    messages_for_compile, fragment_messages = _plan_input(
        request,
        base_messages=[
            ChatMessage(role="system", content="Instructions"),
            ChatMessage(role="user", content="hi"),
        ],
        ranked_fragments=[mandatory, optional],
    )

    class _DropOptionalOnly(DefaultContextDegradationPolicy):
        def ladder_order(self) -> tuple[DegradationStepKind, ...]:
            return (DegradationStepKind.FULL, DegradationStepKind.DROP_OPTIONAL_INJECTIONS)

    planner = ContextPlanner(count_tokens=_count_tokens)
    plan = planner.plan(
        request,
        messages_for_compile=messages_for_compile,
        fragment_messages=fragment_messages,
        ranked_fragments=[mandatory, optional],
        session_history=None,
        resolved_global_budget_tokens=80,
        degradation_policy=_DropOptionalOnly(),
    )
    excluded_refs = {
        ref
        for group in plan.source_groups
        if group.group_id in plan.excluded_group_ids
        for ref in group.source_refs
    }
    included_refs = {
        ref
        for group in plan.source_groups
        if group.group_id in plan.selected_group_ids
        for ref in group.source_refs
    }
    assert "trace-optional" in excluded_refs
    assert "trace-optional" not in included_refs
    assert "trace-mandatory" in included_refs
    assert DegradationStepKind.DROP_OPTIONAL_INJECTIONS.value in plan.degradation_steps
