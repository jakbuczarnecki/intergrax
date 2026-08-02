# © Artur Czarnecki. All rights reserved.

"""CTX-UCL-3 context planner tests."""

from __future__ import annotations

import pytest

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
    ContextFragment,
    ContextFragmentSource,
)
from intergrax.context.planning import (
    MANDATORY_CONTEXT_EXCEEDS_MODEL_LIMIT,
    NO_ELIGIBLE_CONTEXT_OPTIMIZATION_TARGET,
    ContextPlanningError,
)
from intergrax.context.planner import ContextPlanner, group_session_history_snapshot
from intergrax.context.session_history import build_session_history_snapshot
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.context_lifecycle.contracts import (
    ContextOptimizationPolicy,
    ContextOptimizationMode,
    ModelCallExecutionScope,
    OptimizationArtifactType,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _request(
    *,
    scope: ModelCallExecutionScope = ModelCallExecutionScope.PRIMARY_MODEL_CALL,
) -> ContextAssemblyRequest:
    return ContextAssemblyRequest(
        trace_id="trace",
        run_id="run",
        task_id="task",
        tenant_id="tenant",
        assembly_scope="graph_node",
        objective="objective text",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(),
        assembly_options=TaskContextAssemblyOptions(),
        execution_scope=scope,
    )


def _snapshot(messages: list[ChatMessage]) -> object:
    return build_session_history_snapshot(
        tenant_id="tenant",
        context_scope_id="scope",
        revision_id="rev",
        messages=messages,
    )


def test_grouping_system_message_protected() -> None:
    snapshot = _snapshot([ChatMessage(role="system", content="sys", entry_id="s1")])
    groups = group_session_history_snapshot(snapshot, count_tokens=_count_tokens)
    assert len(groups) == 1
    assert groups[0].protected


def test_grouping_user_assistant_turn() -> None:
    snapshot = _snapshot(
        [
            ChatMessage(role="user", content="q", entry_id="u1"),
            ChatMessage(role="assistant", content="a", entry_id="a1"),
        ]
    )
    groups = group_session_history_snapshot(snapshot, count_tokens=_count_tokens)
    assert len(groups) == 1
    assert groups[0].source_refs == ("u1", "a1")


def test_grouping_tool_call_atomic() -> None:
    snapshot = _snapshot(
        [
            ChatMessage(
                role="assistant",
                content="",
                entry_id="a1",
                tool_calls=[{"id": "call-1", "type": "function", "function": {"name": "x"}}],
            ),
            ChatMessage(role="tool", content="result", entry_id="t1", tool_call_id="call-1"),
        ]
    )
    groups = group_session_history_snapshot(snapshot, count_tokens=_count_tokens)
    assert len(groups) == 1
    assert groups[0].protected
    assert groups[0].source_refs == ("a1", "t1")


def test_plan_fits_without_optimization() -> None:
    planner = ContextPlanner(count_tokens=_count_tokens)
    snapshot = _snapshot([ChatMessage(role="user", content="short", entry_id="u1")])
    plan = planner.plan(
        _request(),
        messages_for_compile=[],
        ranked_fragments=[],
        session_history=snapshot,
        resolved_global_budget_tokens=500,
    )
    assert plan.optimization_required is False
    assert plan.artifact_requirement is None


def test_plan_selection_only_drops_droppable() -> None:
    planner = ContextPlanner(count_tokens=_count_tokens)
    droppable = ContextFragment(
        fragment_id="rag-1",
        source=ContextFragmentSource.RAG,
        source_id="rag-1",
        content="x" * 400,
        token_estimate=100,
        relevance_score=0.8,
        freshness_score=0.8,
        confidence_score=0.8,
        mandatory=False,
    )
    snapshot = _snapshot([ChatMessage(role="user", content="short", entry_id="u1")])
    plan = planner.plan(
        _request(),
        messages_for_compile=[],
        ranked_fragments=[droppable],
        session_history=snapshot,
        resolved_global_budget_tokens=20,
    )
    assert plan.optimization_required is False
    assert plan.excluded_group_ids


def test_plan_requires_optimization_for_history() -> None:
    planner = ContextPlanner(count_tokens=_count_tokens)
    messages = [
        ChatMessage(role="user", content="old " * 200, entry_id=f"m{index}")
        for index in range(4)
    ] + [ChatMessage(role="user", content="recent", entry_id="m-recent")]
    snapshot = _snapshot(messages)
    policy = ContextOptimizationPolicy(
        policy_version="pol-1",
        validation_contract_version="val-1",
        enabled=True,
        allow_lossy=True,
        allowed_strategy_ids=("message_sequence.summary.v1",),
        allowed_artifact_types=(OptimizationArtifactType.MESSAGE_SEQUENCE,),
        recent_tail_min_messages=1,
    )
    plan = planner.plan(
        _request(),
        messages_for_compile=[],
        ranked_fragments=[],
        session_history=snapshot,
        resolved_global_budget_tokens=40,
        optimization_policy=policy,
    )
    assert plan.optimization_required is True
    assert plan.artifact_requirement is not None
    assert plan.artifact_requirement.lookup_inputs.artifact_type is OptimizationArtifactType.MESSAGE_SEQUENCE


def test_plan_mandatory_overflow_fails_closed() -> None:
    planner = ContextPlanner(count_tokens=_count_tokens)
    snapshot = _snapshot(
        [
            ChatMessage(
                role="assistant",
                content="x" * 400,
                entry_id="a1",
                tool_calls=[{"id": "call-1", "type": "function", "function": {"name": "x"}}],
            ),
        ]
    )
    with pytest.raises(ContextPlanningError, match=MANDATORY_CONTEXT_EXCEEDS_MODEL_LIMIT):
        planner.plan(
            _request(),
            messages_for_compile=[],
            ranked_fragments=[],
            session_history=snapshot,
            resolved_global_budget_tokens=10,
        )


def test_plan_no_eligible_target_fails_closed() -> None:
    planner = ContextPlanner(count_tokens=_count_tokens)
    tool_fragment = ContextFragment(
        fragment_id="tool-1",
        source=ContextFragmentSource.TOOL_OUTPUT,
        source_id="tool-1",
        content="x" * 800,
        token_estimate=200,
        relevance_score=0.9,
        freshness_score=0.9,
        confidence_score=0.9,
        mandatory=False,
    )
    with pytest.raises(ContextPlanningError, match=NO_ELIGIBLE_CONTEXT_OPTIMIZATION_TARGET):
        planner.plan(
            _request(),
            messages_for_compile=[],
            ranked_fragments=[tool_fragment],
            session_history=None,
            resolved_global_budget_tokens=100,
        )


def test_internal_call_budget_classification() -> None:
    planner = ContextPlanner(count_tokens=_count_tokens)
    snapshot = _snapshot([ChatMessage(role="user", content="hi", entry_id="u1")])
    plan = planner.plan(
        _request(scope=ModelCallExecutionScope.INTERNAL_OPTIMIZATION_CALL),
        messages_for_compile=[],
        ranked_fragments=[],
        session_history=snapshot,
        resolved_global_budget_tokens=200,
    )
    assert plan.budget_class.value == "internal_optimization_input"
