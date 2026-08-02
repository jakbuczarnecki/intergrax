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
from intergrax.context.formatter import DefaultContextFormatter, merge_fragment_messages
from intergrax.context.planning import (
    MANDATORY_CONTEXT_EXCEEDS_MODEL_LIMIT,
    NO_ALLOWED_CONTEXT_OPTIMIZATION_STRATEGY,
    NO_ELIGIBLE_CONTEXT_OPTIMIZATION_TARGET,
    ContextPlanningError,
)
from intergrax.context.planner import ContextPlanner, group_session_history_snapshot
from intergrax.context.session_history import build_session_history_snapshot
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.context_lifecycle.contracts import (
    ContextOptimizationPolicy,
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


def _model_input(
    request: ContextAssemblyRequest,
    *,
    base_messages: list[ChatMessage],
    ranked_fragments: list[ContextFragment] | None = None,
) -> tuple[list[ChatMessage], list[ChatMessage]]:
    ranked = ranked_fragments or []
    formatter = DefaultContextFormatter()
    fragment_messages = formatter.format(ranked, request)
    messages_for_compile = merge_fragment_messages(base_messages, fragment_messages)
    return messages_for_compile, fragment_messages


def _plan(
    planner: ContextPlanner,
    request: ContextAssemblyRequest,
    *,
    base_messages: list[ChatMessage],
    ranked_fragments: list[ContextFragment] | None = None,
    session_history: object | None = None,
    resolved_global_budget_tokens: int,
    optimization_policy: ContextOptimizationPolicy | None = None,
):
    messages_for_compile, fragment_messages = _model_input(
        request,
        base_messages=base_messages,
        ranked_fragments=ranked_fragments,
    )
    return planner.plan(
        request,
        messages_for_compile=messages_for_compile,
        fragment_messages=fragment_messages,
        ranked_fragments=ranked_fragments or [],
        session_history=session_history,
        resolved_global_budget_tokens=resolved_global_budget_tokens,
        optimization_policy=optimization_policy,
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
    assert not groups[0].protected
    assert not groups[0].required
    assert groups[0].source_refs == ("a1", "t1")


def test_plan_fits_without_optimization() -> None:
    planner = ContextPlanner(count_tokens=_count_tokens)
    snapshot = _snapshot([ChatMessage(role="user", content="short", entry_id="u1")])
    plan = _plan(
        planner,
        _request(),
        base_messages=[ChatMessage(role="user", content="current task", entry_id="current")],
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
    plan = _plan(
        planner,
        _request(),
        base_messages=[ChatMessage(role="user", content="current", entry_id="current")],
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
    from intergrax.context.session_history import fragments_from_session_history_snapshot

    ranked = fragments_from_session_history_snapshot(snapshot)
    policy = ContextOptimizationPolicy(
        policy_version="pol-1",
        validation_contract_version="val-1",
        enabled=True,
        allow_lossy=True,
        allowed_strategy_ids=("message_sequence.summary.v1",),
        allowed_artifact_types=(OptimizationArtifactType.MESSAGE_SEQUENCE,),
        recent_tail_min_messages=1,
    )
    plan = _plan(
        planner,
        _request(),
        base_messages=[ChatMessage(role="user", content="recent task", entry_id="current")],
        ranked_fragments=ranked,
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
    from intergrax.context.session_history import fragments_from_session_history_snapshot

    ranked = fragments_from_session_history_snapshot(snapshot)
    with pytest.raises(ContextPlanningError, match=MANDATORY_CONTEXT_EXCEEDS_MODEL_LIMIT):
        _plan(
            planner,
            _request(),
            base_messages=[ChatMessage(role="user", content="task", entry_id="current")],
            ranked_fragments=ranked,
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
        _plan(
            planner,
            _request(),
            base_messages=[ChatMessage(role="user", content="task", entry_id="current")],
            ranked_fragments=[tool_fragment],
            session_history=None,
            resolved_global_budget_tokens=100,
        )


def test_internal_call_budget_classification() -> None:
    planner = ContextPlanner(count_tokens=_count_tokens)
    snapshot = _snapshot([ChatMessage(role="user", content="hi", entry_id="u1")])
    plan = _plan(
        planner,
        _request(scope=ModelCallExecutionScope.INTERNAL_OPTIMIZATION_CALL),
        base_messages=[ChatMessage(role="user", content="hi", entry_id="current")],
        ranked_fragments=[],
        session_history=snapshot,
        resolved_global_budget_tokens=200,
    )
    assert plan.budget_class.value == "internal_optimization_input"


def test_estimated_total_tokens_includes_long_base_user_message() -> None:
    planner = ContextPlanner(count_tokens=_count_tokens)
    long_user = "x" * 400
    base = [
        ChatMessage(role="system", content="sys", entry_id="sys"),
        ChatMessage(role="user", content=long_user, entry_id="current"),
    ]
    plan = _plan(
        planner,
        _request(),
        base_messages=base,
        ranked_fragments=[],
        session_history=None,
        resolved_global_budget_tokens=5000,
    )
    assert plan.estimated_total_tokens > _count_tokens("sys")


def test_estimated_total_tokens_equals_sum_of_messages_for_compile() -> None:
    planner = ContextPlanner(count_tokens=_count_tokens)
    base = [
        ChatMessage(role="system", content="system prompt", entry_id="sys"),
        ChatMessage(role="user", content="final user prompt", entry_id="current"),
    ]
    droppable = ContextFragment(
        fragment_id="rag-1",
        source=ContextFragmentSource.RAG,
        source_id="rag-1",
        content="rag body",
        token_estimate=10,
        relevance_score=0.8,
        freshness_score=0.8,
        confidence_score=0.8,
        mandatory=False,
    )
    messages_for_compile, fragment_messages = _model_input(
        _request(),
        base_messages=base,
        ranked_fragments=[droppable],
    )
    expected = sum(_count_tokens(message.content or "") for message in messages_for_compile)
    plan = planner.plan(
        _request(),
        messages_for_compile=messages_for_compile,
        fragment_messages=fragment_messages,
        ranked_fragments=[droppable],
        session_history=None,
        resolved_global_budget_tokens=5000,
    )
    assert plan.estimated_total_tokens == expected


def test_raw_system_and_final_user_are_required_protected() -> None:
    planner = ContextPlanner(count_tokens=_count_tokens)
    plan = _plan(
        planner,
        _request(),
        base_messages=[
            ChatMessage(role="system", content="sys", entry_id="sys"),
            ChatMessage(role="user", content="task", entry_id="current"),
        ],
        ranked_fragments=[],
        session_history=None,
        resolved_global_budget_tokens=500,
    )
    groups = {group.group_id: group for group in plan.source_groups}
    required = {groups[group_id] for group_id in plan.required_group_ids}
    protected = {groups[group_id] for group_id in plan.protected_group_ids}
    assert any(group.source is ContextFragmentSource.SYSTEM_INSTRUCTIONS for group in required)
    assert any(group.source is ContextFragmentSource.TASK_MESSAGE for group in required)
    assert any(group.source is ContextFragmentSource.SYSTEM_INSTRUCTIONS for group in protected)
    assert any(group.source is ContextFragmentSource.TASK_MESSAGE for group in protected)


def test_formatted_fragment_token_cost_is_counted() -> None:
    planner = ContextPlanner(count_tokens=_count_tokens)
    fragment = ContextFragment(
        fragment_id="rag-1",
        source=ContextFragmentSource.RAG,
        source_id="rag-1",
        content="formatted fragment payload",
        token_estimate=1,
        relevance_score=0.8,
        freshness_score=0.8,
        confidence_score=0.8,
        mandatory=False,
    )
    messages_for_compile, fragment_messages = _model_input(
        _request(),
        base_messages=[ChatMessage(role="user", content="task", entry_id="current")],
        ranked_fragments=[fragment],
    )
    fragment_tokens = _count_tokens(fragment_messages[0].content or "")
    plan = planner.plan(
        _request(),
        messages_for_compile=messages_for_compile,
        fragment_messages=fragment_messages,
        ranked_fragments=[fragment],
        session_history=None,
        resolved_global_budget_tokens=5000,
    )
    assert fragment_tokens > 1
    assert plan.estimated_total_tokens >= fragment_tokens


def test_each_model_facing_message_counted_exactly_once() -> None:
    planner = ContextPlanner(count_tokens=_count_tokens)
    snapshot = _snapshot(
        [
            ChatMessage(role="user", content="old", entry_id="u1"),
            ChatMessage(role="assistant", content="reply", entry_id="a1"),
        ]
    )
    from intergrax.context.session_history import fragments_from_session_history_snapshot

    ranked = fragments_from_session_history_snapshot(snapshot)
    messages_for_compile, fragment_messages = _model_input(
        _request(),
        base_messages=[ChatMessage(role="user", content="task", entry_id="current")],
        ranked_fragments=ranked,
    )
    plan = planner.plan(
        _request(),
        messages_for_compile=messages_for_compile,
        fragment_messages=fragment_messages,
        ranked_fragments=ranked,
        session_history=snapshot,
        resolved_global_budget_tokens=5000,
    )
    assert sum(group.token_estimate for group in plan.source_groups) == plan.estimated_total_tokens


def test_fragment_ranked_length_mismatch_fails_closed() -> None:
    planner = ContextPlanner(count_tokens=_count_tokens)
    with pytest.raises(ContextPlanningError, match="fragment_message_mapping_mismatch"):
        planner.plan(
            _request(),
            messages_for_compile=[ChatMessage(role="user", content="task", entry_id="current")],
            fragment_messages=[ChatMessage(role="system", content="frag", entry_id="f1")],
            ranked_fragments=[],
            session_history=None,
            resolved_global_budget_tokens=500,
        )


def test_complete_old_tool_call_group_is_atomic_and_compressible() -> None:
    planner = ContextPlanner(count_tokens=_count_tokens)
    snapshot = _snapshot(
        [
            ChatMessage(role="user", content="old question", entry_id="u-old"),
            ChatMessage(
                role="assistant",
                content="",
                entry_id="a-old",
                tool_calls=[{"id": "call-1", "type": "function", "function": {"name": "x"}}],
            ),
            ChatMessage(role="tool", content="tool output", entry_id="t-old", tool_call_id="call-1"),
        ]
    )
    from intergrax.context.session_history import fragments_from_session_history_snapshot

    ranked = fragments_from_session_history_snapshot(snapshot)
    plan = _plan(
        planner,
        _request(),
        base_messages=[ChatMessage(role="user", content="task", entry_id="current")],
        ranked_fragments=ranked,
        session_history=snapshot,
        resolved_global_budget_tokens=5000,
    )
    tool_groups = [
        group
        for group in plan.source_groups
        if group.source_refs == ("a-old", "t-old")
    ]
    assert len(tool_groups) == 1
    assert tool_groups[0].compressible


def test_incomplete_tool_call_group_is_required_protected() -> None:
    planner = ContextPlanner(count_tokens=_count_tokens)
    snapshot = _snapshot(
        [
            ChatMessage(
                role="assistant",
                content="",
                entry_id="a1",
                tool_calls=[{"id": "call-1", "type": "function", "function": {"name": "x"}}],
            ),
        ]
    )
    from intergrax.context.session_history import fragments_from_session_history_snapshot

    ranked = fragments_from_session_history_snapshot(snapshot)
    plan = _plan(
        planner,
        _request(),
        base_messages=[ChatMessage(role="user", content="task", entry_id="current")],
        ranked_fragments=ranked,
        session_history=snapshot,
        resolved_global_budget_tokens=5000,
    )
    incomplete = [group for group in plan.source_groups if group.source_refs == ("a1",)]
    assert incomplete
    assert incomplete[0].required and incomplete[0].protected


def test_duplicate_tool_call_id_is_fail_closed_protected() -> None:
    snapshot = _snapshot(
        [
            ChatMessage(
                role="assistant",
                content="",
                entry_id="a1",
                tool_calls=[
                    {"id": "dup", "type": "function", "function": {"name": "x"}},
                    {"id": "dup", "type": "function", "function": {"name": "y"}},
                ],
            ),
        ]
    )
    groups = group_session_history_snapshot(snapshot, count_tokens=_count_tokens)
    assert len(groups) == 1
    assert groups[0].required and groups[0].protected


def test_recent_tail_protects_complete_tool_group_as_whole() -> None:
    planner = ContextPlanner(count_tokens=_count_tokens)
    snapshot = _snapshot(
        [
            ChatMessage(role="user", content="older", entry_id="u-old"),
            ChatMessage(
                role="assistant",
                content="",
                entry_id="a-tail",
                tool_calls=[{"id": "call-1", "type": "function", "function": {"name": "x"}}],
            ),
            ChatMessage(role="tool", content="tail result", entry_id="t-tail", tool_call_id="call-1"),
        ]
    )
    from intergrax.context.session_history import fragments_from_session_history_snapshot

    ranked = fragments_from_session_history_snapshot(snapshot)
    policy = ContextOptimizationPolicy(
        policy_version="pol-1",
        validation_contract_version="val-1",
        enabled=True,
        allow_lossy=True,
        allowed_strategy_ids=("message_sequence.summary.v1",),
        allowed_artifact_types=(OptimizationArtifactType.MESSAGE_SEQUENCE,),
        recent_tail_min_messages=2,
    )
    plan = _plan(
        planner,
        _request(),
        base_messages=[ChatMessage(role="user", content="task", entry_id="current")],
        ranked_fragments=ranked,
        session_history=snapshot,
        resolved_global_budget_tokens=5000,
        optimization_policy=policy,
    )
    tail_group = next(group for group in plan.source_groups if group.source_refs == ("a-tail", "t-tail"))
    assert tail_group.group_id in plan.protected_group_ids


def test_raw_base_session_messages_never_compressible() -> None:
    planner = ContextPlanner(count_tokens=_count_tokens)
    plan = _plan(
        planner,
        _request(),
        base_messages=[
            ChatMessage(role="user", content="history user", entry_id="u-old"),
            ChatMessage(role="user", content="task", entry_id="current"),
        ],
        ranked_fragments=[],
        session_history=None,
        resolved_global_budget_tokens=500,
    )
    session_groups = [
        group for group in plan.source_groups if group.source is ContextFragmentSource.SESSION_HISTORY
    ]
    assert session_groups
    assert all(not group.compressible for group in session_groups)


def test_overflow_without_canonical_snapshot_raises_no_eligible_target() -> None:
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
    policy = ContextOptimizationPolicy(
        policy_version="pol-1",
        validation_contract_version="val-1",
        enabled=True,
        allow_lossy=True,
        allowed_strategy_ids=("message_sequence.summary.v1",),
        allowed_artifact_types=(OptimizationArtifactType.MESSAGE_SEQUENCE,),
    )
    with pytest.raises(ContextPlanningError, match=NO_ELIGIBLE_CONTEXT_OPTIMIZATION_TARGET):
        _plan(
            planner,
            _request(),
            base_messages=[ChatMessage(role="user", content="task", entry_id="current")],
            ranked_fragments=[tool_fragment],
            session_history=None,
            resolved_global_budget_tokens=100,
            optimization_policy=policy,
        )


def test_task_id_not_used_as_artifact_context_scope() -> None:
    planner = ContextPlanner(count_tokens=_count_tokens)
    messages = [
        ChatMessage(role="user", content="old " * 200, entry_id=f"m{index}")
        for index in range(4)
    ] + [ChatMessage(role="user", content="recent", entry_id="m-recent")]
    snapshot = _snapshot(messages)
    from intergrax.context.session_history import fragments_from_session_history_snapshot

    ranked = fragments_from_session_history_snapshot(snapshot)
    policy = ContextOptimizationPolicy(
        policy_version="pol-1",
        validation_contract_version="val-1",
        enabled=True,
        allow_lossy=True,
        allowed_strategy_ids=("message_sequence.summary.v1",),
        allowed_artifact_types=(OptimizationArtifactType.MESSAGE_SEQUENCE,),
        recent_tail_min_messages=1,
    )
    plan = _plan(
        planner,
        _request(),
        base_messages=[ChatMessage(role="user", content="recent task", entry_id="current")],
        ranked_fragments=ranked,
        session_history=snapshot,
        resolved_global_budget_tokens=40,
        optimization_policy=policy,
    )
    assert plan.artifact_requirement is not None
    lookup = plan.artifact_requirement.lookup_inputs
    assert lookup.context_scope_id == "scope"
    assert lookup.context_scope_id != "task"
    assert lookup.tenant_id == "tenant"


def test_missing_policy_does_not_invent_strategy() -> None:
    planner = ContextPlanner(count_tokens=_count_tokens)
    messages = [ChatMessage(role="user", content="old " * 200, entry_id=f"m{index}") for index in range(4)]
    snapshot = _snapshot(messages)
    from intergrax.context.session_history import fragments_from_session_history_snapshot

    ranked = fragments_from_session_history_snapshot(snapshot)
    with pytest.raises(ContextPlanningError, match=NO_ALLOWED_CONTEXT_OPTIMIZATION_STRATEGY):
        _plan(
            planner,
            _request(),
            base_messages=[ChatMessage(role="user", content="task", entry_id="current")],
            ranked_fragments=ranked,
            session_history=snapshot,
            resolved_global_budget_tokens=40,
            optimization_policy=None,
        )


def test_empty_allowed_strategy_list_fails_closed() -> None:
    planner = ContextPlanner(count_tokens=_count_tokens)
    messages = [
        ChatMessage(role="user", content="old " * 200, entry_id=f"m{index}")
        for index in range(4)
    ] + [ChatMessage(role="user", content="recent", entry_id="m-recent")]
    snapshot = _snapshot(messages)
    from intergrax.context.session_history import fragments_from_session_history_snapshot

    ranked = fragments_from_session_history_snapshot(snapshot)
    policy = ContextOptimizationPolicy(
        policy_version="pol-1",
        validation_contract_version="val-1",
        enabled=True,
        allow_lossy=True,
        allowed_strategy_ids=(),
        allowed_artifact_types=(OptimizationArtifactType.MESSAGE_SEQUENCE,),
        recent_tail_min_messages=1,
    )
    with pytest.raises(ContextPlanningError, match=NO_ALLOWED_CONTEXT_OPTIMIZATION_STRATEGY):
        _plan(
            planner,
            _request(),
            base_messages=[ChatMessage(role="user", content="recent task", entry_id="current")],
            ranked_fragments=ranked,
            session_history=snapshot,
            resolved_global_budget_tokens=40,
            optimization_policy=policy,
        )


def test_user_group_remains_before_assistant_tool_group() -> None:
    snapshot = _snapshot(
        [
            ChatMessage(role="user", content="question", entry_id="u1"),
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
    assert len(groups) == 2
    assert groups[0].source_refs == ("u1",)
    assert groups[1].source_refs == ("a1", "t1")


def test_tool_call_without_id_is_required_protected() -> None:
    snapshot = _snapshot(
        [
            ChatMessage(
                role="assistant",
                content="",
                entry_id="a1",
                tool_calls=[{"type": "function", "function": {"name": "x"}}],
            ),
        ]
    )
    groups = group_session_history_snapshot(snapshot, count_tokens=_count_tokens)
    assert len(groups) == 1
    assert groups[0].required and groups[0].protected


def test_non_string_tool_call_id_is_required_protected() -> None:
    snapshot = _snapshot(
        [
            ChatMessage(
                role="assistant",
                content="",
                entry_id="a1",
                tool_calls=[{"id": 123, "type": "function", "function": {"name": "x"}}],
            ),
        ]
    )
    groups = group_session_history_snapshot(snapshot, count_tokens=_count_tokens)
    assert len(groups) == 1
    assert groups[0].required and groups[0].protected


def test_foreign_tool_result_makes_group_required_protected() -> None:
    snapshot = _snapshot(
        [
            ChatMessage(
                role="assistant",
                content="",
                entry_id="a1",
                tool_calls=[{"id": "call-1", "type": "function", "function": {"name": "x"}}],
            ),
            ChatMessage(role="tool", content="foreign", entry_id="t1", tool_call_id="other"),
        ]
    )
    groups = group_session_history_snapshot(snapshot, count_tokens=_count_tokens)
    assert len(groups) == 1
    assert groups[0].required and groups[0].protected


def test_duplicate_tool_result_makes_group_required_protected() -> None:
    snapshot = _snapshot(
        [
            ChatMessage(
                role="assistant",
                content="",
                entry_id="a1",
                tool_calls=[{"id": "call-1", "type": "function", "function": {"name": "x"}}],
            ),
            ChatMessage(role="tool", content="first", entry_id="t1", tool_call_id="call-1"),
            ChatMessage(role="tool", content="dup", entry_id="t2", tool_call_id="call-1"),
        ]
    )
    groups = group_session_history_snapshot(snapshot, count_tokens=_count_tokens)
    assert len(groups) == 1
    assert groups[0].required and groups[0].protected


def test_complete_canonical_tool_group_remains_atomically_compressible() -> None:
    planner = ContextPlanner(count_tokens=_count_tokens)
    snapshot = _snapshot(
        [
            ChatMessage(role="user", content="old question", entry_id="u-old"),
            ChatMessage(
                role="assistant",
                content="",
                entry_id="a-old",
                tool_calls=[{"id": "call-1", "type": "function", "function": {"name": "x"}}],
            ),
            ChatMessage(role="tool", content="tool output", entry_id="t-old", tool_call_id="call-1"),
        ]
    )
    from intergrax.context.session_history import fragments_from_session_history_snapshot

    ranked = fragments_from_session_history_snapshot(snapshot)
    plan = _plan(
        planner,
        _request(),
        base_messages=[ChatMessage(role="user", content="task", entry_id="current")],
        ranked_fragments=ranked,
        session_history=snapshot,
        resolved_global_budget_tokens=5000,
    )
    tool_groups = [group for group in plan.source_groups if group.source_refs == ("a-old", "t-old")]
    assert len(tool_groups) == 1
    assert tool_groups[0].compressible
    assert not tool_groups[0].required
    assert not tool_groups[0].protected
