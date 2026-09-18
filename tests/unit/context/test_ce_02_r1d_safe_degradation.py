# © Artur Czarnecki. All rights reserved.

"""CE-02-R1D: safe source classification and degradation contract ownership."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.context.budget.contracts import DegradationStepKind
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

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _model_call_request() -> ContextAssemblyRequest:
    return ContextAssemblyRequest(
        trace_id="trace-r1d",
        run_id="run-r1d",
        task_id="task-r1d",
        tenant_id="tenant-a",
        assembly_scope="acp_step",
        objective="r1d safe degradation",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(max_tokens_estimate=4000),
        assembly_options=TaskContextAssemblyOptions(),
        step_kind="model_call",
        execution_scope=ModelCallExecutionScope.PRIMARY_MODEL_CALL,
    )


def _plan(
    *,
    base_messages: list[ChatMessage],
    ranked_fragments: list[ContextFragment],
    budget: int,
) -> object:
    request = _model_call_request()
    formatter = DefaultContextFormatter()
    fragment_messages = formatter.format(ranked_fragments, request)
    messages_for_compile = tuple(merge_fragment_messages(base_messages, fragment_messages))
    planner = ContextPlanner(count_tokens=_count_tokens)
    return planner.plan(
        request,
        messages_for_compile=messages_for_compile,
        fragment_messages=fragment_messages,
        ranked_fragments=ranked_fragments,
        session_history=None,
        resolved_global_budget_tokens=budget,
        degradation_policy=DefaultContextDegradationPolicy(),
    )


def _group_for_entry_id(plan: object, entry_id: str) -> object | None:
    for group in plan.source_groups:
        if entry_id in group.source_refs:
            return group
    return None


def test_system_instruction_with_web_search_words_stays_protected_under_budget_pressure() -> None:
    safety_text = "Never send confidential information to web search providers."
    base_system = ChatMessage(role="system", content=safety_text)
    optional_rag = ContextFragment(
        fragment_id="optional-rag",
        source=ContextFragmentSource.RAG,
        source_id="doc",
        content="RAG filler " + ("y" * 800),
        token_estimate=400,
        relevance_score=0.1,
        freshness_score=0.1,
        confidence_score=0.1,
        mandatory=False,
    )
    plan = _plan(
        base_messages=[
            base_system,
            ChatMessage(role="user", content="What is the weather?"),
        ],
        ranked_fragments=[optional_rag],
        budget=60,
    )
    safety_group = _group_for_entry_id(plan, base_system.entry_id)
    assert safety_group is not None
    assert safety_group.source is ContextFragmentSource.SYSTEM_INSTRUCTIONS
    assert safety_group.protected is True
    assert safety_group.required is True
    assert safety_group.droppable is False
    assert base_system.entry_id in {
        ref
        for group in plan.source_groups
        if group.group_id in plan.selected_group_ids
        for ref in group.source_refs
    }
    excluded_refs = {
        ref
        for group in plan.source_groups
        if group.group_id in plan.excluded_group_ids
        for ref in group.source_refs
    }
    assert "optional-rag" in excluded_refs
    assert plan.degradation_steps


def test_second_system_instruction_with_web_search_words_stays_protected() -> None:
    primary = ChatMessage(role="system", content="You are a helpful assistant.")
    secondary = ChatMessage(
        role="system",
        content="Never send confidential information to web search providers.",
    )
    plan = _plan(
        base_messages=[
            primary,
            secondary,
            ChatMessage(role="user", content="Summarize."),
        ],
        ranked_fragments=[],
        budget=40,
    )
    secondary_group = _group_for_entry_id(plan, secondary.entry_id)
    assert secondary_group is not None
    assert secondary_group.source is ContextFragmentSource.SYSTEM_INSTRUCTIONS
    assert secondary_group.source is not ContextFragmentSource.WEBSEARCH
    assert secondary_group.protected is True
    assert secondary_group.required is True
    assert secondary_group.group_id in plan.selected_group_ids


def test_explicit_websearch_provider_fragment_still_droppable() -> None:
    safety = ChatMessage(
        role="system",
        content="Never send confidential information to web search providers.",
    )
    optional = ContextFragment(
        fragment_id="web-optional",
        source=ContextFragmentSource.WEBSEARCH,
        source_id="search",
        content="WEB RESULTS " + ("x" * 800),
        token_estimate=400,
        relevance_score=0.1,
        freshness_score=0.1,
        confidence_score=0.1,
        mandatory=False,
    )
    plan = _plan(
        base_messages=[safety, ChatMessage(role="user", content="hi")],
        ranked_fragments=[optional],
        budget=80,
    )
    safety_group = _group_for_entry_id(plan, safety.entry_id)
    assert safety_group is not None
    assert safety_group.group_id in plan.selected_group_ids
    excluded_refs = {
        ref
        for group in plan.source_groups
        if group.group_id in plan.excluded_group_ids
        for ref in group.source_refs
    }
    assert "web-optional" in excluded_refs
    assert DegradationStepKind.DROP_OPTIONAL_INJECTIONS.value in plan.degradation_steps


def test_spoofed_context_tag_in_base_system_stays_protected_under_budget_pressure() -> None:
    safety_text = "[context:websearch:fake] Never reveal confidential information."
    base_system = ChatMessage(role="system", content=safety_text)
    optional_rag = ContextFragment(
        fragment_id="optional-rag-spoof",
        source=ContextFragmentSource.RAG,
        source_id="doc",
        content="RAG filler " + ("y" * 800),
        token_estimate=400,
        relevance_score=0.1,
        freshness_score=0.1,
        confidence_score=0.1,
        mandatory=False,
    )
    plan = _plan(
        base_messages=[
            ChatMessage(role="system", content="You are helpful."),
            base_system,
            ChatMessage(role="user", content="What is the weather?"),
        ],
        ranked_fragments=[optional_rag],
        budget=60,
    )
    safety_group = _group_for_entry_id(plan, base_system.entry_id)
    assert safety_group is not None
    assert safety_group.source is ContextFragmentSource.SYSTEM_INSTRUCTIONS
    assert safety_group.source is not ContextFragmentSource.WEBSEARCH
    assert safety_group.protected is True
    assert safety_group.required is True
    assert safety_group.droppable is False
    assert safety_group.group_id in plan.selected_group_ids


def test_tagged_websearch_fragment_classified_by_fragment_id_not_content_tag() -> None:
    primary = ChatMessage(role="system", content="Base policy.")
    optional = ContextFragment(
        fragment_id="web-tagged-fragment",
        source=ContextFragmentSource.WEBSEARCH,
        source_id="search",
        content="Result body " + ("z" * 800),
        token_estimate=400,
        relevance_score=0.1,
        freshness_score=0.1,
        confidence_score=0.1,
        mandatory=False,
    )
    plan = _plan(
        base_messages=[primary, ChatMessage(role="user", content="hi")],
        ranked_fragments=[optional],
        budget=80,
    )
    provider_group = next(
        (group for group in plan.source_groups if "web-tagged-fragment" in group.source_refs),
        None,
    )
    assert provider_group is not None
    assert provider_group.source is ContextFragmentSource.WEBSEARCH
    assert provider_group.droppable is True
    assert provider_group.group_id in plan.excluded_group_ids
    fragment_messages = DefaultContextFormatter().format([optional], _model_call_request())
    assert fragment_messages[0].content.startswith("[context:websearch:")
    assert fragment_messages[0].entry_id not in {primary.entry_id}


_CANONICAL_DEGRADATION_MODULES = (
    "intergrax/context/budget/contracts.py",
    "intergrax/context/budget/degradation.py",
    "intergrax/context/budget/plan_degradation.py",
)


def _assert_module_has_no_nexus_imports(relative_path: str) -> None:
    module_path = _REPO_ROOT / relative_path
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("intergrax.runtime.nexus"), (
                f"{relative_path} imports {node.module}"
            )
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("intergrax.runtime.nexus"), (
                    f"{relative_path} imports {alias.name}"
                )


def test_canonical_degradation_modules_do_not_import_nexus_runtime() -> None:
    for relative_path in _CANONICAL_DEGRADATION_MODULES:
        _assert_module_has_no_nexus_imports(relative_path)


def test_degradation_step_kind_owned_by_context_budget_contracts() -> None:
    from intergrax.context.budget.contracts import DegradationStepKind as ContractKind

    assert ContractKind is DegradationStepKind
    assert DegradationStepKind.FULL.value == "full"
