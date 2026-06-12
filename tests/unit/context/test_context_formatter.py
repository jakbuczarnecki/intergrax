# © Artur Czarnecki. All rights reserved.

"""CE-FMT-1: fragment formatter merge."""

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
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.llm.messages import ChatMessage

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_merge_fragment_messages_before_last_user() -> None:
    base = [
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="user", content="question"),
    ]
    injected = [ChatMessage(role="system", content="[context:workspace:f1] code")]
    merged = merge_fragment_messages(base, injected)
    assert len(merged) == 3
    assert merged[1].content.startswith("[context:workspace")
    assert merged[2].content == "question"


def test_formatter_emits_source_tagged_blocks() -> None:
    formatter = DefaultContextFormatter()
    request = ContextAssemblyRequest(
        trace_id="t",
        run_id="r",
        task_id="task",
        tenant_id="tenant",
        assembly_scope="graph_node",
        objective="obj",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(),
        assembly_options=TaskContextAssemblyOptions(),
    )
    fragment = ContextFragment(
        fragment_id="w1",
        source=ContextFragmentSource.WORKSPACE,
        source_id="app.py",
        content="print(1)",
        token_estimate=2,
        relevance_score=0.9,
        freshness_score=0.9,
        confidence_score=0.9,
        mandatory=False,
    )
    messages = formatter.format([fragment], request)
    assert messages[0].role == "system"
    assert "workspace" in messages[0].content
    assert "print(1)" in messages[0].content
