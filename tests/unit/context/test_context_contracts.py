# © Artur Czarnecki. All rights reserved.

"""CE-1.1–CE-1.2: Context Engineering Tier-0 contracts."""

from __future__ import annotations

import pytest

from intergrax.context.contracts import (
    AssembledContext,
    BudgetAllocationResult,
    ContextAssemblyProvenance,
    ContextAssemblyRequest,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
    ContextFragment,
    ContextFragmentSource,
    ContextProviderContext,
    content_hash_for_text,
)
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.llm.messages import ChatMessage

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_context_fragment_source_enum_values() -> None:
    assert ContextFragmentSource.RAG.value == "rag"
    assert ContextFragmentSource.SESSION_HISTORY_SEMANTIC.value == "session_history_semantic"


def test_context_fragment_auto_content_hash() -> None:
    fragment = ContextFragment(
        fragment_id="f1",
        source=ContextFragmentSource.RAG,
        source_id="chunk-1",
        content="hello",
        token_estimate=2,
        relevance_score=0.9,
        freshness_score=0.8,
        confidence_score=0.7,
        mandatory=False,
    )
    assert fragment.content_hash == content_hash_for_text("hello")


def test_context_fragment_rejects_invalid_score() -> None:
    with pytest.raises(ValueError, match="relevance_score"):
        ContextFragment(
            fragment_id="f1",
            source=ContextFragmentSource.RAG,
            source_id="chunk-1",
            content="hello",
            token_estimate=2,
            relevance_score=1.5,
            freshness_score=0.8,
            confidence_score=0.7,
            mandatory=False,
        )


def test_context_assembly_request_repr_omits_objective() -> None:
    request = ContextAssemblyRequest(
        trace_id="trace-1",
        run_id="run-1",
        task_id="task-1",
        tenant_id="tenant-1",
        assembly_scope="acp_step",
        objective="secret user objective text",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(),
        assembly_options=TaskContextAssemblyOptions(),
        step_index=2,
        step_kind="tool_call",
    )
    rendered = repr(request)
    assert "secret user objective" not in rendered
    assert "acp_step" in rendered
    assert "tool_call" in rendered


def test_context_provider_context_repr_hides_handle_values() -> None:
    ctx = ContextProviderContext(
        engine_id="default",
        plugin_ids=("builtin.rag",),
        handles={"api_key": "super-secret"},
    )
    rendered = repr(ctx)
    assert "super-secret" not in rendered
    assert "api_key" in rendered


def test_assembled_context_frozen_shapes() -> None:
    fragment = ContextFragment(
        fragment_id="f1",
        source=ContextFragmentSource.TASK_MESSAGE,
        source_id="task",
        content="do work",
        token_estimate=3,
        relevance_score=1.0,
        freshness_score=1.0,
        confidence_score=1.0,
        mandatory=True,
    )
    assembled = AssembledContext(
        messages=(ChatMessage(role="user", content="do work"),),
        fragments_included=(fragment,),
        fragments_excluded=(),
        provenance=(
            ContextAssemblyProvenance(
                source_type=fragment.source.value,
                source_id=fragment.source_id,
                fragment_id=fragment.fragment_id,
            ),
        ),
        total_tokens=3,
        budget_tokens=4_000,
    )
    assert assembled.schema_version == "assembled_context.v1"
    assert isinstance(
        BudgetAllocationResult(
            included=(fragment,),
            excluded=(),
            total_tokens=3,
            budget_tokens=4_000,
        ),
        BudgetAllocationResult,
    )


def test_context_assembly_request_execution_scope_default_and_validation() -> None:
    from intergrax.runtime.context_lifecycle.contracts import ModelCallExecutionScope

    request = ContextAssemblyRequest(
        trace_id="trace-1",
        run_id="run-1",
        task_id="task-1",
        tenant_id="tenant-1",
        assembly_scope="acp_step",
        objective="objective",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(),
        assembly_options=TaskContextAssemblyOptions(),
    )
    assert request.execution_scope is ModelCallExecutionScope.PRIMARY_MODEL_CALL

    with pytest.raises(ValueError, match="execution_scope"):
        ContextAssemblyRequest(
            trace_id="trace-1",
            run_id="run-1",
            task_id="task-1",
            tenant_id="tenant-1",
            assembly_scope="acp_step",
            objective="objective",
            decision_profile=ContextDecisionSnapshot(),
            budget_policy=ContextBudgetSnapshot(),
            assembly_options=TaskContextAssemblyOptions(),
            execution_scope="primary_model_call",  # type: ignore[arg-type]
        )
