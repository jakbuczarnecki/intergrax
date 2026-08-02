# © Artur Czarnecki. All rights reserved.

"""CTX-UCL-3 engine integration tests."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
    ContextProviderContext,
)
from intergrax.context.session_history import SESSION_HISTORY_SNAPSHOT_HANDLE, build_session_history_snapshot
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.context.context_engine import DefaultNexusContextEngine

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[4]


class _SmallWindowAdapter(LLMAdapter):
    provider = "fake"
    model = "fake-small"

    @property
    def context_window_tokens(self) -> int:
        return 512

    def generate_messages(self, messages, **kwargs) -> LLMAdapterResponse:
        _ = messages, kwargs
        return LLMAdapterResponse(content="ok")


@pytest.mark.asyncio
async def test_engine_attaches_context_plan() -> None:
    adapter = _SmallWindowAdapter()
    config = RuntimeConfig(llm_adapter=adapter, production_mode=False)
    engine = DefaultNexusContextEngine()
    snapshot = build_session_history_snapshot(
        tenant_id="tenant",
        context_scope_id="scope",
        revision_id="rev",
        messages=[ChatMessage(role="user", content="hello", entry_id="m1")],
    )
    request = ContextAssemblyRequest(
        trace_id="t1",
        run_id="r1",
        task_id="task1",
        tenant_id="tenant1",
        assembly_scope="acp_step",
        objective="test",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(max_tokens_estimate=200),
        assembly_options=TaskContextAssemblyOptions(),
    )
    provider_ctx = ContextProviderContext(
        engine_id="default",
        handles={
            "runtime_config": config,
            "messages": [ChatMessage(role="user", content="short prompt", entry_id="current")],
            SESSION_HISTORY_SNAPSHOT_HANDLE: snapshot,
        },
    )
    assembled = await engine.assemble(request, provider_ctx=provider_ctx)
    assert assembled.context_plan is not None
    assert assembled.context_plan.resolved_global_budget_tokens == assembled.budget_tokens


@pytest.mark.asyncio
async def test_engine_plan_total_includes_actual_base_messages() -> None:
    adapter = _SmallWindowAdapter()
    config = RuntimeConfig(llm_adapter=adapter, production_mode=False)
    engine = DefaultNexusContextEngine()
    long_user = "x" * 400
    request = ContextAssemblyRequest(
        trace_id="t1",
        run_id="r1",
        task_id="task1",
        tenant_id="tenant1",
        assembly_scope="acp_step",
        objective="test",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(max_tokens_estimate=200),
        assembly_options=TaskContextAssemblyOptions(),
    )
    provider_ctx = ContextProviderContext(
        engine_id="default",
        handles={
            "runtime_config": config,
            "messages": [
                ChatMessage(role="system", content="system prompt", entry_id="sys"),
                ChatMessage(role="user", content=long_user, entry_id="current"),
            ],
        },
    )
    assembled = await engine.assemble(request, provider_ctx=provider_ctx)
    assert assembled.context_plan is not None
    assert assembled.context_plan.estimated_total_tokens > len(long_user) // 4


@pytest.mark.asyncio
async def test_engine_plan_total_equals_sum_of_pre_compile_model_facing_messages() -> None:
    adapter = _SmallWindowAdapter()
    config = RuntimeConfig(llm_adapter=adapter, production_mode=False)
    engine = DefaultNexusContextEngine()
    snapshot = build_session_history_snapshot(
        tenant_id="tenant",
        context_scope_id="scope",
        revision_id="rev",
        messages=[ChatMessage(role="user", content="history", entry_id="m1")],
    )
    base_messages = [
        ChatMessage(role="system", content="system", entry_id="sys"),
        ChatMessage(role="user", content="current user task", entry_id="current"),
    ]
    request = ContextAssemblyRequest(
        trace_id="t1",
        run_id="r1",
        task_id="task1",
        tenant_id="tenant1",
        assembly_scope="acp_step",
        objective="test",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(max_tokens_estimate=200),
        assembly_options=TaskContextAssemblyOptions(),
    )
    provider_ctx = ContextProviderContext(
        engine_id="default",
        handles={
            "runtime_config": config,
            "messages": base_messages,
            SESSION_HISTORY_SNAPSHOT_HANDLE: snapshot,
        },
    )
    assembled = await engine.assemble(request, provider_ctx=provider_ctx)
    assert assembled.context_plan is not None
    assert assembled.context_plan.estimated_total_tokens > 0
    assert (
        sum(group.token_estimate for group in assembled.context_plan.source_groups)
        == assembled.context_plan.estimated_total_tokens
    )


@pytest.mark.asyncio
async def test_long_required_current_user_message_is_not_reported_as_zero_token_plan() -> None:
    adapter = _SmallWindowAdapter()
    config = RuntimeConfig(llm_adapter=adapter, production_mode=False)
    engine = DefaultNexusContextEngine()
    long_user = "required " * 200
    request = ContextAssemblyRequest(
        trace_id="t1",
        run_id="r1",
        task_id="task1",
        tenant_id="tenant1",
        assembly_scope="acp_step",
        objective="test",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(max_tokens_estimate=200),
        assembly_options=TaskContextAssemblyOptions(),
    )
    provider_ctx = ContextProviderContext(
        engine_id="default",
        handles={
            "runtime_config": config,
            "messages": [ChatMessage(role="user", content=long_user, entry_id="current")],
        },
    )
    assembled = await engine.assemble(request, provider_ctx=provider_ctx)
    assert assembled.context_plan is not None
    assert assembled.context_plan.estimated_total_tokens > 0


def test_no_repository_executor_llm_integration_introduced() -> None:
    forbidden = (
        "OptimizationArtifactRepository",
        "InMemoryOptimizationArtifactRepository",
        "MessageSequenceArtifactExecutor",
        "OptimizationExecutionGuard",
    )
    engine_source = (REPO_ROOT / "intergrax" / "runtime" / "nexus" / "context" / "context_engine.py").read_text(
        encoding="utf-8"
    )
    for token in forbidden:
        assert token not in engine_source


def test_planning_modules_do_not_import_repository() -> None:
    forbidden = {
        "OptimizationArtifactRepository",
        "InMemoryOptimizationArtifactRepository",
        "try_acquire_creation_reservation",
        "store_validated_artifact",
    }
    scan_root = REPO_ROOT / "intergrax" / "context"
    matches: list[str] = []
    for path in scan_root.rglob("*.py"):
        if path.name == "serialization.py":
            continue
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    if alias.name in forbidden:
                        matches.append(f"{path.name}:{alias.name}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden:
                        matches.append(f"{path.name}:{alias.name}")
        for token in forbidden:
            if token in text and "repository" in token.lower():
                if f"import {token}" in text or f"import {token.split('Repository')[0]}" in text:
                    matches.append(f"{path.name}:{token}")
    assert not matches
