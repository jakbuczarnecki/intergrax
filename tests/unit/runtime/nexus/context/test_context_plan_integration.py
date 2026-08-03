# © Artur Czarnecki. All rights reserved.

"""CTX-UCL-3 engine integration tests."""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
    ContextProviderContext,
)
from intergrax.context.registry import ContextPluginRegistry
from intergrax.context.session_history import (
    HandleSessionHistoryProvider,
    SESSION_HISTORY_SNAPSHOT_HANDLE,
    build_session_history_snapshot,
)
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.llm.messages import ChatMessage, compute_model_facing_messages_hash
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.context_lifecycle.contracts import (
    ContextOptimizationDecision,
    ContextOptimizationMode,
    ContextOptimizationPolicy,
    EphemeralArtifactPersistencePolicy,
    OptimizationArtifactType,
)
from intergrax.runtime.context_lifecycle.in_memory_repository import InMemoryOptimizationArtifactRepository
from intergrax.runtime.nexus.context.context_budget import ContextBudgetPolicy
from intergrax.runtime.nexus.context.context_engine import DefaultNexusContextEngine
from intergrax.runtime.nexus.context.ucl_orchestration import (
    NEXUS_UCL_RUNTIME_HANDLE,
    NexusUCLExecutionReason,
    NexusUCLRuntimeDependencies,
)
from intergrax.runtime.token_optimization.message_sequence_artifact import MessageSequenceArtifactExecutor

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[5]


@dataclass(slots=True)
class _RuntimeConfigStub:
    llm_adapter: LLMAdapter
    production_mode: bool = False
    context_budget_policy: ContextBudgetPolicy | None = None
    context_decision_profile: dict[str, Any] | None = None


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
    config = _RuntimeConfigStub(llm_adapter=adapter, production_mode=False)
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
    config = _RuntimeConfigStub(llm_adapter=adapter, production_mode=False)
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
    config = _RuntimeConfigStub(llm_adapter=adapter, production_mode=False)
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
    config = _RuntimeConfigStub(llm_adapter=adapter, production_mode=False)
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


def test_context_engine_delegates_ucl_without_direct_repository_ownership() -> None:
    engine_source = (REPO_ROOT / "intergrax" / "runtime" / "nexus" / "context" / "context_engine.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(engine_source)
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "intergrax.runtime.nexus.context.ucl_orchestration":
            for alias in node.names:
                imported_names.add(alias.name)
    assert "resolve_ucl_context_plan" in imported_names
    forbidden = (
        "InMemoryOptimizationArtifactRepository",
        "MessageSequenceArtifactExecutor",
    )
    for token in forbidden:
        assert token not in engine_source
    assert "InMemoryOptimizationArtifactRepository(" not in engine_source


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


def _ucl_runtime(
    model_calls: list[int],
    *,
    repository: InMemoryOptimizationArtifactRepository | None = None,
) -> NexusUCLRuntimeDependencies:
    def _invoke_model(_call: object) -> LLMAdapterResponse:
        model_calls[0] += 1
        return LLMAdapterResponse(content="engine integration summary")

    executor = MessageSequenceArtifactExecutor(
        preflight=lambda _call: None,
        invoke_model=_invoke_model,
        count_tokens=lambda text: max(1, len(text) // 4),
    )
    return NexusUCLRuntimeDependencies(
        repository=repository or InMemoryOptimizationArtifactRepository(),
        message_sequence_executor=executor,
        strategy_versions={"message_sequence_summarization.v1": "1.0.0"},
        artifact_id_factory=lambda: "artifact-engine-1",
    )


def _optimization_policy() -> ContextOptimizationPolicy:
    return ContextOptimizationPolicy(
        policy_version="policy.v1",
        validation_contract_version="validation.v1",
        enabled=True,
        mode=ContextOptimizationMode.EPHEMERAL_ASSEMBLY,
        allow_lossy=True,
        allow_llm_summarization=True,
        allow_artifact_reuse=True,
        allowed_artifact_types=(OptimizationArtifactType.MESSAGE_SEQUENCE,),
        allowed_strategy_ids=("message_sequence_summarization.v1",),
        ephemeral_artifact_persistence=EphemeralArtifactPersistencePolicy.PERSIST_REUSABLE,
    )


@pytest.mark.asyncio
async def test_engine_ucl_runtime_create_then_reuse() -> None:
    adapter = _SmallWindowAdapter()
    config = _RuntimeConfigStub(
        llm_adapter=adapter,
        production_mode=False,
        context_budget_policy=ContextBudgetPolicy(
            max_tokens_estimate=80,
            max_chars=16_000,
        ),
    )
    registry = ContextPluginRegistry()
    registry.add_provider(HandleSessionHistoryProvider())
    engine = DefaultNexusContextEngine(registry=registry)
    history = [ChatMessage(role="user", content="history " * 80, entry_id="m1")]
    snapshot = build_session_history_snapshot(
        tenant_id="tenant",
        context_scope_id="scope",
        revision_id="rev",
        messages=history,
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
    model_calls = [0]
    repository = InMemoryOptimizationArtifactRepository()
    runtime = _ucl_runtime(model_calls, repository=repository)
    provider_ctx = ContextProviderContext(
        engine_id="default",
        handles={
            "runtime_config": config,
            "messages": [ChatMessage(role="user", content="current", entry_id="current")],
            SESSION_HISTORY_SNAPSHOT_HANDLE: snapshot,
            "context_optimization_policy": _optimization_policy(),
            NEXUS_UCL_RUNTIME_HANDLE: runtime,
        },
    )
    first = await engine.assemble(request, provider_ctx=provider_ctx)
    second = await engine.assemble(
        replace(request, run_id="r2"),
        provider_ctx=provider_ctx,
    )
    assert first.context_plan is not None
    assert first.context_plan.optimization_required is True
    assert model_calls[0] == 1
    first_replacements = [
        message for message in first.messages if message.entry_id.startswith("ucl-artifact-")
    ]
    second_replacements = [
        message for message in second.messages if message.entry_id.startswith("ucl-artifact-")
    ]
    assert len(first_replacements) == 1
    assert len(second_replacements) == 1
    assert (
        first_replacements[0].metadata["optimization_decision"]
        == ContextOptimizationDecision.CREATE_ARTIFACT.value
    )
    assert (
        second_replacements[0].metadata["optimization_decision"]
        == ContextOptimizationDecision.REUSE_ARTIFACT.value
    )
    assert any("engine integration summary" in (message.content or "") for message in first.messages)
    assert any("engine integration summary" in (message.content or "") for message in second.messages)
    assert not any("history " * 10 in (message.content or "") for message in first.messages)
    assert not any("history " * 10 in (message.content or "") for message in second.messages)


@pytest.mark.asyncio
async def test_engine_detects_structural_tool_linkage_mutation() -> None:
    adapter = _SmallWindowAdapter()
    config = _RuntimeConfigStub(llm_adapter=adapter, production_mode=False)
    engine = DefaultNexusContextEngine()
    planned_message = ChatMessage(
        role="assistant",
        content="call tool",
        entry_id="assistant-1",
        tool_calls=[{"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
    )
    mutated_message = ChatMessage(
        role="assistant",
        content="call tool",
        entry_id="assistant-1",
        tool_calls=[{"id": "call-2", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
    )
    assert compute_model_facing_messages_hash([planned_message]) != compute_model_facing_messages_hash(
        [mutated_message]
    )
    resolved_budget = engine._compiler.resolve_global_input_budget(
        config,
        max_output_tokens=None,
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
            "messages": [planned_message],
        },
    )
    degraded = type(
        "CompileResult",
        (),
        {
            "messages": [mutated_message],
            "total_tokens": 1,
            "budget_tokens": resolved_budget,
            "degradation_steps": (),
        },
    )()
    with patch(
        "intergrax.runtime.nexus.context.context_engine.compile_chat_messages",
        return_value=degraded,
    ):
        with pytest.raises(ValueError, match=NexusUCLExecutionReason.FINAL_COMPILE_MUTATED_PLAN.value):
            await engine.assemble(request, provider_ctx=provider_ctx)


@pytest.mark.asyncio
async def test_engine_validation_failure_skips_preflight() -> None:
    adapter = _SmallWindowAdapter()
    config = _RuntimeConfigStub(llm_adapter=adapter, production_mode=False)
    engine = DefaultNexusContextEngine()
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
            "messages": [ChatMessage(role="user", content="short", entry_id="current")],
        },
    )
    preflight_calls = [0]

    def _fake_preflight(*args: object, **kwargs: object) -> object:
        preflight_calls[0] += 1
        return None

    with patch.object(engine._validator, "validate", return_value=type("R", (), {"valid": False, "errors": ("bad",)})()):
        with patch(
            "intergrax.runtime.nexus.context.context_engine.verify_context_preflight",
            side_effect=_fake_preflight,
        ):
            with pytest.raises(ValueError, match="bad"):
                await engine.assemble(request, provider_ctx=provider_ctx)
    assert preflight_calls[0] == 0


@pytest.mark.asyncio
async def test_engine_compile_mutation_raises_ucl_final_compile_mutated_plan() -> None:
    adapter = _SmallWindowAdapter()
    config = _RuntimeConfigStub(llm_adapter=adapter, production_mode=False)
    engine = DefaultNexusContextEngine()
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
            "messages": [ChatMessage(role="user", content="short", entry_id="current")],
        },
    )
    degraded = type(
        "CompileResult",
        (),
        {
            "messages": [ChatMessage(role="user", content="mutated", entry_id="current")],
            "total_tokens": 1,
            "budget_tokens": 100,
            "degradation_steps": ("trim",),
        },
    )()
    with patch(
        "intergrax.runtime.nexus.context.context_engine.compile_chat_messages",
        return_value=degraded,
    ):
        with pytest.raises(ValueError, match=NexusUCLExecutionReason.FINAL_COMPILE_MUTATED_PLAN.value):
            await engine.assemble(request, provider_ctx=provider_ctx)
