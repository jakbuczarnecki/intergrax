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
