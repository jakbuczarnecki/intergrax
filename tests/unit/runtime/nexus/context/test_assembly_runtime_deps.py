# © Artur Czarnecki. All rights reserved.

"""CE-01-R1A: typed ContextAssemblyRuntimeDependencies and fail-closed canonical assembly."""

from __future__ import annotations

import pytest

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
    ContextProviderContext,
)
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.context_lifecycle.contracts import ContextOptimizationPolicy
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.assembly_runtime_deps import (
    build_context_assembly_runtime_dependencies,
)
from intergrax.runtime.nexus.context.context_engine import DefaultNexusContextEngine
from intergrax.runtime.nexus.context.legacy_assembly_runtime_bridge import (
    build_context_assembly_runtime_from_legacy_handles,
    try_build_runtime_from_legacy_handles,
)
from intergrax.runtime.nexus.context.ucl_orchestration import NEXUS_UCL_RUNTIME_HANDLE
from intergrax.runtime.wiring.context_runtime_bridge import CONTEXT_OPTIMIZATION_POLICY_HANDLE
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _Adapter(BaseLLMAdapter):
    provider = "fake"
    model = "fake-runtime-deps"

    @property
    def context_window_tokens(self) -> int:
        return 4096

    def generate_messages(self, messages, **kwargs) -> LLMAdapterResponse:
        _ = messages, kwargs
        return LLMAdapterResponse(content="ok")


def test_context_provider_context_does_not_auto_hydrate_runtime_from_handles() -> None:
    config = RuntimeConfig(llm_adapter=_Adapter(), production_mode=False)
    handles = {
        "runtime_config": config,
        "messages": [ChatMessage(role="user", content="legacy")],
    }
    ctx = ContextProviderContext(engine_id="default", handles=handles)
    assert ctx.runtime is None


def test_explicit_legacy_bridge_builds_typed_runtime() -> None:
    config = RuntimeConfig(llm_adapter=_Adapter(), production_mode=False)
    policy = ContextOptimizationPolicy(
        policy_version="policy.v1",
        validation_contract_version="validation.v1",
        enabled=False,
    )
    handles = {
        "runtime_config": config,
        "messages": [ChatMessage(role="user", content="legacy")],
        "max_output_tokens": 32,
        CONTEXT_OPTIMIZATION_POLICY_HANDLE: policy,
    }
    runtime = build_context_assembly_runtime_from_legacy_handles(handles)
    assert runtime is not None
    assert runtime.runtime_config is config
    assert runtime.base_messages[0].content == "legacy"
    assert runtime.max_output_tokens == 32
    assert runtime.optimization_policy is policy

    ctx = ContextProviderContext(engine_id="default", runtime=runtime, handles=handles)
    assert ctx.runtime is runtime


@pytest.mark.asyncio
async def test_assemble_missing_runtime_fails_closed() -> None:
    config = RuntimeConfig(llm_adapter=_Adapter(), production_mode=False)
    handles = {
        "runtime_config": config,
        "messages": [ChatMessage(role="user", content="should-not-hydrate")],
    }
    ctx = ContextProviderContext(engine_id="default", handles=handles)
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
    with pytest.raises(ValueError, match="runtime is required"):
        await DefaultNexusContextEngine().assemble(request, provider_ctx=ctx)


@pytest.mark.asyncio
async def test_engine_reads_typed_runtime_not_semantic_handles() -> None:
    config = RuntimeConfig(llm_adapter=_Adapter(), production_mode=False)
    runtime = build_context_assembly_runtime_dependencies(
        runtime_config=config,
        messages=[ChatMessage(role="user", content="typed-good")],
        max_output_tokens=48,
    )
    poisoned = {
        "runtime_config": RuntimeConfig(llm_adapter=_Adapter(), production_mode=True),
        "messages": [ChatMessage(role="user", content="legacy-bad")],
        "max_output_tokens": 1,
    }
    ctx = ContextProviderContext(engine_id="default", runtime=runtime, handles=poisoned)
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
    assembled = await DefaultNexusContextEngine().assemble(request, provider_ctx=ctx)
    assert assembled.messages
    user_contents = [m.content for m in assembled.messages if m.role == "user"]
    assert "typed-good" in user_contents
    assert "legacy-bad" not in user_contents


def test_custom_optimization_policy_via_runtime_contract() -> None:
    config = RuntimeConfig(llm_adapter=_Adapter(), production_mode=False)
    policy = ContextOptimizationPolicy(
        policy_version="policy.v1",
        validation_contract_version="validation.v1",
        enabled=True,
    )
    runtime = build_context_assembly_runtime_dependencies(
        runtime_config=config,
        optimization_policy=policy,
    )
    assert runtime.optimization_policy is policy


def test_try_build_runtime_from_legacy_handles_requires_runtime_config() -> None:
    assert try_build_runtime_from_legacy_handles({}) is None
    assert try_build_runtime_from_legacy_handles({NEXUS_UCL_RUNTIME_HANDLE: object()}) is None


def test_ce_q14_gate_detects_canonical_handle_read_fixture() -> None:
    fixture = 'ctx.handles.get("runtime_config")'
    import re

    pattern = re.compile(r'\.handles\.get\(\s*["\']([^"\']+)["\']')
    match = pattern.search(fixture)
    assert match is not None
    assert match.group(1) == "runtime_config"
