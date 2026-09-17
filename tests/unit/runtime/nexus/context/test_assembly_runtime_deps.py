# © Artur Czarnecki. All rights reserved.

"""CE-01-R1: typed ContextAssemblyRuntimeDependencies."""

from __future__ import annotations

import pytest

from intergrax.context.contracts import ContextProviderContext
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.context_lifecycle.contracts import ContextOptimizationPolicy
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.assembly_runtime_deps import (
    build_context_assembly_runtime_dependencies,
    ensure_context_assembly_runtime,
    try_build_runtime_from_legacy_handles,
)
from intergrax.runtime.nexus.context.context_engine import DefaultNexusContextEngine
from intergrax.runtime.nexus.context.ucl_orchestration import NEXUS_UCL_RUNTIME_HANDLE
from intergrax.runtime.wiring.context_runtime_bridge import CONTEXT_OPTIMIZATION_POLICY_HANDLE
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _Adapter(LLMAdapter):
    provider = "fake"
    model = "fake-runtime-deps"

    @property
    def context_window_tokens(self) -> int:
        return 4096

    def generate_messages(self, messages, **kwargs) -> LLMAdapterResponse:
        _ = messages, kwargs
        return LLMAdapterResponse(content="ok")


def test_legacy_handles_hydrate_typed_runtime_once() -> None:
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
    ctx = ContextProviderContext(engine_id="default", handles=handles)
    assert ctx.runtime is not None
    assert ctx.runtime.runtime_config is config
    assert ctx.runtime.base_messages[0].content == "legacy"
    assert ctx.runtime.max_output_tokens == 32
    assert ctx.runtime.optimization_policy is policy


@pytest.mark.asyncio
async def test_engine_reads_typed_runtime_not_semantic_handles() -> None:
    from intergrax.context.contracts import (
        ContextAssemblyRequest,
        ContextBudgetSnapshot,
        ContextDecisionSnapshot,
    )
    from intergrax.contracts.context_assembly import TaskContextAssemblyOptions

    config = RuntimeConfig(llm_adapter=_Adapter(), production_mode=False)
    runtime = build_context_assembly_runtime_dependencies(
        runtime_config=config,
        messages=[ChatMessage(role="user", content="typed path")],
        max_output_tokens=48,
    )
    ctx = ContextProviderContext(engine_id="default", runtime=runtime, handles={})
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


def test_ensure_context_assembly_runtime_is_idempotent() -> None:
    config = RuntimeConfig(llm_adapter=_Adapter(), production_mode=False)
    built = build_context_assembly_runtime_dependencies(runtime_config=config)
    ctx = ContextProviderContext(engine_id="default", runtime=built)
    again = ensure_context_assembly_runtime(ctx)
    assert again.runtime is built


def test_try_build_runtime_from_legacy_handles_requires_runtime_config() -> None:
    assert try_build_runtime_from_legacy_handles({}) is None
    assert try_build_runtime_from_legacy_handles({NEXUS_UCL_RUNTIME_HANDLE: object()}) is None
