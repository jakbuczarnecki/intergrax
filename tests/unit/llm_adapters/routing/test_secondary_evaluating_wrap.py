# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.applications._shared.llm_routing_runtime_bridge import maybe_wrap_secondary_routing_adapter
from intergrax.applications._shared.routing_evaluating_adapter import RoutingEvaluatingLLMAdapter
from intergrax.applications._shared.runtime_config_bridge import materialize_runtime_config
from intergrax.agents.reference_harness import default_reference_harness
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.routing import BudgetBelowRule, LLMRoutingProfile
from intergrax.llm_adapters.routing.context_bridge import LLMRoutingRuntimeSnapshot
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.tools.catalog_tool_planner import CatalogToolPlanner
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.websearch.service.websearch_config import WebSearchConfig, WebSearchLLMConfig
from testing_support.builder import FakeLLMAdapter


@pytest.mark.integration
@pytest.mark.gate
def test_secondary_evaluating_wraps_websearch_adapters_when_flag_set() -> None:
    primary = LLMProfile(provider=LLMProvider.OPENAI, model="gpt-4o-mini")
    local = LLMProfile(provider=LLMProvider.VLLM, model="meta-llama/Llama-3.1-8B")
    env = ApplicationEnvironmentProfile.lab_defaults()
    env.llm_profile = primary
    env.llm_routing_profile = LLMRoutingProfile(
        default_profile=primary,
        allowed_profiles=(primary, local),
        rules=(BudgetBelowRule(threshold=0.2, profile=local),),
    )
    env.llm_routing_evaluating_secondary = True

    map_adapter = FakeLLMAdapter(fixed_text="map")
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        llm_routing_snapshot=LLMRoutingRuntimeSnapshot(metadata={"run_id": "run-sec"}),
        websearch_config=WebSearchConfig(
            llm=WebSearchLLMConfig(map_adapter=map_adapter),
        ),
    )
    wrapped = maybe_wrap_secondary_routing_adapter(map_adapter, env, config)
    assert isinstance(wrapped, RoutingEvaluatingLLMAdapter)


@pytest.mark.integration
@pytest.mark.gate
def test_materialize_runtime_wraps_tool_planner_when_secondary_flag_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = LLMProfile(provider=LLMProvider.OPENAI, model="gpt-4o-mini")
    local = LLMProfile(provider=LLMProvider.VLLM, model="meta-llama/Llama-3.1-8B")
    env = ApplicationEnvironmentProfile.lab_defaults()
    env.llm_profile = primary
    env.llm_routing_profile = LLMRoutingProfile(
        default_profile=primary,
        allowed_profiles=(primary, local),
        rules=(BudgetBelowRule(threshold=0.2, profile=local),),
    )
    env.llm_routing_evaluating_secondary = True

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver._create_base_llm_adapter",
        lambda _env, _profile, hint=None: FakeLLMAdapter(fixed_text="core"),
    )

    request = RuntimeRequest(
        agent_id="lab-agent",
        user_id="user-1",
        session_id="sess-1",
        tenant_id="lab-tenant",
        message="hello",
        metadata={"task_class": "lab_routing", "agent_id": "lab-agent", "run_id": "run-sec"},
    )
    config = materialize_runtime_config(request, default_reference_harness(), env)
    config.tool_planner = CatalogToolPlanner.from_registry(
        llm=FakeLLMAdapter(fixed_text="planner"),
        registry=ToolRegistry(),
    )
    from intergrax.applications._shared.llm_routing_runtime_bridge import (
        wire_secondary_llm_routing_evaluating,
    )

    wire_secondary_llm_routing_evaluating(config, env)
    assert isinstance(config.tool_planner.llm, RoutingEvaluatingLLMAdapter)
