# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.applications._shared.lab_environment_profile import build_lab_environment_profile
from intergrax.applications._shared.llm_resolver import evaluate_llm_routing
from intergrax.applications._shared.runtime_config_bridge import materialize_runtime_config
from intergrax.agents.reference_harness import default_reference_harness
from intergrax.llm_adapters.routing.context_bridge import build_routing_context_from_runtime
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from lab_application.host.settings import LabApplicationSettings


@pytest.mark.integration
@pytest.mark.gate
def test_lab_host_budget_below_rule_switches_profile() -> None:
    env = build_lab_environment_profile(LabApplicationSettings())
    context = build_routing_context_from_runtime(
        tenant_id="lab",
        task_class="lab_routing",
        budget_remaining_ratio=0.1,
    )
    selected, _hint, reason = evaluate_llm_routing(env, routing_context=context)
    assert selected.model == "meta-llama/Llama-3.1-8B"
    assert reason is not None
    assert reason.startswith("rule:")


@pytest.mark.integration
@pytest.mark.gate
def test_materialize_runtime_config_auto_builds_routing_context() -> None:
    env = build_lab_environment_profile(LabApplicationSettings())
    request = RuntimeRequest(
        agent_id="lab-agent",
        user_id="user-1",
        session_id="sess-1",
        tenant_id="lab-tenant",
        message="hello",
        metadata={"task_class": "lab_routing", "agent_id": "lab-agent"},
    )
    config = materialize_runtime_config(request, default_reference_harness(), env)
    assert config.llm_routing_context is not None
    assert config.llm_routing_context.tenant_id == "lab-tenant"
    assert config.llm_routing_context.agent_id == "lab-agent"
    assert config.llm_routing_context.task_class == "lab_routing"
    assert config.llm_adapter is not None
