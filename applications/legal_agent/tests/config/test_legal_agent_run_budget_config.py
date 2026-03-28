# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest

from legal_agent.legal_agent import LegalAgent
from legal_agent.config.legal_agent_config import LegalAgentConfig
from intergrax.runtime.nexus.budget.budget_models import BudgetPolicy, BudgetEnforcementMode, RunBudget
from intergrax.runtime.nexus.policies.runtime_policies import DataCompliancePolicy
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest

from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

pytestmark = pytest.mark.unit


def test_legal_agent_config_rejects_run_budget_without_policy() -> None:
    with pytest.raises(ValueError, match="budget_policy"):
        LegalAgentConfig(
            session_manager=build_in_memory_session_manager(),
            llm_adapter=FakeLLMAdapter(),
            run_budget=RunBudget(max_llm_calls=5),
        )


def test_legal_agent_config_accepts_budget_pair() -> None:
    cfg = LegalAgentConfig(
        session_manager=build_in_memory_session_manager(),
        llm_adapter=FakeLLMAdapter(),
        run_budget=RunBudget(max_rag_invocations=2),
        budget_policy=BudgetPolicy(enforcement_mode=BudgetEnforcementMode.ABORT),
    )
    assert cfg.run_budget is not None
    assert cfg.run_budget.max_rag_invocations == 2


def test_legal_agent_build_context_propagates_run_budget_to_runtime_config() -> None:
    """
    Wiring smoke: Tier-2 config must reach RuntimeContext.config unchanged
    (regressions in LegalAgent.build_context would not fail budget_ticks unit tests).
    """
    run_budget = RunBudget(max_rag_invocations=1, max_tool_calls=3)
    budget_policy = BudgetPolicy(enforcement_mode=BudgetEnforcementMode.ABORT)
    cfg = LegalAgentConfig(
        session_manager=build_in_memory_session_manager(),
        llm_adapter=FakeLLMAdapter(),
        production_mode=False,
        run_budget=run_budget,
        budget_policy=budget_policy,
    )
    agent = LegalAgent(config=cfg)
    request = RuntimeRequest(
        agent_id="legal-budget-smoke",
        user_id="u1",
        session_id="s1",
        message="ping",
        tenant_id="tenant-budget-smoke",
    )
    ctx = agent.build_context(request)
    assert ctx.config.run_budget is run_budget
    assert ctx.config.budget_policy is budget_policy
    assert ctx.config.run_budget.max_rag_invocations == 1
    assert ctx.config.run_budget.max_tool_calls == 3


def test_legal_agent_build_context_propagates_tenant_workspace_to_runtime_config() -> None:
    cfg = LegalAgentConfig(
        session_manager=build_in_memory_session_manager(),
        llm_adapter=FakeLLMAdapter(),
        production_mode=False,
    )
    agent = LegalAgent(config=cfg)
    request = RuntimeRequest(
        agent_id="legal-scope-smoke",
        user_id="u1",
        session_id="s1",
        message="ping",
        tenant_id="tenant-scope",
        workspace_id="ws-scope",
    )
    ctx = agent.build_context(request)
    assert ctx.config.tenant_id == "tenant-scope"
    assert ctx.config.workspace_id == "ws-scope"


def test_legal_agent_build_context_propagates_data_compliance_to_runtime_policies() -> None:
    dc = DataCompliancePolicy(api_trace_export="none", redact_tool_calls_in_api=True)
    cfg = LegalAgentConfig(
        session_manager=build_in_memory_session_manager(),
        llm_adapter=FakeLLMAdapter(),
        production_mode=False,
        data_compliance=dc,
    )
    agent = LegalAgent(config=cfg)
    request = RuntimeRequest(
        agent_id="legal-dc-smoke",
        user_id="u1",
        session_id="s1",
        message="ping",
        tenant_id="t1",
    )
    ctx = agent.build_context(request)
    assert ctx.config.runtime_policies.data_compliance.api_trace_export == "none"
    assert ctx.config.runtime_policies.data_compliance.redact_tool_calls_in_api is True
