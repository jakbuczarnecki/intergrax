# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.event_severity import EventSeverity
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.contracts.runtime_policy_context import (
    AgentDecisionPolicyContext,
    CriticPolicyContext,
    PreModelPhase,
    PreModelPolicyContext,
)
from intergrax.runtime.policy.policy_engine import (
    PolicyEngine,
    coerce_policy_engine,
    coerce_replay_policy_engine,
)
from intergrax.runtime.replay.policy import ExecutionPolicyEngine
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine
from intergrax.runtime.replay.policy import PolicyDecisionType
from intergrax.runtime.replay.policy_config import ExecutionPolicyConfig


@pytest.mark.unit
@pytest.mark.gate
def test_coerce_policy_engine_wraps_runtime_engine():
    runtime = RuntimePolicyEngine()
    facade = coerce_policy_engine(runtime)
    assert isinstance(facade, PolicyEngine)
    assert facade.runtime is runtime


@pytest.mark.unit
@pytest.mark.gate
def test_policy_engine_evaluate_decision_delegates_to_runtime():
    engine = PolicyEngine()
    decision = AgentDecision(
        type=AgentDecisionType.INTERRUPT,
        reason="critical",
        severity=EventSeverity.CRITICAL,
    )
    result = engine.evaluate_decision(
        decision,
        context=AgentDecisionPolicyContext(require_human_on_critical=True),
    )
    assert result.action == PolicyAction.REQUIRE_HUMAN


@pytest.mark.unit
@pytest.mark.gate
def test_evaluate_decision_critical_interrupt_default_requires_human():
    engine = PolicyEngine()
    decision = AgentDecision(
        type=AgentDecisionType.INTERRUPT,
        reason="critical",
        severity=EventSeverity.CRITICAL,
    )
    result = engine.evaluate_decision(decision)
    assert result.action == PolicyAction.REQUIRE_HUMAN
    assert result.reason == "critical_interrupt_requires_human"


@pytest.mark.unit
@pytest.mark.gate
def test_evaluate_decision_critical_interrupt_opt_out_allows():
    engine = PolicyEngine()
    decision = AgentDecision(
        type=AgentDecisionType.INTERRUPT,
        reason="critical",
        severity=EventSeverity.CRITICAL,
    )
    result = engine.evaluate_decision(
        decision,
        context=AgentDecisionPolicyContext(require_human_on_critical=False),
    )
    assert result.action == PolicyAction.ALLOW
    assert result.reason == "default_allow"


@pytest.mark.unit
@pytest.mark.gate
def test_evaluate_decision_complete_with_unresolved_critical_requires_human():
    engine = PolicyEngine()
    decision = AgentDecision(type=AgentDecisionType.COMPLETE, reason="done")
    result = engine.evaluate_decision(
        decision,
        context=AgentDecisionPolicyContext(has_unresolved_critical_interrupt=True),
    )
    assert result.action == PolicyAction.REQUIRE_HUMAN
    assert result.reason == "unresolved_critical_interrupt"


@pytest.mark.unit
@pytest.mark.gate
def test_evaluate_decision_normal_allows():
    engine = PolicyEngine()
    decision = AgentDecision(type=AgentDecisionType.CONTINUE, reason="ok")
    result = engine.evaluate_decision(decision)
    assert result.action == PolicyAction.ALLOW
    assert result.reason == "default_allow"


@pytest.mark.unit
@pytest.mark.gate
def test_evaluate_pre_llm_empty_context_denies():
    engine = PolicyEngine()
    result = engine.evaluate_pre_llm(tenant_id="t1", agent_id="a1", message_count=0)
    assert result.action == PolicyAction.DENY
    assert result.reason == "pre_llm_empty_context"


@pytest.mark.unit
@pytest.mark.gate
def test_evaluate_pre_llm_denied_planning_model():
    engine = PolicyEngine()
    result = engine.evaluate_pre_llm(
        tenant_id="t1",
        agent_id="a1",
        message_count=1,
        context=PreModelPolicyContext(
            phase=PreModelPhase.NEXUS_PLANNING,
            planner_model_id="blocked-model",
            denied_planner_model_ids=("blocked-model",),
        ),
    )
    assert result.action == PolicyAction.DENY
    assert result.reason == "planner_model_denied"


@pytest.mark.unit
@pytest.mark.gate
def test_evaluate_pre_llm_allowed_planning_model():
    engine = PolicyEngine()
    result = engine.evaluate_pre_llm(
        tenant_id="t1",
        agent_id="a1",
        message_count=1,
        context=PreModelPolicyContext(
            phase=PreModelPhase.NEXUS_PLANNING,
            planner_model_id="allowed-model",
            denied_planner_model_ids=("blocked-model",),
        ),
    )
    assert result.action == PolicyAction.ALLOW
    assert result.reason == "pre_llm_default_allow"


@pytest.mark.unit
@pytest.mark.gate
def test_evaluate_critic_verdict_escalate_hitl():
    engine = PolicyEngine()
    result = engine.evaluate_critic_verdict(
        passed=False,
        recommended_action="escalate_hitl",
    )
    assert result.action == PolicyAction.REQUIRE_HUMAN
    assert result.reason == "critic_escalate_hitl"


@pytest.mark.unit
@pytest.mark.gate
def test_evaluate_critic_verdict_required_and_failed_denies():
    engine = PolicyEngine()
    result = engine.evaluate_critic_verdict(
        passed=False,
        recommended_action="fail",
        context=CriticPolicyContext(require_critic_on_completion=True),
    )
    assert result.action == PolicyAction.DENY
    assert result.reason == "critic_completion_required"


@pytest.mark.unit
@pytest.mark.gate
def test_evaluate_critic_verdict_normal_allows():
    engine = PolicyEngine()
    result = engine.evaluate_critic_verdict(passed=True, recommended_action="continue")
    assert result.action == PolicyAction.ALLOW
    assert result.reason == "critic_default_allow"


@pytest.mark.unit
@pytest.mark.gate
def test_evaluate_pre_output_empty_denies():
    engine = PolicyEngine()
    result = engine.evaluate_pre_output(tenant_id="t1", agent_id="a1", output_chars=0)
    assert result.action == PolicyAction.DENY
    assert result.reason == "pre_output_empty"


@pytest.mark.unit
@pytest.mark.gate
def test_evaluate_pre_output_non_empty_allows():
    engine = PolicyEngine()
    result = engine.evaluate_pre_output(tenant_id="t1", agent_id="a1", output_chars=12)
    assert result.action == PolicyAction.ALLOW
    assert result.reason == "pre_output_default_allow"


@pytest.mark.unit
@pytest.mark.gate
def test_coerce_replay_policy_engine_from_execution_engine():
    config = ExecutionPolicyConfig()
    facade = coerce_replay_policy_engine(ExecutionPolicyEngine(config))
    assert facade.replay is not None


@pytest.mark.unit
@pytest.mark.gate
def test_policy_engine_evaluate_replay_without_config_allows():
    engine = PolicyEngine()
    from intergrax.runtime.replay.metrics import ExecutionMetrics
    from intergrax.runtime.replay.regression import RegressionSignals

    metrics = ExecutionMetrics(
        step_count=1,
        total_llm_calls=0,
        total_tool_calls=0,
        total_artifacts=0,
        total_tokens=0,
        duration=None,
        tool_steps_ratio=0.0,
        llm_steps_ratio=0.0,
    )
    regression = RegressionSignals()
    result = engine.evaluate_replay(metrics, regression)
    assert result.decision == PolicyDecisionType.ALLOW


@pytest.mark.unit
@pytest.mark.gate
def test_tools_agent_module_removed_and_tool_plan_decision_canonical():
    """§42.7 legacy ``tools_agent`` removed; ``ToolPlanDecision`` is canonical (TYP-06)."""
    import importlib.util

    from intergrax.tools.core.tool_plan_decision import ToolPlanDecision

    assert importlib.util.find_spec("intergrax.tools.tools_agent") is None
    assert ToolPlanDecision.__module__ == "intergrax.tools.core.tool_plan_decision"
