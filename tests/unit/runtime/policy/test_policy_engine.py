# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.event_severity import EventSeverity
from intergrax.contracts.runtime_policy import PolicyAction
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
    result = engine.evaluate_decision(decision, context={"require_human_on_critical": True})
    assert result.action == PolicyAction.REQUIRE_HUMAN


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
def test_tools_agent_has_no_agent_decision_alias():
    """§42.7 ``AgentDecision`` must not be re-exported from ``tools_agent`` (TYP-06)."""
    import intergrax.tools.tools_agent as tools_agent

    assert not hasattr(tools_agent, "AgentDecision")
    from intergrax.tools.core.tool_plan_decision import ToolPlanDecision

    assert ToolPlanDecision.__module__ == "intergrax.tools.core.tool_plan_decision"
