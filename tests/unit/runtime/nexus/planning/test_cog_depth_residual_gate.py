# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.reasoning_failure import ReasoningFailureKind
from intergrax.runtime.interrupts.handler import ExecutionInterruptHandler
from intergrax.runtime.nexus.observability.planning_metrics import export_planning_metrics, record_planning_failure
from intergrax.runtime.nexus.planning.task_planner import NexusPlan, PlanStep
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine
from intergrax.applications._shared.reasoning_wiring import (
    resolve_engine_planner_prompt_config,
    resolve_tool_planning_config,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.runtime_policy import PolicyAction

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_nexus_plan_immutable_snapshot_during_engine_replan_boundary() -> None:
    plan = NexusPlan(
        task_id="t1",
        classification="single_agent",
        steps=[PlanStep(step_id="s1", agent_id="echo", description="first")],
        plan_metadata={"planner_source": "engine"},
    )
    snapshot = plan.model_copy(deep=True)
    _replan_candidate = plan.model_copy(
        update={
            "steps": [PlanStep(step_id="s2", agent_id="echo", description="replan")],
            "plan_metadata": {"planner_source": "engine", "used_fallback": "true"},
        }
    )
    assert plan.steps == snapshot.steps
    assert plan.plan_metadata == snapshot.plan_metadata
    assert _replan_candidate.steps != snapshot.steps


def test_modify_plan_denied_when_dynamic_replan_disabled() -> None:
    handler = ExecutionInterruptHandler(allow_dynamic_replan=False)
    resolution = handler.resolve_decision(
        AgentDecision(type=AgentDecisionType.MODIFY_PLAN, reason="replan"),
        task_id="t",
        run_id="r",
        agent_id="echo",
    )
    assert resolution.should_fail is True
    assert resolution.policy_decision.reason == "MODIFY_PLAN_NOT_SUPPORTED"


def test_modify_plan_allowed_on_engine_boundary_when_enabled() -> None:
    handler = ExecutionInterruptHandler(allow_dynamic_replan=True)
    resolution = handler.resolve_decision(
        AgentDecision(type=AgentDecisionType.MODIFY_PLAN, reason="replan"),
        task_id="t",
        run_id="r",
        agent_id="echo",
        context={"engine_replan_boundary": True},
    )
    assert resolution.policy_decision.action is PolicyAction.ALLOW
    assert resolution.policy_decision.reason == "engine_replan_allowed"


def test_planner_model_policy_denied() -> None:
    engine = RuntimePolicyEngine()
    decision = engine.evaluate_pre_llm(
        tenant_id="t",
        agent_id="planner",
        message_count=1,
        context={
            "phase": "nexus_planning",
            "planner_model_id": "gpt-4o",
            "denied_planner_model_ids": ("gpt-4o",),
        },
    )
    assert decision.action is PolicyAction.DENY
    assert decision.reason == "planner_model_denied"


def test_planning_failure_metrics_export() -> None:
    record_planning_failure(kind=ReasoningFailureKind.PLANNER_PARSE_FAILED.value)
    metrics = export_planning_metrics()
    assert metrics["ops_planning_failure_planner_parse_failed_total"] >= 1.0


def test_reasoning_wiring_resolves_prompt_configs() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="cog.wire")
    env = env.model_copy(
        update={
            "reasoning_profile": env.reasoning_profile.model_copy(
                update={
                    "tool_planner_prompt_id": "tools_agent_planner",
                    "engine_planner_prompt_id": "planner_default",
                }
            )
        }
    )
    tool_cfg = resolve_tool_planning_config(env)
    engine_cfg = resolve_engine_planner_prompt_config(env)
    assert tool_cfg.planner_instructions
    assert engine_cfg.system_prompt
