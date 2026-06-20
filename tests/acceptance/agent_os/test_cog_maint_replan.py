# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""COG-MAINT-03 — acceptance: dynamic replan after policy interrupt on reference host."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.reasoning_wiring import resolve_replan_policy_context
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.execution_interrupt import ExecutionInterrupt, InterruptType
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.interrupts.handler import ExecutionInterruptHandler

pytestmark = pytest.mark.unit


def _replan_enabled_env() -> ApplicationEnvironmentProfile:
    base = ApplicationEnvironmentProfile.lab_defaults()
    return base.model_copy(
        update={
            "orchestration_profile": base.orchestration_profile.model_copy(
                update={"allow_dynamic_replan": True},
            ),
        },
    )


def test_cog_maint_replan_after_policy_interrupt_on_reference_host() -> None:
    env = _replan_enabled_env()
    policy_ctx = resolve_replan_policy_context(env)
    handler = ExecutionInterruptHandler(allow_dynamic_replan=True)

    interrupt = ExecutionInterrupt(
        interrupt_type=InterruptType.POLICY_REVIEW_REQUIRED,
        source_agent_id="echo",
        task_id="task-1",
        run_id="run-1",
        blocking=True,
    )
    interrupt_resolution = handler.resolve_interrupt(
        interrupt,
        context=policy_ctx,
    )
    assert interrupt_resolution.policy_decision.action in {
        PolicyAction.ALLOW,
        PolicyAction.REQUIRE_HUMAN,
        PolicyAction.DENY,
    }

    replan = handler.resolve_decision(
        AgentDecision(type=AgentDecisionType.MODIFY_PLAN, reason="replan_after_interrupt"),
        task_id="task-1",
        run_id="run-1",
        agent_id="echo",
        context=policy_ctx,
    )
    assert replan.policy_decision.action is PolicyAction.ALLOW
    assert replan.policy_decision.policy_rule_id == "orchestration.allow_dynamic_replan"


def test_cog_maint_replan_denied_when_profile_disabled() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults()
    handler = ExecutionInterruptHandler(allow_dynamic_replan=False)
    replan = handler.resolve_decision(
        AgentDecision(type=AgentDecisionType.MODIFY_PLAN, reason="replan"),
        task_id="task-1",
        run_id="run-1",
        agent_id="echo",
        context=resolve_replan_policy_context(env),
    )
    assert replan.policy_decision.action is PolicyAction.DENY
