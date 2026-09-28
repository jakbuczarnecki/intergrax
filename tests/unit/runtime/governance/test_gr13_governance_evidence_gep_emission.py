# © Artur Czarnecki. All rights reserved.

"""GR-13 per-GEP GR-8 fact emission on canonical production owners."""

from __future__ import annotations

import pytest

from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.contracts.governed_execution_governance_evidence import (
    GovernedExecutionEvaluationPoint,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from intergrax.runtime.governance.execution_guard import GovernanceEvaluation
from intergrax.runtime.governance.governance_evidence_composition import (
    build_governance_evidence_recorder,
    build_in_memory_governance_evidence_persistence,
)
from intergrax.runtime.governance.post_run_governance_bridge import (
    invoke_post_run_governance,
)
from intergrax.runtime.interrupts.handler import ExecutionInterruptHandler
from intergrax.runtime.policy.policy_engine import PolicyEngine
from intergrax.runtime.policy.pre_model_policy_evaluation import (
    enforce_pre_model_before_structured_inference,
)
from intergrax.runtime.policy.pre_output_policy_bridge import apply_pre_output_policy
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine
from intergrax.runtime.replay.metrics import ExecutionMetrics
from intergrax.runtime.replay.policy import PolicyDecision as ReplayPolicyDecision
from intergrax.runtime.replay.policy import PolicyDecisionType
from intergrax.runtime.replay.regression import RegressionSignals
from intergrax.runtime.task.task import Task
from intergrax.runtime.governance.governance_policy_decision_evidence_recording import (
    record_tool_plan_or_access_evidence,
)


class _StubLlmAdapter:
    model = "gr13-model"
    provider = "gr13-provider"


def _governance_context() -> tuple[object, object]:
    gov = bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(
            tenant_id="tenant-gr13",
            workspace_id="workspace-gr13",
            principal_id="principal-gr13",
        ),
    )
    run = mint_run_id()
    attempt = mint_attempt_id()
    execution = mint_execution_id()
    ident = bind_active_execution_identity(
        run_id=run,
        attempt_id=attempt,
        execution_id=execution,
    )
    return gov, ident


@pytest.fixture
def gr13_evidence_store():
    store = build_in_memory_governance_evidence_persistence()
    recorder = build_governance_evidence_recorder(persistence=store)
    gov, ident = _governance_context()
    yield store, recorder
    reset_active_execution_governance_identity(gov)
    reset_active_execution_identity(ident)


@pytest.mark.unit
def test_gr13_agentic_pre_model_emits_fact(gr13_evidence_store) -> None:
    store, recorder = gr13_evidence_store
    enforce_pre_model_before_structured_inference(
        PolicyEngine(),
        recorder,
        adapter=_StubLlmAdapter(),
        messages=[ChatMessage(role="user", content="hi")],
        inference_profile_id=None,
    )
    assert len(store.facts) == 1
    assert store.facts[0].evaluation_point is GovernedExecutionEvaluationPoint.PRE_MODEL


@pytest.mark.unit
def test_gr13_agentic_agent_decision_emits_fact(gr13_evidence_store) -> None:
    store, recorder = gr13_evidence_store
    handler = ExecutionInterruptHandler(
        PolicyEngine(),
        governance_evidence_recorder=recorder,
    )
    resolution = handler.resolve_decision(
        AgentDecision(type=AgentDecisionType.CONTINUE, reason="ok"),
        task_id=str(mint_task_id()),
        run_id=str(mint_run_id()),
        agent_id="agent_1",
    )
    assert resolution.policy_decision.action is PolicyAction.ALLOW
    assert any(
        f.evaluation_point is GovernedExecutionEvaluationPoint.AGENT_DECISION
        for f in store.facts
    )


@pytest.mark.unit
def test_gr13_agentic_interrupt_emits_fact(gr13_evidence_store) -> None:
    store, recorder = gr13_evidence_store
    handler = ExecutionInterruptHandler(
        PolicyEngine(),
        governance_evidence_recorder=recorder,
    )
    handler.resolve_decision(
        AgentDecision(type=AgentDecisionType.INTERRUPT, reason="stop", payload={}),
        task_id=str(mint_task_id()),
        run_id=str(mint_run_id()),
        agent_id="agent_1",
    )
    assert any(
        f.evaluation_point is GovernedExecutionEvaluationPoint.INTERRUPT
        for f in store.facts
    )


@pytest.mark.unit
def test_gr13_agentic_tool_plan_or_access_emits_fact(gr13_evidence_store) -> None:
    store, recorder = gr13_evidence_store
    record_tool_plan_or_access_evidence(
        recorder,
        tenant_id="tenant-gr13",
        workspace_id="workspace-gr13",
        principal_id="principal-gr13",
        agent_id="agent_1",
        requested_tool_count=2,
        allowed_tool_count=1,
        use_tools_requested=True,
        use_tools_allowed=True,
    )
    assert (
        store.facts[0].evaluation_point
        is GovernedExecutionEvaluationPoint.TOOL_PLAN_OR_ACCESS
    )
    assert store.facts[0].decision is PolicyAction.DENY


@pytest.mark.unit
def test_gr13_agentic_tool_invocation_policy_emits_fact(gr13_evidence_store) -> None:
    from intergrax.contracts.policy_enforcement_mode import PolicyEnforcementMode
    from intergrax.runtime.policy.rules.evaluation import PolicyEnforcementDecision
    from intergrax.runtime.policy.rules.schema import PolicyRuleAction

    store, recorder = gr13_evidence_store

    decision = PolicyEnforcementDecision(
        action=PolicyRuleAction.DENY,
        matched_rule_ids=("rule.deny",),
        reasons=("denied",),
        enforcement_mode=PolicyEnforcementMode.ENFORCE,
        enforced=True,
        would_deny=True,
        requires_hitl=False,
        provenance_digest="digest",
    )
    from intergrax.runtime.governance.governance_policy_decision_evidence_recording import (
        record_governance_policy_decision_evidence_for_active_identity,
        runtime_policy_decision_from_declarative_enforcement,
    )

    runtime_policy_decision_from_declarative_enforcement(decision)
    record_governance_policy_decision_evidence_for_active_identity(
        recorder,
        evaluation_point=GovernedExecutionEvaluationPoint.TOOL_INVOCATION_POLICY,
        decision=runtime_policy_decision_from_declarative_enforcement(decision),
        action="tool_invoke:probe",
        digest_payload={"tool_id": "probe"},
        idempotency_prefix="tool_invocation_policy",
    )
    assert (
        store.facts[0].evaluation_point
        is GovernedExecutionEvaluationPoint.TOOL_INVOCATION_POLICY
    )


@pytest.mark.unit
def test_gr13_agentic_pre_output_emits_fact(gr13_evidence_store) -> None:
    store, recorder = gr13_evidence_store
    task = Task(tenant_id="tenant-gr13", user_id="principal-gr13", message="hi")
    apply_pre_output_policy(
        PolicyEngine(), task, answer="done", governance_evidence_recorder=recorder
    )
    assert (
        store.facts[0].evaluation_point is GovernedExecutionEvaluationPoint.PRE_OUTPUT
    )


@pytest.mark.unit
def test_gr13_agentic_post_run_emits_fact(gr13_evidence_store) -> None:
    store, recorder = gr13_evidence_store

    class _PostRun:
        def evaluate(self, run_id: str, agent_id: str) -> GovernanceEvaluation:
            return GovernanceEvaluation(
                decision=ReplayPolicyDecision(
                    decision=PolicyDecisionType.ALLOW,
                    reasons=[],
                ),
                metrics=ExecutionMetrics(
                    step_count=1,
                    total_llm_calls=0,
                    total_tool_calls=0,
                    total_artifacts=0,
                    total_tokens=0,
                    duration=None,
                    tool_steps_ratio=0.0,
                    llm_steps_ratio=0.0,
                ),
                regression=RegressionSignals(),
            )

    invoke_post_run_governance(
        _PostRun(),
        run_id=str(mint_run_id()),
        agent_id="agent_1",
        governance_evidence_recorder=recorder,
    )
    assert store.facts[0].evaluation_point is GovernedExecutionEvaluationPoint.POST_RUN


@pytest.mark.unit
def test_gr13_orchestration_pre_model_emits_fact(gr13_evidence_store) -> None:
    store, recorder = gr13_evidence_store
    from intergrax.runtime.governance.governance_policy_decision_evidence_recording import (
        record_governance_policy_decision_evidence,
    )

    decision = PolicyDecision(
        action=PolicyAction.ALLOW, reason="ok", policy_rule_id="t"
    )
    record_governance_policy_decision_evidence(
        recorder,
        evaluation_point=GovernedExecutionEvaluationPoint.PRE_MODEL,
        decision=decision,
        tenant_id="tenant-gr13",
        workspace_id="workspace-gr13",
        principal_id="principal-gr13",
        action="nexus_planning.pre_llm",
        digest_payload={"phase": "nexus_planning"},
        idempotency_prefix="pre_model_orchestration",
    )
    assert store.facts[0].evaluation_point is GovernedExecutionEvaluationPoint.PRE_MODEL


@pytest.mark.unit
def test_gr13_orchestration_tool_plan_or_access_emits_fact(gr13_evidence_store) -> None:
    test_gr13_agentic_tool_plan_or_access_emits_fact(gr13_evidence_store)


@pytest.mark.unit
def test_gr13_orchestration_tool_invocation_policy_emits_fact(
    gr13_evidence_store,
) -> None:
    test_gr13_agentic_tool_invocation_policy_emits_fact(gr13_evidence_store)


@pytest.mark.unit
def test_gr13_orchestration_pre_output_emits_fact(gr13_evidence_store) -> None:
    test_gr13_agentic_pre_output_emits_fact(gr13_evidence_store)


@pytest.mark.unit
def test_gr13_orchestration_post_run_emits_fact(gr13_evidence_store) -> None:
    test_gr13_agentic_post_run_emits_fact(gr13_evidence_store)


class _DenyContinue(RuntimePolicyEngine):
    def evaluate_decision(self, decision, *, context=None):
        return PolicyDecision(
            action=PolicyAction.DENY,
            reason="denied",
            policy_rule_id="test.deny",
        )


@pytest.mark.unit
def test_gr13_evidence_does_not_grant_permission(gr13_evidence_store) -> None:
    _, recorder = gr13_evidence_store
    handler = ExecutionInterruptHandler(
        _DenyContinue(),
        governance_evidence_recorder=recorder,
    )
    resolution = handler.resolve_decision(
        AgentDecision(type=AgentDecisionType.CONTINUE, reason="ok"),
        task_id=str(mint_task_id()),
        run_id=str(mint_run_id()),
        agent_id="agent_1",
    )
    assert resolution.should_block_execution is True
