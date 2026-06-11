# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax_assistant.intergrax_assistant_agent import IntergraxAssistantAgent
from organization_worker.organization_worker_agent import OrganizationWorkerAgent
from problem_radar.problem_radar_agent import ProblemRadarAgent
from vendor_discovery.vendor_discovery_agent import VendorDiscoveryAgent
from intergrax.contracts.agent_decision import AgentDecisionType
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from intergrax.contracts.agent_run_enums import AgentRunStatus, CognitivePattern
from intergrax.contracts.agent_step import AgentStep
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("factory", "fragment"),
    [
        (IntergraxAssistantAgent, "intergrax_assistant:"),
        (ProblemRadarAgent, "stub-1"),
        (VendorDiscoveryAgent, "stub-1"),
    ],
)
async def test_remaining_batch_typed_run(factory, fragment) -> None:
    agent = factory()
    assert agent.get_contract().cognitive_pattern == CognitivePattern.REFLEX
    result = await agent.run(
        AgentRunRequest(
            input="remaining batch smoke",
            identity=RequestIdentity(tenant_id="t1", user_id="u1"),
        )
    )
    assert result.status == AgentRunStatus.SUCCEEDED
    if isinstance(result.output, dict):
        summary = str(result.output.get("summary") or "")
    else:
        summary = str(result.output or "")
    assert fragment in summary


@pytest.mark.unit
@pytest.mark.gate
def test_organization_worker_decide_after_step_requests_hitl() -> None:
    agent = OrganizationWorkerAgent()
    step = AgentStep(step_id="prepare_vendor_report", step_name="prepare_vendor_report", step_index=0)
    ctx = RuntimeExecutionContext(
        run_id="run1",
        task_id="task1",
        agent_id="organization_worker",
        request=RuntimeRequest(
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            agent_id="organization_worker",
            message="Acme Corp",
        ),
    )
    from intergrax.contracts.agent_step import StepOutput

    decision = agent.decide_after_step(
        step,
        StepOutput(step_id=step.step_id, summary="draft", data={"subject": "Acme Corp"}),
        ctx,
    )
    assert decision.type == AgentDecisionType.REQUEST_HUMAN
    assert decision.human_request is not None


@pytest.mark.unit
@pytest.mark.gate
def test_organization_worker_decide_after_step_completes_when_approved() -> None:
    agent = OrganizationWorkerAgent()
    step = AgentStep(step_id="prepare_vendor_report", step_name="prepare_vendor_report", step_index=0)
    ctx = RuntimeExecutionContext(
        run_id="run1",
        task_id="task1",
        agent_id="organization_worker",
        request=RuntimeRequest(
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            agent_id="organization_worker",
            message="Acme Corp",
            metadata={"human_approved": True},
        ),
    )
    from intergrax.contracts.agent_step import StepOutput

    decision = agent.decide_after_step(
        step,
        StepOutput(step_id=step.step_id, summary="draft", data={"subject": "Acme Corp"}),
        ctx,
    )
    assert decision.type == AgentDecisionType.COMPLETE
    assert "delivered to finance channel" in str(ctx.metadata.get("runtime_answer", ""))
