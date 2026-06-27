# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.authoring.uaep_step_bridge import (
    agent_decision_to_step_outcome,
    build_kernel_session,
    execute_uaep_step_via_kernel,
    trace_summary_from_kernel,
)
from intergrax.contracts.acp_metadata_keys import AcpStructuredDataKey
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.agent_run_enums import StepNextAction, TerminalReason
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.contracts.tool_request import ToolRequest, ToolResponse, ToolResponseStatus
from intergrax.runtime.nexus.orchestration.application_run_summary_builder import (
    build_application_run_summary,
)
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.policy.policy_engine import PolicyEngine


@pytest.mark.unit
@pytest.mark.gate
def test_agent_decision_continue_maps_to_step_outcome() -> None:
    output = StepOutput(step_id="s1", summary="partial")
    outcome = agent_decision_to_step_outcome(
        AgentDecision(type=AgentDecisionType.CONTINUE, reason="next"),
        output,
    )
    assert outcome.next_action == StepNextAction.CONTINUE
    assert outcome.is_terminal is False


@pytest.mark.unit
@pytest.mark.gate
def test_agent_decision_complete_maps_to_terminal_outcome() -> None:
    output = StepOutput(step_id="s2", summary="done")
    outcome = agent_decision_to_step_outcome(
        AgentDecision(type=AgentDecisionType.COMPLETE, reason="finished"),
        output,
    )
    assert outcome.is_terminal is True
    assert outcome.terminal_reason == TerminalReason.GOAL_MET
    assert outcome.output == "done"


@pytest.mark.unit
@pytest.mark.gate
def test_agent_decision_request_human_maps_to_pause_hitl() -> None:
    outcome = agent_decision_to_step_outcome(
        AgentDecision(type=AgentDecisionType.REQUEST_HUMAN, reason="approve"),
        StepOutput(step_id="s3", summary=""),
    )
    assert outcome.next_action == StepNextAction.PAUSE_HITL
    assert outcome.terminal_reason == TerminalReason.HUMAN_REQUIRED


class _CatalogToolUAEPAgent:
    """Minimal UAEP agent stub that invokes one catalog tool in run_step."""

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step, output, ctx
        return AgentDecision(type=AgentDecisionType.COMPLETE, reason="done")

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        await ctx.invoke_tool(
            ToolRequest(
                tool_name="rag.ingest_document",
                agent_id="local_indexer",
                step_id=step.step_id,
                input={"source_path": "/tmp/doc.txt"},
            )
        )
        return StepOutput(step_id=step.step_id, summary="indexed")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_uaep_kernel_bridge_harvests_catalog_tool_calls_for_app_summary() -> None:
    class _Gateway:
        async def invoke(self, request: ToolRequest) -> ToolResponse:
            return ToolResponse(
                request_id=request.request_id,
                status=ToolResponseStatus.SUCCESS,
                output={"used": True, "num_chunks": 1},
                duration_ms=5,
            )

    request = RuntimeRequest(
        agent_id="local_indexer",
        tenant_id="t1",
        user_id="u1",
        session_id="s1",
        message="index",
        metadata={},
    )
    kernel_ctx = build_kernel_session(
        agent_id="local_indexer",
        run_id="run-uaep-bridge",
        task_id="task-uaep-bridge",
        tenant_id="t1",
        max_steps=1,
        policy_engine=PolicyEngine(),
        request=request,
    )
    exec_ctx = RuntimeExecutionContext(
        task_id="task-uaep-bridge",
        run_id="run-uaep-bridge",
        agent_id="local_indexer",
        tool_gateway=_Gateway(),
    )
    step = AgentStep(
        step_id="local_indexer_step",
        step_name="local_indexer_step",
        step_index=0,
    )

    await execute_uaep_step_via_kernel(_CatalogToolUAEPAgent(), step, exec_ctx, kernel_ctx)

    trace_summary = trace_summary_from_kernel(kernel_ctx)
    assert trace_summary["total_tool_calls"] >= 1

    execution = AgentExecutionResult(
        agent_id="local_indexer",
        run_id="run-uaep-bridge",
        status=AgentExecutionStatus.COMPLETED,
        structured_data={AcpStructuredDataKey.TRACE_SUMMARY: trace_summary},
    )
    app_summary = build_application_run_summary(
        task_id="task-uaep-bridge",
        graph_id="graph-1",
        executions=[execution],
    )
    assert app_summary.agent_invocations[0].total_tool_calls >= 1
