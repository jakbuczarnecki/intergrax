"""Order assistant agent — governed tool workflow for order status and updates."""

from __future__ import annotations

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_run import AgentRunError, AgentRunRequest, AgentRunResult
from intergrax.contracts.agent_run_enums import AgentRunErrorCode, AgentRunStatus, TerminalReason
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.task_envelope import TaskEnvelope, routing_capability_from_envelope
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.tools.registry import ToolRegistry

from platform_proofs.scenarios.indirect_prompt_injection.application.order_workflow import (
    execute_order_workflow,
    tool_trace_to_dict,
)
from platform_proofs.scenarios.indirect_prompt_injection.application.runtime_composition import (
    ORDER_ASSISTANT_AGENT_ID,
    ORDER_ASSISTANT_CAPABILITY,
    ScenarioRuntimeComposition,
    build_agent_runtime_context,
)
from platform_proofs.scenarios.indirect_prompt_injection.application.tools import SCENARIO_TOOL_IDS
from platform_proofs.scenarios.indirect_prompt_injection.application.workflows import WorkflowKind

ORDER_ASSISTANT_NODE_ID = f"node_{ORDER_ASSISTANT_AGENT_ID}"


def _domain_payload_from_workflow_result(
    workflow_result: object,
    *,
    workflow_kind: str,
) -> dict[str, object]:
    from platform_proofs.scenarios.indirect_prompt_injection.application.order_workflow import (
        OrderWorkflowResult,
    )

    if not isinstance(workflow_result, OrderWorkflowResult):
        raise TypeError("expected OrderWorkflowResult")
    return {
        "outcome": workflow_result.outcome,
        "terminal_summary": workflow_result.terminal_summary,
        "order_facts": workflow_result.order_facts,
        "retrieved_notes": [
            note.model_dump(mode="json") for note in workflow_result.retrieved_notes
        ],
        "tool_traces": [tool_trace_to_dict(trace) for trace in workflow_result.tool_traces],
        "policy_evaluations": list(workflow_result.policy_evaluations),
        "planner_rounds": list(workflow_result.planner_rounds),
        "write_tool_proposed": workflow_result.write_tool_proposed,
        "write_tool_executed": workflow_result.write_tool_executed,
        "policy_denied": workflow_result.policy_denied,
        "matched_policy_rule_ids": list(workflow_result.matched_policy_rule_ids),
        "model_provider": workflow_result.model_provider,
        "model_name": workflow_result.model_name,
        "workflow_kind": workflow_kind,
    }


def _agent_run_result_from_workflow(
    workflow_result: object,
    *,
    workflow_kind: str,
    run_id: str,
) -> AgentRunResult:
    domain_payload = _domain_payload_from_workflow_result(
        workflow_result,
        workflow_kind=workflow_kind,
    )
    return AgentRunResult(
        status=AgentRunStatus.SUCCEEDED,
        output=str(domain_payload.get("terminal_summary", "")),
        structured_data=domain_payload,
        terminal_reason=TerminalReason.GOAL_MET,
        run_id=run_id,
    )


class OrderAssistantAgent(Agent):
    def __init__(
        self,
        *,
        registry: ToolRegistry,
        runtime_composition: ScenarioRuntimeComposition,
        workflow: WorkflowKind,
        order_id: str = "48291",
        user_message: str = "",
    ) -> None:
        self._registry = registry
        self._runtime_composition = runtime_composition
        self._workflow = workflow
        self._order_id = order_id
        self._user_message = user_message

    def with_execution_inputs(self, *, order_id: str, user_message: str) -> OrderAssistantAgent:
        return OrderAssistantAgent(
            registry=self._registry,
            runtime_composition=self._runtime_composition,
            workflow=self._workflow,
            order_id=order_id,
            user_message=user_message,
        )

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id=ORDER_ASSISTANT_AGENT_ID,
            name="AI Order Assistant",
            description="Production-capable order status and shipping assistant.",
            capabilities=[ORDER_ASSISTANT_CAPABILITY],
            allowed_tools=list(SCENARIO_TOOL_IDS),
        )

    def can_handle(self, task: TaskEnvelope) -> CapabilityMatchResult:
        capability = routing_capability_from_envelope(task)
        if capability in (None, ORDER_ASSISTANT_CAPABILITY):
            return CapabilityMatchResult(
                matched=True,
                agent_id=ORDER_ASSISTANT_AGENT_ID,
                matched_capabilities=[ORDER_ASSISTANT_CAPABILITY],
                score=1.0,
                rationale="order assistant capability",
            )
        return CapabilityMatchResult(matched=False, rationale="capability not supported")

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        return build_agent_runtime_context(request, self._runtime_composition)

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        runtime_state = request.metadata.get("runtime_state")
        if not isinstance(runtime_state, RuntimeState):
            return AgentRunResult(
                status=AgentRunStatus.FAILED,
                errors=[
                    AgentRunError(
                        code=AgentRunErrorCode.INTERNAL_ERROR,
                        message="runtime_state_not_bound_for_tool_runtime",
                    )
                ],
                terminal_reason=TerminalReason.ERROR,
            )

        user_message = self._user_message
        if isinstance(request.input, str) and request.input.strip():
            user_message = request.input.strip()
        elif isinstance(request.input, dict):
            message = request.input.get("message")
            if isinstance(message, str) and message.strip():
                user_message = message.strip()

        metadata_order_id = request.metadata.get("order_id")
        order_id = (
            str(metadata_order_id)
            if isinstance(metadata_order_id, str) and metadata_order_id.strip()
            else self._order_id
        )

        workflow_result = await execute_order_workflow(
            runtime_state=runtime_state,
            registry=self._registry,
            order_id=order_id,
            user_message=user_message,
            workflow=self._workflow,
            agent_id=ORDER_ASSISTANT_AGENT_ID,
        )
        return _agent_run_result_from_workflow(
            workflow_result,
            workflow_kind=self._workflow.value,
            run_id=str(runtime_state.run_id),
        )

    def get_steps(self) -> list[AgentStep]:
        return [
            AgentStep(
                step_id="assist",
                step_name="assist",
                step_index=0,
                trace_label=ORDER_ASSISTANT_CAPABILITY,
                allowed_tools=list(SCENARIO_TOOL_IDS),
            )
        ]

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        _ = step
        runtime_state = ctx.metadata.get("runtime_state")
        if not isinstance(runtime_state, RuntimeState):
            raise RuntimeError("runtime_state_not_bound_for_tool_runtime")

        user_message = self._user_message
        request_message = ""
        if isinstance(ctx.request, RuntimeRequest):
            request_message = ctx.request.message
        if request_message.strip():
            user_message = request_message

        workflow_result = await execute_order_workflow(
            runtime_state=runtime_state,
            registry=self._registry,
            order_id=self._order_id,
            user_message=user_message,
            workflow=self._workflow,
            agent_id=ORDER_ASSISTANT_AGENT_ID,
        )
        domain_payload = _domain_payload_from_workflow_result(
            workflow_result,
            workflow_kind=self._workflow.value,
        )
        return StepOutput(
            step_id=step.step_id,
            summary=workflow_result.terminal_summary,
            data={"domain_summary": domain_payload},
        )

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step, output, ctx
        return AgentDecision(
            type=AgentDecisionType.COMPLETE,
            reason="order assistant workflow complete",
        )
