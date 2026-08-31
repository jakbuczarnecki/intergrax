"""Order assistant agent — governed tool workflow for order status and updates."""

from __future__ import annotations

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.task import TaskContext
from intergrax.tools.registry import ToolRegistry

from platform_proofs.scenarios.indirect_prompt_injection.application.order_provider_client import (
    OrderProviderClient,
)
from platform_proofs.scenarios.indirect_prompt_injection.application.order_workflow import (
    execute_order_workflow,
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


class OrderAssistantAgent(Agent):
    def __init__(
        self,
        *,
        registry: ToolRegistry,
        runtime_composition: ScenarioRuntimeComposition,
        provider_client: OrderProviderClient,
        workflow: WorkflowKind,
        order_id: str = "48291",
        user_message: str = "",
    ) -> None:
        self._registry = registry
        self._runtime_composition = runtime_composition
        self._provider_client = provider_client
        self._workflow = workflow
        self._order_id = order_id
        self._user_message = user_message

    def with_execution_inputs(self, *, order_id: str, user_message: str) -> OrderAssistantAgent:
        return OrderAssistantAgent(
            registry=self._registry,
            runtime_composition=self._runtime_composition,
            provider_client=self._provider_client,
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

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        if task_context.capability in (None, ORDER_ASSISTANT_CAPABILITY):
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

    def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
        _ = context
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
        request_message = ctx.request.message if ctx.request is not None else ""
        if request_message.strip():
            user_message = request_message

        workflow_result = execute_order_workflow(
            runtime_state=runtime_state,
            registry=self._registry,
            order_id=self._order_id,
            user_message=user_message,
            workflow=self._workflow,
            agent_id=ORDER_ASSISTANT_AGENT_ID,
        )
        domain_payload = {
            "outcome": workflow_result.outcome,
            "terminal_summary": workflow_result.terminal_summary,
            "order_facts": workflow_result.order_facts,
            "retrieved_notes": list(workflow_result.retrieved_notes),
            "tool_trace_count": len(workflow_result.tool_traces),
            "policy_evaluations": list(workflow_result.policy_evaluations),
            "planner_rounds": list(workflow_result.planner_rounds),
            "write_tool_proposed": workflow_result.write_tool_proposed,
            "write_tool_executed": workflow_result.write_tool_executed,
            "policy_denied": workflow_result.policy_denied,
            "matched_policy_rule_ids": list(workflow_result.matched_policy_rule_ids),
            "model_provider": workflow_result.model_provider,
            "model_name": workflow_result.model_name,
            "workflow_kind": self._workflow.value,
        }
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
