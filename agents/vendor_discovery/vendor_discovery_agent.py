# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from vendor_discovery.capabilities import CAPABILITIES
from vendor_discovery.contract import build_agent_contract
from vendor_discovery.steps.domain import build_stub_vendor_discovery_output
from intergrax.agents.authoring.stub_llm import PrefixStubLLMAdapter
from intergrax.agents.authoring.acp_stub_reflex import (
    evaluate_complete,
    perceive_run_input,
    reason_passthrough,
)
from intergrax.agents.authoring.patterns.reflex import ReflexAgent
from intergrax.agents.reference_harness import (
    LabHarnessContext,
    build_lab_agent_runtime_context,
    default_reference_harness,
)
from intergrax.contracts.agent_run_enums import CognitivePattern
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.task import TaskContext


class VendorDiscoveryAgent(ReflexAgent):
    """Phase K.2 prototype — typed Reflex pattern (ACP-MIG-5)."""

    contract_id = "vendor_discovery"
    capabilities = tuple(CAPABILITIES)
    cognitive_pattern = CognitivePattern.REFLEX
    main_step_id = "vendor_discovery_step"

    def __init__(self, harness: LabHarnessContext | None = None) -> None:
        self._harness = harness or default_reference_harness()

    def get_contract(self):
        return build_agent_contract()

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        capability = task_context.capability
        supported = set(CAPABILITIES)
        if capability is None or capability in supported:
            return CapabilityMatchResult(
                matched=True,
                agent_id="vendor_discovery",
                matched_capabilities=list(supported),
                score=1.0,
                rationale="vendor discovery capability",
            )
        return CapabilityMatchResult(
            matched=False, rationale="not a vendor discovery capability"
        )

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        return build_lab_agent_runtime_context(
            request=request,
            llm_adapter=PrefixStubLLMAdapter(prefix="vendor_discovery"),
            harness=self._harness,
            enable_websearch=True,
        )

    async def perceive(self, step_ctx: AgentStepContext):
        return perceive_run_input(step_ctx, self)

    async def reason(self, step_ctx: AgentStepContext, observation):
        return reason_passthrough(step_ctx, observation)

    async def act(self, step_ctx: AgentStepContext, reasoning):
        query = (reasoning.thought or "").strip()
        payload = build_stub_vendor_discovery_output(query).model_dump_json(indent=2)
        return {"summary": payload, "answer": payload, "run_id": step_ctx.run_id}

    def evaluate(self, step_ctx: AgentStepContext, output: dict[str, object]):
        return evaluate_complete(step_ctx, output, reason="vendor_discovery_goal_met")
