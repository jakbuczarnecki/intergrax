# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.agents.authoring.acp_stub_reflex import (
    build_agent_runtime_context,
    evaluate_complete,
    perceive_run_input,
    reason_passthrough,
)
from intergrax.agents.authoring.patterns.diagnostic_reflex import DiagnosticReflexAgent
from intergrax.contracts.agent_run import AgentRunResult
from intergrax.contracts.agent_run_enums import CognitivePattern
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.registry.profile import llm_profile_from_env
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload
from intergrax.runtime.task.task import TaskContext
from web_search_qualifier.capabilities import CAPABILITIES
from web_search_qualifier.contract import build_agent_contract
from web_search_qualifier.diagnostics import web_search_diagnostic_from_output
from web_search_qualifier.steps.web_search_job import run_web_search_job


class WebSearchQualifierAgent(DiagnosticReflexAgent):
    """LKW qualification agent for real web search pipeline (DIAG-FUNCTIONAL-Q3)."""

    contract_id = "web_search_qualifier"
    capabilities = tuple(CAPABILITIES)
    cognitive_pattern = CognitivePattern.REFLEX
    main_step_id = "web_search_qualifier_step"

    def __init__(self, llm_adapter: LLMAdapter | None = None) -> None:
        self._llm_adapter = llm_adapter

    def get_contract(self):
        return build_agent_contract()

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        capability = task_context.capability
        supported = set(CAPABILITIES)
        if capability is None or capability in supported:
            return CapabilityMatchResult(
                matched=True,
                agent_id=self.contract_id,
                matched_capabilities=list(supported),
                score=1.0,
                rationale="web search qualification capability",
            )
        return CapabilityMatchResult(matched=False, rationale="capability not supported")

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        adapter = self._llm_adapter
        if adapter is None:
            adapter = llm_profile_from_env().create_adapter()
        return build_agent_runtime_context(request, adapter)

    async def perceive(self, step_ctx: AgentStepContext):
        return perceive_run_input(step_ctx, self)

    async def reason(self, step_ctx: AgentStepContext, observation):
        return reason_passthrough(step_ctx, observation)

    async def act(self, step_ctx: AgentStepContext, reasoning):
        _ = reasoning
        return await run_web_search_job(step_ctx)

    def evaluate(self, step_ctx: AgentStepContext, output: dict[str, object]):
        return evaluate_complete(step_ctx, output, reason="web_search_qualification_complete")

    def build_diagnostic_payloads(self, output: dict[str, object]) -> list[DiagnosticPayload]:
        return [web_search_diagnostic_from_output(output)]

    async def on_run_end(self, result: AgentRunResult) -> None:
        output = result.output
        if not isinstance(output, dict):
            return
        summary = output.get("web_search_summary")
        if isinstance(summary, dict) and summary:
            result.structured_data["web_search_summary"] = dict(summary)
