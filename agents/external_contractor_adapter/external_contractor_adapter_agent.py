# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from external_contractor_adapter.capabilities import CAPABILITIES
from external_contractor_adapter.contract import build_agent_contract
from intergrax.agents.authoring.patterns.reflex import ReflexAgent
from intergrax.agents.authoring.patterns.types import (
    AgentEvaluation,
    CognitiveEvaluation,
    Observation,
    ReasoningResult,
)
from intergrax.contracts.agent_contract_meta import AgentRiskLevel
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.memory.conversational_memory import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from typing import Optional, Sequence

# LLM / catalog (Tier-3 host): intergrax/llm_adapters/USAGE.md — LLMProfile, ModelCatalog,
# optional LLMRoutingProfile on ApplicationEnvironmentProfile; agents use stub LLM below only in tests.


class _ExternalContractorAdapterStubLLM(LLMAdapter):
    provider = "external_contractor_adapter"
    model = "external_contractor_adapter-stub"

    @property
    def context_window_tokens(self) -> int:
        return 128_000

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        for msg in reversed(messages):
            if msg.content:
                return build_adapter_response(content=msg.content[:200])
        return build_adapter_response(content="external_contractor_adapter: (empty)")


class ExternalContractorAdapterAgent(ReflexAgent):
    """Typed cognitive-pattern agent — implement hooks below (ACP §32.0)."""

    contract_id = "external_contractor_adapter"
    capabilities = tuple(CAPABILITIES)
    agent_name = "ExternalContractorAdapterAgent"
    agent_description = (
        "Tier-2 adapter for governed external contractor agents (GEC)"
    )
    risk_level = AgentRiskLevel.LOW
    max_steps = 10

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        from intergrax.agents.defaults import harness_production_mode

        config = RuntimeConfig(
            llm_adapter=_ExternalContractorAdapterStubLLM(),
            enable_rag=False,
            production_mode=harness_production_mode(),
            tenant_id=request.tenant_id,
        )
        session_manager = SessionManager(storage=InMemorySessionStorage())
        return RuntimeContext.build(config=config, session_manager=session_manager)
    async def perceive(self, step_ctx: AgentStepContext) -> Observation:
        _ = step_ctx
        return Observation(summary="TODO: domain perception")

    async def reason(
        self,
        step_ctx: AgentStepContext,
        observation: Observation,
    ) -> ReasoningResult:
        _ = step_ctx
        return ReasoningResult(thought=observation.summary)

    async def act(
        self,
        step_ctx: AgentStepContext,
        reasoning: ReasoningResult,
    ) -> dict[str, object]:
        _ = reasoning
        from external_contractor_adapter.steps.domain_job import run_domain_job
        return await run_domain_job(step_ctx)

    def evaluate(
        self,
        step_ctx: AgentStepContext,
        output: dict[str, object],
    ) -> AgentEvaluation:
        _ = step_ctx
        return AgentEvaluation(
            verdict=CognitiveEvaluation.COMPLETE,
            reason="reflex scaffold complete",
            confidence=0.9,
        )

