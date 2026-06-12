# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Runtime validation agents — stress-test registration, UAEP, and routing (Phase L.4).

These agents implement no business value; they return deterministic stubs for harness tests.
"""

from __future__ import annotations

from typing import Optional, Sequence

from intergrax.agents.harness_reference_agent import HarnessReferenceAgent
from intergrax.agents.reference_harness import (
    LabHarnessContext,
    build_lab_agent_runtime_context,
    default_reference_harness,
)
from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.memory.conversational_memory import ChatMessage
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.task import TaskContext


class _StubLLM(LLMAdapter):
    def __init__(self, *, provider: str, prefix: str) -> None:
        super().__init__()
        self.provider = provider
        self.model = f"{provider}-stub"
        self._prefix = prefix

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
        call = self.usage.begin_call(run_id=run_id)
        try:
            for msg in reversed(messages):
                content = msg.content or ""
                if content:
                    return build_adapter_response(content=f"{self._prefix}: {content[:120]}")
            return build_adapter_response(content=f"{self._prefix}: (empty)")
        finally:
            self.usage.end_call(call, input_tokens=0, output_tokens=1, success=True)


class _MockAgentBase(HarnessReferenceAgent):
    """UAEP mock agent with deterministic stub step output."""

    def __init__(
        self,
        harness: LabHarnessContext | None = None,
        *,
        agent_id: str,
        name: str,
        capability: str,
        prefix: str,
        provider: str,
    ) -> None:
        self._harness = harness or default_reference_harness()
        self._agent_id = agent_id
        self._name = name
        self._capability = capability
        self._prefix = prefix
        self._provider = provider

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id=self._agent_id,
            name=self._name,
            description=f"Runtime validation mock ({self._agent_id}).",
            version="0.1.0",
            capabilities=[self._capability],
            skills=[],
            extra_tools=[],
            risk_level=AgentRiskLevel.LOW,
            lifecycle_state=AgentLifecycleState.DEVELOPMENT,
            owner_team="platform",
            max_steps=5,
        )

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        capability = task_context.capability
        if capability in (None, self._capability):
            return CapabilityMatchResult(
                matched=True,
                agent_id=self._agent_id,
                matched_capabilities=[self._capability],
                score=1.0,
                rationale="mock capability match",
            )
        return CapabilityMatchResult(matched=False, rationale="capability not supported")

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        return build_lab_agent_runtime_context(
            request=request,
            llm_adapter=_StubLLM(provider=self._provider, prefix=self._prefix),
            harness=self._harness,
        )

    def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
        _ = context
        contract = self.get_contract()
        return [
            AgentStep(
                step_id=f"{self._agent_id}_step",
                step_name=f"{self._agent_id}_step",
                step_index=0,
                trace_label=self._capability,
                allowed_tools=list(contract.allowed_tools),
            )
        ]

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        message = ""
        if ctx.request is not None:
            message = (ctx.request.message or "").strip()
        answer = f"{self._prefix}: {message}"
        return StepOutput(
            step_id=step.step_id,
            summary=answer,
            data={"run_id": ctx.run_id, "answer": answer},
        )

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step, output, ctx
        return AgentDecision(type=AgentDecisionType.COMPLETE, reason=f"{self._agent_id} mock finished")


class ResearchMockAgent(_MockAgentBase):
    def __init__(self, harness: LabHarnessContext | None = None) -> None:
        super().__init__(
            harness,
            agent_id="research_mock",
            name="Research Mock Agent",
            capability="lab.research_mock",
            prefix="research-mock",
            provider="research_mock",
        )


class DocumentMockAgent(_MockAgentBase):
    def __init__(self, harness: LabHarnessContext | None = None) -> None:
        super().__init__(
            harness,
            agent_id="document_mock",
            name="Document Mock Agent",
            capability="lab.document_mock",
            prefix="document-mock",
            provider="document_mock",
        )


class ValidatorMockAgent(_MockAgentBase):
    def __init__(self, harness: LabHarnessContext | None = None) -> None:
        super().__init__(
            harness,
            agent_id="validator_mock",
            name="Validator Mock Agent",
            capability="lab.validator_mock",
            prefix="validator-mock",
            provider="validator_mock",
        )


class ComposerMockAgent(_MockAgentBase):
    def __init__(self, harness: LabHarnessContext | None = None) -> None:
        super().__init__(
            harness,
            agent_id="composer_mock",
            name="Composer Mock Agent",
            capability="lab.composer_mock",
            prefix="composer-mock",
            provider="composer_mock",
        )
