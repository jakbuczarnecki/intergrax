# © Artur Czarnecki. All rights reserved.

"""UAEP-compatible gate stubs for graph, orchestration, and debug intake tests."""

from __future__ import annotations
from intergrax.utils import attribute_access

from typing import Any

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_handoff import AgentHandoff
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager


class UaepPipelineStubAgent(Agent):
    """UAEP gate stub with deterministic prefix answers (no legacy pipeline)."""

    run_count: int = 0
    run_log: list[str] = []

    def __init__(
        self,
        *,
        agent_id: str,
        capability: str,
        prefix: str | None = None,
        description: str = "UAEP gate stub",
        answer_separator: str = ": ",
        extra_capabilities: tuple[str, ...] = (),
        always_match: bool = False,
        route_extra: dict[str, object] | None = None,
        track_request_metadata: bool = False,
    ) -> None:
        self._agent_id = agent_id
        self._capability = capability
        self._prefix = prefix if prefix is not None else agent_id
        self._description = description
        self._answer_separator = answer_separator
        self._extra_capabilities = extra_capabilities
        self._always_match = always_match
        self._route_extra = dict(route_extra or {})
        self._track_request_metadata = track_request_metadata
        self.last_metadata: dict[str, Any] = {}
        self.last_request: RuntimeRequest | None = None

    def get_contract(self) -> AgentContract:
        capabilities = list(dict.fromkeys([self._capability, *self._extra_capabilities]))
        return AgentContract(
            id=self._agent_id,
            name=self._agent_id,
            description=self._description,
            capabilities=capabilities,
        )

    def can_handle(self, task_context: object) -> CapabilityMatchResult:
        if self._always_match:
            return CapabilityMatchResult(
                matched=True,
                agent_id=self._agent_id,
                matched_capabilities=[self._capability],
                score=1.0,
            )
        capability = attribute_access.optional(task_context, "capability", None)
        allowed = {self._capability, *self._extra_capabilities, None}
        if capability in allowed:
            return CapabilityMatchResult(
                matched=True,
                agent_id=self._agent_id,
                matched_capabilities=[self._capability],
                score=1.0,
            )
        return CapabilityMatchResult(matched=False)

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        UaepPipelineStubAgent.run_count += 1
        UaepPipelineStubAgent.run_log.append(self._agent_id)
        if self._track_request_metadata:
            self.last_metadata = dict(request.metadata)
            self.last_request = request
        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(
                fixed_text=f"{self._prefix}{self._answer_separator}{request.message}"
            ),
            enable_rag=False,
            production_mode=False,
            tenant_id=request.tenant_id,
        )
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )

    def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
        _ = context
        return [
            AgentStep(
                step_id=f"{self._agent_id}_step",
                step_name=f"{self._agent_id}_step",
                step_index=0,
                trace_label=self._capability,
            )
        ]

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        message = (ctx.request.message or "") if ctx.request is not None else ""
        answer = f"{self._prefix}{self._answer_separator}{message}"
        return StepOutput(step_id=step.step_id, summary=answer, data={"answer": answer})

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step, output, ctx
        raw_handoff = self._route_extra.get("pending_handoff")
        if raw_handoff is not None:
            handoff = (
                raw_handoff
                if isinstance(raw_handoff, AgentHandoff)
                else AgentHandoff.model_validate(raw_handoff)
            )
            return AgentDecision(
                type=AgentDecisionType.MODIFY_PLAN,
                reason="gate stub handoff",
                handoff=handoff,
            )
        return AgentDecision(type=AgentDecisionType.COMPLETE, reason=f"{self._agent_id} gate stub finished")
