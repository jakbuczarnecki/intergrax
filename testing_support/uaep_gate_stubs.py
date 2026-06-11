# © Artur Czarnecki. All rights reserved.

"""UAEP-compatible gate stubs for graph, orchestration, and debug intake tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from intergrax.agents.agent_contract import Agent
from intergrax.agents.authoring.uaep_pipeline_bridge import (
    pipeline_agent_steps,
    pipeline_step_complete,
    run_pipeline_step,
)
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_decision import AgentDecision
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer, RuntimeRequest, RouteInfo
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager


def build_prefix_answer_pipeline(
    prefix: str,
    *,
    separator: str = ": ",
    route_extra: dict[str, object] | None = None,
) -> RuntimePipeline:
    """Return a one-shot pipeline that answers ``{prefix}{separator}{message}``."""

    class _PrefixAnswerPipeline(RuntimePipeline):
        async def _inner_run(self, state: RuntimeState) -> RuntimeAnswer:
            answer = f"{prefix}{separator}{state.request.message}"
            state.raw_answer = answer
            route = RouteInfo(extra=dict(route_extra or {})) if route_extra else RouteInfo()
            state.runtime_answer = RuntimeAnswer(
                run_id=state.run_id,
                answer=answer,
                route=route,
            )
            return state.runtime_answer

    return _PrefixAnswerPipeline()


class UaepPipelineStubAgent(Agent):
    """
    UAEP gate stub backed by a single internal Nexus pipeline step.

    Replaces legacy RuntimeEngine-only stubs removed by ACP-CLOSE-LEG-1.
    """

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
        pipeline_factory: Callable[[RuntimeRequest], RuntimePipeline] | None = None,
        track_request_metadata: bool = False,
    ) -> None:
        self._agent_id = agent_id
        self._capability = capability
        self._prefix = prefix if prefix is not None else agent_id
        self._description = description
        self._answer_separator = answer_separator
        self._extra_capabilities = extra_capabilities
        self._always_match = always_match
        self._route_extra = route_extra
        self._pipeline_factory = pipeline_factory
        self._track_request_metadata = track_request_metadata
        self.last_metadata: dict[str, Any] = {}

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
        capability = getattr(task_context, "capability", None)
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
        if self._pipeline_factory is not None:
            pipeline = self._pipeline_factory(request)
        else:
            pipeline = build_prefix_answer_pipeline(
                self._prefix,
                separator=self._answer_separator,
                route_extra=self._route_extra,
            )
        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(fixed_text=f"{self._prefix}{self._answer_separator}{request.message}"),
            enable_rag=False,
            production_mode=False,
            tenant_id=request.tenant_id,
        )
        config.pipeline = pipeline
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )

    def get_steps(self, context: RuntimeContext) -> list[AgentStep]:
        _ = context
        return pipeline_agent_steps(
            step_id=f"{self._agent_id}_pipeline",
            step_name=f"{self._agent_id}_pipeline",
            trace_label=self._capability,
        )

    async def run_step(self, step: AgentStep, ctx: RuntimeExecutionContext) -> StepOutput:
        return await run_pipeline_step(step, ctx)

    def decide_after_step(
        self,
        step: AgentStep,
        output: StepOutput | None,
        ctx: RuntimeExecutionContext,
    ) -> AgentDecision:
        _ = step, output, ctx
        return pipeline_step_complete(reason=f"{self._agent_id} gate stub finished")
