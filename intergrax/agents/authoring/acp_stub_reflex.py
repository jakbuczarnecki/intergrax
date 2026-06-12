# © Artur Czarnecki. All rights reserved.

"""Shared Reflex stubs for fleet migration (ACP-MIG-4) — preserves legacy answer shapes."""

from __future__ import annotations

from intergrax.agents.authoring.patterns.types import (
    AgentEvaluation,
    CognitiveEvaluation,
    Observation,
    ReasoningResult,
)
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager


def perceive_run_input(step_ctx: AgentStepContext, agent: object) -> Observation:
    message = agent.read_run_input(step_ctx)  # type: ignore[attr-defined]
    return Observation(summary=message or "")


def reason_passthrough(
    step_ctx: AgentStepContext,
    observation: Observation,
) -> ReasoningResult:
    _ = step_ctx
    return ReasoningResult(thought=observation.summary)


def evaluate_complete(step_ctx: AgentStepContext, output: dict[str, object], *, reason: str) -> AgentEvaluation:
    _ = step_ctx, output
    return AgentEvaluation(verdict=CognitiveEvaluation.COMPLETE, reason=reason)


def prefixed_act_output(
    *,
    prefix: str,
    step_ctx: AgentStepContext,
    reasoning: ReasoningResult,
) -> dict[str, object]:
    message = (reasoning.thought or "").strip()
    answer = f"{prefix}: {message}"
    return {"summary": answer, "answer": answer, "run_id": step_ctx.run_id}


def build_agent_runtime_context(
    request: RuntimeRequest,
    llm_adapter: LLMAdapter,
    *,
    enable_rag: bool = False,
    enable_websearch: bool = False,
) -> RuntimeContext:
    from intergrax.agents.defaults import harness_production_mode

    config = RuntimeConfig(
        llm_adapter=llm_adapter,
        enable_rag=enable_rag,
        enable_websearch=enable_websearch,
        production_mode=harness_production_mode(),
        tenant_id=request.tenant_id,
    )
    session_manager = SessionManager(storage=InMemorySessionStorage())
    return RuntimeContext.build(config=config, session_manager=session_manager)


def summary_act_output(step_ctx: AgentStepContext, reasoning: ReasoningResult) -> dict[str, object]:
    raw = (reasoning.thought or "").strip()
    if "--- prior agent outputs ---" in raw:
        _, _, prior = raw.partition("--- prior agent outputs ---")
        summary = f"summary: {prior.strip()[:800]}"
    else:
        summary = f"summary: {raw[:800]}"
    return {"summary": summary, "answer": summary, "run_id": step_ctx.run_id}
