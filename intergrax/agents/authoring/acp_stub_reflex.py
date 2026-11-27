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
from intergrax.runtime.execution.agent_runtime_context import RuntimeContext
from intergrax.runtime.execution.agent_runtime_io import RuntimeRequest
from intergrax.runtime.execution.lab_reference_agent_runtime import build_stub_agent_runtime_context


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
    production_mode: bool = False,
) -> RuntimeContext:
    """Reference-stub runtime context; delegates Tier-1 materialization (FC-1 Part C1)."""
    return build_stub_agent_runtime_context(
        request,
        llm_adapter,
        enable_rag=enable_rag,
        enable_websearch=enable_websearch,
        production_mode=production_mode,
    )


def summary_act_output(step_ctx: AgentStepContext, reasoning: ReasoningResult) -> dict[str, object]:
    raw = (reasoning.thought or "").strip()
    if "--- prior agent outputs ---" in raw:
        _, _, prior = raw.partition("--- prior agent outputs ---")
        summary = f"summary: {prior.strip()[:800]}"
    else:
        summary = f"summary: {raw[:800]}"
    return {"summary": summary, "answer": summary, "run_id": step_ctx.run_id}
