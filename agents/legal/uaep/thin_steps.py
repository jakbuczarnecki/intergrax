# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Thin UAEP steps for Legal sequential analysis (Phase E).

Domain steps are exposed individually to AgentEngine instead of one monolithic
``RuntimeEngine.run()`` boundary. ``LegalAnalysisPipeline`` reuses the same
ordered runners for notebook / legacy RuntimeEngine callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.llm_adapters.tracking.llm_usage_track import LLMUsageTracker
from legal.config.legal_agent_config import LegalAgentConfig
from legal.domain.legal_agent_state import LegalAgentState
from legal.steps.legal_decision_enforcement_step import LegalDecisionEnforcementStep
from legal.steps.legal_decision_step import LegalDecisionStep
from legal.steps.legal_extract_clauses_step import LegalExtractClausesStep
from legal.steps.legal_finalize_answer_step import LegalFinalizeAnswerStep
from legal.steps.legal_normalize_clauses_step import LegalNormalizeClausesStep
from legal.steps.legal_recommendation_step import LegalRecommendationStep
from legal.steps.legal_risk_analysis_step import LegalRiskAnalysisStep
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer
from intergrax.runtime.nexus.runtime_steps.contract import RuntimeStepRunner
from intergrax.runtime.nexus.runtime_steps.setup_steps_tool import SETUP_STEPS

_RUNTIME_STATE_KEY = "legal_runtime_state"


@dataclass(frozen=True)
class LegalUAEPStepDef:
    step_id: str
    step_name: str
    trace_label: str


LEGAL_SEQUENTIAL_STEP_DEFS: tuple[LegalUAEPStepDef, ...] = (
    LegalUAEPStepDef("legal_setup", "legal_setup", "legal.setup"),
    LegalUAEPStepDef(
        "legal_extract_clauses",
        "LegalExtractClausesStep",
        "legal.extract_clauses",
    ),
    LegalUAEPStepDef(
        "legal_normalize_clauses",
        "LegalNormalizeClausesStep",
        "legal.normalize_clauses",
    ),
    LegalUAEPStepDef(
        "legal_risk_analysis",
        "LegalRiskAnalysisStep",
        "legal.risk_analysis",
    ),
    LegalUAEPStepDef(
        "legal_recommendation",
        "LegalRecommendationStep",
        "legal.recommendation",
    ),
    LegalUAEPStepDef("legal_decision", "LegalDecisionStep", "legal.decision"),
    LegalUAEPStepDef(
        "legal_decision_enforcement",
        "LegalDecisionEnforcementStep",
        "legal.decision_enforcement",
    ),
    LegalUAEPStepDef(
        "legal_finalize_answer",
        "LegalFinalizeAnswerStep",
        "legal.finalize_answer",
    ),
)

FINAL_SEQUENTIAL_STEP_ID = LEGAL_SEQUENTIAL_STEP_DEFS[-1].step_id


def legal_sequential_agent_steps(
    *,
    allowed_tools: Sequence[str] | None = None,
) -> list[AgentStep]:
    tools = list(allowed_tools or [])
    return [
        AgentStep(
            step_id=defn.step_id,
            step_name=defn.step_name,
            step_index=index,
            trace_label=defn.trace_label,
            allowed_tools=tools,
        )
        for index, defn in enumerate(LEGAL_SEQUENTIAL_STEP_DEFS)
    ]


def get_or_create_runtime_state(
    ctx: RuntimeExecutionContext,
) -> RuntimeState:
    existing = ctx.metadata.get(_RUNTIME_STATE_KEY)
    if isinstance(existing, RuntimeState):
        return existing

    request = ctx.request
    runtime_context = ctx.domain_context
    if request is None or runtime_context is None:
        raise RuntimeError("UAEP context missing request or domain_context.")

    state = RuntimeState(
        context=runtime_context,
        request=request,
        run_id=ctx.run_id,
        llm_usage_tracker=LLMUsageTracker(run_id=ctx.run_id),
    )
    state.configure_llm_tracker()
    ctx.metadata[_RUNTIME_STATE_KEY] = state
    return state


async def _run_setup(state: RuntimeState, config: LegalAgentConfig) -> None:
    await RuntimeStepRunner.execute_pipeline(SETUP_STEPS, state)
    if state.agent_state is None:
        state.agent_state = LegalAgentState(config=config)
    elif not isinstance(state.agent_state, LegalAgentState):
        raise TypeError("state.agent_state must be LegalAgentState for Legal UAEP.")


_DOMAIN_RUNNERS: dict[str, Callable[[], object]] = {
    "legal_extract_clauses": LegalExtractClausesStep,
    "legal_normalize_clauses": LegalNormalizeClausesStep,
    "legal_risk_analysis": LegalRiskAnalysisStep,
    "legal_recommendation": LegalRecommendationStep,
    "legal_decision": LegalDecisionStep,
    "legal_decision_enforcement": LegalDecisionEnforcementStep,
    "legal_finalize_answer": LegalFinalizeAnswerStep,
}


async def run_sequential_step_by_id(
    step_id: str,
    state: RuntimeState,
    *,
    config: LegalAgentConfig,
) -> None:
    if step_id == "legal_setup":
        await _run_setup(state, config)
        return

    runner_factory = _DOMAIN_RUNNERS.get(step_id)
    if runner_factory is None:
        raise ValueError(f"Unknown legal UAEP step_id: {step_id}")

    if state.agent_state is None:
        raise RuntimeError("Legal agent_state is not initialized; run legal_setup first.")

    step_impl = runner_factory()
    await step_impl.run(state=state)


async def run_sequential_pipeline_on_state(
    state: RuntimeState,
    *,
    config: LegalAgentConfig,
) -> RuntimeAnswer:
    for defn in LEGAL_SEQUENTIAL_STEP_DEFS:
        await run_sequential_step_by_id(defn.step_id, state, config=config)

    if state.runtime_answer is None:
        raise RuntimeError("LegalFinalizeAnswerStep did not set state.runtime_answer.")
    return state.runtime_answer


def _step_summary(step_id: str, state: RuntimeState) -> str:
    if step_id == "legal_finalize_answer" and state.runtime_answer is not None:
        return (state.runtime_answer.answer or "").strip()
    if step_id == "legal_setup":
        return "session and history ready"
    agent_state = state.agent_state
    if not isinstance(agent_state, LegalAgentState):
        return step_id
    if step_id == "legal_extract_clauses":
        return f"clauses={len(agent_state.clauses)}"
    if step_id == "legal_decision" and agent_state.decision is not None:
        return f"decision={agent_state.decision.status}"
    return step_id.replace("_", " ")


async def run_legal_uaep_step(
    step: AgentStep,
    ctx: RuntimeExecutionContext,
    *,
    config: LegalAgentConfig,
) -> StepOutput:
    state = get_or_create_runtime_state(ctx)
    await run_sequential_step_by_id(step.step_id, state, config=config)

    if step.step_id == FINAL_SEQUENTIAL_STEP_ID and state.runtime_answer is not None:
        ctx.metadata["runtime_answer"] = state.runtime_answer

    summary = _step_summary(step.step_id, state)
    return StepOutput(
        step_id=step.step_id,
        summary=summary,
        data={"run_id": ctx.run_id, "legal_step": step.step_id},
    )
