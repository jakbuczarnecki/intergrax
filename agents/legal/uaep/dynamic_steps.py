# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Thin UAEP macro-steps for Legal dynamic pipeline (Phase E.4).

Routing/replan waves stay inside ``legal_dynamic_waves``; UAEP exposes setup,
tool plan, route, waves, and finalize as separate runtime-controlled boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Set

from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from legal.config.legal_agent_config import LegalAgentConfig
from legal.domain.legal_agent_state import LegalAgentState
from legal.memory.legal_memory_policy import resolve_session_prior_workspace_snapshot
from legal.pipeline.legal_execution_loop import (
    run_legal_dynamic_waves_phase,
    run_legal_finalize_phase,
    run_legal_initial_route_phase,
    run_legal_tool_plan_phase,
)
from legal.pipeline.legal_pipeline_routing import LegalRoutingResult
from legal.uaep.thin_steps import get_or_create_runtime_state
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer
from intergrax.runtime.nexus.runtime_steps.contract import RuntimeStepRunner
from intergrax.runtime.nexus.runtime_steps.setup_steps_tool import SETUP_STEPS

_ROUTING_KEY = "legal_dynamic_routing"
_COMPLETED_KEY = "legal_dynamic_completed"


@dataclass(frozen=True)
class LegalDynamicUAEPStepDef:
    step_id: str
    step_name: str
    trace_label: str


LEGAL_DYNAMIC_STEP_DEFS: tuple[LegalDynamicUAEPStepDef, ...] = (
    LegalDynamicUAEPStepDef("legal_setup_dynamic", "legal_setup_dynamic", "legal.setup"),
    LegalDynamicUAEPStepDef("legal_tool_plan", "legal_tool_plan", "legal.tool_plan"),
    LegalDynamicUAEPStepDef("legal_route", "legal_route", "legal.route"),
    LegalDynamicUAEPStepDef("legal_dynamic_waves", "legal_dynamic_waves", "legal.waves"),
    LegalDynamicUAEPStepDef(
        "legal_finalize_answer",
        "LegalFinalizeAnswerStep",
        "legal.finalize_answer",
    ),
)

FINAL_DYNAMIC_STEP_ID = LEGAL_DYNAMIC_STEP_DEFS[-1].step_id


def legal_dynamic_agent_steps(
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
        for index, defn in enumerate(LEGAL_DYNAMIC_STEP_DEFS)
    ]


def _require_agent_state(state: RuntimeState) -> LegalAgentState:
    if not isinstance(state.agent_state, LegalAgentState):
        raise TypeError("state.agent_state must be LegalAgentState for Legal dynamic UAEP.")
    return state.agent_state


async def _run_setup_dynamic(state: RuntimeState, config: LegalAgentConfig) -> None:
    await RuntimeStepRunner.execute_pipeline(SETUP_STEPS, state)
    prior_snapshot = resolve_session_prior_workspace_snapshot(
        session=state.session,
        request=state.request,
        policy=config.memory_policy,
    )
    if state.agent_state is None:
        state.agent_state = LegalAgentState(
            config=config,
            session_prior_workspace_snapshot=prior_snapshot,
        )
    elif not isinstance(state.agent_state, LegalAgentState):
        raise TypeError("state.agent_state must be LegalAgentState for Legal dynamic UAEP.")
    else:
        state.agent_state = state.agent_state.model_copy(
            update={"session_prior_workspace_snapshot": prior_snapshot},
        )


def _get_completed(ctx: RuntimeExecutionContext) -> Set[str]:
    raw = ctx.metadata.get(_COMPLETED_KEY)
    if isinstance(raw, set):
        return raw
    completed: Set[str] = set()
    ctx.metadata[_COMPLETED_KEY] = completed
    return completed


def _get_routing(ctx: RuntimeExecutionContext) -> LegalRoutingResult:
    routing = ctx.metadata.get(_ROUTING_KEY)
    if not isinstance(routing, LegalRoutingResult):
        raise RuntimeError("legal_dynamic_routing missing; run legal_route first.")
    return routing


async def run_dynamic_step_by_id(
    step_id: str,
    state: RuntimeState,
    ctx: RuntimeExecutionContext,
    *,
    config: LegalAgentConfig,
) -> None:
    if step_id == "legal_setup_dynamic":
        await _run_setup_dynamic(state, config)
        return

    agent_state = _require_agent_state(state)

    if step_id == "legal_tool_plan":
        await run_legal_tool_plan_phase(
            state=state,
            agent_state=agent_state,
            config=config,
        )
        return

    if step_id == "legal_route":
        routing = await run_legal_initial_route_phase(
            state=state,
            agent_state=agent_state,
            config=config,
        )
        ctx.metadata[_ROUTING_KEY] = routing
        return

    if step_id == "legal_dynamic_waves":
        routing = _get_routing(ctx)
        completed = _get_completed(ctx)
        final_routing = await run_legal_dynamic_waves_phase(
            state=state,
            agent_state=agent_state,
            config=config,
            routing=routing,
            completed=completed,
        )
        ctx.metadata[_ROUTING_KEY] = final_routing
        return

    if step_id == "legal_finalize_answer":
        await run_legal_finalize_phase(state=state)
        return

    raise ValueError(f"Unknown legal dynamic UAEP step_id: {step_id}")


async def run_dynamic_pipeline_on_state(
    state: RuntimeState,
    *,
    config: LegalAgentConfig,
) -> RuntimeAnswer:
    await _run_setup_dynamic(state, config)
    agent_state = _require_agent_state(state)
    await run_legal_tool_plan_phase(
        state=state,
        agent_state=agent_state,
        config=config,
    )
    routing = await run_legal_initial_route_phase(
        state=state,
        agent_state=agent_state,
        config=config,
    )
    await run_legal_dynamic_waves_phase(
        state=state,
        agent_state=agent_state,
        config=config,
        routing=routing,
    )
    await run_legal_finalize_phase(state=state)
    if state.runtime_answer is None:
        raise RuntimeError("LegalFinalizeAnswerStep did not set state.runtime_answer.")
    return state.runtime_answer


def _dynamic_step_summary(step_id: str, state: RuntimeState) -> str:
    if step_id == "legal_finalize_answer" and state.runtime_answer is not None:
        return (state.runtime_answer.answer or "").strip()
    if step_id == "legal_setup_dynamic":
        return "dynamic session and history ready"
    agent_state = state.agent_state
    if isinstance(agent_state, LegalAgentState):
        if step_id == "legal_tool_plan" and agent_state.last_legal_tool_plan is not None:
            return f"tool_intent={agent_state.last_legal_tool_plan.intent}"
        if step_id == "legal_dynamic_waves":
            waves = agent_state.legal_dynamic_loop_waves
            stages = len(agent_state.legal_stages_completed_this_run)
            return f"waves={waves} stages={stages}"
    return step_id.replace("_", " ")


async def run_legal_dynamic_uaep_step(
    step: AgentStep,
    ctx: RuntimeExecutionContext,
    *,
    config: LegalAgentConfig,
) -> StepOutput:
    state = get_or_create_runtime_state(ctx)
    await run_dynamic_step_by_id(step.step_id, state, ctx, config=config)

    if step.step_id == FINAL_DYNAMIC_STEP_ID and state.runtime_answer is not None:
        ctx.metadata["runtime_answer"] = state.runtime_answer

    return StepOutput(
        step_id=step.step_id,
        summary=_dynamic_step_summary(step.step_id, state),
        data={"run_id": ctx.run_id, "legal_step": step.step_id},
    )
