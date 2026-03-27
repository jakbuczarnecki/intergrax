# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Production-style controller for LegalDynamicPipeline: plan → execute stages →
optional LLM evaluation → bounded replan (routing union) → finalize.
"""

from __future__ import annotations

import json
from typing import Any, List, Set

from intergrax.agents_packages.legal_agent.config.legal_agent_config import LegalAgentConfig
from intergrax.agents_packages.legal_agent.prompts.legal_agent_llm_prompts import (
    LEGAL_RUN_EVALUATION_SYSTEM,
    legal_run_evaluation_user,
)
from intergrax.agents_packages.legal_agent.domain.legal_agent_state import LegalAgentState
from intergrax.agents_packages.legal_agent.domain.legal_dynamic_execution_gates import (
    LegalDynamicLoopGates,
)
from intergrax.agents_packages.legal_agent.pipeline.legal_pipeline_routing import (
    LegalEvaluationResult,
    LegalPipelineRouting,
    LegalRoutingResult,
)
from intergrax.agents_packages.legal_agent.governance.legal_tool_plan_governance import (
    enforce_legal_tool_plan_governance,
)
from intergrax.agents_packages.legal_agent.runtime.legal_tool_runtime_bridge import (
    run_legal_tool_runtime_bridge,
    sync_legal_tool_runtime_feedback,
)
from intergrax.agents_packages.legal_agent.runtime.tool_decision_component import (
    decide_legal_tool_plan,
)
from intergrax.agents_packages.legal_agent.steps.legal_finalize_answer_step import (
    LegalFinalizeAnswerStep,
)
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel


async def _evaluate_legal_run_llm(
    *,
    state: RuntimeState,
    routing: LegalRoutingResult,
    agent_state: LegalAgentState,
) -> LegalEvaluationResult:
    llm = state.context.config.llm_adapter
    metrics = LegalPipelineRouting.workspace_metrics_json(agent_state, runtime_state=state)
    stages = ", ".join(agent_state.legal_stages_completed_this_run) or "(none)"
    user = legal_run_evaluation_user(
        user_message=(state.request.message or "").strip(),
        workspace_metrics_json=metrics,
        stages_completed=stages,
        current_routing_json=json.dumps(routing.model_dump(), sort_keys=True),
    )
    messages = [
        ChatMessage(role="system", content=LEGAL_RUN_EVALUATION_SYSTEM),
        ChatMessage(role="user", content=user),
    ]
    try:
        raw = llm.generate_structured(
            messages,
            LegalEvaluationResult,
            run_id=state.run_id,
        )
        if isinstance(raw, LegalEvaluationResult):
            return raw
    except Exception:
        agent_state.legal_run_evaluator_degraded = True
        return LegalEvaluationResult(
            complete=True,
            replan=False,
            rationale="evaluator LLM failed; assuming complete",
        )
    agent_state.legal_run_evaluator_degraded = True
    return LegalEvaluationResult(
        complete=True,
        replan=False,
        rationale="evaluator LLM failed; assuming complete",
    )


async def run_legal_dynamic_execution_loop(
    *,
    state: RuntimeState,
    agent_state: LegalAgentState,
    config: LegalAgentConfig,
) -> None:
    """
    Runs routed stages (possibly over several iterations), then
    :class:`LegalFinalizeAnswerStep`. Mutates ``agent_state`` and ``state``.

    At the start of each loop: tool plan from LLM → static org governance
    (:func:`~intergrax.agents_packages.legal_agent.governance.legal_tool_plan_governance.enforce_legal_tool_plan_governance`)
    → optional :attr:`~intergrax.agents_packages.legal_agent.config.legal_agent_config.LegalAgentConfig.legal_tool_plan_governance`
    → Nexus bridge. ``last_legal_tool_plan`` stores the **final** plan after all clamps.
    """
    agent_state.legal_stages_completed_this_run = []
    completed: Set[str] = set()
    agent_state.legal_dynamic_loop_waves = 0
    agent_state.legal_run_evaluator_degraded = False
    agent_state.clause_extraction_retrieval_outcome = None

    tool_plan = await decide_legal_tool_plan(state=state, legal_config=config)
    tool_plan = enforce_legal_tool_plan_governance(
        plan=tool_plan,
        state=state,
        legal_config=config,
    )
    if config.legal_tool_plan_governance is not None:
        tool_plan = config.legal_tool_plan_governance.adjust_legal_tool_plan(
            tool_plan,
            state=state,
            legal_config=config,
        )
    agent_state.last_legal_tool_plan = tool_plan
    await run_legal_tool_runtime_bridge(state=state, plan=tool_plan)
    sync_legal_tool_runtime_feedback(agent_state, state)

    routing = await LegalPipelineRouting.obtain_initial(
        state=state,
        config=config,
        agent_state=agent_state,
    )
    last_fp: str | None = LegalPipelineRouting.routing_fingerprint(routing)
    same_fp = 0
    max_iter = config.legal_loop_max_iterations
    max_same = config.legal_loop_max_same_routing_repeats

    state.trace_event(
        component=TraceComponent.PIPELINE,
        step="LegalExecutionLoop",
        message=f"start fingerprint={last_fp}",
        level=TraceLevel.INFO,
    )

    for iteration in range(max_iter):
        stage_runners = LegalPipelineRouting.build_step_runners(
            routing,
            include_finalize=False,
        )
        to_run: List[Any] = []
        for step in stage_runners:
            flag = LegalPipelineRouting.stage_flag_for_runner(step)
            if flag is None or flag not in completed:
                to_run.append(step)

        flags_this_wave = [LegalPipelineRouting.stage_flag_for_runner(s) for s in to_run]
        flags_this_wave = [f for f in flags_this_wave if f]

        state.trace_event(
            component=TraceComponent.PIPELINE,
            step="LegalExecutionLoop",
            message=(
                f"wave={iteration + 1}/{max_iter} run_steps={flags_this_wave or ['(none)']} "
                f"fp={LegalPipelineRouting.routing_fingerprint(routing)}"
            ),
            level=TraceLevel.INFO,
        )

        for step in to_run:
            await step.run(state=state)
            flag = LegalPipelineRouting.stage_flag_for_runner(step)
            if flag:
                completed.add(flag)
                agent_state.legal_stages_completed_this_run.append(flag)

        sync_legal_tool_runtime_feedback(agent_state, state)
        agent_state.legal_dynamic_loop_waves = iteration + 1

        if not config.use_legal_run_evaluator:
            break

        if (
            config.legal_loop_early_exit
            and to_run
            and LegalDynamicLoopGates.post_wave_early_exit_ok(agent_state, completed, config)
        ):
            state.trace_event(
                component=TraceComponent.PIPELINE,
                step="LegalExecutionLoop",
                message=(
                    "early exit: decision confidence and policy/blocking gates satisfied; "
                    "skipping evaluator/replan"
                ),
                level=TraceLevel.INFO,
            )
            break

        ev = await _evaluate_legal_run_llm(
            state=state,
            routing=routing,
            agent_state=agent_state,
        )

        state.trace_event(
            component=TraceComponent.PIPELINE,
            step="LegalExecutionLoop",
            message=(
                f"eval complete={ev.complete} replan={ev.replan} "
                f"missing={ev.missing_aspects!r}"
            ),
            level=TraceLevel.INFO,
        )

        if ev.complete or not ev.replan:
            break

        if iteration >= max_iter - 1:
            break

        routing = await LegalPipelineRouting.obtain_replan(
            state=state,
            config=config,
            agent_state=agent_state,
            prior=routing,
            iteration=iteration + 1,
            evaluation_rationale=ev.rationale,
            missing_aspects=list(ev.missing_aspects or []),
            workspace_metrics_json=LegalPipelineRouting.workspace_metrics_json(
                agent_state,
                runtime_state=state,
            ),
        )

        fp = LegalPipelineRouting.routing_fingerprint(routing)
        if fp == last_fp:
            same_fp += 1
        else:
            same_fp = 0
        last_fp = fp

        if same_fp >= max_same:
            state.trace_event(
                component=TraceComponent.PIPELINE,
                step="LegalExecutionLoop",
                message=(
                    f"stop replan: same routing fingerprint repeated "
                    f"({same_fp} >= {max_same})"
                ),
                level=TraceLevel.WARNING,
            )
            break

    finalize = LegalFinalizeAnswerStep()
    await finalize.run(state=state)
