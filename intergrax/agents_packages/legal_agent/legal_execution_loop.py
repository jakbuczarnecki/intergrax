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

from intergrax.agents_packages.legal_agent.legal_agent_config import LegalAgentConfig
from intergrax.agents_packages.legal_agent.legal_agent_llm_prompts import (
    LEGAL_RUN_EVALUATION_SYSTEM,
    legal_run_evaluation_user,
)
from intergrax.agents_packages.legal_agent.legal_agent_state import LegalAgentState
from intergrax.agents_packages.legal_agent.legal_pipeline_routing import (
    LegalEvaluationResult,
    LegalRoutingResult,
    build_legal_step_runners_from_routing,
    legal_routing_fingerprint,
    legal_stage_flag_for_runner,
    obtain_initial_legal_routing,
    obtain_replan_legal_routing,
)
from intergrax.agents_packages.legal_agent.steps.legal_finalize_answer_step import (
    LegalFinalizeAnswerStep,
)
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel


def legal_workspace_metrics_json(agent_state: LegalAgentState) -> str:
    """Compact JSON for evaluator / replanner (no clause bodies)."""
    pol_v = agent_state.policy_violations
    payload = {
        "clause_count": len(agent_state.clauses),
        "sensitive_flag_count": len(agent_state.sensitive_flags),
        "legal_check_count": len(agent_state.legal_checks),
        "compliance_result_count": len(agent_state.compliance_results),
        "policy_violation_count": len(pol_v) if pol_v else 0,
        "recommendation_count": len(agent_state.recommendations),
        "uncertainty_count": len(agent_state.uncertainties),
        "has_decision": agent_state.decision is not None,
        "decision_status": agent_state.decision.status if agent_state.decision else None,
        "decision_confidence": (
            agent_state.decision.confidence if agent_state.decision else None
        ),
        "blocking_issues_count": (
            len(agent_state.decision.blocking_issues)
            if agent_state.decision and agent_state.decision.blocking_issues
            else 0
        ),
        "decision_enforcement_modified": agent_state.decision_enforcement_modified,
        "final_opinion_present": agent_state.final_opinion is not None,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


async def _evaluate_legal_run_llm(
    *,
    state: RuntimeState,
    routing: LegalRoutingResult,
    agent_state: LegalAgentState,
) -> LegalEvaluationResult:
    llm = state.context.config.llm_adapter
    metrics = legal_workspace_metrics_json(agent_state)
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
        pass
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
    """
    agent_state.legal_stages_completed_this_run = []
    completed: Set[str] = set()

    routing = await obtain_initial_legal_routing(state=state, config=config)
    last_fp: str | None = legal_routing_fingerprint(routing)
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
        stage_runners = build_legal_step_runners_from_routing(
            routing,
            include_finalize=False,
        )
        to_run: List[Any] = []
        for step in stage_runners:
            flag = legal_stage_flag_for_runner(step)
            if flag is None or flag not in completed:
                to_run.append(step)

        flags_this_wave = [legal_stage_flag_for_runner(s) for s in to_run]
        flags_this_wave = [f for f in flags_this_wave if f]

        state.trace_event(
            component=TraceComponent.PIPELINE,
            step="LegalExecutionLoop",
            message=(
                f"wave={iteration + 1}/{max_iter} run_steps={flags_this_wave or ['(none)']} "
                f"fp={legal_routing_fingerprint(routing)}"
            ),
            level=TraceLevel.INFO,
        )

        for step in to_run:
            await step.run(state=state)
            flag = legal_stage_flag_for_runner(step)
            if flag:
                completed.add(flag)
                agent_state.legal_stages_completed_this_run.append(flag)

        if not config.use_legal_run_evaluator:
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

        routing = await obtain_replan_legal_routing(
            state=state,
            config=config,
            agent_state=agent_state,
            prior=routing,
            iteration=iteration + 1,
            evaluation_rationale=ev.rationale,
            missing_aspects=list(ev.missing_aspects or []),
            workspace_metrics_json=legal_workspace_metrics_json(agent_state),
        )

        fp = legal_routing_fingerprint(routing)
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
