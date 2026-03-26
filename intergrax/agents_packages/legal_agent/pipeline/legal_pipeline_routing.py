# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
LLM-driven routing for Legal Agent pipeline stages.

Chooses which analysis steps to run for the current request. Dependencies are
closed deterministically (e.g. risk analysis implies clause extraction when
document context is required). Final synthesis always runs.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, Field

from intergrax.agents_packages.legal_agent.config.legal_agent_config import LegalAgentConfig
from intergrax.agents_packages.legal_agent.memory.legal_memory_policy import build_legal_conversation_snippet
from intergrax.agents_packages.legal_agent.prompts.legal_agent_llm_prompts import (
    LEGAL_PIPELINE_REPLAN_SYSTEM,
    LEGAL_PIPELINE_ROUTING_SYSTEM,
    legal_pipeline_replan_user,
    legal_pipeline_routing_user,
)
from intergrax.agents_packages.legal_agent.domain.legal_agent_state import LegalAgentState
from intergrax.agents_packages.legal_agent.steps.legal_decision_enforcement_step import (
    LegalDecisionEnforcementStep,
)
from intergrax.agents_packages.legal_agent.steps.legal_decision_step import LegalDecisionStep
from intergrax.agents_packages.legal_agent.steps.legal_extract_clauses_step import (
    LegalExtractClausesStep,
)
from intergrax.agents_packages.legal_agent.steps.legal_finalize_answer_step import (
    LegalFinalizeAnswerStep,
)
from intergrax.agents_packages.legal_agent.steps.legal_normalize_clauses_step import (
    LegalNormalizeClausesStep,
)
from intergrax.agents_packages.legal_agent.steps.legal_policy_compliance_step import (
    LegalPolicyComplianceStep,
)
from intergrax.agents_packages.legal_agent.steps.legal_recommendation_step import (
    LegalRecommendationStep,
)
from intergrax.agents_packages.legal_agent.steps.legal_risk_analysis_step import (
    LegalRiskAnalysisStep,
)
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel


class LegalRoutingResult(BaseModel):
    """
    Per-stage toggles from the routing model. ``LegalFinalizeAnswerStep`` is always
    appended by the pipeline (not part of this schema).
    """

    run_extract: bool = Field(default=True, description="Ingest / RAG clause extraction")
    run_normalize: bool = Field(default=True, description="Merge / dedupe clauses")
    run_policy_compliance: bool = Field(
        default=True,
        description="Check clauses vs organization policy (skipped in-step if policy empty)",
    )
    run_risk_analysis: bool = Field(default=True, description="Legal checks + sensitive flags")
    run_recommendations: bool = Field(default=True, description="Structured recommendations")
    run_decision: bool = Field(default=True, description="LLM decision status")
    run_enforcement: bool = Field(default=True, description="Deterministic policy guardrails")


class LegalEvaluationResult(BaseModel):
    """LLM verdict after a batch of legal stages (no replan details — see rationale)."""

    complete: bool = Field(
        default=True,
        description="True if the run is ready for finalize / user answer.",
    )
    replan: bool = Field(
        default=False,
        description="True if more stages should run before finalize.",
    )
    missing_aspects: List[str] = Field(default_factory=list)
    rationale: str = ""


def legal_workspace_metrics_json(
    agent_state: LegalAgentState,
    *,
    runtime_state: Optional[RuntimeState] = None,
) -> str:
    """Compact JSON for routing, evaluator, and replanner (no clause bodies)."""
    pol_v = agent_state.policy_violations
    payload: Dict[str, Any] = {
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

    p = agent_state.last_legal_tool_plan
    if p is not None:
        payload["legal_tool_intent"] = p.intent
        payload["legal_tool_confidence"] = p.confidence
        payload["legal_tool_requested_rag"] = p.use_rag
        payload["legal_tool_requested_websearch"] = p.use_websearch
        payload["legal_tool_requested_tools"] = p.use_tools

    if agent_state.legal_tool_runtime_feedback_json:
        try:
            payload["legal_tool_runtime_feedback"] = json.loads(
                agent_state.legal_tool_runtime_feedback_json
            )
        except json.JSONDecodeError:
            payload["legal_tool_runtime_feedback"] = agent_state.legal_tool_runtime_feedback_json

    if runtime_state is not None:
        payload["runtime_used_rag"] = runtime_state.used_rag
        payload["runtime_used_websearch"] = runtime_state.used_websearch
        payload["runtime_used_tools"] = runtime_state.used_tools
        payload["runtime_tool_trace_count"] = len(runtime_state.tool_traces or [])

    prior = agent_state.session_prior_workspace_snapshot
    if prior is not None:
        payload["session_prior_legal_run"] = prior.model_dump()

    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


# (flag attribute name, step factory) — execution order
_LEGAL_STAGE_FACTORIES: Tuple[Tuple[str, Callable[[], Any]], ...] = (
    ("run_extract", lambda: LegalExtractClausesStep()),
    ("run_normalize", lambda: LegalNormalizeClausesStep()),
    ("run_policy_compliance", lambda: LegalPolicyComplianceStep()),
    ("run_risk_analysis", lambda: LegalRiskAnalysisStep()),
    ("run_recommendations", lambda: LegalRecommendationStep()),
    ("run_decision", lambda: LegalDecisionStep()),
    ("run_enforcement", lambda: LegalDecisionEnforcementStep()),
)

_STAGE_FLAG_BY_TYPE: Dict[Type[Any], str] = {
    LegalExtractClausesStep: "run_extract",
    LegalNormalizeClausesStep: "run_normalize",
    LegalPolicyComplianceStep: "run_policy_compliance",
    LegalRiskAnalysisStep: "run_risk_analysis",
    LegalRecommendationStep: "run_recommendations",
    LegalDecisionStep: "run_decision",
    LegalDecisionEnforcementStep: "run_enforcement",
}


def _apply_dependency_closure(
    r: LegalRoutingResult,
    *,
    has_attachments: bool,
) -> LegalRoutingResult:
    """Ensure downstream stages have required predecessors."""
    d = r.model_copy()

    if has_attachments:
        d.run_extract = True

    if d.run_enforcement:
        d.run_decision = True
    if d.run_decision or d.run_recommendations:
        d.run_risk_analysis = True
    if d.run_risk_analysis or d.run_policy_compliance or d.run_normalize:
        d.run_extract = True
    if d.run_normalize:
        d.run_extract = True

    return d


def _default_full_routing() -> LegalRoutingResult:
    return LegalRoutingResult()


def legal_stage_flag_for_runner(runner: Any) -> Optional[str]:
    """Stage flag name for a pipeline runner instance, or None for finalize / unknown."""
    return _STAGE_FLAG_BY_TYPE.get(type(runner))


def merge_legal_routing_union(a: LegalRoutingResult, b: LegalRoutingResult) -> LegalRoutingResult:
    """Boolean OR per stage (server-side merge with replan output)."""
    da = a.model_dump()
    db = b.model_dump()
    keys = tuple(LegalRoutingResult.model_fields.keys())
    return LegalRoutingResult(**{k: bool(da.get(k) or db.get(k)) for k in keys})


def legal_routing_fingerprint(r: LegalRoutingResult) -> str:
    return json.dumps(r.model_dump(), sort_keys=True)


def build_legal_step_runners_from_routing(
    plan: LegalRoutingResult,
    *,
    include_finalize: bool = True,
) -> List[Any]:
    """Materialize step instances from a routing model (finalize optional for execution loops)."""
    data = plan.model_dump()
    runners: List[Any] = []
    for flag, factory in _LEGAL_STAGE_FACTORIES:
        if data.get(flag):
            runners.append(factory())
    if include_finalize:
        runners.append(LegalFinalizeAnswerStep())
    return runners


async def _llm_legal_routing(
    *,
    state: RuntimeState,
    system: str,
    user_content: str,
) -> LegalRoutingResult:
    llm = state.context.config.llm_adapter
    messages = [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=user_content),
    ]
    try:
        raw = llm.generate_structured(
            messages,
            LegalRoutingResult,
            run_id=state.run_id,
        )
        if isinstance(raw, LegalRoutingResult):
            return raw
    except Exception:
        pass
    return _default_full_routing()


async def obtain_initial_legal_routing(
    *,
    state: RuntimeState,
    config: LegalAgentConfig,
    agent_state: LegalAgentState,
) -> LegalRoutingResult:
    """First routing plan for the request (LLM or full default + dependency closure)."""
    req = state.request
    has_attachments = bool(req.attachments)
    snippet = build_legal_conversation_snippet(state, policy=config.memory_policy)
    metrics = legal_workspace_metrics_json(agent_state, runtime_state=state)

    if config.use_llm_legal_route_planner:
        plan = await _llm_legal_routing(
            state=state,
            system=LEGAL_PIPELINE_ROUTING_SYSTEM,
            user_content=legal_pipeline_routing_user(
                user_message=(req.message or "").strip(),
                has_attachments=has_attachments,
                conversation_snippet=snippet,
                workspace_metrics_json=metrics,
            ),
        )
    else:
        plan = _default_full_routing()

    return _apply_dependency_closure(plan, has_attachments=has_attachments)


async def obtain_replan_legal_routing(
    *,
    state: RuntimeState,
    config: LegalAgentConfig,
    agent_state: LegalAgentState,
    prior: LegalRoutingResult,
    iteration: int,
    evaluation_rationale: str,
    missing_aspects: List[str],
    workspace_metrics_json: str,
) -> LegalRoutingResult:
    """
    LLM replan merged with ``prior`` (union). Falls back to full routing union when LLM replanner off.
    """
    req = state.request
    has_attachments = bool(req.attachments)
    snippet = build_legal_conversation_snippet(state, policy=config.memory_policy)

    if config.use_llm_legal_route_planner and config.use_legal_route_replanner:
        proposed = await _llm_legal_routing(
            state=state,
            system=LEGAL_PIPELINE_REPLAN_SYSTEM,
            user_content=legal_pipeline_replan_user(
                user_message=(req.message or "").strip(),
                has_attachments=has_attachments,
                conversation_snippet=snippet,
                iteration=iteration,
                prior_routing_json=json.dumps(prior.model_dump(), sort_keys=True),
                stages_completed=", ".join(agent_state.legal_stages_completed_this_run or [])
                or "(none)",
                evaluation_rationale=evaluation_rationale or "[empty]",
                missing_aspects_json=json.dumps(missing_aspects, ensure_ascii=False),
                workspace_metrics_json=workspace_metrics_json,
            ),
        )
    else:
        proposed = _default_full_routing()

    merged = merge_legal_routing_union(prior, proposed)
    return _apply_dependency_closure(merged, has_attachments=has_attachments)


async def plan_legal_step_runners(
    *,
    state: RuntimeState,
    agent_state: LegalAgentState,
    config: LegalAgentConfig,
) -> List[Any]:
    """
    Returns ordered runtime step instances (including finalize) to execute.
    """
    plan = await obtain_initial_legal_routing(
        state=state,
        config=config,
        agent_state=agent_state,
    )
    runners = build_legal_step_runners_from_routing(plan, include_finalize=True)

    enabled = [flag for flag, _ in _LEGAL_STAGE_FACTORIES if plan.model_dump().get(flag)]
    state.trace_event(
        component=TraceComponent.PIPELINE,
        step="LegalPipelineRouting",
        message=f"Legal run plan: stages={enabled} + finalize",
        level=TraceLevel.INFO,
    )

    return runners
