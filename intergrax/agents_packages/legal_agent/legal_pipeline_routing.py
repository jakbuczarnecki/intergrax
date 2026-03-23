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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

from pydantic import BaseModel, Field

from intergrax.agents_packages.legal_agent.legal_agent_config import LegalAgentConfig
from intergrax.agents_packages.legal_agent.legal_agent_llm_prompts import (
    LEGAL_PIPELINE_REPLAN_SYSTEM,
    LEGAL_PIPELINE_ROUTING_SYSTEM,
    legal_pipeline_replan_user,
    legal_pipeline_routing_user,
)
from intergrax.agents_packages.legal_agent.legal_agent_state import LegalAgentState
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


def _history_snippet(state: RuntimeState, *, max_messages: int = 12) -> str:
    lines: List[str] = []
    msgs: Sequence[Any] = state.messages_for_llm or state.built_history_messages or []
    tail = list(msgs)[-max_messages:]
    for m in tail:
        role = getattr(m, "role", "?")
        content = getattr(m, "content", "") or ""
        text = content if isinstance(content, str) else str(content)
        lines.append(f"{role}: {text[:500]}")
    return "\n".join(lines) if lines else "(no prior turns in context)"


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
) -> LegalRoutingResult:
    """First routing plan for the request (LLM or full default + dependency closure)."""
    req = state.request
    has_attachments = bool(req.attachments)
    snippet = _history_snippet(state)

    if config.use_llm_legal_route_planner:
        plan = await _llm_legal_routing(
            state=state,
            system=LEGAL_PIPELINE_ROUTING_SYSTEM,
            user_content=legal_pipeline_routing_user(
                user_message=(req.message or "").strip(),
                has_attachments=has_attachments,
                conversation_snippet=snippet,
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
    snippet = _history_snippet(state)

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
    plan = await obtain_initial_legal_routing(state=state, config=config)
    runners = build_legal_step_runners_from_routing(plan, include_finalize=True)

    enabled = [flag for flag, _ in _LEGAL_STAGE_FACTORIES if plan.model_dump().get(flag)]
    state.trace_event(
        component=TraceComponent.PIPELINE,
        step="LegalPipelineRouting",
        message=f"Legal run plan: stages={enabled} + finalize",
        level=TraceLevel.INFO,
    )

    return runners
