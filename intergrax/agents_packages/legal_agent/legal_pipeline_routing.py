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

from typing import Any, Callable, List, Sequence, Tuple

from pydantic import BaseModel, Field

from intergrax.agents_packages.legal_agent.legal_agent_config import LegalAgentConfig
from intergrax.agents_packages.legal_agent.legal_agent_llm_prompts import (
    LEGAL_PIPELINE_ROUTING_SYSTEM,
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


async def plan_legal_step_runners(
    *,
    state: RuntimeState,
    agent_state: LegalAgentState,
    config: LegalAgentConfig,
) -> List[Any]:
    """
    Returns ordered runtime step instances (including finalize) to execute.
    """
    req = state.request
    has_attachments = bool(req.attachments)
    snippet = _history_snippet(state)

    if config.use_llm_legal_route_planner:
        llm = state.context.config.llm_adapter
        messages = [
            ChatMessage(role="system", content=LEGAL_PIPELINE_ROUTING_SYSTEM),
            ChatMessage(
                role="user",
                content=legal_pipeline_routing_user(
                    user_message=(req.message or "").strip(),
                    has_attachments=has_attachments,
                    conversation_snippet=snippet,
                ),
            ),
        ]

        try:
            raw = llm.generate_structured(
                messages,
                LegalRoutingResult,
                run_id=state.run_id,
            )
            if not isinstance(raw, LegalRoutingResult):
                raise TypeError("routing model type mismatch")
            plan = raw
        except Exception:
            plan = _default_full_routing()
    else:
        plan = _default_full_routing()

    plan = _apply_dependency_closure(plan, has_attachments=has_attachments)

    runners: List[Any] = []
    data = plan.model_dump()
    for flag, factory in _LEGAL_STAGE_FACTORIES:
        if data.get(flag):
            runners.append(factory())

    runners.append(LegalFinalizeAnswerStep())

    enabled = [flag for flag, _ in _LEGAL_STAGE_FACTORIES if data.get(flag)]
    state.trace_event(
        component=TraceComponent.PIPELINE,
        step="LegalPipelineRouting",
        message=f"Legal run plan: stages={enabled} + finalize",
        level=TraceLevel.INFO,
    )

    return runners
