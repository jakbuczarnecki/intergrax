# © Artur Czarnecki. All rights reserved.

"""LLM-as-judge catalog tool (Phase CRIT-V-2.1)."""

from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, Field

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.tools.providers.eval.contracts import EvalJudgeInput, EvalJudgeOutput
from intergrax.tools.providers.eval.service import _append_critic_observation
from intergrax.tools.registry.wiring import ToolWiringContext

EVAL_JUDGE_TOOL_ID = "eval.judge"


class _JudgeLLMResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    reasons: list[str] = Field(default_factory=list)


def _require_llm_adapter(ctx: ToolWiringContext) -> LLMAdapter:
    adapter = ctx.extras.get("llm_adapter")
    if adapter is None or not isinstance(adapter, LLMAdapter):
        raise RuntimeError("llm_adapter_not_configured")
    return adapter


def _build_judge_messages(params: EvalJudgeInput) -> list[ChatMessage]:
    criteria_block = "\n".join(f"- {item}" for item in params.criteria) or "- Output is correct and complete."
    reference = (params.reference_context or "").strip()
    reference_block = f"\nReference context:\n{reference}\n" if reference else ""
    user_content = (
        f"Rubric id: {params.rubric_id}\n"
        f"Pass threshold (minimum score): {params.min_score}\n"
        f"Criteria:\n{criteria_block}\n"
        f"{reference_block}\n"
        f"Candidate output:\n{params.output_text}\n\n"
        "Score the candidate from 0.0 to 1.0 against the criteria. "
        "Set passed=true only when score >= threshold."
    )
    return [
        ChatMessage(
            role="system",
            content=(
                "You are an evaluation judge. Return structured JSON only. "
                "Be strict, factual, and independent from the candidate author."
            ),
        ),
        ChatMessage(role="user", content=user_content),
    ]


def eval_judge(ctx: ToolWiringContext, params: EvalJudgeInput) -> EvalJudgeOutput:
    adapter = _require_llm_adapter(ctx)
    structured = adapter.generate_structured(
        _build_judge_messages(params),
        _JudgeLLMResult,
        temperature=0.0,
        run_id=params.run_id,
    )
    result = structured.parsed
    passed = result.passed and result.score >= params.min_score
    reasons = list(result.reasons)
    if result.score < params.min_score and not any("threshold" in item.lower() for item in reasons):
        reasons.append(f"score {result.score:.2f} below threshold {params.min_score:.2f}")

    observation_recorded = _append_critic_observation(
        ctx,
        record=params.record_observation,
        observation_id=params.observation_id or f"judge-{uuid4().hex[:12]}",
        run_id=params.run_id,
        agent_id=params.agent_id,
        scenario_id=params.scenario_id or f"critic.judge:{params.rubric_id}",
        mode=params.mode,
        passed=passed,
        score=result.score,
        candidate_profile_version_id=params.candidate_profile_version_id,
    )

    return EvalJudgeOutput(
        rubric_id=params.rubric_id,
        score=result.score,
        passed=passed,
        reasons=reasons,
        observation_recorded=observation_recorded,
    )
