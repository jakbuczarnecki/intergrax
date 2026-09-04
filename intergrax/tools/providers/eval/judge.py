# © Artur Czarnecki. All rights reserved.

"""LLM-as-judge catalog tool."""

from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, Field

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.tools.providers.eval.contracts import EvalJudgeInput, EvalJudgeOutput
from intergrax.tools.providers.eval.judge_messages import build_eval_judge_messages_from_input
from intergrax.tools.providers.eval.service import _append_eval_observation
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


def eval_judge(ctx: ToolWiringContext, params: EvalJudgeInput) -> EvalJudgeOutput:
    adapter = _require_llm_adapter(ctx)
    structured = adapter.generate_structured(
        build_eval_judge_messages_from_input(params),
        _JudgeLLMResult,
        temperature=0.0,
        run_id=params.run_id,
    )
    result = structured.parsed
    passed = result.passed and result.score >= params.min_score
    reasons = list(result.reasons)
    if result.score < params.min_score and not any("threshold" in item.lower() for item in reasons):
        reasons.append(f"score {result.score:.2f} below threshold {params.min_score:.2f}")

    observation_recorded = _append_eval_observation(
        ctx,
        record=params.record_observation,
        observation_id=params.observation_id or f"judge-{uuid4().hex[:12]}",
        run_id=params.run_id,
        agent_id=params.agent_id,
        scenario_id=params.scenario_id or f"eval.judge:{params.rubric_id}",
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
