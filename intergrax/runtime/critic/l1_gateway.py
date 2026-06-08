# © Artur Czarnecki. All rights reserved.

"""L1 semantic / trajectory critic gateway — Phase CRIT-V-3.3."""

from __future__ import annotations

from intergrax.runtime.critic.contracts import CriticLayer, CriticRequest, LayerVerdict, RubricSpec
from intergrax.runtime.critic.eval_tool_client import CriticEvalToolClient
from intergrax.tools.providers.eval.contracts import EvalJudgeInput, EvalTrajectoryInput


class L1Gateway:
    """Invokes ``eval.judge`` and ``eval.trajectory`` via a typed Tier-0 client."""

    def __init__(self, *, tool_client: CriticEvalToolClient | None = None) -> None:
        self._tool_client = tool_client

    def verify_semantic(self, request: CriticRequest) -> LayerVerdict:
        if self._tool_client is None:
            return LayerVerdict(
                layer=CriticLayer.L1_SEMANTIC,
                passed=False,
                errors=["L1 semantic tool client not configured"],
            )
        rubric = request.rubric
        if rubric is None:
            return LayerVerdict(
                layer=CriticLayer.L1_SEMANTIC,
                passed=False,
                errors=["missing rubric for L1 semantic verification"],
            )
        output_text = _resolve_output_text(request)
        if not output_text.strip():
            return LayerVerdict(
                layer=CriticLayer.L1_SEMANTIC,
                passed=False,
                errors=["empty output text for L1 semantic verification"],
            )

        result = self._tool_client.judge(
            EvalJudgeInput(
                output_text=output_text,
                rubric_id=rubric.rubric_id,
                criteria=list(rubric.criteria),
                reference_context=rubric.reference_context,
                min_score=rubric.min_score,
                run_id=request.run_id,
                agent_id=request.agent_id,
            ),
        )
        return LayerVerdict(
            layer=CriticLayer.L1_SEMANTIC,
            passed=result.passed,
            score=result.score,
            errors=[] if result.passed else list(result.reasons),
            warnings=[] if result.passed else [],
        )

    def verify_trajectory(self, request: CriticRequest) -> LayerVerdict:
        if self._tool_client is None:
            return LayerVerdict(
                layer=CriticLayer.L1_TRAJECTORY,
                passed=False,
                errors=["L1 trajectory tool client not configured"],
            )
        tenant_id = str(request.context.get("tenant_id") or "default")
        min_score = float(request.context.get("trajectory_min_score", 0.75))

        result = self._tool_client.trajectory(
            EvalTrajectoryInput(
                run_id=request.run_id,
                tenant_id=tenant_id,
                min_score=min_score,
                agent_id=request.agent_id,
            ),
        )
        return LayerVerdict(
            layer=CriticLayer.L1_TRAJECTORY,
            passed=result.passed,
            score=result.score,
            errors=[] if result.passed else list(result.reasons),
        )


def _resolve_output_text(request: CriticRequest) -> str:
    if request.execution is not None and (request.execution.summary or "").strip():
        return request.execution.summary
    if request.answer is not None and (request.answer.answer or "").strip():
        return request.answer.answer
    return ""
