# © Artur Czarnecki. All rights reserved.

"""Tier-0 eval tool client for L1 critic gateway (Phase CRIT-V-FOLLOWUP)."""

from __future__ import annotations

from intergrax.runtime.critic.eval_tool_client import CriticEvalToolClient
from intergrax.tools.providers.eval.contracts import EvalJudgeInput, EvalJudgeOutput, EvalTrajectoryInput, EvalTrajectoryOutput
from intergrax.tools.providers.eval.judge import eval_judge
from intergrax.tools.providers.eval.trajectory import eval_trajectory
from intergrax.tools.registry.wiring import ToolWiringContext


class ToolRegistryCriticEvalClient:
    """Invoke ``eval.judge`` / ``eval.trajectory`` via Tier-0 service functions."""

    def __init__(self, ctx: ToolWiringContext) -> None:
        self._ctx = ctx

    def judge(self, params: EvalJudgeInput) -> EvalJudgeOutput:
        return eval_judge(self._ctx, params)

    def trajectory(self, params: EvalTrajectoryInput) -> EvalTrajectoryOutput:
        return eval_trajectory(self._ctx, params)


def as_critic_eval_client(ctx: ToolWiringContext) -> CriticEvalToolClient:
    """Return a typed client backed by the supplied wiring context."""
    return ToolRegistryCriticEvalClient(ctx)
