# © Artur Czarnecki. All rights reserved.

"""Typed client for L1 critic tools (eval.judge / eval.trajectory) — Phase CRIT-V-3.3."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.tools.providers.eval.contracts import EvalJudgeInput, EvalJudgeOutput, EvalTrajectoryInput, EvalTrajectoryOutput


@runtime_checkable
class CriticEvalToolClient(Protocol):
    """Invoke Tier-0 eval critic tools without direct LLM access in Tier-1."""

    def judge(self, params: EvalJudgeInput) -> EvalJudgeOutput: ...

    def trajectory(self, params: EvalTrajectoryInput) -> EvalTrajectoryOutput: ...
