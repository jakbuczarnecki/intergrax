# © Artur Czarnecki. All rights reserved.

"""CVL-MAINT-01 — optional eval.trajectory_judge skill path."""

from __future__ import annotations

from intergrax.runtime.critic.trajectory_judge_path import (
    TrajectoryEvalMode,
    resolve_trajectory_eval_mode,
    trajectory_judge_skill_id,
)

DEFAULT_SKILL = "eval.trajectory_judge"
HEURISTIC_SKILL = "eval.trajectory"


def test_trajectory_judge_opt_in_returns_judge_skill() -> None:
    assert resolve_trajectory_eval_mode(use_judge=True) is TrajectoryEvalMode.JUDGE
    assert trajectory_judge_skill_id(use_judge=True) == DEFAULT_SKILL


def test_trajectory_heuristic_default() -> None:
    assert resolve_trajectory_eval_mode(use_judge=False) is TrajectoryEvalMode.HEURISTIC
    assert trajectory_judge_skill_id(use_judge=False) == HEURISTIC_SKILL
