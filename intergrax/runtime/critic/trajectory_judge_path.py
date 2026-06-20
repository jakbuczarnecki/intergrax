# © Artur Czarnecki. All rights reserved.

"""Optional trajectory judge eval path (CVL-MAINT-01)."""

from __future__ import annotations

from enum import Enum


class TrajectoryEvalMode(str, Enum):
    HEURISTIC = "heuristic"
    JUDGE = "judge"


def resolve_trajectory_eval_mode(*, use_judge: bool = False) -> TrajectoryEvalMode:
    return TrajectoryEvalMode.JUDGE if use_judge else TrajectoryEvalMode.HEURISTIC


def trajectory_judge_skill_id(*, use_judge: bool = False) -> str:
    if use_judge:
        return "eval.trajectory_judge"
    return "eval.trajectory"
