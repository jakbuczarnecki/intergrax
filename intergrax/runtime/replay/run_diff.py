# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations
from dataclasses import dataclass
from typing import List

from intergrax.runtime.replay.models import ReconstructedRun


@dataclass(slots=True)
class RunDiff:
    step_count_changed: bool
    llm_call_delta: int
    tool_call_delta: int
    artifact_delta: int
    final_answer_changed: bool
    step_type_changes: List[str]


class RunDiffEngine:
    """
    Compares two reconstructed runs on execution semantics level.
    """

    def diff(self, a: ReconstructedRun, b: ReconstructedRun) -> RunDiff:
        step_types_a = [s.step_type for s in a.steps]
        step_types_b = [s.step_type for s in b.steps]

        return RunDiff(
            step_count_changed=len(a.steps) != len(b.steps),
            llm_call_delta=len(b.llm_calls) - len(a.llm_calls),
            tool_call_delta=len(b.tool_calls) - len(a.tool_calls),
            artifact_delta=len(b.artifacts) - len(a.artifacts),
            final_answer_changed=a.final_answer != b.final_answer,
            step_type_changes=[
                f"{i}: {sa} -> {sb}"
                for i, (sa, sb) in enumerate(zip(step_types_a, step_types_b))
                if sa != sb
            ],
        )
