# © Artur Czarnecki. All rights reserved.
# Intergrax platform proof — TOOLS-ITERATIVE-SQL-INVESTIGATION (PP-3C).

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ScenarioId(StrEnum):
    A = "A"
    B = "B"
    C = "C"


@dataclass(frozen=True, slots=True)
class InvestigationScenario:
    scenario_id: ScenarioId
    question: str


SCENARIO_A = InvestigationScenario(
    scenario_id=ScenarioId.A,
    question=(
        "North has a higher delay rate than other regions. Investigate what is driving "
        "the difference and identify the strongest supported operational explanation."
    ),
)

SCENARIO_B = InvestigationScenario(
    scenario_id=ScenarioId.B,
    question=(
        "Parcel weight appears associated with delays. Investigate whether heavier weight "
        "itself is the likely operational driver, or whether the association changes when "
        "you examine relevant observable segments in the available data. Do not infer "
        "causation from correlation alone."
    ),
)

SCENARIO_C = InvestigationScenario(
    scenario_id=ScenarioId.C,
    question="Are staffing shortages the reason for the delays?",
)

ALL_SCENARIOS: tuple[InvestigationScenario, ...] = (SCENARIO_A, SCENARIO_B, SCENARIO_C)

MAX_TOOL_ITERATIONS = 8
MAX_TOOL_CALLS_PER_ROUND = 2
