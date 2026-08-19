# © Artur Czarnecki. All rights reserved.

"""Typed proof result contracts for the governed hybrid knowledge flagship proof."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SemanticDecisionV1(StrEnum):
    YES = "YES"
    NO = "NO"
    CANNOT_DETERMINE = "CANNOT DETERMINE"


class FlagshipProofScenarioResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(..., min_length=1)
    passed: bool
    expected: SemanticDecisionV1 | str
    observed: SemanticDecisionV1 | str
    http_read_count: int | None = None
    llm_call_count: int | None = None
    admissibility: str | None = None
    ask_status: str | None = None
    run_id: str | None = None
    key_metrics: dict[str, Any] = Field(default_factory=dict)


class FlagshipProofResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_1: FlagshipProofScenarioResultV1
    scenario_2: FlagshipProofScenarioResultV1
    scenario_3: FlagshipProofScenarioResultV1
    scenario_4: FlagshipProofScenarioResultV1
    overall_status: str
    scenario_1_run_id: str | None = None

    @property
    def all_passed(self) -> bool:
        return all(
            scenario.passed
            for scenario in (
                self.scenario_1,
                self.scenario_2,
                self.scenario_3,
                self.scenario_4,
            )
        )

    @property
    def passed_count(self) -> int:
        return sum(
            1
            for scenario in (
                self.scenario_1,
                self.scenario_2,
                self.scenario_3,
                self.scenario_4,
            )
            if scenario.passed
        )
