# © Artur Czarnecki. All rights reserved.

"""Typed controlled-alignment scenario contract (DS-E2E-15J-L1.R4.R4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from platform_proofs.scenarios.ai_incident_investigation.application.completion_alignment_correction import (
    CompletionAlignmentDirection,
)


@dataclass(frozen=True, slots=True)
class ControlledAlignmentScenario:
    scenario_id: str
    expected_direction: CompletionAlignmentDirection
    expected_correctable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "expected_direction": self.expected_direction.value,
            "expected_correctable": self.expected_correctable,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ControlledAlignmentScenario:
        direction_raw = payload.get("expected_direction")
        correctable = payload.get("expected_correctable")
        scenario_id = payload.get("scenario_id")
        if not isinstance(scenario_id, str):
            raise ValueError("scenario_id must be a string")
        if not isinstance(direction_raw, str):
            raise ValueError("expected_direction must be a string")
        if not isinstance(correctable, bool):
            raise ValueError("expected_correctable must be a bool")
        return cls(
            scenario_id=scenario_id,
            expected_direction=CompletionAlignmentDirection(direction_raw),
            expected_correctable=correctable,
        )


MODEL_OVERCOMMIT_SCENARIO = ControlledAlignmentScenario(
    scenario_id="model_overcommit_supported_diagnosis_without_state",
    expected_direction=CompletionAlignmentDirection.MODEL_OVERCOMMIT,
    expected_correctable=True,
)

__all__ = [
    "ControlledAlignmentScenario",
    "MODEL_OVERCOMMIT_SCENARIO",
]
